"""
Joint state-action diffusion denoiser for MPE.

Architecture: Residual MLP with FiLM conditioning.
Training: Warm-start separated DDPM objective.
Inference: Sliding-window buffer + truncated reverse diffusion.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collect_joint_data import extract_landmarks, get_role_encoding


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def make_beta_schedule(T, beta_start=1e-4, beta_end=2e-2):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_bar


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Anchor helpers
# ---------------------------------------------------------------------------

def select_anchor_data(anchor_type, init_obs, landmarks, roles):
    """
    Select anchor signal from collected arrays.

    Args:
        anchor_type: "none" | "init_obs" | "landmarks" | "roles" | "landmarks+roles"
        init_obs:   [N, Ds]
        landmarks:  [N, D_land]
        roles:      [N, n_agents]

    Returns:
        [N, anchor_dim] numpy array, or None for "none".
    """
    if anchor_type == "none":
        return None
    elif anchor_type == "init_obs":
        return init_obs
    elif anchor_type == "landmarks":
        return landmarks
    elif anchor_type == "roles":
        return roles
    elif anchor_type == "landmarks+roles":
        return np.concatenate([landmarks, roles], axis=-1)
    else:
        raise ValueError("Unknown anchor_type: {}".format(anchor_type))


def build_anchor_at_reset(obs_n, env, num_adversaries, anchor_type):
    """
    Build the clean anchor signal at episode reset (before any corruption).

    Returns: float32 numpy array [anchor_dim], or None.
    """
    if anchor_type == "none":
        return None
    elif anchor_type == "init_obs":
        return np.concatenate(obs_n, axis=0).astype(np.float32)
    elif anchor_type == "landmarks":
        return extract_landmarks(env, None).astype(np.float32)
    elif anchor_type == "roles":
        return get_role_encoding(env, num_adversaries).astype(np.float32)
    elif anchor_type == "landmarks+roles":
        lm = extract_landmarks(env, None).astype(np.float32)
        rl = get_role_encoding(env, num_adversaries).astype(np.float32)
        return np.concatenate([lm, rl])
    else:
        return None


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """t: [B] integer timesteps → [B, dim]"""
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device).float() / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class FiLMResidualBlock(nn.Module):
    """Residual block with FiLM conditioning."""

    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.film_proj = nn.Linear(cond_dim, 2 * hidden_dim)
        # Zero-init: block starts as identity through residual
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

    def forward(self, x, cond):
        """x: [B, hidden_dim], cond: [B, cond_dim]"""
        residual = x
        h = self.linear1(x)
        h = self.norm(h)
        film_params = self.film_proj(cond)
        gamma, beta = film_params.chunk(2, dim=-1)
        h = (1.0 + gamma) * h + beta
        h = F.relu(h)
        h = self.linear2(h)
        return residual + h


class AnchorEncoder(nn.Module):
    """2-layer MLP that projects anchor signal to embedding."""

    def __init__(self, input_dim, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, xi):
        return self.net(xi)


class JointDiffusionDenoiser(nn.Module):
    """
    Joint state-action diffusion denoiser for MPE.

    Input:  [B, H, Ds+Da] noisy joint window
    Output: [B, H, Ds+Da] predicted noise

    Args:
        horizon:         temporal window length H
        state_dim:       joint state dimension Ds
        action_dim:      joint action dimension Da
        anchor_dim:      anchor signal dimension (0 = no anchor)
        hidden_dim:      width of residual blocks
        n_blocks:        number of FiLM residual blocks
        anchor_embed_dim: anchor encoder output dimension
        time_embed_dim:  sinusoidal time embedding dimension
    """

    def __init__(self, horizon, state_dim, action_dim,
                 anchor_dim=0, hidden_dim=256, n_blocks=4,
                 anchor_embed_dim=128, time_embed_dim=128):
        super().__init__()
        self.horizon = horizon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.joint_dim = state_dim + action_dim
        self.input_dim = horizon * self.joint_dim
        self.anchor_dim = anchor_dim

        cond_dim = hidden_dim

        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_proj = nn.Linear(time_embed_dim, cond_dim)

        if anchor_dim > 0:
            self.anchor_enc = AnchorEncoder(anchor_dim, anchor_embed_dim)
            self.anchor_proj = nn.Linear(anchor_embed_dim, cond_dim)
        else:
            self.anchor_enc = None
            self.anchor_proj = None

        self.input_proj = nn.Linear(self.input_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            FiLMResidualBlock(hidden_dim, cond_dim)
            for _ in range(n_blocks)
        ])

        self.output_proj = nn.Linear(hidden_dim, self.input_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x_noisy, t, anchor=None):
        """
        x_noisy: [B, H, Ds+Da]
        t:       [B] integer timesteps
        anchor:  [B, anchor_dim] or None

        Returns: [B, H, Ds+Da] predicted noise
        """
        B = x_noisy.shape[0]
        x_flat = x_noisy.reshape(B, -1)

        t_emb = self.time_embed(t)
        cond = self.time_proj(t_emb)

        if self.anchor_enc is not None and anchor is not None:
            a_emb = self.anchor_enc(anchor)
            a_proj = self.anchor_proj(a_emb)
            cond = cond + a_proj

        h = self.input_proj(x_flat)
        for block in self.blocks:
            h = block(h, cond)

        eps_pred = self.output_proj(h)
        return eps_pred.reshape(B, self.horizon, self.joint_dim)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_joint_denoiser(arglist):
    """
    Train the joint state-action diffusion denoiser.

    Warm-start separated training objective:
    - Apply synthetic corruption to clean data → x0_corrupted
    - Forward-diffuse from x0_corrupted at step k → x_k
    - Supervise against target computed relative to x0_CLEAN
    """
    data = np.load(arglist.joint_diffusion_data_path, allow_pickle=True)
    states = data["states"]       # [N, H, Ds]
    actions = data["actions"]     # [N, H, Da]
    init_obs = data["init_obs"]   # [N, Ds]
    landmarks = data["landmarks"] # [N, D_land]
    roles = data["roles"]         # [N, n_agents]

    N, H, Ds = states.shape
    _, _, Da = actions.shape

    joint_clean = np.concatenate([states, actions], axis=-1).astype(np.float32)  # [N, H, Ds+Da]
    joint_clean_t = torch.from_numpy(joint_clean)

    mean = joint_clean_t.mean(dim=(0, 1), keepdim=True)   # [1, 1, Ds+Da]
    std = joint_clean_t.std(dim=(0, 1), keepdim=True) + 1e-6
    joint_norm = (joint_clean_t - mean) / std

    anchor_type = arglist.anchor_type
    anchor_data = select_anchor_data(anchor_type, init_obs, landmarks, roles)
    anchor_dim = 0 if anchor_data is None else anchor_data.shape[1]
    anchor_t = torch.from_numpy(anchor_data).float() if anchor_data is not None else None

    device = torch.device("cpu")
    model = JointDiffusionDenoiser(
        horizon=H,
        state_dim=Ds,
        action_dim=Da,
        anchor_dim=anchor_dim,
        hidden_dim=arglist.denoiser_hidden_dim,
        n_blocks=arglist.denoiser_n_blocks,
    ).to(device)

    print("[JointDenoiser] params: {:,}".format(count_params(model)))
    print("[JointDenoiser] Training on {} windows, H={}, Ds={}, Da={}, anchor={}".format(
        N, H, Ds, Da, anchor_type))

    K = arglist.diffusion_steps
    betas, alphas, alphas_bar = make_beta_schedule(K)
    alphas_bar = alphas_bar.to(device)
    alphas = alphas.to(device)

    corruption_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0]
    lam = arglist.channel_weight_lambda

    opt = torch.optim.AdamW(model.parameters(), lr=arglist.diffusion_lr, weight_decay=1e-4)
    batch_size = arglist.diffusion_batch_size

    std_dev = std.to(device)   # [1, 1, Ds+Da]

    for epoch in range(arglist.diffusion_epochs):
        perm = torch.randperm(N)
        epoch_loss = epoch_loss_a = epoch_loss_s = 0.0

        for b_start in range(0, N, batch_size):
            b_end = min(N, b_start + batch_size)
            idx = perm[b_start:b_end]
            B = len(idx)

            x0_clean = joint_norm[idx].to(device)   # [B, H, Ds+Da]

            # Sample per-sample corruption levels
            alpha_s = torch.tensor(
                np.random.choice(corruption_levels, size=B), dtype=torch.float32
            ).to(device)
            alpha_a = torch.tensor(
                np.random.choice(corruption_levels, size=B), dtype=torch.float32
            ).to(device)

            # Build corruption in normalized space
            noise_s = alpha_s.view(B, 1, 1) * torch.randn(B, H, Ds, device=device)
            noise_s = noise_s / std_dev[:, :, :Ds]
            noise_a = alpha_a.view(B, 1, 1) * torch.randn(B, H, Da, device=device)
            noise_a = noise_a / std_dev[:, :, Ds:]

            corruption = torch.cat([noise_s, noise_a], dim=-1)
            x0_corrupted = x0_clean + corruption

            # Forward diffusion from corrupted starting point
            k = torch.randint(1, K, (B,), device=device)
            eps = torch.randn_like(x0_corrupted)
            a_bar_k = alphas_bar[k].view(B, 1, 1)
            x_k = torch.sqrt(a_bar_k) * x0_corrupted + torch.sqrt(1.0 - a_bar_k) * eps

            # Warm-start separated target (relative to CLEAN x0)
            eps_star = (x_k - torch.sqrt(a_bar_k) * x0_clean) / torch.sqrt(1.0 - a_bar_k)

            anchor_batch = None
            if anchor_t is not None:
                anchor_batch = anchor_t[idx].to(device)

            eps_pred = model(x_k, k, anchor_batch)

            loss_a = F.mse_loss(eps_pred[:, :, Ds:], eps_star[:, :, Ds:])
            loss_s = F.mse_loss(eps_pred[:, :, :Ds], eps_star[:, :, :Ds])
            loss = loss_a + lam * loss_s

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            epoch_loss += loss.item() * B
            epoch_loss_a += loss_a.item() * B
            epoch_loss_s += loss_s.item() * B

        epoch_loss /= N
        epoch_loss_a /= N
        epoch_loss_s /= N

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print("[JointDenoiser] Epoch {}/{} — total: {:.6f}  action: {:.6f}  state: {:.6f}".format(
                epoch + 1, arglist.diffusion_epochs, epoch_loss, epoch_loss_a, epoch_loss_s))

    save_dict = {
        "model_state_dict": model.state_dict(),
        "horizon": H,
        "state_dim": Ds,
        "action_dim": Da,
        "anchor_dim": anchor_dim,
        "anchor_type": anchor_type,
        "hidden_dim": arglist.denoiser_hidden_dim,
        "n_blocks": arglist.denoiser_n_blocks,
        "diffusion_steps": K,
        "mean": mean,
        "std": std,
        "channel_weight_lambda": lam,
    }
    model_path = arglist.joint_denoiser_model_path
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    torch.save(save_dict, model_path)
    print("[JointDenoiser] Saved to {}".format(model_path))


# ---------------------------------------------------------------------------
# Inference globals and buffer
# ---------------------------------------------------------------------------

JOINT_DENOISER = None
JOINT_DENOISER_CONSTS = {}


def load_joint_denoiser(arglist):
    """Load trained joint denoiser model and set module-level globals."""
    global JOINT_DENOISER, JOINT_DENOISER_CONSTS

    ckpt = torch.load(arglist.joint_denoiser_model_path, map_location="cpu")

    model = JointDiffusionDenoiser(
        horizon=ckpt["horizon"],
        state_dim=ckpt["state_dim"],
        action_dim=ckpt["action_dim"],
        anchor_dim=ckpt["anchor_dim"],
        hidden_dim=ckpt["hidden_dim"],
        n_blocks=ckpt["n_blocks"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    K = ckpt["diffusion_steps"]
    betas, alphas, alphas_bar = make_beta_schedule(K)

    JOINT_DENOISER = model
    JOINT_DENOISER_CONSTS = {
        "betas": betas,
        "alphas": alphas,
        "alphas_bar": alphas_bar,
        "mean": ckpt["mean"],
        "std": ckpt["std"],
        "K": K,
        "H": ckpt["horizon"],
        "Ds": ckpt["state_dim"],
        "Da": ckpt["action_dim"],
        "anchor_type": ckpt["anchor_type"],
    }
    print("[JointDenoiser] Loaded model (anchor={}, H={}, Ds={}, Da={})".format(
        ckpt["anchor_type"], ckpt["horizon"], ckpt["state_dim"], ckpt["action_dim"]))


class JointDenoiserBuffer:
    """
    Sliding window buffer that accumulates (state, action) pairs during an
    episode and returns the joint window needed for denoising.
    """

    def __init__(self, H, Ds, Da):
        self.H = H
        self.Ds = Ds
        self.Da = Da
        self.states = []
        self.actions = []
        self.anchor = None

    def reset(self, anchor_vec=None):
        self.states = []
        self.actions = []
        self.anchor = anchor_vec

    def push(self, state_vec, action_vec):
        self.states.append(state_vec.copy())
        self.actions.append(action_vec.copy())

    def get_window(self):
        """Returns [1, H, Ds+Da] tensor, or None if buffer is empty."""
        T = len(self.states)
        if T == 0:
            return None

        s_arr = np.array(self.states, dtype=np.float32)   # [T, Ds]
        a_arr = np.array(self.actions, dtype=np.float32)  # [T, Da]

        if T >= self.H:
            s_win = s_arr[-self.H:]
            a_win = a_arr[-self.H:]
        else:
            pad = self.H - T
            s_win = np.concatenate([np.tile(s_arr[0:1], (pad, 1)), s_arr], axis=0)
            a_win = np.concatenate([np.tile(a_arr[0:1], (pad, 1)), a_arr], axis=0)

        joint = np.concatenate([s_win, a_win], axis=-1)   # [H, Ds+Da]
        return torch.from_numpy(joint).float().unsqueeze(0)  # [1, H, Ds+Da]


@torch.no_grad()
def joint_denoise_action(buffer, k_start=10):
    """
    Denoise the current action using the joint state-action denoiser.

    Args:
        buffer:  JointDenoiserBuffer with current episode data (corrupted)
        k_start: warm-start truncation timestep

    Returns:
        clean_action: [Da] numpy array, or None if buffer empty.
    """
    model = JOINT_DENOISER
    C = JOINT_DENOISER_CONSTS
    Ds = C["Ds"]

    x_tilde = buffer.get_window()
    if x_tilde is None:
        return None

    x_tilde = (x_tilde - C["mean"]) / C["std"]

    anchor = None
    if buffer.anchor is not None:
        anchor = torch.from_numpy(buffer.anchor).float().unsqueeze(0)

    x = x_tilde.clone()
    alphas = C["alphas"]
    alphas_bar = C["alphas_bar"]

    # Forward-diffuse to k_start so input matches training distribution x_k
    a_bar_k = alphas_bar[k_start]
    x = torch.sqrt(a_bar_k) * x + torch.sqrt(1.0 - a_bar_k) * torch.randn_like(x)

    for k in reversed(range(k_start + 1)):
        k_tensor = torch.tensor([k])
        eps_pred = model(x, k_tensor, anchor)

        alpha_bar_k = alphas_bar[k]
        alpha_k = alphas[k]

        x0_hat = (x - torch.sqrt(1.0 - alpha_bar_k) * eps_pred) / torch.sqrt(alpha_bar_k)

        if k > 0:
            z = torch.randn_like(x)
            x = torch.sqrt(alpha_k) * x0_hat + torch.sqrt(1.0 - alpha_k) * z
        else:
            x = x0_hat

    x_denorm = x * C["std"] + C["mean"]
    clean_action = x_denorm[0, -1, Ds:].numpy()  # last timestep, action portion
    return clean_action


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

def _split_actions(action_vec, n_agents, action_dim_per_agent):
    split = []
    start = 0
    for dim in action_dim_per_agent:
        split.append(action_vec[start:start + dim])
        start += dim
    return split


def testRobustnessAP_joint(arglist, use_denoiser=True, k_start=10, obs_noise_std=0.0):
    """
    Evaluate the policy under action (and optionally observation) noise.
    Optionally denoises using the joint denoiser.

    The joint denoiser must already be loaded via load_joint_denoiser()
    before calling this function.

    Returns: mean total reward over all test episodes.
    """
    import tensorflow as tf
    import maddpg.common.tf_util as U
    from train import make_env, get_trainers, resolve_checkpoint, apply_action_disruption

    tf.reset_default_graph()
    with U.single_threaded_session():
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

        U.initialize()
        load_dir, exp_name = resolve_checkpoint(arglist, None, None)
        U.load_state(load_dir, exp_name=exp_name)

        C = JOINT_DENOISER_CONSTS if use_denoiser else {}
        Ds = C.get("Ds", 0)
        Da = C.get("Da", 0)
        H = C.get("H", arglist.diffusion_horizon)
        anchor_type = C.get("anchor_type", "none")

        n_episodes = arglist.num_test_episodes
        max_episode_len = arglist.max_episode_len
        all_rewards = []

        env.llm_disturb_iteration = 0
        env.previous_reward = 0

        for ep in range(n_episodes):
            obs_n = env.reset()
            episode_reward = np.zeros(env.n)

            if use_denoiser:
                anchor_vec = build_anchor_at_reset(
                    obs_n, env, num_adversaries, anchor_type
                )
                if getattr(arglist, "corrupt_anchor_std", 0.0) > 0.0 and anchor_vec is not None:
                    anchor_vec = anchor_vec + np.random.normal(
                        0, arglist.corrupt_anchor_std, size=anchor_vec.shape
                    ).astype(np.float32)
                buffer = JointDenoiserBuffer(H, Ds, Da)
                buffer.reset(anchor_vec)

            for step in range(max_episode_len):
                # Optionally corrupt observations before action selection
                if obs_noise_std > 0.0:
                    obs_n_in = [
                        obs + np.random.normal(0, obs_noise_std, size=obs.shape).astype(np.float32)
                        for obs in obs_n
                    ]
                else:
                    obs_n_in = obs_n

                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n_in)]
                n_agents = len(action_n)
                action_dim_per_agent = [len(a) for a in action_n]

                action_n_noisy = [
                    apply_action_disruption(a, 0, env, arglist)
                    for a in action_n
                ]

                action_n_out = action_n_noisy

                if use_denoiser:
                    state_vec = np.concatenate(obs_n_in, axis=0).astype(np.float32)
                    action_vec_noisy = np.concatenate(action_n_noisy, axis=0).astype(np.float32)
                    buffer.push(state_vec, action_vec_noisy)

                    action_vec_clean = joint_denoise_action(buffer, k_start=k_start)
                    if action_vec_clean is not None:
                        # Fix 1: restore adversary action slices — in simple_tag,
                        # noisy adversary actions make them less effective (higher reward),
                        # so denoising adversaries back to clean hurts the metric.
                        adv_dims = sum(action_dim_per_agent[:num_adversaries])
                        action_vec_clean[:adv_dims] = action_vec_noisy[:adv_dims]

                        # Fix 3: store denoised action in buffer so next step's window
                        # reflects what the environment actually received.
                        buffer.actions[-1] = action_vec_clean.copy()

                        action_n_out = _split_actions(
                            action_vec_clean, n_agents, action_dim_per_agent
                        )

                new_obs_n, rew_n, done_n, _ = env.step(action_n_out)
                episode_reward += rew_n
                obs_n = new_obs_n

                if arglist.display:
                    import time
                    env.render()
                    time.sleep(0.05)

                if all(done_n):
                    break

            all_rewards.append(episode_reward)

        mean_rewards = np.mean(all_rewards, axis=0)
        print("Average reward per agent over {} episodes: {}".format(n_episodes, mean_rewards))
        print("Average total reward: {:.3f}".format(np.mean(np.sum(all_rewards, axis=1))))
        return np.mean(np.sum(all_rewards, axis=1))
