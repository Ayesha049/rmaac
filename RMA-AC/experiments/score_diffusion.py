"""
Score-Based Diffusion Denoiser for MPE joint state-action.

Phase 1: Score-only correction (lam_q=0).
Phase 2: Score + MADDPG critic guidance (lam_q > 0).

Reference design:
    Train:  s_theta(x_tilde, sigma) = -eps/sigma   (DSM, sigma ~ LogUniform)
    Infer:  a_hat = a_noisy + eta * s_theta(a_noisy, sigma_est, anchor)
            Optionally: + lambda_Q * grad_a Q(s, a)  [Phase 2]

Key differences from JointDiffusionDenoiser (DDPM):
    - Continuous noise level sigma (not discrete timestep T).
    - LogSigmaEmbedding instead of SinusoidalTimeEmbedding.
    - Output is the score, not the predicted noise epsilon.
    - Inference is 1-3 Langevin steps; no iterative reverse process.
    - Self-calibrating: large score magnitude = far from clean manifold.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from joint_diffusion import (
    AnchorEncoder,
    FiLMResidualBlock,
    JointDenoiserBuffer,
    build_anchor_at_reset,
    count_params,
    select_anchor_data,
)


# ---------------------------------------------------------------------------
# Continuous sigma embedding
# ---------------------------------------------------------------------------

class LogSigmaEmbedding(nn.Module):
    """
    Maps log(sigma) -> dim-d embedding via a 2-layer MLP.

    Unlike the sinusoidal timestep embedding, this handles continuous sigma
    and generalises across the entire LogUniform training range without
    needing a discrete index.
    """

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, sigma):
        """sigma: [B] float -> [B, dim]"""
        log_s = torch.log(sigma.clamp(min=1e-6)).unsqueeze(-1)  # [B, 1]
        return self.net(log_s)


# ---------------------------------------------------------------------------
# Score network
# ---------------------------------------------------------------------------

class ScoreNetwork(nn.Module):
    """
    Score network s_theta(x_noisy, sigma, anchor) ~ grad_x log p_sigma(x).

    Identical FiLM residual architecture to JointDiffusionDenoiser but:
      - conditioned on continuous sigma via LogSigmaEmbedding, and
      - output is the score (target -eps/sigma), not the noise eps.

    Args:
        horizon:          temporal window length H
        state_dim:        joint state dimension Ds
        action_dim:       joint action dimension Da
        anchor_dim:       anchor signal dimension (0 = no anchor)
        hidden_dim:       FiLM block width
        n_blocks:         number of FiLM residual blocks
        anchor_embed_dim: anchor encoder output dimension
        sigma_embed_dim:  log-sigma embedding dimension
    """

    def __init__(self, horizon, state_dim, action_dim,
                 anchor_dim=0, hidden_dim=256, n_blocks=4,
                 anchor_embed_dim=128, sigma_embed_dim=128):
        super().__init__()
        self.horizon = horizon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.joint_dim = state_dim + action_dim
        self.input_dim = horizon * self.joint_dim
        self.anchor_dim = anchor_dim

        cond_dim = hidden_dim

        self.sigma_embed = LogSigmaEmbedding(sigma_embed_dim)
        self.sigma_proj = nn.Linear(sigma_embed_dim, cond_dim)

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

    def forward(self, x_noisy, sigma, anchor=None):
        """
        x_noisy: [B, H, Ds+Da]  (normalized to training distribution)
        sigma:   [B]  float noise levels
        anchor:  [B, anchor_dim] or None

        Returns: [B, H, Ds+Da] predicted score ~ -eps/sigma
        """
        B = x_noisy.shape[0]
        x_flat = x_noisy.reshape(B, -1)

        s_emb = self.sigma_embed(sigma)
        cond = self.sigma_proj(s_emb)

        if self.anchor_enc is not None and anchor is not None:
            a_emb = self.anchor_enc(anchor)
            cond = cond + self.anchor_proj(a_emb)

        h = self.input_proj(x_flat)
        for block in self.blocks:
            h = block(h, cond)

        score = self.output_proj(h)
        return score.reshape(B, self.horizon, self.joint_dim)


# ---------------------------------------------------------------------------
# Module-level globals (loaded once per process)
# ---------------------------------------------------------------------------

SCORE_NET = None
SCORE_NET_CONSTS = {}


# ---------------------------------------------------------------------------
# Training: Denoising Score Matching
# ---------------------------------------------------------------------------

def train_score_network(arglist):
    """
    Train the score network with Denoising Score Matching (DSM).

    For each clean sample x0 (normalised):
      - Draw sigma ~ LogUniform(sigma_min, sigma_max).
      - Perturb: x_tilde = x0 + sigma * eps,  eps ~ N(0, I).
      - Train s_theta(x_tilde, sigma) to predict -eps/sigma.
      - Loss: ||s_theta(x_tilde, sigma) + eps/sigma||^2
              with action channels weighted 1.0 and state channels lambda.

    Saves checkpoint to arglist.score_model_path.
    """
    data = np.load(arglist.joint_diffusion_data_path, allow_pickle=True)
    states = data["states"]       # [N, H, Ds]
    actions = data["actions"]     # [N, H, Da]
    init_obs = data["init_obs"]   # [N, Ds]
    landmarks = data["landmarks"] # [N, D_land]
    roles = data["roles"]         # [N, n_agents]

    N, H, Ds = states.shape
    _, _, Da = actions.shape

    joint_clean = np.concatenate([states, actions], axis=-1).astype(np.float32)
    joint_clean_t = torch.from_numpy(joint_clean)

    mean = joint_clean_t.mean(dim=(0, 1), keepdim=True)   # [1, 1, Ds+Da]
    std = joint_clean_t.std(dim=(0, 1), keepdim=True) + 1e-6
    joint_norm = (joint_clean_t - mean) / std

    anchor_type = arglist.anchor_type
    anchor_data = select_anchor_data(anchor_type, init_obs, landmarks, roles)
    anchor_dim = 0 if anchor_data is None else anchor_data.shape[1]
    anchor_t = torch.from_numpy(anchor_data).float() if anchor_data is not None else None

    device = torch.device("cpu")
    model = ScoreNetwork(
        horizon=H,
        state_dim=Ds,
        action_dim=Da,
        anchor_dim=anchor_dim,
        hidden_dim=arglist.denoiser_hidden_dim,
        n_blocks=arglist.denoiser_n_blocks,
    ).to(device)

    print("[ScoreNet] params: {:,}".format(count_params(model)))
    print("[ScoreNet] Training on {} windows, H={}, Ds={}, Da={}, anchor={}".format(
        N, H, Ds, Da, anchor_type))
    print("[ScoreNet] sigma in LogUniform({}, {})".format(
        arglist.score_sigma_min, arglist.score_sigma_max))

    sigma_min = arglist.score_sigma_min
    sigma_max = arglist.score_sigma_max
    lam = arglist.channel_weight_lambda

    opt = torch.optim.AdamW(model.parameters(), lr=arglist.diffusion_lr, weight_decay=1e-4)
    batch_size = arglist.diffusion_batch_size

    for epoch in range(arglist.diffusion_epochs):
        perm = torch.randperm(N)
        epoch_loss = epoch_loss_a = epoch_loss_s = 0.0

        for b_start in range(0, N, batch_size):
            idx = perm[b_start:min(N, b_start + batch_size)]
            B = len(idx)

            x0 = joint_norm[idx].to(device)   # [B, H, Ds+Da], normalised

            # sigma ~ LogUniform per sample
            log_sigma = torch.empty(B, device=device).uniform_(
                math.log(sigma_min), math.log(sigma_max)
            )
            sigma = torch.exp(log_sigma)   # [B]

            # Perturb in normalised space: x_tilde = x0 + sigma * eps
            eps = torch.randn_like(x0)
            x_noisy = x0 + sigma.view(B, 1, 1) * eps

            # Target score: -eps / sigma
            score_target = -eps / sigma.view(B, 1, 1)

            anchor_batch = anchor_t[idx].to(device) if anchor_t is not None else None

            score_pred = model(x_noisy, sigma, anchor_batch)

            # Per-channel weighted MSE
            loss_a = F.mse_loss(score_pred[:, :, Ds:], score_target[:, :, Ds:])
            loss_s = F.mse_loss(score_pred[:, :, :Ds], score_target[:, :, :Ds])
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
            print("[ScoreNet] Epoch {}/{} — total: {:.6f}  action: {:.6f}  state: {:.6f}".format(
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
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "mean": mean,
        "std": std,
    }
    model_path = arglist.score_model_path
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    torch.save(save_dict, model_path)
    print("[ScoreNet] Saved to {}".format(model_path))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_score_network(arglist):
    """Load trained score network into module-level globals."""
    global SCORE_NET, SCORE_NET_CONSTS

    ckpt = torch.load(arglist.score_model_path, map_location="cpu")

    model = ScoreNetwork(
        horizon=ckpt["horizon"],
        state_dim=ckpt["state_dim"],
        action_dim=ckpt["action_dim"],
        anchor_dim=ckpt["anchor_dim"],
        hidden_dim=ckpt["hidden_dim"],
        n_blocks=ckpt["n_blocks"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    SCORE_NET = model
    SCORE_NET_CONSTS = {
        "mean": ckpt["mean"],
        "std": ckpt["std"],
        "H": ckpt["horizon"],
        "Ds": ckpt["state_dim"],
        "Da": ckpt["action_dim"],
        "anchor_type": ckpt["anchor_type"],
        "sigma_min": ckpt["sigma_min"],
        "sigma_max": ckpt["sigma_max"],
    }
    print("[ScoreNet] Loaded (anchor={}, H={}, Ds={}, Da={})".format(
        ckpt["anchor_type"], ckpt["horizon"], ckpt["state_dim"], ckpt["action_dim"]))


# ---------------------------------------------------------------------------
# Inference: single/few-step Langevin correction
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_correct_action(buffer, sigma_est=0.5, eta=0.1, n_steps=1,
                         critic_grad=None, lam_q=0.0):
    """
    Apply score-based Langevin correction to the latest action in the buffer.

    Correction (in normalised space, n_steps iterations):
        s = s_theta(x, sigma_est, anchor)
        x = x + eta * (s  +  lambda_Q * critic_grad_norm)   [Phase 2: lam_q > 0]

    Self-calibrating: when x is already clean, ||s|| is small so the step is
    small; when x is far from the manifold, ||s|| is large and correction is
    proportionally stronger.

    Args:
        buffer:      JointDenoiserBuffer with current episode window
        sigma_est:   estimated noise level sigma (tuned to expected corruption)
        eta:         Langevin step size
        n_steps:     number of correction steps (1 = single-step, Phase 1)
        critic_grad: [Da] numpy array grad_a Q(s, a) in raw action space,
                     or None.  Phase 2 only.
        lam_q:       critic gradient weight lambda_Q (0 = score-only).

    Returns:
        corrected action [Da] numpy array, or None if buffer is empty.
    """
    C = SCORE_NET_CONSTS
    model = SCORE_NET
    Ds = C["Ds"]

    x_raw = buffer.get_window()    # [1, H, Ds+Da]
    if x_raw is None:
        return None

    mean = C["mean"]               # [1, 1, Ds+Da]
    std = C["std"]
    x = (x_raw - mean) / std      # normalise

    anchor = None
    if buffer.anchor is not None:
        anchor = torch.from_numpy(buffer.anchor).float().unsqueeze(0)

    sigma_t = torch.tensor([sigma_est], dtype=torch.float32)

    for _ in range(n_steps):
        s = model(x, sigma_t, anchor)   # [1, H, Ds+Da]

        # Phase 2: add critic gradient (applied only at the last timestep's
        # action slice; normalised by std to stay in the same space as the score).
        if critic_grad is not None and lam_q > 0.0:
            cg_norm = torch.from_numpy(
                critic_grad.astype(np.float32)
            ).view(1, -1) / std[0, 0, Ds:]            # [1, Da]
            s[:, -1, Ds:] = s[:, -1, Ds:] + lam_q * cg_norm

        x = x + eta * s

    x_denorm = x * std + mean
    return x_denorm[0, -1, Ds:].numpy()   # last timestep, action slice


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

def _split_actions(action_vec, action_dim_per_agent):
    """Split a flat joint action vector into per-agent list."""
    out, start = [], 0
    for dim in action_dim_per_agent:
        out.append(action_vec[start:start + dim])
        start += dim
    return out


def testRobustnessScore(arglist, sigma_est=0.5, eta=0.1, n_steps=1,
                        lam_q=0.0, obs_noise_std=0.0):
    """
    Evaluate the policy under (obs, act) noise with score-based correction.

    Mirrors testRobustnessAP_joint() from joint_diffusion.py, replacing the
    DDPM reverse process with the Langevin score correction.

    Phase 1 (lam_q=0): score correction only — no critic gradient.
    Phase 2 (lam_q>0): score + critic gradient from frozen MADDPG critics.
                       critic_grad_fn must be wired up inside this function
                       once the MADDPG trainer exposes its Q-function.

    Returns:
        mean total reward (sum over agents, mean over episodes).
    """
    import tensorflow as tf
    import maddpg.common.tf_util as U
    from train import (
        make_env, get_trainers, resolve_checkpoint, apply_action_disruption
    )

    tf.reset_default_graph()
    with U.single_threaded_session():
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

        U.initialize()
        load_dir, exp_name = resolve_checkpoint(arglist, None, None)
        U.load_state(load_dir, exp_name=exp_name)

        C = SCORE_NET_CONSTS
        Ds = C["Ds"]
        Da = C["Da"]
        H = C["H"]
        anchor_type = C["anchor_type"]

        all_rewards = []
        env.llm_disturb_iteration = 0
        env.previous_reward = 0

        for ep in range(arglist.num_test_episodes):
            obs_n = env.reset()
            episode_reward = np.zeros(env.n)

            anchor_vec = build_anchor_at_reset(obs_n, env, num_adversaries, anchor_type)
            buf = JointDenoiserBuffer(H, Ds, Da)
            buf.reset(anchor_vec)

            action_dim_per_agent = [len(trainers[i].action(obs_n[i]))
                                    for i in range(env.n)]

            for step in range(arglist.max_episode_len):
                # Corrupt observations before action selection
                if obs_noise_std > 0.0:
                    obs_in = [
                        o + np.random.normal(0, obs_noise_std, o.shape).astype(np.float32)
                        for o in obs_n
                    ]
                else:
                    obs_in = obs_n

                # Get actions from policy
                action_n = [a.action(o) for a, o in zip(trainers, obs_in)]

                # Corrupt actions
                action_n_noisy = [
                    apply_action_disruption(a, 0, env, arglist)
                    for a in action_n
                ]

                # Push joint (state, action) window to buffer
                state_vec = np.concatenate(obs_in, axis=0).astype(np.float32)
                action_vec = np.concatenate(action_n_noisy, axis=0).astype(np.float32)
                buf.push(state_vec, action_vec)

                # ── Score-based Langevin correction ──────────────────────
                # Phase 2 note: to add critic gradient, compute
                #   critic_grad = finite_diff_q_grad(trainers, obs_in, action_n_noisy)
                # and pass it here.  Requires trainer Q-network access (TF-side).
                action_vec_clean = score_correct_action(
                    buf,
                    sigma_est=sigma_est,
                    eta=eta,
                    n_steps=n_steps,
                    critic_grad=None,
                    lam_q=lam_q,
                )

                if action_vec_clean is not None:
                    # Restore adversary action slices — denoising adversary
                    # actions back to clean reduces their disruption effectiveness.
                    adv_dims = sum(action_dim_per_agent[:num_adversaries])
                    action_vec_clean[:adv_dims] = action_vec[:adv_dims]
                    buf.actions[-1] = action_vec_clean.copy()
                    action_n_out = _split_actions(action_vec_clean, action_dim_per_agent)
                else:
                    action_n_out = action_n_noisy

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

        mean_total = np.mean(np.sum(all_rewards, axis=1))
        print("[ScoreNet] n_steps={} sigma={} eta={} lam_q={} "
              "obs_noise={} — mean reward: {:.3f}".format(
                  n_steps, sigma_est, eta, lam_q, obs_noise_std, mean_total))
        return mean_total
