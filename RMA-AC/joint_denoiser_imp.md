# Joint State–Action Diffusion Denoiser for MPE — Claude Code Implementation Instructions

## Overview

This document provides step-by-step implementation instructions for Claude Code to build a **joint state–action diffusion denoiser** adapted for the Multi-Agent Particle Environment (MPE) testbed. The denoiser is a test-time defense module that cleans corrupted joint actions using a conditional DDPM trained on clean expert rollouts, with optional clean-anchor conditioning.

The codebase has 4 training scripts (`train-maddpg.py`, `train-m3ddpg.py`, `train-rmaac.py`, `train-rmaac-v2.py`) that share identical diffusion code (currently an action-only 2-layer MLP denoiser). We will create a **shared module** and upgrade all 4 scripts to use it.

---

## Part 0: Shared Module Structure

### Task 0.1 — Create `joint_diffusion.py` as a shared module

Create a new file `joint_diffusion.py` in the same directory as the training scripts. All diffusion-related code (model, training, data collection, inference) will live here. The 4 training scripts will `import joint_diffusion` and call its functions.

This replaces the duplicated `TrajectoryDiffusion`, `train_diffusion()`, `collect_diffusion_data()`, `load_diffusion_model()`, `diffusion_denoise_action()`, `make_beta_schedule()`, `q_sample()`, `concat_actions()`, and `split_actions()` currently copy-pasted across all 4 scripts.

Keep the existing action-only denoiser code working as a fallback (selectable via `--denoiser-type legacy`).

---

## Part 1: Data Collection — `collect_joint_diffusion_data()`

### Task 1.1 — Modify data collection to save anchor signals

The existing `collect_diffusion_data()` saves `states: [N, H, Ds]` and `actions: [N, H, Da]`. The new version must additionally save anchor signals per episode.

**Function signature:**
```python
def collect_joint_diffusion_data(env, trainers, arglist):
```

**What to collect per episode (clean rollout, no noise):**

```python
obs_n = env.reset()

# === ANCHOR SIGNALS (captured ONCE at t=0, before any corruption) ===
init_obs = np.concatenate(obs_n, axis=0)  # A1: initial joint observation [Ds]

# A2: landmark/goal positions (scenario-specific, extracted from obs)
landmark_positions = extract_landmarks(env, arglist.scenario)  # see Task 1.2

# A3: agent role encoding
role_encoding = get_role_encoding(env, arglist.num_adversaries)  # see Task 1.3

# === TRAJECTORY (same as existing code) ===
for t in range(max_episode_len):
    state_vec = np.concatenate(obs_n, axis=0)      # [Ds]
    action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
    action_vec = np.concatenate(action_n, axis=0)   # [Da]
    ep_states.append(state_vec)
    ep_actions.append(action_vec)
    obs_n, rew_n, done_n, info_n = env.step(action_n)
```

**Save format (`.npz`):**
```python
np.savez(path,
    states=states,                # [N, H, Ds]
    actions=actions,              # [N, H, Da]
    init_obs=init_obs_all,        # [N, Ds]       — anchor A1
    landmarks=landmarks_all,      # [N, D_land]   — anchor A2
    roles=roles_all,              # [N, n_agents]  — anchor A3
    state_dim=state_dim,
    action_dim=action_dim,
    n_agents=env.n,
    scenario=arglist.scenario
)
```

**Important:** Keep the existing sliding-window logic. Currently `collect_diffusion_data()` truncates each episode to exactly `H` steps (`ep_states[:H]`). The current default is `H=25` which equals `max_episode_len=25`, so each episode yields exactly 1 window. Keep this behavior — MPE episodes are short enough that 1 window per episode is the right granularity. If `H < max_episode_len`, use sliding windows with stride 1 to extract `(T - H + 1)` windows per episode, each sharing the same anchor signals from that episode.

### Task 1.2 — Implement `extract_landmarks()`

Landmark/goal positions are scenario-specific. They are constant within an episode and set during `env.reset()`. Extract them from `env.world.landmarks`.

```python
def extract_landmarks(env, scenario_name):
    """
    Extract landmark positions from the environment.
    Returns a flat numpy array of landmark (x,y) positions.
    These are constant per episode and invariant to observation/action corruption.
    """
    positions = []
    for landmark in env.world.landmarks:
        positions.extend(landmark.state.p_pos.tolist())  # [x, y] per landmark
    
    if len(positions) == 0:
        # Scenarios without landmarks: return zeros
        return np.zeros(2, dtype=np.float32)
    
    return np.array(positions, dtype=np.float32)
```

**Expected landmark dims per scenario:**
- `simple_spread`: 3 landmarks → 6-d
- `simple_tag`: 2 landmarks → 4-d
- `simple_adversary`: 2 landmarks → 4-d
- `simple_push`: 2 landmarks (1 landmark + 1 "goal" landmark) → 4-d
- `simple_crypto`: 2 landmarks → 4-d
- `simple_speaker_listener`: 3 landmarks → 6-d

**Verify these dimensions** by printing `len(env.world.landmarks)` for each scenario at data-collection time. If a scenario has no landmarks, use a zero-length array and anchor A2 falls back to A0 (no anchor) for that scenario.

### Task 1.3 — Implement `get_role_encoding()`

```python
def get_role_encoding(env, num_adversaries):
    """
    One-hot-ish encoding of agent roles.
    adversary agents = 1.0, good agents = 0.0
    Returns [n_agents] array.
    """
    roles = np.zeros(env.n, dtype=np.float32)
    for i in range(min(env.n, num_adversaries)):
        roles[i] = 1.0
    return roles
```

### Task 1.4 — CLI args for data collection

Add to `parse_args()` in all 4 training scripts (or better, in the shared module):

```python
parser.add_argument("--num-collect-episodes", type=int, default=5000,
    help="number of episodes for diffusion data collection")
parser.add_argument("--joint-diffusion-data-path", type=str,
    default="../../joint_diffusion_data.npz",
    help="where to save joint (state+action+anchor) data")
```

Add a new mode `"collect_joint_diffusion"` to the `--mode` choices.

---

## Part 2: Model Architecture — `JointDiffusionDenoiser`

### Task 2.1 — Implement the residual MLP denoiser

This is a **residual MLP with FiLM conditioning**, NOT a U-Net. MPE has low-dimensional joint state-action spaces (~50-70 dims) and short episodes (25 steps), so a fully-connected architecture is appropriate.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        """t: [B] integer timesteps → [B, dim] embeddings"""
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
    """
    Residual block with FiLM conditioning.
    Linear → LayerNorm → FiLM(gamma, beta) → ReLU → Linear → residual add
    
    FiLM gamma weights are initialized to zero so the network
    starts as identity-conditioned (output = input through residual).
    """
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # FiLM projection: cond → (gamma, beta)
        self.film_proj = nn.Linear(cond_dim, 2 * hidden_dim)
        # Zero-init so gamma=0, beta=0 at start → output of modulation = 0
        # → after ReLU and linear2, residual block contributes ~nothing initially
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)
    
    def forward(self, x, cond):
        """
        x:    [B, hidden_dim]
        cond: [B, cond_dim]
        """
        residual = x
        h = self.linear1(x)
        h = self.norm(h)
        
        # FiLM modulation
        film_params = self.film_proj(cond)
        gamma, beta = film_params.chunk(2, dim=-1)
        h = (1.0 + gamma) * h + beta
        
        h = F.relu(h)
        h = self.linear2(h)
        return residual + h


class AnchorEncoder(nn.Module):
    """
    Two-layer MLP that projects anchor signal ξ to embedding c ∈ R^{D_c}.
    For anchor A0 (no anchor), this module is bypassed and c = zeros.
    """
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
    
    Architecture: Residual MLP with FiLM conditioning.
    - Input: flattened joint window [B, H * (Ds + Da)]
    - Conditioning: sinusoidal timestep embedding + anchor embedding
    - Output: predicted noise [B, H * (Ds + Da)], reshaped to [B, H, Ds+Da]
    
    Args:
        horizon (int): temporal window length H
        state_dim (int): joint state dimension Ds (sum of all agent obs dims)
        action_dim (int): joint action dimension Da (sum of all agent action dims)
        anchor_dim (int): dimension of anchor signal (0 for no anchor)
        hidden_dim (int): width of residual blocks (default 256)
        n_blocks (int): number of FiLM residual blocks (default 4)
        anchor_embed_dim (int): anchor encoder output dim (default 128)
        time_embed_dim (int): sinusoidal time embedding dim (default 128)
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
        
        # Conditioning dimension (what goes into FiLM)
        cond_dim = hidden_dim  # project everything to hidden_dim
        
        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_proj = nn.Linear(time_embed_dim, cond_dim)
        
        # Anchor embedding (only if anchor_dim > 0)
        if anchor_dim > 0:
            self.anchor_enc = AnchorEncoder(anchor_dim, anchor_embed_dim)
            self.anchor_proj = nn.Linear(anchor_embed_dim, cond_dim)
        else:
            self.anchor_enc = None
            self.anchor_proj = None
        
        # Input projection: flatten joint window → hidden_dim
        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        
        # Residual blocks with FiLM
        self.blocks = nn.ModuleList([
            FiLMResidualBlock(hidden_dim, cond_dim)
            for _ in range(n_blocks)
        ])
        
        # Output projection: hidden_dim → H * (Ds + Da)
        self.output_proj = nn.Linear(hidden_dim, self.input_dim)
        # Zero-init output so initial predictions are zero noise
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x_noisy, t, anchor=None):
        """
        x_noisy: [B, H, Ds+Da] — noised joint window
        t:       [B] — integer diffusion timesteps (0..K-1)
        anchor:  [B, anchor_dim] or None — clean anchor signal
        
        Returns: [B, H, Ds+Da] — predicted noise
        """
        B = x_noisy.shape[0]
        
        # Flatten input
        x_flat = x_noisy.reshape(B, -1)  # [B, H*(Ds+Da)]
        
        # Build conditioning vector
        t_emb = self.time_embed(t)           # [B, time_embed_dim]
        cond = self.time_proj(t_emb)         # [B, cond_dim]
        
        if self.anchor_enc is not None and anchor is not None:
            a_emb = self.anchor_enc(anchor)      # [B, anchor_embed_dim]
            a_proj = self.anchor_proj(a_emb)     # [B, cond_dim]
            cond = cond + a_proj                 # additive fusion
        
        # Forward through residual blocks
        h = self.input_proj(x_flat)          # [B, hidden_dim]
        for block in self.blocks:
            h = block(h, cond)
        
        # Output
        eps_pred = self.output_proj(h)       # [B, H*(Ds+Da)]
        eps_pred = eps_pred.reshape(B, self.horizon, self.joint_dim)
        return eps_pred
```

### Task 2.2 — Verify parameter count is reasonable

After implementing, add a helper that prints parameter count:

```python
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

**Expected rough parameter counts** (H=25, hidden=256, 4 blocks):
- `simple_spread` (Ds≈54, Da≈15): input_dim=25*69=1725 → ~2.5M params
- `simple_tag` (Ds≈64, Da≈20): input_dim=25*84=2100 → ~2.8M params

These are manageable on CPU. If training is too slow, reduce `hidden_dim` to 128 or `n_blocks` to 3.

---

## Part 3: Training Loop — `train_joint_denoiser()`

### Task 3.1 — Implement the warm-start separated training objective

This is the critical difference from the existing action-only denoiser. The existing code trains with standard DDPM noise prediction (`loss = MSE(eps_pred, eps)`). The new training objective uses **warm-start separated targets**: the forward process starts from a synthetically corrupted version of the data, but the supervision target is computed relative to the clean data.

```python
def train_joint_denoiser(arglist):
    """
    Train the joint state-action diffusion denoiser.
    
    Key difference from existing train_diffusion():
    1. Input is joint [H, Ds+Da] not action-only [H, Da]
    2. Training uses warm-start separated targets
    3. Loss is per-channel weighted (action channel weighted higher)
    4. Anchor signals are loaded and used for conditioning
    """
    # ---- Load data ----
    data = np.load(arglist.joint_diffusion_data_path, allow_pickle=True)
    states = data["states"]       # [N, H, Ds]
    actions = data["actions"]     # [N, H, Da]
    init_obs = data["init_obs"]   # [N, Ds]
    landmarks = data["landmarks"] # [N, D_land]
    roles = data["roles"]         # [N, n_agents]
    
    N, H, Ds = states.shape
    _, _, Da = actions.shape
    
    # ---- Build joint windows: [N, H, Ds+Da] ----
    joint_clean = np.concatenate([states, actions], axis=-1)  # [N, H, Ds+Da]
    joint_clean_t = torch.from_numpy(joint_clean).float()
    
    # ---- Normalize per-channel ----
    # Compute mean/std over [N, H] for each of the (Ds+Da) features
    mean = joint_clean_t.mean(dim=(0, 1), keepdim=True)   # [1, 1, Ds+Da]
    std = joint_clean_t.std(dim=(0, 1), keepdim=True) + 1e-6
    joint_clean_t = (joint_clean_t - mean) / std
    
    # ---- Select anchor based on --anchor-type ----
    anchor_data = select_anchor_data(
        arglist.anchor_type, init_obs, landmarks, roles
    )  # returns [N, anchor_dim] numpy array, or None for A0
    
    anchor_dim = 0 if anchor_data is None else anchor_data.shape[1]
    if anchor_data is not None:
        anchor_t = torch.from_numpy(anchor_data).float()
    
    # ---- Build model ----
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
    
    # ---- Diffusion schedule ----
    K = arglist.diffusion_steps
    betas, alphas, alphas_bar = make_beta_schedule(K)
    betas = betas.to(device)
    alphas = alphas.to(device)
    alphas_bar = alphas_bar.to(device)
    
    # ---- Training corruption distribution ----
    # Sample (alpha_s, alpha_a) uniformly from this grid during training
    corruption_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0]
    
    # ---- Channel weight lambda ----
    # lambda < Da/Ds to prioritize action fidelity
    lam = arglist.channel_weight_lambda  # default 0.1
    
    # ---- Optimizer ----
    opt = torch.optim.AdamW(model.parameters(), lr=arglist.diffusion_lr,
                            weight_decay=1e-4)
    batch_size = arglist.diffusion_batch_size
    
    print("[JointDenoiser] Training on {} windows, H={}, Ds={}, Da={}, anchor={}".format(
        N, H, Ds, Da, arglist.anchor_type))
    
    for epoch in range(arglist.diffusion_epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0
        epoch_loss_a = 0.0
        epoch_loss_s = 0.0
        n_batches = 0
        
        for b_start in range(0, N, batch_size):
            b_end = min(N, b_start + batch_size)
            idx = perm[b_start:b_end]
            B = len(idx)
            
            # Clean joint window (normalized)
            x0_clean = joint_clean_t[idx].to(device)  # [B, H, Ds+Da]
            
            # ---- Synthetic corruption ----
            # Sample corruption levels per sample
            alpha_s = torch.tensor(
                np.random.choice(corruption_levels, size=B)
            ).float().to(device)
            alpha_a = torch.tensor(
                np.random.choice(corruption_levels, size=B)
            ).float().to(device)
            
            # Apply corruption to the UNNORMALIZED clean data, then renormalize
            # (corruption is in the original data scale, not normalized scale)
            # Build corruption noise in normalized space:
            #   In original space: noise_s ~ N(0, alpha_s^2)
            #   In normalized space: noise_s / std_s
            noise_s = alpha_s.view(B, 1, 1) * torch.randn(B, H, Ds, device=device)
            noise_s = noise_s / std[:, :, :Ds]  # scale to normalized space
            
            noise_a = alpha_a.view(B, 1, 1) * torch.randn(B, H, Da, device=device)
            noise_a = noise_a / std[:, :, Ds:]
            
            corruption = torch.cat([noise_s, noise_a], dim=-1)  # [B, H, Ds+Da]
            x0_corrupted = x0_clean + corruption  # corrupted in normalized space
            
            # ---- Forward diffusion from CORRUPTED starting point ----
            k = torch.randint(1, K, (B,), device=device)  # 1..K-1
            eps = torch.randn_like(x0_corrupted)
            
            a_bar_k = alphas_bar[k].view(B, 1, 1)
            x_k = torch.sqrt(a_bar_k) * x0_corrupted + torch.sqrt(1.0 - a_bar_k) * eps
            
            # ---- Warm-start separated target ----
            # eps_star = (x_k - sqrt(alpha_bar_k) * x0_CLEAN) / sqrt(1 - alpha_bar_k)
            eps_star = (x_k - torch.sqrt(a_bar_k) * x0_clean) / torch.sqrt(1.0 - a_bar_k)
            
            # ---- Forward pass ----
            anchor_batch = None
            if anchor_data is not None:
                anchor_batch = anchor_t[idx].to(device)
            
            eps_pred = model(x_k, k, anchor_batch)
            
            # ---- Per-channel weighted loss ----
            loss_a = F.mse_loss(eps_pred[:, :, Ds:], eps_star[:, :, Ds:])
            loss_s = F.mse_loss(eps_pred[:, :, :Ds], eps_star[:, :, :Ds])
            loss = loss_a + lam * loss_s
            
            opt.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            
            epoch_loss += loss.item() * B
            epoch_loss_a += loss_a.item() * B
            epoch_loss_s += loss_s.item() * B
            n_batches += 1
        
        epoch_loss /= N
        epoch_loss_a /= N
        epoch_loss_s /= N
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print("[JointDenoiser] Epoch {}/{} — total: {:.6f}  action: {:.6f}  state: {:.6f}".format(
                epoch + 1, arglist.diffusion_epochs, epoch_loss, epoch_loss_a, epoch_loss_s))
    
    # ---- Save ----
    save_dict = {
        "model_state_dict": model.state_dict(),
        "horizon": H,
        "state_dim": Ds,
        "action_dim": Da,
        "anchor_dim": anchor_dim,
        "anchor_type": arglist.anchor_type,
        "hidden_dim": arglist.denoiser_hidden_dim,
        "n_blocks": arglist.denoiser_n_blocks,
        "diffusion_steps": K,
        "mean": mean,          # [1, 1, Ds+Da]
        "std": std,            # [1, 1, Ds+Da]
        "channel_weight_lambda": lam,
    }
    os.makedirs(os.path.dirname(arglist.joint_denoiser_model_path), exist_ok=True)
    torch.save(save_dict, arglist.joint_denoiser_model_path)
    print("[JointDenoiser] Saved to {}".format(arglist.joint_denoiser_model_path))
```

### Task 3.2 — Implement `select_anchor_data()`

```python
def select_anchor_data(anchor_type, init_obs, landmarks, roles):
    """
    Select and return the anchor signal based on anchor_type.
    
    Args:
        anchor_type: str — "none" / "init_obs" / "landmarks" / "roles" / "landmarks+roles"
        init_obs:    [N, Ds] initial joint observations
        landmarks:   [N, D_land] landmark positions
        roles:       [N, n_agents] role encodings
    
    Returns:
        [N, anchor_dim] numpy array, or None for "none"
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
```

### Task 3.3 — CLI args for training

Add these arguments to `parse_args()`:

```python
# --- Joint denoiser settings ---
parser.add_argument("--denoiser-type", type=str, default="joint",
    choices=["legacy", "joint"],
    help="'legacy' = existing action-only MLP, 'joint' = new joint state-action denoiser")
parser.add_argument("--anchor-type", type=str, default="none",
    choices=["none", "init_obs", "landmarks", "roles", "landmarks+roles"],
    help="Clean anchor signal for the joint denoiser")
parser.add_argument("--denoiser-hidden-dim", type=int, default=256,
    help="Hidden dimension for residual blocks")
parser.add_argument("--denoiser-n-blocks", type=int, default=4,
    help="Number of FiLM residual blocks")
parser.add_argument("--channel-weight-lambda", type=float, default=0.1,
    help="Weight for state channel loss (action channel weight = 1.0)")
parser.add_argument("--joint-denoiser-model-path", type=str,
    default="../../joint_denoiser.pt",
    help="Where to save/load the joint denoiser model")
```

Add mode `"train_joint_denoiser"` to the `--mode` choices.

---

## Part 4: Inference — `joint_denoise_action()`

### Task 4.1 — Implement the warm-start reverse diffusion at deployment

This replaces the existing `diffusion_denoise_action()`. The key differences:
1. Input is the full joint `(state, action)` window, not just the action
2. Reverse diffusion starts from the corrupted data (warm start), not pure noise
3. Only the action portion of the denoised output is extracted and returned

```python
# ---- Global state (same pattern as existing code) ----
JOINT_DENOISER = None
JOINT_DENOISER_CONSTS = {}


def load_joint_denoiser(arglist):
    """Load trained joint denoiser model and constants."""
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
        "mean": ckpt["mean"],       # [1, 1, Ds+Da]
        "std": ckpt["std"],         # [1, 1, Ds+Da]
        "K": K,
        "H": ckpt["horizon"],
        "Ds": ckpt["state_dim"],
        "Da": ckpt["action_dim"],
        "anchor_type": ckpt["anchor_type"],
    }
    print("[JointDenoiser] Loaded model (anchor={})".format(ckpt["anchor_type"]))


class JointDenoiserBuffer:
    """
    Sliding window buffer that accumulates (state, action) pairs
    during an episode and provides the joint window for denoising.
    
    Also stores the episode anchor (set once at reset).
    """
    def __init__(self, H, Ds, Da):
        self.H = H
        self.Ds = Ds
        self.Da = Da
        self.states = []   # list of [Ds] arrays
        self.actions = []  # list of [Da] arrays
        self.anchor = None
    
    def reset(self, anchor_vec=None):
        """Call at episode start. anchor_vec is the clean anchor signal."""
        self.states = []
        self.actions = []
        self.anchor = anchor_vec
    
    def push(self, state_vec, action_vec):
        """Add a (state, action) pair to the buffer."""
        self.states.append(state_vec.copy())
        self.actions.append(action_vec.copy())
    
    def get_window(self):
        """
        Return the most recent H (state, action) pairs as a joint window.
        If fewer than H steps have been taken, left-pad with the earliest entry.
        Returns [1, H, Ds+Da] tensor.
        """
        T = len(self.states)
        if T == 0:
            return None
        
        # Build arrays
        s_arr = np.array(self.states)  # [T, Ds]
        a_arr = np.array(self.actions) # [T, Da]
        
        if T >= self.H:
            s_window = s_arr[-self.H:]
            a_window = a_arr[-self.H:]
        else:
            # Left-pad by repeating the first entry
            pad_len = self.H - T
            s_pad = np.tile(s_arr[0:1], (pad_len, 1))
            a_pad = np.tile(a_arr[0:1], (pad_len, 1))
            s_window = np.concatenate([s_pad, s_arr], axis=0)
            a_window = np.concatenate([a_pad, a_arr], axis=0)
        
        joint = np.concatenate([s_window, a_window], axis=-1)  # [H, Ds+Da]
        return torch.from_numpy(joint).float().unsqueeze(0)    # [1, H, Ds+Da]


@torch.no_grad()
def joint_denoise_action(buffer, k_start=10):
    """
    Denoise the current action using the joint state-action denoiser.
    
    Args:
        buffer: JointDenoiserBuffer with current episode data
        k_start: truncation timestep for warm-start reverse diffusion
    
    Returns:
        clean_action_vec: [Da] numpy array — denoised joint action
    """
    model = JOINT_DENOISER
    C = JOINT_DENOISER_CONSTS
    Ds = C["Ds"]
    Da = C["Da"]
    
    # ---- Get joint window ----
    x_tilde = buffer.get_window()  # [1, H, Ds+Da]
    if x_tilde is None:
        return None
    
    # ---- Normalize ----
    x_tilde = (x_tilde - C["mean"]) / C["std"]
    
    # ---- Prepare anchor ----
    anchor = None
    if buffer.anchor is not None:
        anchor = torch.from_numpy(buffer.anchor).float().unsqueeze(0)  # [1, anchor_dim]
    
    # ---- Warm-start: initialize reverse diffusion from corrupted data ----
    x = x_tilde.clone()  # x_{k_start} ← x_tilde
    
    alphas = C["alphas"]
    alphas_bar = C["alphas_bar"]
    
    # ---- Reverse diffusion: k_start, k_start-1, ..., 1, 0 ----
    for k in reversed(range(k_start + 1)):
        k_tensor = torch.tensor([k])
        
        eps_pred = model(x, k_tensor, anchor)
        
        alpha_bar_k = alphas_bar[k]
        alpha_k = alphas[k]
        
        # Predict x0
        x0_hat = (x - torch.sqrt(1.0 - alpha_bar_k) * eps_pred) / torch.sqrt(alpha_bar_k)
        
        if k > 0:
            # Add noise for next step
            z = torch.randn_like(x)
            x = torch.sqrt(alpha_k) * x0_hat + torch.sqrt(1.0 - alpha_k) * z
        else:
            x = x0_hat
    
    # ---- Extract action from last timestep ----
    # x is [1, H, Ds+Da], take the last timestep's action portion
    x_denorm = x * C["std"] + C["mean"]
    clean_action = x_denorm[0, -1, Ds:].numpy()  # [Da]
    
    return clean_action
```

### Task 4.2 — Integrate into the test harness

Modify `testRobustnessAP()` in all 4 scripts (or better, in shared module). The key change is using the `JointDenoiserBuffer` to accumulate the sliding window during evaluation.

The modified evaluation loop should look like:

```python
def testRobustnessAP_joint(arglist, use_denoiser=True, k_start=10):
    """Test with action perturbation, optionally using joint denoiser."""
    # ... [existing setup: env, trainers, load model] ...
    
    if use_denoiser:
        load_joint_denoiser(arglist)
        C = JOINT_DENOISER_CONSTS
        Ds, Da, H = C["Ds"], C["Da"], C["H"]
    
    for ep in range(n_episodes):
        obs_n = env.reset()
        episode_reward = np.zeros(env.n)
        
        if use_denoiser:
            # ---- Initialize buffer and anchor ----
            buffer = JointDenoiserBuffer(H, Ds, Da)
            anchor_vec = build_anchor_at_reset(
                obs_n, env, arglist.num_adversaries, C["anchor_type"]
            )
            buffer.reset(anchor_vec)
        
        for step in range(max_episode_len):
            # Clean actions from policy
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            n_agents = len(action_n)
            action_dim_per_agent = [len(a) for a in action_n]
            
            # Apply action noise
            action_n_noisy = [
                apply_action_disruption(action, 0, env, arglist)
                for action in action_n
            ]
            
            action_n_clean = action_n_noisy  # default: no denoising
            
            if use_denoiser:
                state_vec = np.concatenate(obs_n, axis=0)
                action_vec_noisy = np.concatenate(action_n_noisy, axis=0)
                
                # Push corrupted (state, action) into buffer
                buffer.push(state_vec, action_vec_noisy)
                
                # Denoise
                action_vec_clean = joint_denoise_action(buffer, k_start=k_start)
                
                if action_vec_clean is not None:
                    action_n_clean = split_actions(
                        action_vec_clean, n_agents, action_dim_per_agent
                    )
            
            # Step environment
            new_obs_n, rew_n, done_n, info_n = env.step(action_n_clean)
            episode_reward += rew_n
            obs_n = new_obs_n
            
            if all(done_n):
                break
        
        all_rewards.append(episode_reward)
    
    # ... [existing reporting code] ...
```

### Task 4.3 — Implement `build_anchor_at_reset()`

```python
def build_anchor_at_reset(obs_n, env, num_adversaries, anchor_type):
    """
    Build the clean anchor signal at episode reset.
    Called once per episode before any corruption is applied.
    
    Returns: numpy array [anchor_dim], or None
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
```

---

## Part 5: Evaluation Harness — Hyperparameter Sweeps

### Task 5.1 — Add the joint denoiser evaluation mode

Add mode `"test_joint"` to `--mode` choices. The evaluation sweep should cover:

```python
elif arglist.mode == "test_joint":
    load_joint_denoiser(arglist)
    
    arglist.noise_type = "gauss"
    
    # ---- Sweep parameters ----
    obs_noise_list = [0.0, 0.4, 0.8, 1.2]
    act_noise_list = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
    k_start_list = [5, 10, 20, 40]
    
    # ---- Baseline ----
    rew_clean = testWithoutP(arglist)
    print("Clean baseline: {:.3f}".format(rew_clean))
    
    results = []
    
    for obs_std in obs_noise_list:
        arglist.noise_sigma = obs_std
        for act_std in act_noise_list:
            arglist.act_noise = act_std
            
            # No denoiser
            rew_noisy = testRobustnessAP_joint(
                arglist, use_denoiser=False
            )
            
            # With denoiser at each k_start
            diff_rewards = {}
            for k_s in k_start_list:
                rew_denoised = testRobustnessAP_joint(
                    arglist, use_denoiser=True, k_start=k_s
                )
                diff_rewards[k_s] = rew_denoised
            
            best_rew = max(diff_rewards.values())
            best_k = max(diff_rewards, key=diff_rewards.get)
            
            # Recovery ratio
            if abs(rew_clean - rew_noisy) > 1e-6:
                recovery = (best_rew - rew_noisy) / (rew_clean - rew_noisy)
            else:
                recovery = 1.0
            
            row = {
                "obs_noise_std": obs_std,
                "act_noise_std": act_std,
                "reward_clean": rew_clean,
                "reward_noisy": rew_noisy,
                "best_reward_denoised": best_rew,
                "best_k_start": best_k,
                "recovery_ratio": recovery,
            }
            for k_s in k_start_list:
                row["reward_k{}".format(k_s)] = diff_rewards[k_s]
            
            results.append(row)
    
    # Save CSV
    df = pd.DataFrame(results)
    csv_path = "{}_joint_denoiser_{}_sweep.csv".format(
        arglist.exp_name, arglist.anchor_type
    )
    df.to_csv(csv_path, index=False)
    print("Saved to {}".format(csv_path))
```

### Task 5.2 — Observation noise in the test harness

The existing `testRobustnessAP()` only applies action noise. For the joint denoiser, we need observation noise too. Modify the test loop so that when `arglist.noise_sigma > 0`, observation noise is also applied:

```python
# Inside the test loop, before action selection:
if arglist.noise_sigma > 0:
    obs_n_noisy = [
        obs + np.random.normal(0, arglist.noise_sigma, size=obs.shape).astype(np.float32)
        for obs in obs_n
    ]
    action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n_noisy)]
else:
    action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
```

**Important:** The state vector pushed into the denoiser buffer should be the **corrupted** observation (what the policy actually saw), not the true observation. This matches the deployment setting where the denoiser only sees corrupted signals.

---

## Part 6: Full Experiment Script — `run_experiments.sh`

### Task 6.1 — Create a bash script for the complete experiment pipeline

```bash
#!/bin/bash
# run_joint_denoiser_experiments.sh
#
# Complete experiment pipeline for the joint state-action diffusion denoiser.
# Run from the directory containing the training scripts.
#
# Usage:
#   bash run_joint_denoiser_experiments.sh <algorithm> <scenario>
#   Example: bash run_joint_denoiser_experiments.sh maddpg simple_spread

ALG=$1        # maddpg | m3ddpg | rmaac | rmaac-v2
SCENARIO=$2   # simple_spread | simple_tag | simple_adversary | ...

SCRIPT="train-${ALG}.py"
EXP="${ALG}_${SCENARIO}"
MODEL_DIR="../../models/${EXP}"
DATA_DIR="../../diffusion_data"
RESULT_DIR="../../results/${EXP}"

mkdir -p ${DATA_DIR} ${RESULT_DIR}

# ============================================================
# Step 1: Train baseline MARL policy (skip if already trained)
# ============================================================
if [ ! -d "${MODEL_DIR}" ]; then
    echo "=== Step 1: Training ${ALG} on ${SCENARIO} ==="
    python ${SCRIPT} \
        --scenario ${SCENARIO} \
        --mode train \
        --exp-name ${EXP} \
        --save-dir ${MODEL_DIR} \
        --num-episodes 60000
else
    echo "=== Step 1: Skipping training (model exists at ${MODEL_DIR}) ==="
fi

# ============================================================
# Step 2: Collect clean trajectory data with anchors
# ============================================================
DATA_PATH="${DATA_DIR}/${EXP}_joint.npz"

if [ ! -f "${DATA_PATH}" ]; then
    echo "=== Step 2: Collecting joint diffusion data ==="
    python ${SCRIPT} \
        --scenario ${SCENARIO} \
        --mode collect_joint_diffusion \
        --exp-name ${EXP} \
        --save-dir ${MODEL_DIR} \
        --num-collect-episodes 5000 \
        --joint-diffusion-data-path ${DATA_PATH}
else
    echo "=== Step 2: Skipping data collection (exists at ${DATA_PATH}) ==="
fi

# ============================================================
# Step 3: Train joint denoisers (one per anchor type)
# ============================================================
for ANCHOR in none init_obs landmarks landmarks+roles; do
    MODEL_PATH="${DATA_DIR}/${EXP}_denoiser_${ANCHOR}.pt"
    
    if [ ! -f "${MODEL_PATH}" ]; then
        echo "=== Step 3: Training joint denoiser (anchor=${ANCHOR}) ==="
        python ${SCRIPT} \
            --scenario ${SCENARIO} \
            --mode train_joint_denoiser \
            --exp-name ${EXP} \
            --save-dir ${MODEL_DIR} \
            --joint-diffusion-data-path ${DATA_PATH} \
            --joint-denoiser-model-path ${MODEL_PATH} \
            --anchor-type ${ANCHOR} \
            --denoiser-hidden-dim 256 \
            --denoiser-n-blocks 4 \
            --diffusion-steps 100 \
            --diffusion-epochs 200 \
            --diffusion-lr 1e-4 \
            --diffusion-batch-size 128 \
            --channel-weight-lambda 0.1
    else
        echo "=== Step 3: Skipping denoiser training (${ANCHOR}, exists) ==="
    fi
done

# ============================================================
# Step 4: Evaluate under corruption grid
# ============================================================
for ANCHOR in none init_obs landmarks landmarks+roles; do
    MODEL_PATH="${DATA_DIR}/${EXP}_denoiser_${ANCHOR}.pt"
    
    echo "=== Step 4: Evaluating (anchor=${ANCHOR}) ==="
    python ${SCRIPT} \
        --scenario ${SCENARIO} \
        --mode test_joint \
        --exp-name ${EXP} \
        --save-dir ${MODEL_DIR} \
        --joint-denoiser-model-path ${MODEL_PATH} \
        --anchor-type ${ANCHOR} \
        --num-test-episodes 400 \
        --plots-dir ${RESULT_DIR}
done

echo "=== Done! Results in ${RESULT_DIR} ==="
```

### Task 6.2 — Master sweep across all scenarios and algorithms

```bash
#!/bin/bash
# run_all_experiments.sh

SCENARIOS="simple_spread simple_tag simple_adversary simple_crypto simple_push simple_speaker_listener"
ALGORITHMS="maddpg m3ddpg rmaac"

for ALG in ${ALGORITHMS}; do
    for SCENARIO in ${SCENARIOS}; do
        echo "========================================"
        echo "  ${ALG} × ${SCENARIO}"
        echo "========================================"
        bash run_joint_denoiser_experiments.sh ${ALG} ${SCENARIO}
    done
done
```

---

## Part 7: Ablation Experiments

### Task 7.1 — Joint vs. action-only ablation

Run the existing legacy denoiser and the new joint denoiser on the same corruption grid, same scenario, same MARL policy. Compare CSVs. The legacy denoiser uses `--mode test` (existing code). The joint denoiser uses `--mode test_joint`.

### Task 7.2 — Channel weight lambda sweep

```bash
for LAM in 0.01 0.1 0.5 1.0; do
    python train-maddpg.py \
        --scenario simple_spread \
        --mode train_joint_denoiser \
        --anchor-type landmarks \
        --channel-weight-lambda ${LAM} \
        --joint-denoiser-model-path ../../models/denoiser_lam${LAM}.pt \
        ...  # [other args]
    
    python train-maddpg.py \
        --scenario simple_spread \
        --mode test_joint \
        --joint-denoiser-model-path ../../models/denoiser_lam${LAM}.pt \
        ...
done
```

### Task 7.3 — Warm-start vs. cold-start ablation

This is tested at inference time on the SAME trained model. Set `k_start = K` (= `diffusion_steps`, typically 100) for cold start. The warm-start uses small `k_start` values (5, 10, 20). Already covered by the k_start sweep in the evaluation harness.

### Task 7.4 — Architecture sweep (hidden_dim, n_blocks)

```bash
for HDIM in 128 256; do
    for NBLK in 3 4 6; do
        python train-maddpg.py \
            --scenario simple_spread \
            --mode train_joint_denoiser \
            --denoiser-hidden-dim ${HDIM} \
            --denoiser-n-blocks ${NBLK} \
            --joint-denoiser-model-path ../../models/denoiser_h${HDIM}_b${NBLK}.pt \
            ...
    done
done
```

### Task 7.5 — Anchor corruption validation

Train a denoiser with `--anchor-type landmarks`. At test time, corrupt the anchor by adding noise to the landmark positions passed to `build_anchor_at_reset()`. This validates the formulation's prediction that corrupting the anchor degrades performance:

```python
# In build_anchor_at_reset(), add optional noise:
if arglist.corrupt_anchor_std > 0:
    anchor_vec += np.random.normal(0, arglist.corrupt_anchor_std, size=anchor_vec.shape)
```

Add CLI arg: `--corrupt-anchor-std` (default 0.0).

Sweep: `corrupt_anchor_std ∈ {0.0, 0.2, 0.5, 1.0, 2.0}`.

---

## Part 8: Integration Checklist

### Modifications to each training script

For each of `train-maddpg.py`, `train-m3ddpg.py`, `train-rmaac.py`, `train-rmaac-v2.py`:

1. **Add** `import joint_diffusion` at the top
2. **Add** new CLI args (Task 3.3, plus the modes)
3. **Add** new mode handlers in the `if arglist.mode == ...` block:
   - `"collect_joint_diffusion"` → calls `joint_diffusion.collect_joint_diffusion_data()`
   - `"train_joint_denoiser"` → calls `joint_diffusion.train_joint_denoiser()`
   - `"test_joint"` → calls the new evaluation sweep
4. **Keep** all existing modes (`train`, `test`, `collect_diffusion`, `train_diffusion`) working unchanged
5. **Keep** the existing `TrajectoryDiffusion` class and `diffusion_denoise_action()` working as the `--denoiser-type legacy` path

### File structure after implementation

```
experiments/
├── joint_diffusion.py          # NEW: shared module
├── train-maddpg.py             # MODIFIED: imports joint_diffusion, new modes
├── train-m3ddpg.py             # MODIFIED: same
├── train-rmaac.py              # MODIFIED: same
├── train-rmaac-v2.py           # MODIFIED: same
├── run_joint_denoiser_experiments.sh  # NEW
├── run_all_experiments.sh             # NEW
└── ...
```

### Default hyperparameters summary

| Parameter | Default | Notes |
|-----------|---------|-------|
| `diffusion_steps` (K) | 100 | Number of forward diffusion steps |
| `diffusion_horizon` (H) | 25 | = max_episode_len for MPE |
| `denoiser_hidden_dim` | 256 | Width of residual MLP blocks |
| `denoiser_n_blocks` | 4 | Number of FiLM residual blocks |
| `anchor_embed_dim` | 128 | Anchor encoder output dim |
| `time_embed_dim` | 128 | Sinusoidal time embedding dim |
| `channel_weight_lambda` | 0.1 | State channel loss weight (action = 1.0) |
| `diffusion_lr` | 1e-4 | AdamW learning rate |
| `diffusion_batch_size` | 128 | Training batch size |
| `diffusion_epochs` | 200 | Training epochs |
| `num_collect_episodes` | 5000 | Episodes for data collection |
| `k_start` (eval) | 10 | Default warm-start truncation step |
| `corruption_levels` (train) | [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0] | Sampled during training |

### Sanity checks to run after implementation

1. **Data collection check:** After collecting, print `states.shape`, `actions.shape`, `init_obs.shape`, `landmarks.shape`, `roles.shape` and verify dims match the scenario.
2. **Model forward pass check:** Create a model, pass random tensors through it, verify output shape is `[B, H, Ds+Da]`.
3. **Training loss check:** Loss should decrease over epochs. Action loss should decrease faster than state loss (since lambda < 1 gives it more weight implicitly by being unweighted).
4. **Denoising check:** With `k_start=0`, the denoiser should return roughly the input (no denoising). With `k_start=K`, it should return something independent of the input (cold start / prior sample).
5. **No-noise check:** At `(obs_noise=0, act_noise=0)` with denoiser on, reward should be close to (and not much worse than) the clean baseline. If the denoiser hurts clean performance significantly, `k_start` is too high or training has a bug.
6. **Legacy compatibility:** The existing `--mode test` with `--denoiser-type legacy` should produce identical results to before.