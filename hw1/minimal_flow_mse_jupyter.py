# %% [markdown]
# # HW1 minimal walkthrough: MSE vs Flow Matching
#
# This is a notebook-style single file. Open it in VS Code/Jupyter as cells and
# run from top to bottom. It imports no homework source files: data loading,
# normalization, dataset construction, models, training, evaluation, and display
# are all in this file.

# %%
from __future__ import annotations

import math
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import gym_pusht  # noqa: F401, registers gym_pusht/PushT-v0
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import zarr
from IPython.display import HTML, Image, display
from torch import nn
from torch.utils.data import DataLoader, Dataset


# %% [markdown]
# ## 0. Hard-coded config
#
# Change these values directly in Jupyter. There is no command-line interface.

# %%
ROOT = Path.cwd()
if not (ROOT / "pyproject.toml").exists() and (Path.cwd() / "hw1").exists():
    ROOT = Path.cwd() / "hw1"

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "minimal_outputs_jupyter"
PUSHT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
ZARR_PATH = DATA_DIR / "pusht" / "pusht_cchi_v7_replay.zarr"
ENV_ID = "gym_pusht/PushT-v0"

SEED = 42
CHUNK_SIZE = 8
BATCH_SIZE = 128
HIDDEN_DIMS = (256, 256, 256)
LR = 3e-4
EPOCHS = 3
FLOW_STEPS = 10
EVAL_EPISODES = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)
print("ROOT =", ROOT)
print("DEVICE =", DEVICE)
print("OUT_DIR =", OUT_DIR)


# %% [markdown]
# ## 1. Load original Push-T data
#
# Raw data meaning:
# - `states[t]`: current environment state, shape `(5,)`
# - `actions[t]`: expert action, shape `(2,)`
# - `episode_ends`: cumulative end indices, so each episode is a slice

# %%
def ensure_pusht_data() -> Path:
    if ZARR_PATH.exists():
        return ZARR_PATH
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "pusht.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(PUSHT_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(DATA_DIR)
    return ZARR_PATH


def load_pusht_zarr(zarr_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = zarr.open(zarr_path, mode="r")
    states = np.asarray(root["data"]["state"][:], dtype=np.float32)
    actions = np.asarray(root["data"]["action"][:], dtype=np.float32)
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    return states, actions, episode_ends


zarr_path = ensure_pusht_data()
states, actions, episode_ends = load_pusht_zarr(zarr_path)

print("states.shape =", states.shape)
print("actions.shape =", actions.shape)
print("num episodes =", len(episode_ends))
print("first 5 raw states:\n", states[:5])
print("first 5 raw actions:\n", actions[:5])
print("first 10 episode_ends:", episode_ends[:10])


# %% [markdown]
# ## 2. Display original data
#
# This cell shows one expert trajectory before any model or normalization.

# %%
def svg_polyline(points: np.ndarray, color: str, width: int = 520, height: int = 360) -> str:
    xy = points[:, :2].astype(float)
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    scale = np.maximum(hi - lo, 1e-6)
    norm = (xy - lo) / scale
    px = 30 + norm[:, 0] * (width - 60)
    py = height - 30 - norm[:, 1] * (height - 60)
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(px, py))
    dots = "\n".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" fill="{color}" />'
        for x, y in zip(px[::8], py[::8])
    )
    return f"""
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      <rect x="0" y="0" width="{width}" height="{height}" fill="white" stroke="#ccc"/>
      <polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>
      {dots}
      <text x="16" y="24" font-size="14" fill="#222">raw expert trajectory, first two state dims</text>
    </svg>
    """


ep0_start, ep0_end = 0, int(episode_ends[0])
raw_episode_states = states[ep0_start:ep0_end]
raw_episode_actions = actions[ep0_start:ep0_end]
display(HTML(svg_polyline(raw_episode_states, "#2563eb")))

print("episode 0 length =", len(raw_episode_states))
print("episode 0 first 10 states:\n", raw_episode_states[:10])
print("episode 0 first 10 expert actions:\n", raw_episode_actions[:10])


# %% [markdown]
# ## 3. Normalize and build action chunks
#
# We train on normalized values because neural networks learn much more easily
# when every feature has similar scale. Each sample is:
# `state_t -> expert actions[t:t+CHUNK_SIZE]`.

# %%
@dataclass(frozen=True)
class Normalizer:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @classmethod
    def from_data(cls, states: np.ndarray, actions: np.ndarray) -> "Normalizer":
        eps = 1e-6
        return cls(
            states.mean(axis=0),
            np.maximum(states.std(axis=0), eps),
            actions.mean(axis=0),
            np.maximum(actions.std(axis=0), eps),
        )

    def normalize_state(self, x: np.ndarray) -> np.ndarray:
        return (x - self.state_mean) / self.state_std

    def normalize_action(self, x: np.ndarray) -> np.ndarray:
        return (x - self.action_mean) / self.action_std

    def denormalize_action(self, x: np.ndarray) -> np.ndarray:
        return x * self.action_std + self.action_mean


def build_valid_indices(episode_ends: np.ndarray, chunk_size: int) -> np.ndarray:
    episode_starts = np.concatenate(([0], episode_ends[:-1]))
    valid: list[int] = []
    for start, end in zip(episode_starts, episode_ends, strict=True):
        last_start = end - chunk_size
        if last_start >= start:
            valid.extend(range(start, last_start + 1))
    return np.asarray(valid, dtype=np.int64)


class PushtChunkDataset(Dataset):
    def __init__(self, states, actions, episode_ends, normalizer, chunk_size):
        self.states = states
        self.actions = actions
        self.normalizer = normalizer
        self.chunk_size = chunk_size
        self.indices = build_valid_indices(episode_ends, chunk_size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        t = int(self.indices[i])
        state = self.normalizer.normalize_state(self.states[t])
        chunk = self.normalizer.normalize_action(self.actions[t : t + self.chunk_size])
        return torch.from_numpy(state).float(), torch.from_numpy(chunk).float()


normalizer = Normalizer.from_data(states, actions)
dataset = PushtChunkDataset(states, actions, episode_ends, normalizer, CHUNK_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

one_state, one_chunk = dataset[0]
print("dataset length =", len(dataset))
print("one normalized state shape =", tuple(one_state.shape))
print("one normalized action chunk shape =", tuple(one_chunk.shape))
print("one normalized state =", one_state)
print("one normalized action chunk =\n", one_chunk)
print("same chunk restored to raw action units =\n", normalizer.denormalize_action(one_chunk.numpy()))


# %% [markdown]
# ## 4. Define MSE and Flow Matching policies

# %%
def make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for hidden in HIDDEN_DIMS:
        layers += [nn.Linear(last, hidden), nn.ReLU()]
        last = hidden
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class MSEPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, chunk_size: int):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.mlp = make_mlp(state_dim, chunk_size * action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        pred = self.sample_actions(state)
        return ((pred - action_chunk) ** 2).mean()

    def sample_actions(self, state: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        del num_steps
        flat = self.mlp(state)
        return flat.reshape(-1, self.chunk_size, self.action_dim)


class FlowMatchingPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, chunk_size: int):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        flat_action_dim = chunk_size * action_dim
        self.mlp = make_mlp(state_dim + flat_action_dim + 1, flat_action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        x1 = action_chunk.reshape(batch, -1)      # expert action chunk
        x0 = torch.randn_like(x1)                 # noise action chunk
        t = torch.rand(batch, 1, device=state.device)
        xt = (1 - t) * x0 + t * x1                # point between noise and expert
        pred_v = self.mlp(torch.cat([state, xt, t], dim=-1))
        target_v = x1 - x0                        # velocity that moves x0 to x1
        return ((pred_v - target_v) ** 2).mean()

    def sample_actions(self, state: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        batch = state.shape[0]
        x = torch.randn(batch, self.chunk_size * self.action_dim, device=state.device)
        dt = 1.0 / num_steps
        for k in range(num_steps):
            t = torch.full((batch, 1), k / num_steps, device=state.device)
            v = self.mlp(torch.cat([state, x, t], dim=-1))
            x = x + v * dt
        return x.reshape(batch, self.chunk_size, self.action_dim)


state_dim = states.shape[1]
action_dim = actions.shape[1]
mse_model = MSEPolicy(state_dim, action_dim, CHUNK_SIZE).to(DEVICE)
flow_model = FlowMatchingPolicy(state_dim, action_dim, CHUNK_SIZE).to(DEVICE)
print(mse_model)
print(flow_model)


# %% [markdown]
# ## 5. Train one policy
#
# Run this cell for MSE, then change `POLICY_TO_TRAIN = "flow"` and run again.

# %%
def train_policy(model: nn.Module, name: str, epochs: int = EPOCHS) -> list[float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    losses: list[float] = []

    for epoch in range(epochs):
        for state, action_chunk in loader:
            state = state.to(DEVICE)
            action_chunk = action_chunk.to(DEVICE)
            loss = model.compute_loss(state, action_chunk)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        print(f"{name} epoch {epoch + 1}/{epochs}: last_loss={losses[-1]:.4f}")

    ckpt_path = OUT_DIR / f"{name}_jupyter.pt"
    torch.save(model.state_dict(), ckpt_path)
    print("saved", ckpt_path)
    return losses


POLICY_TO_TRAIN = "mse"  # change to "flow", or run both cells below
mse_losses = train_policy(mse_model, "mse", EPOCHS)


# %%
# Optional: train flow matching too. This is slower than MSE on CPU.
flow_losses = train_policy(flow_model, "flow", EPOCHS)


# %% [markdown]
# ## 6. Show model output on one dataset sample
#
# This is not environment evaluation yet. It compares the model's predicted
# action chunk against the original expert chunk for one training sample.

# %%
@torch.no_grad()
def show_one_chunk(model: nn.Module, name: str, sample_i: int = 0) -> None:
    model.eval()
    state, expert_norm = dataset[sample_i]
    pred_norm = model.sample_actions(
        state.unsqueeze(0).to(DEVICE), num_steps=FLOW_STEPS
    ).cpu().numpy()[0]
    expert_raw = normalizer.denormalize_action(expert_norm.numpy())
    pred_raw = normalizer.denormalize_action(pred_norm)
    print(f"{name}: sample {sample_i}")
    print("expert raw action chunk:\n", np.round(expert_raw, 2))
    print("model predicted raw action chunk:\n", np.round(pred_raw, 2))
    print("per-step L2 error:", np.round(np.linalg.norm(expert_raw - pred_raw, axis=1), 2))


show_one_chunk(mse_model, "mse")
show_one_chunk(flow_model, "flow")


# %% [markdown]
# ## 7. Evaluate and collect final model demo data
#
# The returned `records` are the final demonstration data: every environment
# step stores raw observation, selected action, reward, and done flag.

# %%
@torch.no_grad()
def predict_action_chunk(model, obs, env, flow_steps=FLOW_STEPS) -> np.ndarray:
    state = torch.from_numpy(normalizer.normalize_state(obs)).float().to(DEVICE)
    pred_norm = model.sample_actions(state.unsqueeze(0), num_steps=flow_steps).cpu().numpy()[0]
    pred_raw = normalizer.denormalize_action(pred_norm)
    return np.clip(pred_raw, env.action_space.low, env.action_space.high)


def rollout_policy(model: nn.Module, name: str, seed: int = 1000):
    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    done = False
    chunk_i = CHUNK_SIZE
    action_chunk = None
    frames = []
    records = []
    max_reward = -math.inf
    step = 0

    while not done:
        if action_chunk is None or chunk_i >= CHUNK_SIZE:
            action_chunk = predict_action_chunk(model, obs, env)
            chunk_i = 0

        action = action_chunk[chunk_i].astype(np.float32)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frames.append(env.render())
        records.append(
            {
                "step": step,
                "obs": np.round(obs, 3).tolist(),
                "action": np.round(action, 3).tolist(),
                "reward": round(float(reward), 4),
                "done": bool(done),
            }
        )
        obs = next_obs
        chunk_i += 1
        step += 1
        max_reward = max(max_reward, float(reward))

    gif_path = OUT_DIR / f"{name}_jupyter_rollout.gif"
    imageio.mimsave(gif_path, frames, fps=20)
    env.close()
    print(f"{name} rollout max_reward = {max_reward:.3f}")
    print("saved gif:", gif_path)
    return records, gif_path


mse_records, mse_gif = rollout_policy(mse_model, "mse")
display(Image(filename=str(mse_gif)))
print("first 12 final demo records:")
for row in mse_records[:12]:
    print(row)


# %%
# Optional flow rollout.
flow_records, flow_gif = rollout_policy(flow_model, "flow")
display(Image(filename=str(flow_gif)))
print("first 12 final demo records:")
for row in flow_records[:12]:
    print(row)

