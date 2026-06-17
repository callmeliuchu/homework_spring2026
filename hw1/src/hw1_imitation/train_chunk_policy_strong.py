"""Train stronger action-chunk MSE or Flow Matching policies for Push-T.

The starter training script only optimizes supervised loss.  This experiment
script also performs deterministic environment evaluation on fixed seeds and
saves the checkpoint with the best mean max reward.

Examples:
    uv run python src/hw1_imitation/train_chunk_policy_strong.py \
        --policy-type mse --init-checkpoint mse.pt --output-checkpoint mse_chunk_strong.pt

    uv run python src/hw1_imitation/train_chunk_policy_strong.py \
        --policy-type flow --init-checkpoint flow.pt --output-checkpoint flow_chunk_strong.pt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import tyro
from torch import nn
from torch.utils.data import DataLoader, random_split

from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from hw1_imitation.model import build_policy

ENV_ID = "gym_pusht/PushT-v0"


@dataclass
class ChunkTrainConfig:
    data_dir: Path = Path("data")
    policy_type: Literal["mse", "flow"] = "mse"
    init_checkpoint: Path | None = None
    output_checkpoint: Path = Path("chunk_policy_strong.pt")
    rollout_dir: Path = Path("rollouts_chunk_strong")

    chunk_size: int = 8
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    batch_size: int = 512
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 80
    train_fraction: float = 0.95

    flow_num_steps: int = 10
    eval_interval: int = 10
    eval_episodes: int = 50
    eval_seed_base: int = 10_000
    max_episode_steps: int | None = 300
    save_gif: bool = True
    seed: int = 42


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(config: ChunkTrainConfig):
    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)
    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )
    return normalizer, dataset, states.shape[1], actions.shape[1]


@torch.no_grad()
def evaluate_policy(
    model: nn.Module,
    normalizer: Normalizer,
    config: ChunkTrainConfig,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    action_low = env.action_space.low
    action_high = env.action_space.high
    returns: list[float] = []
    max_rewards: list[float] = []
    frames: list[np.ndarray] = []

    model.eval()
    for episode_idx in range(config.eval_episodes):
        obs, _ = env.reset(seed=config.eval_seed_base + episode_idx)
        done = False
        step = 0
        chunk_i = config.chunk_size
        action_chunk: np.ndarray | None = None
        ep_return = 0.0
        ep_max_reward = -float("inf")

        while not done:
            if action_chunk is None or chunk_i >= config.chunk_size:
                state = torch.from_numpy(normalizer.normalize_state(obs)).float().to(device)
                pred = (
                    model.sample_actions(
                        state.unsqueeze(0),
                        num_steps=config.flow_num_steps,
                    )
                    .cpu()
                    .numpy()[0]
                )
                action_chunk = normalizer.denormalize_action(pred)
                action_chunk = np.clip(action_chunk, action_low, action_high)
                chunk_i = 0

            obs, reward, terminated, truncated, _ = env.step(
                action_chunk[chunk_i].astype(np.float32)
            )
            done = terminated or truncated
            step += 1
            chunk_i += 1
            ep_return += float(reward)
            ep_max_reward = max(ep_max_reward, float(reward))
            if config.save_gif and episode_idx == 0:
                frames.append(env.render())
            if config.max_episode_steps is not None and step >= config.max_episode_steps:
                done = True

        returns.append(ep_return)
        max_rewards.append(ep_max_reward)

    env.close()
    if config.save_gif and frames:
        config.rollout_dir.mkdir(parents=True, exist_ok=True)
        gif_path = config.rollout_dir / f"{config.policy_type}_epoch_{epoch:04d}.gif"
        imageio.mimsave(gif_path, frames, fps=20)
        print(f"[eval] saved {gif_path}")
    return float(np.mean(returns)), float(np.mean(max_rewards))


def main() -> None:
    config = tyro.cli(ChunkTrainConfig, description="Train stronger action-chunk policies.")
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    normalizer, dataset, state_dim, action_dim = load_data(config)
    train_len = int(len(dataset) * config.train_fraction)
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(config.seed),
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = build_policy(
        config.policy_type,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device)
    if config.init_checkpoint is not None and config.init_checkpoint.exists():
        model.load_state_dict(torch.load(config.init_checkpoint, map_location=device))
        print(f"loaded init checkpoint: {config.init_checkpoint}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_max_reward = -float("inf")

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for state, action_chunk in train_loader:
            state = state.to(device)
            action_chunk = action_chunk.to(device)
            loss = model.compute_loss(state, action_chunk)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for state, action_chunk in val_loader:
                state = state.to(device)
                action_chunk = action_chunk.to(device)
                val_losses.append(float(model.compute_loss(state, action_chunk).item()))

        print(
            f"[epoch {epoch:04d}] "
            f"train_loss={np.mean(train_losses):.6f} "
            f"val_loss={np.mean(val_losses):.6f}"
        )

        if epoch % config.eval_interval == 0 or epoch == config.num_epochs:
            eval_return, eval_max_reward = evaluate_policy(model, normalizer, config, device, epoch)
            print(
                f"[eval {epoch:04d}] return={eval_return:.3f} "
                f"max_reward={eval_max_reward:.3f}"
            )
            if eval_max_reward >= best_max_reward:
                best_max_reward = eval_max_reward
                torch.save(model.state_dict(), config.output_checkpoint)
                print(
                    f"[save best] {config.output_checkpoint} "
                    f"best_max_reward={best_max_reward:.3f}"
                )
            else:
                print(f"[no save] best_max_reward={best_max_reward:.3f}")


if __name__ == "__main__":
    main()
