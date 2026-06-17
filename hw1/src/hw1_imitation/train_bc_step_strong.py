"""Train a stronger chunk_size=1 behavior-cloning policy for Push-T.

This script is deliberately separate from the starter `train.py` so experiments
do not overwrite existing homework files.  It trains a single-step MSE policy:

    normalized state -> normalized action

and periodically evaluates deterministic rollouts on fixed seeds, saving the
checkpoint with the best mean max reward.

Example:
    uv run python src/hw1_imitation/train_bc_step_strong.py \
        --init-checkpoint mse1.pt \
        --output-checkpoint mse1_strong.pt \
        --num-epochs 80
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
from hw1_imitation.policy_gradient_step import ENV_ID, StepGaussianPolicy


@dataclass
class BCStepConfig:
    data_dir: Path = Path("data")
    init_checkpoint: Path | None = Path("mse1.pt")
    output_checkpoint: Path = Path("mse1_strong.pt")
    rollout_dir: Path = Path("rollouts_bc_step")

    hidden_dims: tuple[int, ...] = (256, 256, 256)
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-5
    num_epochs: int = 80
    train_fraction: float = 0.95

    eval_interval: int = 10
    eval_episodes: int = 20
    eval_seed_base: int = 10_000
    max_episode_steps: int | None = 300
    save_gif: bool = True
    seed: int = 42


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(config: BCStepConfig):
    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)
    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=1,
        normalizer=normalizer,
    )
    return normalizer, dataset, states.shape[1], actions.shape[1]


@torch.no_grad()
def evaluate_policy(
    model: nn.Module,
    normalizer: Normalizer,
    config: BCStepConfig,
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
        ep_return = 0.0
        ep_max_reward = -float("inf")
        while not done:
            state_np = normalizer.normalize_state(obs).astype(np.float32)
            state = torch.from_numpy(state_np).float().to(device).unsqueeze(0)
            action_norm = model(state).cpu().numpy()[0]
            action_raw = normalizer.denormalize_action(action_norm)
            action_raw = np.clip(action_raw, action_low, action_high)

            obs, reward, terminated, truncated, _ = env.step(action_raw.astype(np.float32))
            done = terminated or truncated
            step += 1
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
        gif_path = config.rollout_dir / f"bc_step_epoch_{epoch:04d}.gif"
        imageio.mimsave(gif_path, frames, fps=20)
        print(f"[eval] saved {gif_path}")
    return float(np.mean(returns)), float(np.mean(max_rewards))


def main() -> None:
    config = tyro.cli(BCStepConfig, description="Train a stronger single-step BC policy.")
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    policy = StepGaussianPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config.hidden_dims,
        init_log_std=-4.0,
    ).to(device)
    if config.init_checkpoint is not None and config.init_checkpoint.exists():
        policy.load_bc_mean(config.init_checkpoint, device)
        print(f"loaded init checkpoint: {config.init_checkpoint}")
    model = policy.mlp
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    best_max_reward = -float("inf")
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for state, action_chunk in train_loader:
            state = state.to(device)
            action = action_chunk[:, 0, :].to(device)
            pred = model(state)
            loss = ((pred - action) ** 2).mean()

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
                action = action_chunk[:, 0, :].to(device)
                pred = model(state)
                val_losses.append(float(((pred - action) ** 2).mean().item()))

        print(
            f"[epoch {epoch:04d}] "
            f"train_loss={np.mean(train_losses):.6f} "
            f"val_loss={np.mean(val_losses):.6f}"
        )

        if epoch % config.eval_interval == 0 or epoch == config.num_epochs:
            eval_return, eval_max_reward = evaluate_policy(
                model,
                normalizer,
                config,
                device,
                epoch,
            )
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
