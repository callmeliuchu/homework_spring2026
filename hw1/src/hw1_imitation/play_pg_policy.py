"""Play or record a PPO fine-tuned Push-T chunk policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import tyro

from hw1_imitation.data import Normalizer, download_pusht, load_pusht_zarr
from hw1_imitation.policy_gradient import ENV_ID, GaussianChunkPolicy, PPOConfig
from hw1_imitation.policy_gradient_flow import FlowPPOConfig, GaussianFlowChunkPolicy
from hw1_imitation.policy_gradient_step import StepGaussianPolicy, StepPPOConfig


@dataclass
class PlayPGConfig:
    checkpoint_path: Path = Path("pg_chunk_ppo.pt")
    data_dir: Path = Path("data")
    gif_path: Path | None = Path("rollouts_pg/play_pg_policy.gif")

    chunk_size: int = 8
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    init_log_std: float = -1.0

    num_episodes: int = 1
    max_steps: int | None = None
    seed: int = 0
    render_mode: Literal["human", "rgb_array"] = "rgb_array"
    action_mode: Literal["mean", "sample"] = "mean"


def load_normalizer(data_dir: Path) -> tuple[Normalizer, int, int]:
    zarr_path = download_pusht(data_dir)
    states, actions, _ = load_pusht_zarr(zarr_path)
    return Normalizer.from_data(states, actions), states.shape[1], actions.shape[1]


def config_value(checkpoint: dict, name: str, fallback):
    saved_config = checkpoint.get("config")
    if saved_config is None:
        return fallback
    if isinstance(saved_config, dict):
        return saved_config.get(name, fallback)
    return getattr(saved_config, name, fallback)


def load_pg_policy(
    config: PlayPGConfig,
    device: torch.device,
) -> tuple[GaussianChunkPolicy, Normalizer, int]:
    normalizer, state_dim, action_dim = load_normalizer(config.data_dir)
    # Older checkpoints were saved by running policy_gradient.py as __main__,
    # so pickle looks for __main__.PPOConfig while loading.
    setattr(sys.modules["__main__"], "PPOConfig", PPOConfig)
    setattr(sys.modules["__main__"], "FlowPPOConfig", FlowPPOConfig)
    setattr(sys.modules["__main__"], "StepPPOConfig", StepPPOConfig)
    checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=False)

    chunk_size = int(config_value(checkpoint, "chunk_size", config.chunk_size))
    hidden_dims = tuple(config_value(checkpoint, "hidden_dims", config.hidden_dims))
    init_log_std = float(config_value(checkpoint, "init_log_std", config.init_log_std))
    if checkpoint.get("policy_class") == "StepGaussianPolicy":
        model = StepGaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            init_log_std=init_log_std,
        ).to(device)
        chunk_size = 1
    elif checkpoint.get("policy_class") == "GaussianFlowChunkPolicy":
        model = GaussianFlowChunkPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
            flow_num_steps=int(config_value(checkpoint, "flow_num_steps", 10)),
            init_log_std=init_log_std,
        ).to(device)
    else:
        model = GaussianChunkPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
            init_log_std=init_log_std,
        ).to(device)
    model.load_state_dict(checkpoint["policy"])
    model.eval()
    return model, normalizer, chunk_size


@torch.no_grad()
def predict_chunk(
    model: GaussianChunkPolicy,
    normalizer: Normalizer,
    obs: np.ndarray,
    device: torch.device,
    action_low: np.ndarray,
    action_high: np.ndarray,
    action_mode: Literal["mean", "sample"],
) -> np.ndarray:
    state = torch.from_numpy(normalizer.normalize_state(obs)).float().to(device)
    if action_mode == "mean":
        action_norm = model.mean(state.unsqueeze(0)).cpu().numpy()[0]
    else:
        action_norm, _, _ = model.sample(state.unsqueeze(0))
        action_norm = action_norm.cpu().numpy()[0]
    if action_norm.ndim == 1:
        action_norm = action_norm[None, :]
    action = normalizer.denormalize_action(action_norm)
    return np.clip(action, action_low, action_high)


def run_episode(
    env: gym.Env,
    model: GaussianChunkPolicy,
    normalizer: Normalizer,
    device: torch.device,
    *,
    chunk_size: int,
    max_steps: int | None,
    seed: int,
    action_mode: Literal["mean", "sample"],
    collect_frames: bool,
) -> tuple[float, list[np.ndarray]]:
    obs, _ = env.reset(seed=seed)
    action_low = env.action_space.low
    action_high = env.action_space.high

    frames: list[np.ndarray] = []
    done = False
    episode_return = 0.0
    step = 0
    chunk_index = chunk_size
    action_chunk: np.ndarray | None = None

    while not done:
        if action_chunk is None or chunk_index >= chunk_size:
            action_chunk = predict_chunk(
                model,
                normalizer,
                obs,
                device,
                action_low,
                action_high,
                action_mode,
            )
            chunk_index = 0

        obs, reward, terminated, truncated, _ = env.step(
            action_chunk[chunk_index].astype(np.float32)
        )
        if env.render_mode == "human":
            env.render()
        if collect_frames:
            frames.append(env.render())

        episode_return += float(reward)
        done = terminated or truncated
        chunk_index += 1
        step += 1

        if max_steps is not None and step >= max_steps:
            break

    return episode_return, frames


def main() -> None:
    config = tyro.cli(PlayPGConfig, description="Play a PPO fine-tuned Push-T policy.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalizer, chunk_size = load_pg_policy(config, device)

    env = gym.make(ENV_ID, obs_type="state", render_mode=config.render_mode)
    all_frames: list[np.ndarray] = []
    rewards: list[float] = []
    try:
        for episode_idx in range(config.num_episodes):
            reward, frames = run_episode(
                env,
                model,
                normalizer,
                device,
                chunk_size=chunk_size,
                max_steps=config.max_steps,
                seed=config.seed + episode_idx,
                action_mode=config.action_mode,
                collect_frames=config.gif_path is not None,
            )
            rewards.append(reward)
            all_frames.extend(frames)
            print(f"episode {episode_idx}: return={reward:.3f}")
    finally:
        env.close()

    print(f"mean return: {np.mean(rewards):.3f}")
    if config.gif_path is not None and all_frames:
        config.gif_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(config.gif_path, all_frames, fps=20)
        print(f"saved gif: {config.gif_path}")


if __name__ == "__main__":
    main()
