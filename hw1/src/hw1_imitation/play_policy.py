"""Play a trained Push-T policy checkpoint in the environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
import tyro

from hw1_imitation.data import Normalizer, download_pusht, load_pusht_zarr
from hw1_imitation.model import PolicyType, build_policy

ENV_ID = "gym_pusht/PushT-v0"


@dataclass
class PlayConfig:
    checkpoint_path: Path
    data_dir: Path = Path("data")
    policy_type: PolicyType = "mse"
    chunk_size: int = 8
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    flow_num_steps: int = 10
    num_episodes: int = 1
    max_steps: int | None = None
    seed: int = 0
    render_mode: Literal["human", "rgb_array"] = "human"


def load_normalizer(data_dir: Path) -> tuple[Normalizer, int, int]:
    zarr_path = download_pusht(data_dir)
    states, actions, _ = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)
    return normalizer, states.shape[1], actions.shape[1]


def load_policy(config: PlayConfig, device: torch.device) -> tuple[torch.nn.Module, Normalizer]:
    normalizer, state_dim, action_dim = load_normalizer(config.data_dir)
    model = build_policy(
        config.policy_type,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device)

    state_dict = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, normalizer


def run_episode(
    env: gym.Env,
    model: torch.nn.Module,
    normalizer: Normalizer,
    device: torch.device,
    *,
    chunk_size: int,
    flow_num_steps: int,
    max_steps: int | None,
    seed: int,
) -> float:
    obs, _ = env.reset(seed=seed)
    action_low = env.action_space.low
    action_high = env.action_space.high

    done = False
    episode_reward = 0.0
    step = 0
    chunk_index = chunk_size
    action_chunk: np.ndarray | None = None

    while not done:
        if action_chunk is None or chunk_index >= chunk_size:
            state = torch.from_numpy(normalizer.normalize_state(obs)).float().to(device)
            with torch.no_grad():
                pred_chunk = (
                    model.sample_actions(state.unsqueeze(0), num_steps=flow_num_steps)
                    .cpu()
                    .numpy()[0]
                )
            action_chunk = normalizer.denormalize_action(pred_chunk)
            action_chunk = np.clip(action_chunk, action_low, action_high)
            chunk_index = 0

        action = action_chunk[chunk_index].astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        if env.render_mode == "human":
            env.render()

        episode_reward += float(reward)
        done = terminated or truncated
        chunk_index += 1
        step += 1

        if max_steps is not None and step >= max_steps:
            break

    return episode_reward


def main() -> None:
    config = tyro.cli(PlayConfig, description="Play a trained Push-T policy.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalizer = load_policy(config, device)

    env = gym.make(ENV_ID, obs_type="state", render_mode=config.render_mode)
    try:
        rewards = []
        for episode_idx in range(config.num_episodes):
            reward = run_episode(
                env,
                model,
                normalizer,
                device,
                chunk_size=config.chunk_size,
                flow_num_steps=config.flow_num_steps,
                max_steps=config.max_steps,
                seed=config.seed + episode_idx,
            )
            rewards.append(reward)
            print(f"episode {episode_idx}: reward={reward:.3f}")
        print(f"mean reward: {np.mean(rewards):.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
