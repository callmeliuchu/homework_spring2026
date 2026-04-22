"""Score a trained Push-T policy checkpoint over many episodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
import tyro

from hw1_imitation.play_policy import load_policy

ENV_ID = "gym_pusht/PushT-v0"


@dataclass
class ScoreConfig:
    checkpoint_path: Path
    data_dir: Path = Path("data")
    policy_type: str = "mse"
    chunk_size: int = 8
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    flow_num_steps: int = 10
    num_episodes: int = 100
    max_steps: int | None = None
    seed: int = 0
    verbose: bool = False


def score_policy(config: ScoreConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalizer = load_policy(config, device)

    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    action_low = env.action_space.low
    action_high = env.action_space.high

    episode_returns: list[float] = []
    episode_max_rewards: list[float] = []
    episode_lengths: list[int] = []

    try:
        for episode_idx in range(config.num_episodes):
            obs, _ = env.reset(seed=config.seed + episode_idx)
            done = False
            chunk_index = config.chunk_size
            action_chunk: np.ndarray | None = None
            episode_return = 0.0
            episode_max_reward = 0.0
            step = 0

            while not done:
                if action_chunk is None or chunk_index >= config.chunk_size:
                    state = (
                        torch.from_numpy(normalizer.normalize_state(obs))
                        .float()
                        .to(device)
                    )
                    with torch.no_grad():
                        pred_chunk = (
                            model.sample_actions(
                                state.unsqueeze(0), num_steps=config.flow_num_steps
                            )
                            .cpu()
                            .numpy()[0]
                        )
                    action_chunk = normalizer.denormalize_action(pred_chunk)
                    action_chunk = np.clip(action_chunk, action_low, action_high)
                    chunk_index = 0

                action = action_chunk[chunk_index].astype(np.float32)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                episode_max_reward = max(episode_max_reward, float(reward))
                done = terminated or truncated
                chunk_index += 1
                step += 1

                if config.max_steps is not None and step >= config.max_steps:
                    break

            episode_returns.append(episode_return)
            episode_max_rewards.append(episode_max_reward)
            episode_lengths.append(step)

            if config.verbose:
                print(
                    f"episode {episode_idx:03d} | "
                    f"return={episode_return:.3f} | "
                    f"max_reward={episode_max_reward:.3f} | "
                    f"steps={step}"
                )
    finally:
        env.close()

    returns = np.asarray(episode_returns)
    max_rewards = np.asarray(episode_max_rewards)
    lengths = np.asarray(episode_lengths)

    print(f"checkpoint: {config.checkpoint_path}")
    print(f"policy_type: {config.policy_type}")
    print(f"episodes: {config.num_episodes}")
    print(f"mean_return: {returns.mean():.4f}")
    print(f"std_return: {returns.std():.4f}")
    print(f"mean_max_reward: {max_rewards.mean():.4f}")
    print(f"std_max_reward: {max_rewards.std():.4f}")
    print(f"success_rate@0.8: {(max_rewards >= 0.8).mean():.4f}")
    print(f"mean_episode_length: {lengths.mean():.2f}")


def main() -> None:
    config = tyro.cli(ScoreConfig, description="Score a trained Push-T policy.")
    score_policy(config)


if __name__ == "__main__":
    main()
