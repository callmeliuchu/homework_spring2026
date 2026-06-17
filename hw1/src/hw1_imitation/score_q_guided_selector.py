"""Evaluate a Q-guided Flow bundle by selecting the best chunk among samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
import tyro

from hw1_imitation.data import Normalizer, download_pusht, load_pusht_zarr
from hw1_imitation.model import FlowMatchingPolicy, build_policy
from hw1_imitation.q_guided_flow_finetune import (
    ENV_ID,
    TwinQ,
    sample_flow_flat,
)


@dataclass
class SelectorScoreConfig:
    bundle_path: Path
    data_dir: Path = Path("data")
    num_candidates: int = 8
    num_episodes: int = 100
    seed: int = 10_000
    max_steps: int = 300
    chunk_size: int = 8
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    flow_num_steps: int = 10


def load_normalizer(data_dir: Path) -> tuple[Normalizer, int, int]:
    states, actions, _ = load_pusht_zarr(download_pusht(data_dir))
    return Normalizer.from_data(states, actions), states.shape[1], actions.shape[1]


@torch.no_grad()
def select_chunk(
    flow: FlowMatchingPolicy,
    critic: TwinQ,
    normalizer: Normalizer,
    obs: np.ndarray,
    config: SelectorScoreConfig,
    device: torch.device,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    state_np = normalizer.normalize_state(obs).astype(np.float32)
    state = torch.from_numpy(state_np).to(device).unsqueeze(0)
    state_batch = state.repeat(config.num_candidates, 1)
    action_flat = sample_flow_flat(flow, state_batch, config)  # type: ignore[arg-type]
    q1, q2 = critic(state_batch, action_flat)
    best_idx = torch.minimum(q1, q2).argmax().item()
    action_chunk = action_flat[best_idx].cpu().numpy().reshape(config.chunk_size, flow.action_dim)
    raw_chunk = normalizer.denormalize_action(action_chunk)
    return np.clip(raw_chunk, action_low, action_high)


def main() -> None:
    config = tyro.cli(SelectorScoreConfig, description="Score Q-selected Flow chunks.")
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalizer, state_dim, action_dim = load_normalizer(config.data_dir)
    bundle = torch.load(config.bundle_path, map_location=device, weights_only=False)

    flow = build_policy(
        "flow",
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device)
    flow.load_state_dict(bundle["flow"])
    flow.eval()

    critic = TwinQ(
        state_dim,
        config.chunk_size * action_dim,
        config.hidden_dims,
    ).to(device)
    critic.load_state_dict(bundle["critic"])
    critic.eval()

    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    returns: list[float] = []
    max_rewards: list[float] = []
    lengths: list[int] = []
    try:
        for episode_idx in range(config.num_episodes):
            obs, _ = env.reset(seed=config.seed + episode_idx)
            done = False
            step = 0
            chunk_i = config.chunk_size
            raw_chunk: np.ndarray | None = None
            ep_return = 0.0
            ep_max_reward = 0.0

            while not done and step < config.max_steps:
                if raw_chunk is None or chunk_i >= config.chunk_size:
                    raw_chunk = select_chunk(
                        flow,
                        critic,
                        normalizer,
                        obs,
                        config,
                        device,
                        env.action_space.low,
                        env.action_space.high,
                    )
                    chunk_i = 0

                obs, reward, terminated, truncated, _ = env.step(raw_chunk[chunk_i].astype(np.float32))
                ep_return += float(reward)
                ep_max_reward = max(ep_max_reward, float(reward))
                done = terminated or truncated
                step += 1
                chunk_i += 1

            returns.append(ep_return)
            max_rewards.append(ep_max_reward)
            lengths.append(step)
    finally:
        env.close()

    returns_np = np.asarray(returns)
    max_rewards_np = np.asarray(max_rewards)
    lengths_np = np.asarray(lengths)
    print(f"bundle: {config.bundle_path}")
    print(f"num_candidates: {config.num_candidates}")
    print(f"episodes: {config.num_episodes}")
    print(f"mean_return: {returns_np.mean():.4f}")
    print(f"std_return: {returns_np.std():.4f}")
    print(f"mean_max_reward: {max_rewards_np.mean():.4f}")
    print(f"std_max_reward: {max_rewards_np.std():.4f}")
    print(f"success_rate@0.8: {(max_rewards_np >= 0.8).mean():.4f}")
    print(f"mean_episode_length: {lengths_np.mean():.2f}")


if __name__ == "__main__":
    main()
