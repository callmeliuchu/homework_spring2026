#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

import configs
from agents import agents
from infrastructure import pytorch_util as ptu


def save_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("No frames collected.")
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=1000 / fps)


def rollout_episode(env, agent, max_steps: int) -> tuple[list[np.ndarray], dict]:
    observation, _ = env.reset()
    frames: list[np.ndarray] = []
    total_reward = 0.0
    success = 0.0
    length = 0

    for _ in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))

        action = agent.get_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        length += 1
        success = float(info.get("success", 0.0))

        if terminated or truncated:
            break

    return frames, {
        "success": success,
        "return": total_reward,
        "length": length,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default="sacbc")
    parser.add_argument("--env_name", type=str, default="cube-single-play-singletask-task1-v0")
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--gif_path", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--which_gpu", type=int, default=0)
    args = parser.parse_args()

    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    config = configs.configs[args.base_config](args.env_name)
    env, dataset = config["make_env_and_dataset"]()
    example_batch = dataset.sample(1)
    agent_cls = agents[config["agent"]]
    agent = agent_cls(
        example_batch["observations"].shape[1:],
        example_batch["actions"].shape[-1],
        **config["agent_kwargs"],
    )
    state_dict = torch.load(args.checkpoint_path, map_location=ptu.device)
    agent.load_state_dict(state_dict)
    agent.eval()

    max_steps = args.max_steps or env.spec.max_episode_steps or env.max_episode_steps

    best_frames: list[np.ndarray] = []
    best_stats: dict | None = None

    try:
        for episode in range(args.episodes):
            frames, stats = rollout_episode(env, agent, max_steps)
            print(
                f"episode {episode}: success={stats['success']:.1f}, "
                f"return={stats['return']:.2f}, length={stats['length']}"
            )

            if best_stats is None:
                best_frames, best_stats = frames, stats
                continue

            current_key = (stats["success"], stats["return"])
            best_key = (best_stats["success"], best_stats["return"])
            if current_key > best_key:
                best_frames, best_stats = frames, stats
    finally:
        env.close()

    if best_stats is None:
        raise RuntimeError("No rollout collected.")

    save_gif(best_frames, args.gif_path, args.fps)
    print(f"saved gif to: {args.gif_path}")
    print(
        f"selected rollout: success={best_stats['success']:.1f}, "
        f"return={best_stats['return']:.2f}, length={best_stats['length']}"
    )


if __name__ == "__main__":
    main()
