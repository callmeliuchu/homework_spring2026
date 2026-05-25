#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

import configs
from agents import agents
from infrastructure import pytorch_util as ptu
from infrastructure import utils


def save_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("No frames collected.")
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=1000 / fps)


def load_agent(checkpoint_path: Path, base_config: str, env_name: str, use_gpu: bool, gpu_id: int):
    ptu.init_gpu(use_gpu=use_gpu, gpu_id=gpu_id)
    config = configs.configs[base_config](env_name)
    env, dataset = config["make_env_and_dataset"]()
    example_batch = dataset.sample(1)
    agent_cls = agents[config["agent"]]
    agent = agent_cls(
        example_batch["observations"].shape[1:],
        example_batch["actions"].shape[-1],
        **config["agent_kwargs"],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()
    return env, agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--base_config", type=str, default="sacbc")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--mode", type=str, default="gif", choices=("gif", "window"))
    parser.add_argument("--gif_path", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--which_gpu", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "gif" and args.gif_path is None:
        parser.error("--gif_path is required when --mode=gif")

    env, agent = load_agent(
        checkpoint_path=args.checkpoint_path,
        base_config=args.base_config,
        env_name=args.env_name,
        use_gpu=not args.no_gpu,
        gpu_id=args.which_gpu,
    )
    ep_len = args.max_steps or env.spec.max_episode_steps or env.max_episode_steps

    all_stats = []
    try:
        for episode_idx in range(args.episodes):
            traj = utils.sample_trajectory(env, agent, ep_len, render=True)
            stats = traj["episode_statistics"]
            all_stats.append(stats)
            print(
                f"episode={episode_idx} success={stats['s']} "
                f"return={stats['r']:.2f} length={stats['l']}"
            )

            if args.mode == "gif":
                if args.episodes == 1:
                    gif_path = args.gif_path
                else:
                    gif_path = args.gif_path.with_name(
                        f"{args.gif_path.stem}_ep{episode_idx}{args.gif_path.suffix}"
                    )
                save_gif(list(traj["image_obs"]), gif_path, args.fps)
                print(f"saved gif to: {gif_path}")
            else:
                for frame in traj["image_obs"]:
                    cv2.imshow("hw5-policy-preview", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(int(1000 / args.fps)) & 0xFF == 27:
                        return
    finally:
        env.close()
        cv2.destroyAllWindows()

    if all_stats:
        success_rate = float(np.mean([s["s"] for s in all_stats]))
        print(f"average_success_rate={success_rate:.3f}")


if __name__ == "__main__":
    main()
