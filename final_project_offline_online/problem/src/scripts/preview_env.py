#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import ogbench


def save_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("No frames collected.")
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=1000 / fps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="cube-single-play-singletask-task1-v0",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument(
        "--mode",
        type=str,
        default="window",
        choices=("window", "gif"),
        help="window: live OpenCV preview, gif: save frames to a GIF.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--gif_path", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "gif" and args.gif_path is None:
        parser.error("--gif_path is required when --mode=gif")

    env = ogbench.make_env_and_datasets(args.env_name, env_only=True)
    frames: list[np.ndarray] = []

    try:
        for episode in range(args.episodes):
            _, _ = env.reset()
            episode_return = 0.0

            for _ in range(args.max_steps):
                frame = env.render()
                if frame is not None:
                    if args.mode == "window":
                        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.imshow("offline-online-preview", bgr)
                        if cv2.waitKey(1) & 0xFF == 27:
                            return
                    else:
                        frames.append(np.asarray(frame))

                action = env.action_space.sample()
                _, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                time.sleep(args.sleep)

                if terminated or truncated:
                    break

            print(f"episode {episode}: return={episode_return:.2f}")

        if args.mode == "gif":
            save_gif(frames, args.gif_path, args.fps)
            print(f"saved gif to: {args.gif_path}")
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
