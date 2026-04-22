#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import gym
import imageio.v2 as imageio
import numpy as np


def reset_env(env: gym.Env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def step_env(env: gym.Env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        return obs, reward, done, info
    obs, reward, done, info = out
    return obs, reward, done, info


def save_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("No frames were collected to save.")

    path.parent.mkdir(parents=True, exist_ok=True)
    target_h, target_w = frames[0].shape[:2]
    normalized_frames = []
    for frame in frames:
        if frame.shape[:2] != (target_h, target_w):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        normalized_frames.append(frame)
    imageio.mimsave(path, normalized_frames, duration=1000 / fps)


def collect_frames(render_output) -> list[np.ndarray]:
    if render_output is None:
        return []
    if isinstance(render_output, list):
        frames = []
        for item in render_output:
            if isinstance(item, np.ndarray) and item.ndim >= 2:
                frames.append(np.asarray(item))
        return frames
    if isinstance(render_output, np.ndarray) and render_output.ndim >= 2:
        return [np.asarray(render_output)]
    return []


def run_preview(
    env_name: str,
    episodes: int,
    max_steps: int,
    sleep: float,
    mode: str,
    video_path: Path | None,
    fps: int,
) -> None:
    render_mode = "human" if mode == "human" else "rgb_array"
    env = gym.make(env_name, render_mode=render_mode)

    try:
        all_frames: list[np.ndarray] = []
        for episode in range(episodes):
            _ = reset_env(env)
            episode_return = 0.0

            for _ in range(max_steps):
                action = env.action_space.sample()
                _, reward, done, _ = step_env(env, action)
                episode_return += float(reward)

                if mode == "human":
                    time.sleep(sleep)
                else:
                    all_frames.extend(collect_frames(env.render()))

                if done:
                    break

            print(f"episode {episode}: return={episode_return:.2f}")

        if mode == "rgb_array" and video_path is not None:
            save_video(all_frames, video_path, fps=fps)
            print(f"saved video to: {video_path}")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument(
        "--mode",
        type=str,
        default="human",
        choices=("human", "rgb_array"),
        help="Use human for a live window, or rgb_array to save a video.",
    )
    parser.add_argument(
        "--video_path",
        type=Path,
        default=None,
        help="Required when mode=rgb_array. Example: previews/cartpole.gif",
    )
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if args.mode == "rgb_array" and args.video_path is None:
        parser.error("--video_path is required when --mode=rgb_array")

    run_preview(
        env_name=args.env_name,
        episodes=args.episodes,
        max_steps=args.max_steps,
        sleep=args.sleep,
        mode=args.mode,
        video_path=args.video_path,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
