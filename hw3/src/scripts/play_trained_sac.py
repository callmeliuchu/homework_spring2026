#!/usr/bin/env python3

import argparse
import os
import sys
import tempfile
import time
from itertools import count
from pathlib import Path

import cv2
import gym
import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from gym.wrappers.clip_action import ClipAction
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gym.wrappers.rescale_action import RescaleAction

from agents.sac_agent import SoftActorCritic
from configs import sac_config
from infrastructure import pytorch_util as ptu


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
        return [np.asarray(frame) for frame in render_output if isinstance(frame, np.ndarray)]
    if isinstance(render_output, np.ndarray):
        return [np.asarray(render_output)]
    return []


def resolve_render_mode(env_name: str, mode: str) -> str:
    if mode == "human" and env_name.endswith("-v4"):
        return "rgb_array"
    return "human" if mode == "human" else "rgb_array"


def show_frame(frame: np.ndarray, sleep: float) -> None:
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("play_trained_sac", frame_bgr)
    cv2.waitKey(max(1, int(sleep * 1000)))


def make_config(config_file: str) -> tuple[dict, dict]:
    with open(config_file, "r") as f:
        config_kwargs = yaml.safe_load(f)

    base_config_name = config_kwargs.pop("base_config")
    config = sac_config.configs[base_config_name](**config_kwargs)
    return config, {"base_config": base_config_name, **config_kwargs}


def make_eval_env(config_kwargs: dict, mode: str) -> gym.Env:
    env_name = config_kwargs["env_name"]
    render_mode = resolve_render_mode(env_name, mode)

    return RecordEpisodeStatistics(
        ClipAction(
            RescaleAction(
                gym.make(env_name, render_mode=render_mode),
                -1,
                1,
            )
        )
    )


def load_agent(
    run_dir: str, config_file: str, device: torch.device
) -> tuple[SoftActorCritic, dict]:
    model_path = os.path.join(run_dir, "agent.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing checkpoint file: {model_path}")

    ptu.device = device
    config, config_kwargs = make_config(config_file)
    env = make_eval_env(config_kwargs, mode="rgb_array")
    try:
        agent = SoftActorCritic(
            env.observation_space.shape,
            env.action_space.shape[0],
            **config["agent_kwargs"],
        ).to(device)
    finally:
        env.close()

    state_dict = torch.load(model_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
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
        help="Required when mode=rgb_array. Example: previews/halfcheetah_trained.gif",
    )
    parser.add_argument(
        "--forever",
        action="store_true",
        help="Continuously restart episodes until interrupted (Ctrl+C).",
    )
    args = parser.parse_args()

    if args.mode == "rgb_array" and args.video_path is None:
        parser.error("--video_path is required when --mode=rgb_array")

    ptu.init_gpu(use_gpu=False)
    device = ptu.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent, _ = load_agent(args.run_dir, args.config_file, device)
    config, config_kwargs = make_config(args.config_file)
    env = make_eval_env(config_kwargs, mode=args.mode)
    ep_len = config["ep_len"] or env.spec.max_episode_steps
    max_steps = min(args.max_steps, ep_len)

    try:
        all_frames: list[np.ndarray] = []
        episode_iter = count() if args.forever else range(args.episodes)
        for episode in episode_iter:
            obs = reset_env(env)
            episode_return = 0.0

            for _ in range(max_steps):
                action = agent.get_action(obs)
                obs, reward, done, _ = step_env(env, action)
                episode_return += float(reward)

                if args.mode == "human":
                    frames = collect_frames(env.render())
                    if frames:
                        show_frame(frames[-1], args.sleep)
                    else:
                        time.sleep(args.sleep)
                else:
                    all_frames.extend(collect_frames(env.render()))

                if done:
                    break

            print(f"episode {episode}: return={episode_return:.2f}")

        if args.mode == "rgb_array":
            save_video(all_frames, args.video_path, fps=args.fps)
            print(f"saved video to: {args.video_path}")
    finally:
        env.close()
        if args.mode == "human":
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
