#!/usr/bin/env python3

import argparse
import json
import os
import time
from itertools import count

import gym
import numpy as np
import torch

from agents.pg_agent import PGAgent
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


def load_agent(run_dir: str, device: torch.device) -> tuple[PGAgent, dict]:
    flags_path = os.path.join(run_dir, "flags.json")
    model_path = os.path.join(run_dir, "agent.pt")
    if not os.path.exists(flags_path):
        raise FileNotFoundError(f"Missing flags file: {flags_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing checkpoint file: {model_path}")

    with open(flags_path, "r") as f:
        cfg = json.load(f)

    env = gym.make(cfg["env_name"], render_mode=None)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
    env.close()

    agent = PGAgent(
        ob_dim=ob_dim,
        ac_dim=ac_dim,
        discrete=discrete,
        n_layers=cfg["n_layers"],
        layer_size=cfg["layer_size"],
        gamma=cfg["discount"],
        learning_rate=cfg["learning_rate"],
        use_baseline=cfg["use_baseline"],
        use_reward_to_go=cfg["use_reward_to_go"],
        baseline_learning_rate=cfg["baseline_learning_rate"],
        baseline_gradient_steps=cfg["baseline_gradient_steps"],
        gae_lambda=cfg["gae_lambda"],
        normalize_advantages=cfg["normalize_advantages"],
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--forever",
        action="store_true",
        help="Continuously restart episodes until interrupted (Ctrl+C).",
    )
    args = parser.parse_args()

    ptu.init_gpu(use_gpu=False)
    device = ptu.device

    agent, cfg = load_agent(args.run_dir, device)
    env = gym.make(cfg["env_name"], render_mode="human")

    try:
        episode_iter = count() if args.forever else range(args.episodes)
        for episode in episode_iter:
            obs = reset_env(env)
            episode_return = 0.0

            for _ in range(args.max_steps):
                action = agent.actor.get_action(obs)
                if isinstance(action, np.ndarray):
                    if action.shape == ():
                        action = action.item()
                    elif action.size == 1:
                        action = action.reshape(()).item()
                obs, reward, done, _ = step_env(env, action)
                episode_return += float(reward)
                time.sleep(args.sleep)
                if done:
                    break

            print(f"episode {episode}: return={episode_return:.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

