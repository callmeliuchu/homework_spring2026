#!/usr/bin/env python3

import argparse
import time

import gym


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="CartPole-v0")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--sleep", type=float, default=0.02)
    args = parser.parse_args()

    env = gym.make(args.env_name, render_mode="human")

    try:
        for episode in range(args.episodes):
            obs = reset_env(env)
            episode_return = 0.0

            for _ in range(args.max_steps):
                action = env.action_space.sample()
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
