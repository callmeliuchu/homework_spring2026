"""Online advantage-weighted RL fine-tuning for a Flow action-chunk policy.

This script is intentionally simpler than PPO/TD3/Q-guided actor learning:

1. Start from a strong Flow BC checkpoint.
2. Roll out the current Flow policy in the real environment.
3. Compute Monte Carlo return-to-go for each executed action chunk.
4. Train a value baseline V(s) on those returns.
5. Update Flow by cloning its own high-advantage chunks more strongly.
6. Keep a small expert BC loss so the policy does not drift far from expert data.

It is still RL because the weights come from environment returns, but it avoids
asking a critic to score out-of-distribution actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import tyro
from torch import nn
from torch.utils.data import DataLoader

from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from hw1_imitation.model import FlowMatchingPolicy, build_policy
from hw1_imitation.q_guided_flow_finetune import ENV_ID, sample_flow_flat, weighted_flow_loss


@dataclass
class OnlineAWRConfig:
    data_dir: Path = Path("data")
    init_checkpoint: Path = Path("flow_chunk_strong_best.pt")
    output_checkpoint: Path = Path("flow_online_awr_best.pt")
    rollout_dir: Path = Path("rollouts_online_awr")

    chunk_size: int = 8
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    flow_num_steps: int = 10

    num_iters: int = 40
    episodes_per_iter: int = 8
    max_episode_steps: int = 300
    replay_size: int = 50_000

    batch_size: int = 256
    expert_batch_size: int = 256
    value_updates_per_iter: int = 300
    flow_updates_per_iter: int = 80
    value_lr: float = 3e-4
    flow_lr: float = 5e-6
    gamma: float = 0.99

    advantage_temperature: float = 1.0
    weight_clip: float = 8.0
    elite_quantile: float = 0.0
    expert_coef: float = 0.03
    rl_coef: float = 1.0

    eval_interval: int = 5
    eval_episodes: int = 50
    eval_seed_base: int = 10_000
    save_gif: bool = False
    seed: int = 10_000


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_mlp(in_dim: int, out_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for hidden in hidden_dims:
        layers += [nn.Linear(last, hidden), nn.ReLU()]
        last = hidden
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.net = make_mlp(state_dim, 1, hidden_dims)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class ReturnReplay:
    def __init__(self, state_dim: int, action_dim: int, capacity: int) -> None:
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

    def add_many(self, states: list[np.ndarray], actions: list[np.ndarray], returns: list[float]) -> None:
        for state, action, ret in zip(states, actions, returns, strict=True):
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.returns[self.ptr] = ret
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx], device=device),
            torch.as_tensor(self.actions[idx], device=device),
            torch.as_tensor(self.returns[idx], device=device),
        )

    def return_stats(self) -> tuple[float, float]:
        values = self.returns[: self.size]
        return float(values.mean()), float(values.std() + 1e-6)

    def return_quantile(self, quantile: float) -> float:
        return float(np.quantile(self.returns[: self.size], quantile))


def load_data(config: OnlineAWRConfig):
    states, actions, episode_ends = load_pusht_zarr(download_pusht(config.data_dir))
    normalizer = Normalizer.from_data(states, actions)
    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )
    return normalizer, dataset, states.shape[1], actions.shape[1]


@torch.no_grad()
def collect_rollouts(
    flow: FlowMatchingPolicy,
    normalizer: Normalizer,
    replay: ReturnReplay,
    config: OnlineAWRConfig,
    device: torch.device,
    iteration: int,
) -> tuple[float, float]:
    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    episode_returns: list[float] = []
    episode_max_rewards: list[float] = []
    flow.eval()

    try:
        for episode_idx in range(config.episodes_per_iter):
            obs, _ = env.reset(seed=config.seed + iteration * 1000 + episode_idx)
            done = False
            step = 0
            ep_states: list[np.ndarray] = []
            ep_actions: list[np.ndarray] = []
            ep_rewards: list[float] = []
            ep_discounts: list[float] = []
            ep_return = 0.0
            ep_max_reward = 0.0

            while not done and step < config.max_episode_steps:
                state_np = normalizer.normalize_state(obs).astype(np.float32)
                state = torch.from_numpy(state_np).to(device).unsqueeze(0)
                action_flat = sample_flow_flat(flow, state, config).cpu().numpy()[0]  # type: ignore[arg-type]
                action_chunk = action_flat.reshape(config.chunk_size, flow.action_dim)
                raw_chunk = np.clip(
                    normalizer.denormalize_action(action_chunk),
                    env.action_space.low,
                    env.action_space.high,
                )

                chunk_reward = 0.0
                chunk_steps = 0
                for action in raw_chunk:
                    obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
                    chunk_reward += float(reward)
                    ep_return += float(reward)
                    ep_max_reward = max(ep_max_reward, float(reward))
                    step += 1
                    chunk_steps += 1
                    done = terminated or truncated or step >= config.max_episode_steps
                    if done:
                        break

                ep_states.append(state_np)
                ep_actions.append(action_flat.astype(np.float32))
                ep_rewards.append(chunk_reward)
                ep_discounts.append(config.gamma**chunk_steps)

            running_return = 0.0
            ep_returns_to_go = [0.0] * len(ep_rewards)
            for idx in reversed(range(len(ep_rewards))):
                running_return = ep_rewards[idx] + ep_discounts[idx] * running_return
                ep_returns_to_go[idx] = running_return

            replay.add_many(ep_states, ep_actions, ep_returns_to_go)
            episode_returns.append(ep_return)
            episode_max_rewards.append(ep_max_reward)
    finally:
        env.close()

    return float(np.mean(episode_returns)), float(np.mean(episode_max_rewards))


def train_value(
    value_net: ValueNet,
    optimizer: torch.optim.Optimizer,
    replay: ReturnReplay,
    config: OnlineAWRConfig,
    device: torch.device,
) -> dict[str, float]:
    value_net.train()
    ret_mean, ret_std = replay.return_stats()
    stats: dict[str, float] = {}
    for _ in range(config.value_updates_per_iter):
        state, _, ret = replay.sample(config.batch_size, device)
        target = (ret - ret_mean) / ret_std
        pred = value_net(state)
        loss = ((pred - target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
        optimizer.step()
        stats = {"value_loss": float(loss.item()), "return_mean": ret_mean, "return_std": ret_std}
    return stats


def train_flow(
    flow: FlowMatchingPolicy,
    value_net: ValueNet,
    optimizer: torch.optim.Optimizer,
    replay: ReturnReplay,
    next_expert_batch,
    config: OnlineAWRConfig,
    device: torch.device,
) -> dict[str, float]:
    flow.train()
    value_net.eval()
    ret_mean, ret_std = replay.return_stats()
    stats: dict[str, float] = {}

    for _ in range(config.flow_updates_per_iter):
        state, action, ret = replay.sample(config.batch_size, device)
        target = (ret - ret_mean) / ret_std
        with torch.no_grad():
            advantage = target - value_net(state)
            weights = torch.exp((advantage / config.advantage_temperature).clamp(-5.0, 5.0))
            weights = weights.clamp(max=config.weight_clip)
            if config.elite_quantile > 0.0:
                threshold = replay.return_quantile(config.elite_quantile)
                weights = weights * (ret >= threshold).float()
        rl_loss = weighted_flow_loss(flow, state, action, weights)

        expert_state, expert_chunk = next_expert_batch()
        expert_state = expert_state.to(device)
        expert_action = expert_chunk.reshape(expert_chunk.shape[0], -1).to(device)
        expert_loss = weighted_flow_loss(
            flow,
            expert_state,
            expert_action,
            torch.ones(expert_state.shape[0], device=device),
        )

        loss = config.rl_coef * rl_loss + config.expert_coef * expert_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        optimizer.step()
        stats = {
            "flow_loss": float(loss.item()),
            "rl_loss": float(rl_loss.item()),
            "expert_loss": float(expert_loss.item()),
            "weight_mean": float(weights.mean().item()),
            "weight_max": float(weights.max().item()),
        }
    return stats


@torch.no_grad()
def evaluate(
    flow: FlowMatchingPolicy,
    normalizer: Normalizer,
    config: OnlineAWRConfig,
    device: torch.device,
    iteration: int,
) -> tuple[float, float]:
    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    returns: list[float] = []
    max_rewards: list[float] = []
    frames: list[np.ndarray] = []
    flow.eval()

    try:
        for episode_idx in range(config.eval_episodes):
            obs, _ = env.reset(seed=config.eval_seed_base + episode_idx)
            done = False
            step = 0
            chunk_i = config.chunk_size
            raw_chunk: np.ndarray | None = None
            ep_return = 0.0
            ep_max_reward = 0.0
            while not done and step < config.max_episode_steps:
                if raw_chunk is None or chunk_i >= config.chunk_size:
                    state_np = normalizer.normalize_state(obs).astype(np.float32)
                    state = torch.from_numpy(state_np).to(device).unsqueeze(0)
                    action_flat = sample_flow_flat(flow, state, config).cpu().numpy()[0]  # type: ignore[arg-type]
                    action_chunk = action_flat.reshape(config.chunk_size, flow.action_dim)
                    raw_chunk = np.clip(
                        normalizer.denormalize_action(action_chunk),
                        env.action_space.low,
                        env.action_space.high,
                    )
                    chunk_i = 0

                obs, reward, terminated, truncated, _ = env.step(raw_chunk[chunk_i].astype(np.float32))
                if config.save_gif and episode_idx == 0:
                    frames.append(env.render())
                ep_return += float(reward)
                ep_max_reward = max(ep_max_reward, float(reward))
                done = terminated or truncated
                step += 1
                chunk_i += 1
            returns.append(ep_return)
            max_rewards.append(ep_max_reward)
    finally:
        env.close()

    if config.save_gif and frames:
        config.rollout_dir.mkdir(parents=True, exist_ok=True)
        gif_path = config.rollout_dir / f"online_awr_iter_{iteration:04d}.gif"
        imageio.mimsave(gif_path, frames, fps=20)
        print(f"[eval] saved {gif_path}")
    return float(np.mean(returns)), float(np.mean(max_rewards))


def main() -> None:
    config = tyro.cli(OnlineAWRConfig, description="Online AWR fine-tuning for Flow policy.")
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    normalizer, dataset, state_dim, action_dim = load_data(config)
    expert_loader = DataLoader(
        dataset,
        batch_size=config.expert_batch_size,
        shuffle=True,
        drop_last=True,
    )
    expert_iter = iter(expert_loader)

    def next_expert_batch():
        nonlocal expert_iter
        try:
            return next(expert_iter)
        except StopIteration:
            expert_iter = iter(expert_loader)
            return next(expert_iter)

    flow = build_policy(
        "flow",
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device)
    flow.load_state_dict(torch.load(config.init_checkpoint, map_location=device))
    value_net = ValueNet(state_dim, config.hidden_dims).to(device)
    replay = ReturnReplay(state_dim, config.chunk_size * action_dim, config.replay_size)

    value_optimizer = torch.optim.AdamW(value_net.parameters(), lr=config.value_lr)
    flow_optimizer = torch.optim.AdamW(flow.parameters(), lr=config.flow_lr)

    start_return, start_max_reward = evaluate(flow, normalizer, config, device, iteration=0)
    best_max_reward = start_max_reward
    torch.save(flow.state_dict(), config.output_checkpoint)
    print(f"[start] return={start_return:.3f} max_reward={start_max_reward:.3f}")
    print(f"[save initial best] {config.output_checkpoint}")

    for iteration in range(1, config.num_iters + 1):
        rollout_return, rollout_max_reward = collect_rollouts(
            flow,
            normalizer,
            replay,
            config,
            device,
            iteration,
        )
        value_stats = train_value(value_net, value_optimizer, replay, config, device)
        flow_stats = train_flow(
            flow,
            value_net,
            flow_optimizer,
            replay,
            next_expert_batch,
            config,
            device,
        )
        print(
            f"[iter {iteration:04d}] "
            f"rollout_return={rollout_return:.3f} "
            f"rollout_max_reward={rollout_max_reward:.3f} "
            f"replay={replay.size} "
            f"value={value_stats.get('value_loss', float('nan')):.3f} "
            f"ret_mean={value_stats.get('return_mean', float('nan')):.3f} "
            f"flow={flow_stats.get('flow_loss', float('nan')):.3f} "
            f"rl={flow_stats.get('rl_loss', float('nan')):.3f} "
            f"expert={flow_stats.get('expert_loss', float('nan')):.3f} "
            f"w_mean={flow_stats.get('weight_mean', float('nan')):.3f} "
            f"w_max={flow_stats.get('weight_max', float('nan')):.3f}"
        )

        if iteration % config.eval_interval == 0 or iteration == config.num_iters:
            eval_return, eval_max_reward = evaluate(flow, normalizer, config, device, iteration)
            print(f"[eval {iteration:04d}] return={eval_return:.3f} max_reward={eval_max_reward:.3f}")
            if eval_max_reward >= best_max_reward:
                best_max_reward = eval_max_reward
                torch.save(flow.state_dict(), config.output_checkpoint)
                print(f"[save best] {config.output_checkpoint} best_max_reward={best_max_reward:.3f}")
            else:
                print(f"[no save] best_max_reward={best_max_reward:.3f}")


if __name__ == "__main__":
    main()
