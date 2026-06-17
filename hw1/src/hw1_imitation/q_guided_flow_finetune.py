"""Q-guided fine-tuning for a trained Flow action-chunk Push-T policy.

This is a stronger RL fine-tuning experiment than plain PPO:

1. Load a strong Flow policy as the behavior prior.
2. Collect action-chunk rollouts from the current Flow policy.
3. Train twin Q networks on macro transitions: (state, action_chunk, reward_sum, next_state).
4. Update the Flow policy toward chunks with higher Q values.
5. Keep small expert/prior behavior losses so the Flow does not drift too far.
6. Save the checkpoint that wins fixed-seed environment evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
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

ENV_ID = "gym_pusht/PushT-v0"


@dataclass
class QGuidedFlowConfig:
    data_dir: Path = Path("data")
    init_checkpoint: Path = Path("flow_chunk_strong_best.pt")
    output_checkpoint: Path = Path("flow_q_guided_best.pt")
    rollout_dir: Path = Path("rollouts_q_guided_flow")

    chunk_size: int = 8
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    batch_size: int = 256
    expert_batch_size: int = 256
    flow_num_steps: int = 10

    num_iters: int = 40
    episodes_per_iter: int = 4
    max_episode_steps: int = 300
    warmup_iters: int = 2
    critic_updates_per_iter: int = 200
    flow_updates_per_iter: int = 50
    replay_size: int = 80_000

    gamma: float = 0.99
    tau: float = 0.005
    critic_lr: float = 3e-4
    flow_lr: float = 1e-5
    target_noise: float = 0.03
    target_noise_clip: float = 0.08

    q_coef: float = 0.002
    aw_coef: float = 0.05
    aw_temperature: float = 20.0
    aw_weight_clip: float = 10.0
    expert_coef: float = 0.02
    prior_coef: float = 0.01

    eval_interval: int = 5
    eval_episodes: int = 50
    eval_seed_base: int = 10_000
    save_gif: bool = False
    seed: int = 42


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


class TwinQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.q1 = make_mlp(state_dim + action_dim, 1, hidden_dims)
        self.q2 = make_mlp(state_dim + action_dim, 1, hidden_dims)

    def forward(self, state: torch.Tensor, action_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action_flat], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_value(self, state: torch.Tensor, action_flat: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([state, action_flat], dim=-1)).squeeze(-1)


class ChunkReplay:
    def __init__(self, state_dim: int, action_dim: int, capacity: int) -> None:
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        discount: float,
        done: bool,
    ) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.discounts[self.ptr] = discount
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx], device=device),
            torch.as_tensor(self.actions[idx], device=device),
            torch.as_tensor(self.rewards[idx], device=device),
            torch.as_tensor(self.next_states[idx], device=device),
            torch.as_tensor(self.discounts[idx], device=device),
            torch.as_tensor(self.dones[idx], device=device),
        )


def load_data(config: QGuidedFlowConfig):
    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)
    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )
    return normalizer, dataset, states.shape[1], actions.shape[1]


def sample_flow_flat(
    flow: FlowMatchingPolicy,
    state: torch.Tensor,
    config: QGuidedFlowConfig,
    *,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    batch = state.shape[0]
    flat_dim = config.chunk_size * flow.action_dim
    x = torch.randn(batch, flat_dim, device=state.device, dtype=state.dtype) if noise is None else noise
    dt = 1.0 / config.flow_num_steps
    for k in range(config.flow_num_steps):
        t = torch.full((batch, 1), k / config.flow_num_steps, device=state.device, dtype=state.dtype)
        velocity = flow.mlp(torch.cat([state, x, t], dim=-1))
        x = x + velocity * dt
    return x


def weighted_flow_loss(
    flow: FlowMatchingPolicy,
    state: torch.Tensor,
    action_flat: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    batch = state.shape[0]
    noise = torch.randn_like(action_flat)
    t = torch.rand(batch, 1, device=state.device, dtype=state.dtype)
    xt = noise * (1.0 - t) + t * action_flat
    pred = flow.mlp(torch.cat([state, xt, t], dim=-1))
    target = action_flat - noise
    per_sample = ((pred - target) ** 2).mean(dim=-1)
    normalized_weights = weights / (weights.mean().detach() + 1e-6)
    return (per_sample * normalized_weights.detach()).mean()


@torch.no_grad()
def collect_rollouts(
    flow: FlowMatchingPolicy,
    normalizer: Normalizer,
    replay: ChunkReplay,
    config: QGuidedFlowConfig,
    device: torch.device,
    iteration: int,
) -> tuple[float, float]:
    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    returns: list[float] = []
    max_rewards: list[float] = []
    flow.eval()

    try:
        for episode_idx in range(config.episodes_per_iter):
            obs, _ = env.reset(seed=config.seed + iteration * 1000 + episode_idx)
            done = False
            step = 0
            ep_return = 0.0
            ep_max_reward = 0.0

            while not done and step < config.max_episode_steps:
                state_np = normalizer.normalize_state(obs).astype(np.float32)
                state = torch.from_numpy(state_np).to(device).unsqueeze(0)
                action_flat = sample_flow_flat(flow, state, config).cpu().numpy()[0]
                action_chunk = action_flat.reshape(config.chunk_size, flow.action_dim)
                raw_chunk = np.clip(
                    normalizer.denormalize_action(action_chunk),
                    env.action_space.low,
                    env.action_space.high,
                )

                chunk_reward = 0.0
                chunk_steps = 0
                for action in raw_chunk:
                    next_obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
                    chunk_reward += float(reward)
                    ep_return += float(reward)
                    ep_max_reward = max(ep_max_reward, float(reward))
                    step += 1
                    chunk_steps += 1
                    done = terminated or truncated or step >= config.max_episode_steps
                    if done:
                        break

                next_state_np = normalizer.normalize_state(next_obs).astype(np.float32)
                replay.add(
                    state_np,
                    action_flat.astype(np.float32),
                    chunk_reward,
                    next_state_np,
                    config.gamma**chunk_steps,
                    done,
                )
                obs = next_obs

            returns.append(ep_return)
            max_rewards.append(ep_max_reward)
    finally:
        env.close()

    return float(np.mean(returns)), float(np.mean(max_rewards))


def train_critic(
    flow_target: FlowMatchingPolicy,
    critic: TwinQ,
    critic_target: TwinQ,
    optimizer: torch.optim.Optimizer,
    replay: ChunkReplay,
    config: QGuidedFlowConfig,
    device: torch.device,
) -> dict[str, float]:
    stats: dict[str, float] = {}
    critic.train()
    for _ in range(config.critic_updates_per_iter):
        state, action, reward, next_state, discount, done = replay.sample(config.batch_size, device)
        with torch.no_grad():
            next_action = sample_flow_flat(flow_target, next_state, config)
            noise = (torch.randn_like(next_action) * config.target_noise).clamp(
                -config.target_noise_clip,
                config.target_noise_clip,
            )
            next_action = next_action + noise
            q1_t, q2_t = critic_target(next_state, next_action)
            target_q = reward + discount * (1.0 - done) * torch.minimum(q1_t, q2_t)

        q1, q2 = critic(state, action)
        loss = ((q1 - target_q) ** 2).mean() + ((q2 - target_q) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        optimizer.step()
        stats = {"critic_loss": float(loss.item()), "q_mean": float(q1.mean().item())}
    return stats


def train_flow(
    flow: FlowMatchingPolicy,
    prior_flow: FlowMatchingPolicy,
    critic: TwinQ,
    optimizer: torch.optim.Optimizer,
    next_expert_batch,
    replay: ChunkReplay,
    config: QGuidedFlowConfig,
    device: torch.device,
) -> dict[str, float]:
    flow.train()
    critic.eval()
    stats: dict[str, float] = {}

    for _ in range(config.flow_updates_per_iter):
        state, action, _, _, _, _ = replay.sample(config.batch_size, device)
        candidate = sample_flow_flat(flow, state, config)
        q_value = critic.q1_value(state, candidate)
        q_loss = -q_value.mean()

        with torch.no_grad():
            replay_q = critic.q1_value(state, action)
            candidate_q = critic.q1_value(state, candidate.detach())
            advantage = replay_q - candidate_q
            weights = torch.exp((advantage / config.aw_temperature).clamp(-5.0, 5.0))
            weights = weights.clamp(max=config.aw_weight_clip)
        aw_loss = weighted_flow_loss(flow, state, action, weights)

        expert_state, expert_chunk = next_expert_batch()
        expert_state = expert_state.to(device)
        expert_action = expert_chunk.reshape(expert_chunk.shape[0], -1).to(device)
        expert_loss = weighted_flow_loss(
            flow,
            expert_state,
            expert_action,
            torch.ones(expert_state.shape[0], device=device),
        )

        prior_noise = torch.randn(
            state.shape[0],
            config.chunk_size * flow.action_dim,
            device=device,
        )
        current_prior_sample = sample_flow_flat(flow, state, config, noise=prior_noise)
        with torch.no_grad():
            frozen_prior_sample = sample_flow_flat(prior_flow, state, config, noise=prior_noise)
        prior_loss = ((current_prior_sample - frozen_prior_sample) ** 2).mean()

        loss = (
            config.q_coef * q_loss
            + config.aw_coef * aw_loss
            + config.expert_coef * expert_loss
            + config.prior_coef * prior_loss
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        optimizer.step()

        stats = {
            "flow_loss": float(loss.item()),
            "q_loss": float(q_loss.item()),
            "aw_loss": float(aw_loss.item()),
            "expert_loss": float(expert_loss.item()),
            "prior_loss": float(prior_loss.item()),
            "actor_q": float(q_value.mean().item()),
        }
    return stats


@torch.no_grad()
def evaluate(
    flow: FlowMatchingPolicy,
    normalizer: Normalizer,
    config: QGuidedFlowConfig,
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
            ep_return = 0.0
            ep_max_reward = 0.0
            chunk_i = config.chunk_size
            raw_chunk: np.ndarray | None = None

            while not done and step < config.max_episode_steps:
                if raw_chunk is None or chunk_i >= config.chunk_size:
                    state_np = normalizer.normalize_state(obs).astype(np.float32)
                    state = torch.from_numpy(state_np).to(device).unsqueeze(0)
                    action_flat = sample_flow_flat(flow, state, config).cpu().numpy()[0]
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
                done = terminated or truncated
                step += 1
                chunk_i += 1
                ep_return += float(reward)
                ep_max_reward = max(ep_max_reward, float(reward))

            returns.append(ep_return)
            max_rewards.append(ep_max_reward)
    finally:
        env.close()

    if config.save_gif and frames:
        config.rollout_dir.mkdir(parents=True, exist_ok=True)
        gif_path = config.rollout_dir / f"q_guided_iter_{iteration:04d}.gif"
        imageio.mimsave(gif_path, frames, fps=20)
        print(f"[eval] saved {gif_path}")
    return float(np.mean(returns)), float(np.mean(max_rewards))


def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p, p_t in zip(source.parameters(), target.parameters(), strict=True):
            p_t.data.mul_(1.0 - tau).add_(tau * p.data)


def main() -> None:
    config = tyro.cli(QGuidedFlowConfig, description="Q-guided Flow fine-tuning.")
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

    prior_flow = build_policy(
        "flow",
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device)
    prior_flow.load_state_dict(torch.load(config.init_checkpoint, map_location=device))
    prior_flow.eval()
    for param in prior_flow.parameters():
        param.requires_grad = False

    flow_target = build_policy(
        "flow",
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device)
    flow_target.load_state_dict(flow.state_dict())

    flat_action_dim = config.chunk_size * action_dim
    critic = TwinQ(state_dim, flat_action_dim, config.hidden_dims).to(device)
    critic_target = TwinQ(state_dim, flat_action_dim, config.hidden_dims).to(device)
    critic_target.load_state_dict(critic.state_dict())

    replay = ChunkReplay(state_dim, flat_action_dim, config.replay_size)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=config.critic_lr)
    flow_optimizer = torch.optim.AdamW(flow.parameters(), lr=config.flow_lr)

    best_max_reward = -float("inf")
    start_return, start_max_reward = evaluate(flow, normalizer, config, device, iteration=0)
    print(f"[start] return={start_return:.3f} max_reward={start_max_reward:.3f}")

    for iteration in range(1, config.num_iters + 1):
        rollout_return, rollout_max_reward = collect_rollouts(
            flow,
            normalizer,
            replay,
            config,
            device,
            iteration,
        )

        critic_stats: dict[str, float] = {}
        flow_stats: dict[str, float] = {}
        if replay.size >= config.batch_size and iteration > config.warmup_iters:
            critic_stats = train_critic(
                flow_target,
                critic,
                critic_target,
                critic_optimizer,
                replay,
                config,
                device,
            )
            flow_stats = train_flow(
                flow,
                prior_flow,
                critic,
                flow_optimizer,
                next_expert_batch,
                replay,
                config,
                device,
            )
            polyak_update(critic, critic_target, config.tau)
            polyak_update(flow, flow_target, config.tau)

        print(
            f"[iter {iteration:04d}] "
            f"rollout_return={rollout_return:.3f} "
            f"rollout_max_reward={rollout_max_reward:.3f} "
            f"replay={replay.size} "
            f"critic={critic_stats.get('critic_loss', float('nan')):.3f} "
            f"q={critic_stats.get('q_mean', float('nan')):.3f} "
            f"flow={flow_stats.get('flow_loss', float('nan')):.3f} "
            f"actor_q={flow_stats.get('actor_q', float('nan')):.3f} "
            f"aw={flow_stats.get('aw_loss', float('nan')):.3f} "
            f"expert={flow_stats.get('expert_loss', float('nan')):.3f} "
            f"prior={flow_stats.get('prior_loss', float('nan')):.3f}"
        )

        if iteration % config.eval_interval == 0 or iteration == config.num_iters:
            eval_return, eval_max_reward = evaluate(flow, normalizer, config, device, iteration)
            print(f"[eval {iteration:04d}] return={eval_return:.3f} max_reward={eval_max_reward:.3f}")
            if eval_max_reward >= best_max_reward:
                best_max_reward = eval_max_reward
                torch.save(flow.state_dict(), config.output_checkpoint)
                bundle_path = config.output_checkpoint.with_name(
                    f"{config.output_checkpoint.stem}_bundle.pt"
                )
                torch.save(
                    {
                        "flow": flow.state_dict(),
                        "critic": critic.state_dict(),
                        "config": asdict(config),
                    },
                    bundle_path,
                )
                print(f"[save best] {config.output_checkpoint} best_max_reward={best_max_reward:.3f}")
                print(f"[save bundle] {bundle_path}")
            else:
                print(f"[no save] best_max_reward={best_max_reward:.3f}")


if __name__ == "__main__":
    main()
