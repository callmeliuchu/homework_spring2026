"""Residual TD3 fine-tuning on top of a frozen Flow action-chunk policy.

The Flow policy is already a strong action prior.  This script freezes it and
learns a small action-chunk residual:

    chunk_norm = flow_chunk_norm(obs) + residual_scale * actor(obs)

The residual actor is trained with TD3.  This avoids needing a flow log-prob
and keeps the learned correction small enough that the strong BC/Flow behavior
is not immediately destroyed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import tyro
from torch import nn

from hw1_imitation.data import Normalizer, download_pusht, load_pusht_zarr
from hw1_imitation.model import FlowMatchingPolicy, build_policy

ENV_ID = "gym_pusht/PushT-v0"


@dataclass
class ResidualTD3Config:
    data_dir: Path = Path("data")
    flow_checkpoint: Path = Path("flow_chunk_strong_best.pt")
    residual_checkpoint: Path | None = None
    output_checkpoint: Path = Path("residual_td3_flow_best.pt")
    rollout_dir: Path = Path("rollouts_residual_td3")

    chunk_size: int = 8
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    flow_num_steps: int = 10
    flow_init: str = "random"  # "random" matches the original Flow sampler; "zero" is deterministic.

    num_iters: int = 50
    episodes_per_iter: int = 4
    max_episode_steps: int = 300
    updates_per_iter: int = 400
    warmup_iters: int = 2

    replay_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    residual_scale: float = 0.05
    exploration_noise: float = 0.05
    target_noise: float = 0.02
    target_noise_clip: float = 0.05
    policy_delay: int = 2
    residual_l2: float = 1e-3

    eval_interval: int = 5
    eval_episodes: int = 20
    eval_seed_base: int = 10_000
    save_gif: bool = True
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


class ResidualActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.net = make_mlp(state_dim, action_dim, hidden_dims)
        last_layer = self.net[-1]
        assert isinstance(last_layer, nn.Linear)
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(state))


class TwinCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.q1 = make_mlp(state_dim + action_dim, 1, hidden_dims)
        self.q2 = make_mlp(state_dim + action_dim, 1, hidden_dims)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([state, action], dim=-1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int) -> None:
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
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
            torch.as_tensor(self.dones[idx], device=device),
        )


def load_normalizer(data_dir: Path) -> tuple[Normalizer, int, int]:
    states, actions, _ = load_pusht_zarr(download_pusht(data_dir))
    return Normalizer.from_data(states, actions), states.shape[1], actions.shape[1]


@torch.no_grad()
def flow_action_chunk(
    flow: FlowMatchingPolicy,
    state: torch.Tensor,
    config: ResidualTD3Config,
) -> torch.Tensor:
    batch = state.shape[0]
    flat_dim = config.chunk_size * flow.action_dim
    if config.flow_init == "random":
        x = torch.randn(batch, flat_dim, device=state.device)
    else:
        x = torch.zeros(batch, flat_dim, device=state.device)
    dt = 1.0 / config.flow_num_steps
    for k in range(config.flow_num_steps):
        t = torch.full((batch, 1), k / config.flow_num_steps, device=state.device)
        v = flow.mlp(torch.cat([state, x, t], dim=-1))
        x = x + v * dt
    return x


def final_action_chunk_norm(
    flow: FlowMatchingPolicy,
    actor: ResidualActor,
    state: torch.Tensor,
    config: ResidualTD3Config,
    *,
    noise_std: float = 0.0,
) -> torch.Tensor:
    base = flow_action_chunk(flow, state, config)
    residual = actor(state)
    if noise_std > 0:
        residual = residual + torch.randn_like(residual) * noise_std
    residual = residual.clamp(-1.0, 1.0)
    return base + config.residual_scale * residual


def collect_episodes(
    env: gym.Env,
    flow: FlowMatchingPolicy,
    actor: ResidualActor,
    normalizer: Normalizer,
    replay: ReplayBuffer,
    config: ResidualTD3Config,
    device: torch.device,
    iteration: int,
) -> tuple[float, float]:
    action_low = env.action_space.low
    action_high = env.action_space.high
    episode_returns: list[float] = []
    episode_max_rewards: list[float] = []
    actor.eval()

    for ep in range(config.episodes_per_iter):
        obs, _ = env.reset(seed=config.seed + iteration * 1000 + ep)
        done = False
        step = 0
        ep_return = 0.0
        ep_max_reward = -float("inf")
        while not done and step < config.max_episode_steps:
            state_np = normalizer.normalize_state(obs).astype(np.float32)
            state = torch.from_numpy(state_np).to(device).unsqueeze(0)
            with torch.no_grad():
                action_chunk_norm = final_action_chunk_norm(
                    flow,
                    actor,
                    state,
                    config,
                    noise_std=config.exploration_noise,
                ).cpu().numpy()[0]
            action_chunk = action_chunk_norm.reshape(config.chunk_size, flow.action_dim)
            action_raw_chunk = np.clip(
                normalizer.denormalize_action(action_chunk),
                action_low,
                action_high,
            )
            chunk_reward = 0.0
            for action_raw in action_raw_chunk:
                next_obs, reward, terminated, truncated, _ = env.step(action_raw.astype(np.float32))
                chunk_reward += float(reward)
                step += 1
                ep_return += float(reward)
                ep_max_reward = max(ep_max_reward, float(reward))
                done = terminated or truncated or step >= config.max_episode_steps
                if done:
                    break
            next_state_np = normalizer.normalize_state(next_obs).astype(np.float32)
            replay.add(state_np, action_chunk_norm, chunk_reward, next_state_np, done)
            obs = next_obs
        episode_returns.append(ep_return)
        episode_max_rewards.append(ep_max_reward)
    return float(np.mean(episode_returns)), float(np.mean(episode_max_rewards))


def td3_updates(
    flow: FlowMatchingPolicy,
    actor: ResidualActor,
    actor_target: ResidualActor,
    critic: TwinCritic,
    critic_target: TwinCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    config: ResidualTD3Config,
    device: torch.device,
    total_updates: int,
) -> tuple[int, dict[str, float]]:
    stats: dict[str, float] = {}
    for _ in range(config.updates_per_iter):
        total_updates += 1
        state, action, reward, next_state, done = replay.sample(config.batch_size, device)
        with torch.no_grad():
            target_action = final_action_chunk_norm(flow, actor_target, next_state, config)
            noise = (torch.randn_like(target_action) * config.target_noise).clamp(
                -config.target_noise_clip,
                config.target_noise_clip,
            )
            target_action = target_action + noise
            q1_t, q2_t = critic_target(next_state, target_action)
            target_q = reward + config.gamma * (1.0 - done) * torch.minimum(q1_t, q2_t)

        q1, q2 = critic(state, action)
        critic_loss = ((q1 - target_q) ** 2).mean() + ((q2 - target_q) ** 2).mean()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_optimizer.step()

        actor_loss_value = float("nan")
        if total_updates % config.policy_delay == 0:
            actor_action = final_action_chunk_norm(flow, actor, state, config)
            residual = actor(state)
            actor_loss = -critic.q1_value(state, actor_action).mean()
            actor_loss = actor_loss + config.residual_l2 * (residual**2).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_optimizer.step()
            actor_loss_value = float(actor_loss.item())

            with torch.no_grad():
                for p, p_t in zip(actor.parameters(), actor_target.parameters(), strict=True):
                    p_t.data.mul_(1.0 - config.tau).add_(config.tau * p.data)
                for p, p_t in zip(critic.parameters(), critic_target.parameters(), strict=True):
                    p_t.data.mul_(1.0 - config.tau).add_(config.tau * p.data)

        stats = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss_value,
            "q_mean": float(q1.mean().item()),
        }
    return total_updates, stats


@torch.no_grad()
def evaluate(
    flow: FlowMatchingPolicy,
    actor: ResidualActor,
    normalizer: Normalizer,
    config: ResidualTD3Config,
    device: torch.device,
    iteration: int,
) -> tuple[float, float]:
    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    returns: list[float] = []
    max_rewards: list[float] = []
    frames: list[np.ndarray] = []
    actor.eval()

    for ep in range(config.eval_episodes):
        obs, _ = env.reset(seed=config.eval_seed_base + ep)
        done = False
        step = 0
        ep_return = 0.0
        ep_max_reward = -float("inf")
        while not done and step < config.max_episode_steps:
            state_np = normalizer.normalize_state(obs).astype(np.float32)
            state = torch.from_numpy(state_np).to(device).unsqueeze(0)
            action_chunk_norm = final_action_chunk_norm(flow, actor, state, config).cpu().numpy()[0]
            action_chunk = action_chunk_norm.reshape(config.chunk_size, flow.action_dim)
            action_raw_chunk = np.clip(
                normalizer.denormalize_action(action_chunk),
                env.action_space.low,
                env.action_space.high,
            )
            for action_raw in action_raw_chunk:
                obs, reward, terminated, truncated, _ = env.step(action_raw.astype(np.float32))
                if config.save_gif and ep == 0:
                    frames.append(env.render())
                done = terminated or truncated
                step += 1
                ep_return += float(reward)
                ep_max_reward = max(ep_max_reward, float(reward))
                if done or step >= config.max_episode_steps:
                    break
        returns.append(ep_return)
        max_rewards.append(ep_max_reward)

    env.close()
    if config.save_gif and frames:
        config.rollout_dir.mkdir(parents=True, exist_ok=True)
        gif_path = config.rollout_dir / f"residual_td3_iter_{iteration:04d}.gif"
        imageio.mimsave(gif_path, frames, fps=20)
        print(f"[eval] saved {gif_path}")
    return float(np.mean(returns)), float(np.mean(max_rewards))


def main() -> None:
    config = tyro.cli(ResidualTD3Config, description="Residual TD3 on frozen Flow policy.")
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    normalizer, state_dim, action_dim = load_normalizer(config.data_dir)
    flow = build_policy(
        "flow",
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device)
    flow.load_state_dict(torch.load(config.flow_checkpoint, map_location=device))
    flow.eval()
    for p in flow.parameters():
        p.requires_grad = False

    chunk_action_dim = config.chunk_size * action_dim
    actor = ResidualActor(state_dim, chunk_action_dim, config.hidden_dims).to(device)
    actor_target = ResidualActor(state_dim, chunk_action_dim, config.hidden_dims).to(device)
    actor_target.load_state_dict(actor.state_dict())
    critic = TwinCritic(state_dim, chunk_action_dim, config.hidden_dims).to(device)
    critic_target = TwinCritic(state_dim, chunk_action_dim, config.hidden_dims).to(device)
    critic_target.load_state_dict(critic.state_dict())

    if config.residual_checkpoint is not None:
        checkpoint = torch.load(
            config.residual_checkpoint,
            map_location=device,
            weights_only=False,
        )
        actor.load_state_dict(checkpoint["actor"])
        actor_target.load_state_dict(actor.state_dict())
        if "critic" in checkpoint:
            critic.load_state_dict(checkpoint["critic"])
            critic_target.load_state_dict(critic.state_dict())
        print(f"loaded residual checkpoint: {config.residual_checkpoint}")

    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=config.actor_lr)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=config.critic_lr)
    replay = ReplayBuffer(state_dim, chunk_action_dim, config.replay_size)

    best_max_reward = -float("inf")
    total_updates = 0
    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    try:
        start_return, start_max_reward = evaluate(flow, actor, normalizer, config, device, 0)
        print(f"[start] return={start_return:.3f} max_reward={start_max_reward:.3f}")
        for iteration in range(1, config.num_iters + 1):
            rollout_return, rollout_max_reward = collect_episodes(
                env,
                flow,
                actor,
                normalizer,
                replay,
                config,
                device,
                iteration,
            )
            stats: dict[str, float] = {}
            if replay.size >= config.batch_size and iteration > config.warmup_iters:
                total_updates, stats = td3_updates(
                    flow,
                    actor,
                    actor_target,
                    critic,
                    critic_target,
                    actor_optimizer,
                    critic_optimizer,
                    replay,
                    config,
                    device,
                    total_updates,
                )
            print(
                f"[iter {iteration:04d}] "
                f"rollout_return={rollout_return:.3f} "
                f"rollout_max_reward={rollout_max_reward:.3f} "
                f"replay={replay.size} "
                f"critic={stats.get('critic_loss', float('nan')):.3f} "
                f"actor={stats.get('actor_loss', float('nan')):.3f} "
                f"q={stats.get('q_mean', float('nan')):.3f}"
            )

            if iteration % config.eval_interval == 0 or iteration == config.num_iters:
                eval_return, eval_max_reward = evaluate(
                    flow,
                    actor,
                    normalizer,
                    config,
                    device,
                    iteration,
                )
                print(
                    f"[eval {iteration:04d}] return={eval_return:.3f} "
                    f"max_reward={eval_max_reward:.3f}"
                )
                if eval_max_reward >= best_max_reward:
                    best_max_reward = eval_max_reward
                    torch.save(
                        {
                            "actor": actor.state_dict(),
                            "critic": critic.state_dict(),
                            "config": asdict(config),
                        },
                        config.output_checkpoint,
                    )
                    print(
                        f"[save best] {config.output_checkpoint} "
                        f"best_max_reward={best_max_reward:.3f}"
                    )
                else:
                    print(f"[no save] best_max_reward={best_max_reward:.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
