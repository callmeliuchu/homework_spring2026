"""Per-step PPO fine-tuning for a chunk_size=1 Push-T BC policy.

This is the "proper" PPO version for your `mse1.pt` checkpoint:

    state_t -> Gaussian action_t -> env.step(action_t)

Unlike the chunk PPO experiments, every environment step gets a fresh
observation and a fresh action sample.  The BC checkpoint supplies the policy
mean network, and PPO learns both that mean and a Gaussian log std.

Example:
    uv run python src/hw1_imitation/policy_gradient_step.py \
        --bc-checkpoint mse1.pt \
        --num-iters 100 \
        --episodes-per-iter 16 \
        --eval-interval 5
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import tyro
from torch import nn
from torch.distributions import Normal
from torch.utils.data import DataLoader

from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)

ENV_ID = "gym_pusht/PushT-v0"


@dataclass
class StepPPOConfig:
    data_dir: Path = Path("data")
    bc_checkpoint: Path = Path("mse1.pt")
    output_checkpoint: Path = Path("pg_step_ppo.pt")
    rollout_dir: Path = Path("rollouts_pg_step")

    hidden_dims: tuple[int, ...] = (256, 256, 256)
    batch_size: int = 128

    num_iters: int = 100
    episodes_per_iter: int = 8
    max_episode_steps: int | None = None
    ppo_epochs: int = 1
    minibatch_size: int = 10000

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    policy_lr: float = 3e-6
    value_lr: float = 1e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    bc_coef: float = 0.0
    target_kl: float = 0.03

    init_log_std: float = -4.0
    eval_interval: int = 5
    eval_episodes: int = 10
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


class StepGaussianPolicy(nn.Module):
    """Gaussian policy in normalized action space."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        init_log_std: float,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.mlp = make_mlp(state_dim, action_dim, hidden_dims)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

    def load_bc_mean(self, checkpoint_path: Path, device: torch.device) -> None:
        state_dict = torch.load(checkpoint_path, map_location=device)
        if state_dict and all(not key.startswith("mlp.") for key in state_dict):
            self.mlp.load_state_dict(state_dict)
            return
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        allowed_missing = {"log_std"}
        if set(missing) != allowed_missing or unexpected:
            raise RuntimeError(
                f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}. "
                "This script expects a chunk_size=1 MSE checkpoint like mse1.pt."
            )

    def mean(self, state: torch.Tensor) -> torch.Tensor:
        return self.mlp(state)

    def distribution(self, state: torch.Tensor) -> Normal:
        mean = self.mean(state)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std)

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def log_prob_entropy(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(state)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.net = make_mlp(state_dim, 1, hidden_dims)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


@dataclass
class StepRolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    episode_returns: list[float]
    episode_max_rewards: list[float]


def load_dataset(
    config: StepPPOConfig,
) -> tuple[Normalizer, PushtChunkDataset, int, int]:
    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)
    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=1,
        normalizer=normalizer,
    )
    return normalizer, dataset, states.shape[1], actions.shape[1]


def compute_gae_for_episode(
    rewards: list[float],
    values: list[float],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    advantages = [0.0] * len(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * gae_lambda * last_gae
        advantages[t] = last_gae
    returns = [adv + val for adv, val in zip(advantages, values, strict=True)]
    return returns, advantages


@torch.no_grad()
def collect_rollouts(
    policy: StepGaussianPolicy,
    value_net: ValueNet,
    normalizer: Normalizer,
    config: StepPPOConfig,
    device: torch.device,
    iteration: int,
) -> StepRolloutBatch:
    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    action_low = env.action_space.low
    action_high = env.action_space.high

    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_log_probs: list[float] = []
    all_returns: list[float] = []
    all_advantages: list[float] = []
    episode_returns: list[float] = []
    episode_max_rewards: list[float] = []

    policy.eval()
    value_net.eval()
    for episode_idx in range(config.episodes_per_iter):
        obs, _ = env.reset(seed=config.seed + iteration * 1000 + episode_idx)
        done = False
        step = 0

        ep_states: list[np.ndarray] = []
        ep_actions: list[np.ndarray] = []
        ep_log_probs: list[float] = []
        ep_rewards: list[float] = []
        ep_values: list[float] = []
        ep_return = 0.0
        ep_max_reward = -float("inf")

        while not done:
            state_np = normalizer.normalize_state(obs).astype(np.float32)
            state = torch.from_numpy(state_np).to(device).unsqueeze(0)
            action_norm, log_prob, _ = policy.sample(state)
            value = value_net(state)

            action_norm_np = action_norm.cpu().numpy()[0]
            action_raw = normalizer.denormalize_action(action_norm_np)
            action_raw = np.clip(action_raw, action_low, action_high)

            next_obs, reward, terminated, truncated, _ = env.step(action_raw.astype(np.float32))
            step += 1
            time_limit = config.max_episode_steps is not None and step >= config.max_episode_steps
            done = terminated or truncated or time_limit

            ep_states.append(state_np)
            ep_actions.append(action_norm_np)
            ep_log_probs.append(float(log_prob.item()))
            ep_rewards.append(float(reward))
            ep_values.append(float(value.item()))
            ep_return += float(reward)
            ep_max_reward = max(ep_max_reward, float(reward))
            obs = next_obs

        if terminated or truncated:
            last_value = 0.0
        else:
            state_np = normalizer.normalize_state(obs).astype(np.float32)
            state = torch.from_numpy(state_np).to(device).unsqueeze(0)
            last_value = float(value_net(state).item())

        ep_returns, ep_advantages = compute_gae_for_episode(
            ep_rewards,
            ep_values,
            last_value,
            config.gamma,
            config.gae_lambda,
        )
        all_states.extend(ep_states)
        all_actions.extend(ep_actions)
        all_log_probs.extend(ep_log_probs)
        all_returns.extend(ep_returns)
        all_advantages.extend(ep_advantages)
        episode_returns.append(ep_return)
        episode_max_rewards.append(ep_max_reward)

    env.close()
    advantages = np.asarray(all_advantages, dtype=np.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return StepRolloutBatch(
        states=torch.as_tensor(np.asarray(all_states), dtype=torch.float32, device=device),
        actions=torch.as_tensor(np.asarray(all_actions), dtype=torch.float32, device=device),
        old_log_probs=torch.as_tensor(all_log_probs, dtype=torch.float32, device=device),
        returns=torch.as_tensor(all_returns, dtype=torch.float32, device=device),
        advantages=torch.as_tensor(advantages, dtype=torch.float32, device=device),
        episode_returns=episode_returns,
        episode_max_rewards=episode_max_rewards,
    )


def ppo_update(
    policy: StepGaussianPolicy,
    value_net: ValueNet,
    optimizer: torch.optim.Optimizer,
    bc_loader_iter,
    batch: StepRolloutBatch,
    config: StepPPOConfig,
    device: torch.device,
) -> dict[str, float]:
    policy.train()
    value_net.train()
    num_samples = batch.states.shape[0]
    last_stats: dict[str, float] = {}

    for _ in range(config.ppo_epochs):
        permutation = torch.randperm(num_samples, device=device)
        for start in range(0, num_samples, config.minibatch_size):
            idx = permutation[start : start + config.minibatch_size]
            state = batch.states[idx]
            action = batch.actions[idx]
            old_log_prob = batch.old_log_probs[idx]
            returns = batch.returns[idx]
            advantages = batch.advantages[idx]

            new_log_prob, entropy = policy.log_prob_entropy(state, action)
            ratio = (new_log_prob - old_log_prob).exp()
            unclipped = ratio * advantages
            clipped = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()

            value = value_net(state)
            value_loss = ((value - returns) ** 2).mean()
            entropy_loss = -entropy.mean()

            if config.bc_coef > 0.0:
                bc_state, bc_action_chunk = next(bc_loader_iter)
                bc_state = bc_state.to(device)
                bc_action = bc_action_chunk[:, 0, :].to(device)
                bc_loss = ((policy.mean(bc_state) - bc_action) ** 2).mean()
            else:
                bc_loss = torch.zeros((), device=device)

            loss = (
                policy_loss
                + config.value_coef * value_loss
                + config.entropy_coef * entropy_loss
                + config.bc_coef * bc_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value_net.parameters()), 1.0)
            optimizer.step()

            with torch.no_grad():
                updated_log_prob, _ = policy.log_prob_entropy(state, action)
                approx_kl = (old_log_prob - updated_log_prob).mean().item()
            last_stats = {
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.mean().item()),
                "bc_loss": float(bc_loss.item()),
                "approx_kl": float(approx_kl),
            }
            if approx_kl > config.target_kl:
                return last_stats

    return last_stats


@torch.no_grad()
def evaluate_policy(
    policy: StepGaussianPolicy,
    normalizer: Normalizer,
    config: StepPPOConfig,
    device: torch.device,
    iteration: int,
) -> tuple[float, float]:
    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    action_low = env.action_space.low
    action_high = env.action_space.high
    returns: list[float] = []
    max_rewards: list[float] = []
    frames: list[np.ndarray] = []

    policy.eval()
    for episode_idx in range(config.eval_episodes):
        obs, _ = env.reset(seed=config.eval_seed_base + episode_idx)
        done = False
        step = 0
        ep_return = 0.0
        ep_max_reward = -float("inf")
        while not done:
            state_np = normalizer.normalize_state(obs).astype(np.float32)
            state = torch.from_numpy(state_np).to(device).unsqueeze(0)
            action_norm = policy.mean(state).cpu().numpy()[0]
            action_raw = normalizer.denormalize_action(action_norm)
            action_raw = np.clip(action_raw, action_low, action_high)

            obs, reward, terminated, truncated, _ = env.step(action_raw.astype(np.float32))
            done = terminated or truncated
            step += 1
            ep_return += float(reward)
            ep_max_reward = max(ep_max_reward, float(reward))
            if config.save_gif and episode_idx == 0:
                frames.append(env.render())
            if config.max_episode_steps is not None and step >= config.max_episode_steps:
                done = True

        returns.append(ep_return)
        max_rewards.append(ep_max_reward)

    env.close()
    if config.save_gif and frames:
        config.rollout_dir.mkdir(parents=True, exist_ok=True)
        gif_path = config.rollout_dir / f"step_ppo_iter_{iteration:04d}.gif"
        imageio.mimsave(gif_path, frames, fps=20)
        print(f"[eval] saved {gif_path}")
    return float(np.mean(returns)), float(np.mean(max_rewards))


def main() -> None:
    config = tyro.cli(
        StepPPOConfig,
        description="Per-step PPO fine-tune a chunk_size=1 Push-T BC policy.",
    )
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    normalizer, dataset, state_dim, action_dim = load_dataset(config)
    if config.bc_coef > 0.0:
        bc_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
        bc_loader_iter = cycle(bc_loader)
    else:
        bc_loader_iter = None

    policy = StepGaussianPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config.hidden_dims,
        init_log_std=config.init_log_std,
    ).to(device)
    policy.load_bc_mean(config.bc_checkpoint, device)
    value_net = ValueNet(state_dim=state_dim, hidden_dims=config.hidden_dims).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": policy.parameters(), "lr": config.policy_lr},
            {"params": value_net.parameters(), "lr": config.value_lr},
        ]
    )

    start_return, start_max_reward = evaluate_policy(policy, normalizer, config, device, iteration=0)
    best_max_reward = start_max_reward
    print(
        f"[start] deterministic BC eval_return={start_return:.3f} "
        f"eval_max_reward={start_max_reward:.3f}"
    )

    for iteration in range(1, config.num_iters + 1):
        batch = collect_rollouts(policy, value_net, normalizer, config, device, iteration)
        stats = ppo_update(policy, value_net, optimizer, bc_loader_iter, batch, config, device)
        print(
            f"[iter {iteration:04d}] "
            f"rollout_return={np.mean(batch.episode_returns):.3f} "
            f"rollout_max_reward={np.mean(batch.episode_max_rewards):.3f} "
            f"loss={stats.get('loss', float('nan')):.3f} "
            f"pi={stats.get('policy_loss', float('nan')):.3f} "
            f"v={stats.get('value_loss', float('nan')):.3f} "
            f"bc={stats.get('bc_loss', float('nan')):.3f} "
            f"ent={stats.get('entropy', float('nan')):.3f} "
            f"kl={stats.get('approx_kl', float('nan')):.4f}"
        )

        if iteration % config.eval_interval == 0 or iteration == config.num_iters:
            eval_return, eval_max_reward = evaluate_policy(
                policy,
                normalizer,
                config,
                device,
                iteration,
            )
            print(
                f"[eval {iteration:04d}] deterministic_return={eval_return:.3f} "
                f"deterministic_max_reward={eval_max_reward:.3f}"
            )
            if eval_max_reward >= best_max_reward:
                best_max_reward = eval_max_reward
                torch.save(
                    {
                        "policy": policy.state_dict(),
                        "value_net": value_net.state_dict(),
                        "config": asdict(config),
                        "policy_class": "StepGaussianPolicy",
                    },
                    config.output_checkpoint,
                )
                print(
                    f"[save best] {config.output_checkpoint} "
                    f"best_max_reward={best_max_reward:.3f}"
                )
            else:
                print(f"[no save] best_max_reward={best_max_reward:.3f}")


if __name__ == "__main__":
    main()
