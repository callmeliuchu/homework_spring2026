"""PPO fine-tuning for the HW1 Push-T imitation policy.

This is a first, intentionally simple version:

1. Load the expert dataset only for normalization and optional BC regularization.
2. Load a trained MSE behavior-cloning checkpoint as the policy mean network.
3. Add a learnable Gaussian std so actions can be sampled and log-probs computed.
4. Treat one action chunk as one PPO action, with the reward equal to the sum of
   rewards collected while executing that chunk.
5. Fine-tune with PPO + value loss + entropy bonus + a small BC loss.

Example:
    uv run python src/hw1_imitation/policy_gradient.py \
        --bc-checkpoint mse.pt \
        --num-iters 20 \
        --episodes-per-iter 8 \
        --eval-interval 5
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
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
class PPOConfig:
    data_dir: Path = Path("data")
    bc_checkpoint: Path = Path("mse.pt")
    output_checkpoint: Path = Path("pg_chunk_ppo.pt")
    rollout_dir: Path = Path("rollouts_pg")

    chunk_size: int = 8
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    batch_size: int = 128

    num_iters: int = 50
    episodes_per_iter: int = 8
    max_episode_steps: int | None = None
    ppo_epochs: int = 5
    minibatch_size: int = 64

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    policy_lr: float = 1e-4
    value_lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    bc_coef: float = 0.05
    target_kl: float = 0.03

    init_log_std: float = -1.0
    eval_interval: int = 5
    eval_episodes: int = 3
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
        # Match MSEPolicy exactly so its checkpoint is a true warm start.
        layers += [nn.Linear(last, hidden), nn.ReLU()]
        last = hidden
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class GaussianChunkPolicy(nn.Module):
    """MSE policy mean network + learnable Gaussian std over normalized chunks."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...],
        init_log_std: float,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.mlp = make_mlp(state_dim, chunk_size * action_dim, hidden_dims)
        self.log_std = nn.Parameter(torch.full((chunk_size, action_dim), init_log_std))

    def load_bc_mean(self, checkpoint_path: Path, device: torch.device) -> None:
        """Load an MSEPolicy checkpoint into the shared `mlp` mean network."""

        state_dict = torch.load(checkpoint_path, map_location=device)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        allowed_missing = {"log_std"}
        if set(missing) != allowed_missing or unexpected:
            raise RuntimeError(
                f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )

    def mean(self, state: torch.Tensor) -> torch.Tensor:
        flat = self.mlp(state)
        return flat.reshape(-1, self.chunk_size, self.action_dim)

    def distribution(self, state: torch.Tensor) -> Normal:
        mean = self.mean(state)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std)

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=(1, 2))
        entropy = dist.entropy().sum(dim=(1, 2))
        return action, log_prob, entropy

    def log_prob_entropy(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(state)
        log_prob = dist.log_prob(action_chunk).sum(dim=(1, 2))
        entropy = dist.entropy().sum(dim=(1, 2))
        return log_prob, entropy


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.net = make_mlp(state_dim, 1, hidden_dims)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


@dataclass
class RolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    chunk_rewards: list[float]
    episode_returns: list[float]


def load_dataset(
    config: PPOConfig,
) -> tuple[Normalizer, PushtChunkDataset, int, int]:
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


def discount_chunk_reward(rewards: list[float], gamma: float) -> float:
    return float(sum((gamma**i) * r for i, r in enumerate(rewards)))


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = 0.0 if dones[t] else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * gae_lambda * (0.0 if dones[t] else last_gae)
        advantages[t] = last_gae
    returns = advantages + np.asarray(values[:-1], dtype=np.float32)
    return returns, advantages


@torch.no_grad()
def collect_rollouts(
    policy: GaussianChunkPolicy,
    value_net: ValueNet,
    normalizer: Normalizer,
    config: PPOConfig,
    device: torch.device,
    iteration: int,
) -> RolloutBatch:
    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    action_low = env.action_space.low
    action_high = env.action_space.high

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    log_probs: list[float] = []
    values: list[float] = []
    rewards: list[float] = []
    dones: list[bool] = []
    episode_returns: list[float] = []

    policy.eval()
    value_net.eval()
    for episode_idx in range(config.episodes_per_iter):
        obs, _ = env.reset(seed=config.seed + iteration * 1000 + episode_idx)
        done = False
        env_step = 0
        episode_return = 0.0

        while not done:
            state_np = normalizer.normalize_state(obs).astype(np.float32)
            state = torch.from_numpy(state_np).to(device).unsqueeze(0)
            action_norm, log_prob, _ = policy.sample(state)
            value = value_net(state)

            action_norm_np = action_norm.cpu().numpy()[0]
            action_raw = normalizer.denormalize_action(action_norm_np)
            action_raw = np.clip(action_raw, action_low, action_high)

            chunk_step_rewards: list[float] = []
            for chunk_i in range(config.chunk_size):
                obs, reward, terminated, truncated, _ = env.step(
                    action_raw[chunk_i].astype(np.float32)
                )
                chunk_step_rewards.append(float(reward))
                episode_return += float(reward)
                env_step += 1
                done = terminated or truncated
                if config.max_episode_steps is not None and env_step >= config.max_episode_steps:
                    done = True
                if done:
                    break

            states.append(state_np)
            actions.append(action_norm_np)
            log_probs.append(float(log_prob.item()))
            values.append(float(value.item()))
            rewards.append(discount_chunk_reward(chunk_step_rewards, config.gamma))
            dones.append(done)

        episode_returns.append(episode_return)

    env.close()
    values.append(0.0)
    returns, advantages = compute_gae(
        rewards,
        values,
        dones,
        gamma=config.gamma**config.chunk_size,
        gae_lambda=config.gae_lambda,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return RolloutBatch(
        states=torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device),
        actions=torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=device),
        old_log_probs=torch.as_tensor(log_probs, dtype=torch.float32, device=device),
        returns=torch.as_tensor(returns, dtype=torch.float32, device=device),
        advantages=torch.as_tensor(advantages, dtype=torch.float32, device=device),
        chunk_rewards=rewards,
        episode_returns=episode_returns,
    )


def ppo_update(
    policy: GaussianChunkPolicy,
    value_net: ValueNet,
    optimizer: torch.optim.Optimizer,
    bc_loader_iter,
    batch: RolloutBatch,
    config: PPOConfig,
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

            bc_state, bc_action = next(bc_loader_iter)
            bc_state = bc_state.to(device)
            bc_action = bc_action.to(device)
            bc_loss = ((policy.mean(bc_state) - bc_action) ** 2).mean()

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

            approx_kl = (old_log_prob - new_log_prob).mean().item()
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
    policy: GaussianChunkPolicy,
    normalizer: Normalizer,
    config: PPOConfig,
    device: torch.device,
    iteration: int,
) -> float:
    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    action_low = env.action_space.low
    action_high = env.action_space.high
    returns: list[float] = []
    frames: list[np.ndarray] = []

    policy.eval()
    for episode_idx in range(config.eval_episodes):
        obs, _ = env.reset(seed=10_000 + iteration * 100 + episode_idx)
        done = False
        episode_return = 0.0
        step = 0
        while not done:
            state_np = normalizer.normalize_state(obs).astype(np.float32)
            state = torch.from_numpy(state_np).to(device).unsqueeze(0)
            action_norm = policy.mean(state).cpu().numpy()[0]
            action_raw = normalizer.denormalize_action(action_norm)
            action_raw = np.clip(action_raw, action_low, action_high)

            for chunk_i in range(config.chunk_size):
                obs, reward, terminated, truncated, _ = env.step(
                    action_raw[chunk_i].astype(np.float32)
                )
                episode_return += float(reward)
                done = terminated or truncated
                step += 1
                if config.save_gif and episode_idx == 0:
                    frames.append(env.render())
                if config.max_episode_steps is not None and step >= config.max_episode_steps:
                    done = True
                if done:
                    break
        returns.append(episode_return)

    env.close()
    if config.save_gif and frames:
        config.rollout_dir.mkdir(parents=True, exist_ok=True)
        gif_path = config.rollout_dir / f"ppo_iter_{iteration:04d}.gif"
        imageio.mimsave(gif_path, frames, fps=20)
        print(f"[eval] saved {gif_path}")
    return float(np.mean(returns))


def main() -> None:
    config = tyro.cli(PPOConfig, description="PPO fine-tune a Push-T BC policy.")
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    normalizer, dataset, state_dim, action_dim = load_dataset(config)
    bc_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    bc_loader_iter = cycle(bc_loader)

    policy = GaussianChunkPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=config.chunk_size,
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

    start_eval = evaluate_policy(policy, normalizer, config, device, iteration=0)
    print(f"[start] deterministic BC mean eval_return={start_eval:.3f}")

    for iteration in range(1, config.num_iters + 1):
        batch = collect_rollouts(policy, value_net, normalizer, config, device, iteration)
        stats = ppo_update(policy, value_net, optimizer, bc_loader_iter, batch, config, device)
        print(
            f"[iter {iteration:04d}] "
            f"rollout_return={np.mean(batch.episode_returns):.3f} "
            f"chunk_reward={np.mean(batch.chunk_rewards):.3f} "
            f"loss={stats.get('loss', float('nan')):.3f} "
            f"pi={stats.get('policy_loss', float('nan')):.3f} "
            f"v={stats.get('value_loss', float('nan')):.3f} "
            f"bc={stats.get('bc_loss', float('nan')):.3f} "
            f"ent={stats.get('entropy', float('nan')):.3f} "
            f"kl={stats.get('approx_kl', float('nan')):.4f}"
        )

        if iteration % config.eval_interval == 0 or iteration == config.num_iters:
            eval_return = evaluate_policy(policy, normalizer, config, device, iteration)
            print(f"[eval {iteration:04d}] deterministic_return={eval_return:.3f}")
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "value_net": value_net.state_dict(),
                    "config": asdict(config),
                },
                config.output_checkpoint,
            )
            print(f"[save] {config.output_checkpoint}")


if __name__ == "__main__":
    main()
