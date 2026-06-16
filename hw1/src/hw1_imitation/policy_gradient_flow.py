"""PPO fine-tuning for a Flow Matching Push-T policy.

Important simplification for this first version:

Flow Matching does not give an easy action log-probability. PPO needs
log_prob(action), so this script uses the differentiable flow rollout as the
mean of a Gaussian action-chunk policy:

    mean_chunk = flow_ode(state, initial_noise=0)
    action_chunk ~ Normal(mean_chunk, learned_std)

This is not "exact likelihood PPO for flows"; it is a practical residual
Gaussian PPO wrapper around a trained flow policy.

Example:
    uv run python src/hw1_imitation/policy_gradient_flow.py \
        --flow-checkpoint flow.pt \
        --num-iters 20 \
        --episodes-per-iter 8 \
        --eval-interval 5
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import tyro
from torch import nn
from torch.distributions import Normal
from torch.utils.data import DataLoader

from hw1_imitation.policy_gradient import (
    ValueNet,
    collect_rollouts,
    evaluate_policy,
    load_dataset,
    ppo_update,
    set_seed,
)


@dataclass
class FlowPPOConfig:
    data_dir: Path = Path("data")
    flow_checkpoint: Path = Path("flow.pt")
    output_checkpoint: Path = Path("pg_flow_ppo.pt")
    rollout_dir: Path = Path("rollouts_pg_flow")

    chunk_size: int = 8
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    batch_size: int = 128
    flow_num_steps: int = 10

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


def make_flow_mlp(
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...],
) -> nn.Sequential:
    """Match FlowMatchingPolicy.mlp exactly for checkpoint compatibility."""

    layers: list[nn.Module] = []
    last = state_dim + chunk_size * action_dim + 1
    for hidden in hidden_dims:
        layers += [nn.Linear(last, hidden), nn.ReLU()]
        last = hidden
    layers.append(nn.Linear(last, chunk_size * action_dim))
    return nn.Sequential(*layers)


class GaussianFlowChunkPolicy(nn.Module):
    """Flow ODE mean + learnable Gaussian std over normalized action chunks."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...],
        flow_num_steps: int,
        init_log_std: float,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.flow_num_steps = flow_num_steps
        self.mlp = make_flow_mlp(state_dim, action_dim, chunk_size, hidden_dims)
        self.log_std = nn.Parameter(torch.full((chunk_size, action_dim), init_log_std))

    def load_flow(self, checkpoint_path: Path, device: torch.device) -> None:
        state_dict = torch.load(checkpoint_path, map_location=device)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        allowed_missing = {"log_std"}
        if set(missing) != allowed_missing or unexpected:
            raise RuntimeError(
                f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )

    def mean(self, state: torch.Tensor) -> torch.Tensor:
        """Differentiable Euler integration from zero noise to an action chunk."""

        batch = state.shape[0]
        flat_dim = self.chunk_size * self.action_dim
        x = torch.zeros(batch, flat_dim, device=state.device, dtype=state.dtype)
        dt = 1.0 / self.flow_num_steps
        for k in range(self.flow_num_steps):
            t = torch.full((batch, 1), k / self.flow_num_steps, device=state.device)
            velocity = self.mlp(torch.cat([state, x, t], dim=-1))
            x = x + velocity * dt
        return x.reshape(batch, self.chunk_size, self.action_dim)

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


def main() -> None:
    config = tyro.cli(
        FlowPPOConfig,
        description="PPO fine-tune a Flow Matching Push-T policy.",
    )
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print("Note: PPO uses a Gaussian wrapper around the flow mean, not exact flow likelihood.")

    normalizer, dataset, state_dim, action_dim = load_dataset(config)
    bc_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    bc_loader_iter = cycle(bc_loader)

    policy = GaussianFlowChunkPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
        flow_num_steps=config.flow_num_steps,
        init_log_std=config.init_log_std,
    ).to(device)
    policy.load_flow(config.flow_checkpoint, device)
    value_net = ValueNet(state_dim=state_dim, hidden_dims=config.hidden_dims).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": policy.parameters(), "lr": config.policy_lr},
            {"params": value_net.parameters(), "lr": config.value_lr},
        ]
    )

    start_eval = evaluate_policy(policy, normalizer, config, device, iteration=0)
    print(f"[start] deterministic flow-mean eval_return={start_eval:.3f}")

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
                    "policy_class": "GaussianFlowChunkPolicy",
                },
                config.output_checkpoint,
            )
            print(f"[save] {config.output_checkpoint}")


if __name__ == "__main__":
    main()
