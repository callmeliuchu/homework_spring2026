"""Monte Carlo Q diagnostic for Flow action-chunk selection.

This trains Q(s, action_chunk) from realized return-to-go labels collected
under the Flow behavior policy.  It avoids TD bootstrapping so we can test
whether the Q-selector problem is caused by unstable Bellman targets or by
out-of-distribution action ranking itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
import tyro
from torch import nn

from hw1_imitation.data import Normalizer, download_pusht, load_pusht_zarr
from hw1_imitation.model import FlowMatchingPolicy, build_policy
from hw1_imitation.q_guided_flow_finetune import ENV_ID, TwinQ, sample_flow_flat


@dataclass
class MCQConfig:
    data_dir: Path = Path("data")
    init_checkpoint: Path = Path("flow_chunk_strong_best.pt")
    output_bundle: Path = Path("flow_mc_q_bundle.pt")
    chunk_size: int = 8
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    flow_num_steps: int = 10
    num_collect_episodes: int = 120
    max_episode_steps: int = 300
    gamma: float = 0.99
    train_updates: int = 4000
    batch_size: int = 256
    critic_lr: float = 3e-4
    seed: int = 10_000


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_normalizer(data_dir: Path) -> tuple[Normalizer, int, int]:
    states, actions, _ = load_pusht_zarr(download_pusht(data_dir))
    return Normalizer.from_data(states, actions), states.shape[1], actions.shape[1]


@torch.no_grad()
def collect_mc_dataset(
    flow: FlowMatchingPolicy,
    normalizer: Normalizer,
    config: MCQConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_returns: list[float] = []
    episode_returns: list[float] = []

    try:
        for episode_idx in range(config.num_collect_episodes):
            obs, _ = env.reset(seed=config.seed + episode_idx)
            done = False
            step = 0
            ep_states: list[np.ndarray] = []
            ep_actions: list[np.ndarray] = []
            ep_rewards: list[float] = []
            ep_discounts: list[float] = []
            ep_return = 0.0

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
            ep_returns = [0.0] * len(ep_rewards)
            for idx in reversed(range(len(ep_rewards))):
                running_return = ep_rewards[idx] + ep_discounts[idx] * running_return
                ep_returns[idx] = running_return

            all_states.extend(ep_states)
            all_actions.extend(ep_actions)
            all_returns.extend(ep_returns)
            episode_returns.append(ep_return)
            if (episode_idx + 1) % 20 == 0:
                print(
                    f"[collect {episode_idx + 1:04d}] "
                    f"mean_episode_return={np.mean(episode_returns):.3f} "
                    f"samples={len(all_states)}"
                )
    finally:
        env.close()

    return (
        np.asarray(all_states, dtype=np.float32),
        np.asarray(all_actions, dtype=np.float32),
        np.asarray(all_returns, dtype=np.float32),
    )


def train_mc_q(
    critic: TwinQ,
    states: np.ndarray,
    actions: np.ndarray,
    returns: np.ndarray,
    config: MCQConfig,
    device: torch.device,
) -> None:
    optimizer = torch.optim.AdamW(critic.parameters(), lr=config.critic_lr)
    returns_mean = float(returns.mean())
    returns_std = float(returns.std() + 1e-6)
    targets = (returns - returns_mean) / returns_std
    n = states.shape[0]
    critic.train()

    for update in range(1, config.train_updates + 1):
        idx = np.random.randint(0, n, size=config.batch_size)
        state = torch.as_tensor(states[idx], device=device)
        action = torch.as_tensor(actions[idx], device=device)
        target = torch.as_tensor(targets[idx], device=device)
        q1, q2 = critic(state, action)
        loss = ((q1 - target) ** 2).mean() + ((q2 - target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        optimizer.step()

        if update % 500 == 0 or update == config.train_updates:
            with torch.no_grad():
                full_idx = np.random.randint(0, n, size=min(2048, n))
                q1_eval, _ = critic(
                    torch.as_tensor(states[full_idx], device=device),
                    torch.as_tensor(actions[full_idx], device=device),
                )
                target_eval = torch.as_tensor(targets[full_idx], device=device)
                corr = torch.corrcoef(torch.stack([q1_eval, target_eval]))[0, 1]
            print(
                f"[q update {update:05d}] loss={loss.item():.4f} "
                f"sample_corr={corr.item():.3f}"
            )


def main() -> None:
    config = tyro.cli(MCQConfig, description="Train MC-return Q for Flow selector.")
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
    flow.load_state_dict(torch.load(config.init_checkpoint, map_location=device))
    flow.eval()

    states, actions, returns = collect_mc_dataset(flow, normalizer, config, device)
    print(
        f"[dataset] samples={len(states)} "
        f"return_mean={returns.mean():.3f} return_std={returns.std():.3f}"
    )

    critic = TwinQ(state_dim, config.chunk_size * action_dim, config.hidden_dims).to(device)
    train_mc_q(critic, states, actions, returns, config, device)

    torch.save(
        {
            "flow": flow.state_dict(),
            "critic": critic.state_dict(),
            "config": asdict(config),
        },
        config.output_bundle,
    )
    print(f"[save bundle] {config.output_bundle}")


if __name__ == "__main__":
    main()
