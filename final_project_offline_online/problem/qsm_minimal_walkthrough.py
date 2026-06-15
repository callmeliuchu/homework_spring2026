# %%
"""
Minimal QSM walkthrough.

This file is intentionally self-contained: it does not import anything from
src/.  Run it cell-by-cell in Jupyter/VSCode, or run the whole file as:

    python qsm_minimal_walkthrough.py

Each `# %%` block is one logical step:
1. hard-coded experiment config
2. offline dataset loading
3. minimal network definitions
4. QSM agent definition
5. one-batch debug update
6. training loop
7. evaluation and optional GIF export
8. optional checkpoint loading
"""

# %%
# 1) Imports and hard-coded config.
# Change only this block when you want a different experiment.

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

import gymnasium
import imageio.v2 as imageio
import numpy as np
import ogbench
import torch
from torch import nn


ENV_NAME = "cube-single-play-singletask-task1-v0"
SEED = 0

# QSM hyperparameters.  alpha=0.0005 was the best current sweep result.
ALPHA = 0.0005
INV_TEMP = 50.0
FLOW_STEPS = 10
DISCOUNT = 0.99
TARGET_UPDATE_RATE = 0.005

# Network/training hyperparameters.
HIDDEN_SIZE = 512
NUM_LAYERS = 4
LEARNING_RATE = 3e-4
BATCH_SIZE = 256

# Keep this small for learning/debug.  Use 500_000 for the real run.
TRAINING_STEPS = 20_000
LOG_INTERVAL = 1_000
EVAL_INTERVAL = 5_000
NUM_EVAL_EPISODES = 10

USE_GPU = torch.cuda.is_available()
GPU_ID = 0
DEVICE = torch.device(f"cuda:{GPU_ID}" if USE_GPU else "cpu")

OUT_DIR = Path("qsm_minimal_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional: load an existing trained checkpoint instead of training from scratch.
# Example:
# PRETRAINED_CHECKPOINT = Path("exp/.../best_agent.pt")
PRETRAINED_CHECKPOINT: Path | None = None

print("device:", DEVICE)
print("output dir:", OUT_DIR.resolve())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# %%
# 2) Offline dataset loading.
# OGBench returns a fixed dataset collected by some behavior policy.
# Offline RL can train only from this dataset; it does not interact with env
# during gradient updates.  The env is used later only for evaluation/rendering.


class EpisodeMonitor(gymnasium.Wrapper):
    """Small wrapper that records per-episode return/length/success."""

    def __init__(self, env, filter_regexes: list[str] | None = None):
        super().__init__(env)
        self.filter_regexes = filter_regexes or []
        self.total_timesteps = 0
        self._reset_stats()

    def _reset_stats(self) -> None:
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Remove privileged/proprio keys from logs; policy still receives obs.
        for pattern in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(pattern, key):
                    del info[key]

        self.reward_sum += float(reward)
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if terminated or truncated:
            info["episode"] = {
                "final_reward": float(reward),
                "return": self.reward_sum,
                "length": self.episode_length,
                "duration": time.time() - self.start_time,
            }

        return observation, reward, terminated, truncated, info


class ReplayBuffer:
    """Minimal numpy replay buffer for the fixed offline dataset."""

    def __init__(self, observations, actions, rewards, next_observations, dones):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.next_observations = next_observations
        self.dones = dones
        self.size = len(observations)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_observations": self.next_observations[idx],
            "dones": self.dones[idx],
        }

    def __len__(self) -> int:
        return self.size


def to_tensor(x: np.ndarray) -> torch.Tensor:
    """Convert numpy arrays to torch tensors on DEVICE."""
    t = torch.from_numpy(x)
    if t.dtype == torch.float64:
        t = t.float()
    return t.to(DEVICE)


def batch_to_torch(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Convert a sampled replay batch into tensors used by the agent."""
    return {
        "observations": to_tensor(batch["observations"]).view(len(batch["observations"]), -1),
        "actions": to_tensor(batch["actions"]),
        "rewards": to_tensor(batch["rewards"]),
        "next_observations": to_tensor(batch["next_observations"]).view(len(batch["next_observations"]), -1),
        "dones": to_tensor(batch["dones"]),
    }


def make_env_and_dataset(env_name: str):
    env, train_dataset, _ = ogbench.make_env_and_datasets(env_name)
    env = EpisodeMonitor(env, filter_regexes=[r".*privileged.*", r".*proprio.*"])

    # OGBench uses masks=1 for non-terminal transitions, so done = 1 - mask.
    dataset = ReplayBuffer(
        observations=train_dataset["observations"],
        actions=train_dataset["actions"],
        rewards=train_dataset["rewards"],
        next_observations=train_dataset["next_observations"],
        dones=1 - train_dataset["masks"],
    )
    return env, dataset


env, dataset = make_env_and_dataset(ENV_NAME)
example_batch_np = dataset.sample(4)
print("env:", env.spec.id)
print("dataset size:", len(dataset))
print("observation shape:", dataset.observations.shape)
print("action shape:", dataset.actions.shape)
print("reward range:", float(dataset.rewards.min()), float(dataset.rewards.max()))
print("done mean:", float(dataset.dones.mean()))


# %%
# 3) Minimal network definitions.
# The actor predicts diffusion noise epsilon.  The critic is an ensemble of
# two Q-functions; target_critic is a slow-moving copy for bootstrapping.


def build_mlp(input_size: int, output_size: int, n_layers: int, size: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(nn.LayerNorm(size))
        layers.append(nn.Tanh())
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(nn.Identity())
    return nn.Sequential(*layers)


class EnsembleMLP(nn.Module):
    """Runs n independent MLPs and stacks outputs as (n, batch, output_dim)."""

    def __init__(self, mlps: nn.ModuleList):
        super().__init__()
        self.mlps = mlps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([mlp(x) for mlp in self.mlps], dim=0)


def build_ensemble_mlp(
    input_size: int,
    output_size: int,
    n_layers: int,
    size: int,
    n: int,
) -> EnsembleMLP:
    return EnsembleMLP(
        nn.ModuleList([build_mlp(input_size, output_size, n_layers, size) for _ in range(n)])
    )


class VectorFieldPolicy(nn.Module):
    """
    Diffusion/score policy.

    Input:  state s, noisy action a_t, normalized time t.
    Output: predicted noise epsilon_theta(s, a_t, t).
    """

    def __init__(self, ob_dim: int, ac_dim: int):
        super().__init__()
        self.net = build_mlp(
            input_size=ob_dim + ac_dim + 1,
            output_size=ac_dim,
            n_layers=NUM_LAYERS,
            size=HIDDEN_SIZE,
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor, times: torch.Tensor):
        return self.net(torch.cat([obs, actions, times], dim=-1))


class EnsembleCritic(nn.Module):
    """Two Q-functions Q_i(s, a).  Shape returned: (2, batch)."""

    def __init__(self, ob_dim: int, ac_dim: int):
        super().__init__()
        self.net = build_ensemble_mlp(
            input_size=ob_dim + ac_dim,
            output_size=1,
            n_layers=NUM_LAYERS,
            size=HIDDEN_SIZE,
            n=2,
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, actions], dim=-1)).squeeze(-1)


OBS_DIM = int(np.prod(dataset.observations.shape[1:]))
ACT_DIM = int(dataset.actions.shape[-1])
print("OBS_DIM:", OBS_DIM, "ACT_DIM:", ACT_DIM)


# %%
# 4) QSM agent.
# Key idea:
# - Critic learns Q(s, a) using Bellman backup.
# - Actor is still a diffusion noise predictor.
# - Q guidance modifies the target noise:
#       eps_target = noise - alpha * inv_temp * sigma_t * grad_a Q(s, a_t)
# - The actor loss is MSE(eps_pred, eps_target).


class QSMAgent(nn.Module):
    def __init__(self, ob_dim: int, ac_dim: int):
        super().__init__()
        self.ac_dim = ac_dim
        self.actor = VectorFieldPolicy(ob_dim, ac_dim)
        self.critic = EnsembleCritic(ob_dim, ac_dim)
        self.target_critic = EnsembleCritic(ob_dim, ac_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

        self.discount = DISCOUNT
        self.target_update_rate = TARGET_UPDATE_RATE
        self.alpha = ALPHA
        self.inv_temp = INV_TEMP
        self.flow_steps = FLOW_STEPS

        betas = self.cosine_beta_schedule(FLOW_STEPS)
        alphas = 1.0 - betas
        alpha_hats = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_hats", alpha_hats)

        self.to(DEVICE)

    @staticmethod
    def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine DDPM noise schedule."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alpha_hats = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alpha_hats = alpha_hats / alpha_hats[0]
        betas = 1 - alpha_hats[1:] / alpha_hats[:-1]
        return torch.clamp(betas, min=1e-5, max=0.999)

    @torch.no_grad()
    def ddpm_sampler(self, observations: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Reverse diffusion: start from Gaussian noise and denoise into an action."""
        x = noise
        for t in reversed(range(self.flow_steps)):
            t_batch = torch.full(
                (x.shape[0], 1),
                t / self.flow_steps,
                device=x.device,
                dtype=x.dtype,
            )
            eps_pred = self.actor(observations, x, t_batch)

            alpha = self.alphas[t]
            alpha_hat = self.alpha_hats[t]
            beta = self.betas[t]

            # DDPM posterior mean update.
            x = (1 / alpha.sqrt()) * (
                x - ((1 - alpha) / (1 - alpha_hat).sqrt()) * eps_pred
            )

            # Add noise except at the final denoising step.
            if t > 0:
                x = x + beta.sqrt() * torch.randn_like(x)

        return torch.clamp(x, -1, 1)

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Policy evaluation: sample one action by DDPM denoising."""
        obs = to_tensor(observation[None]).view(1, -1)
        noise = torch.randn(1, self.ac_dim, device=DEVICE)
        action = self.ddpm_sampler(obs, noise)[0]
        return action.detach().cpu().numpy()

    def update_critic(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, float]:
        """Bellman update for Q(s,a), using diffusion policy for next action."""
        with torch.no_grad():
            next_noise = torch.randn_like(actions)
            next_actions = self.ddpm_sampler(next_observations, next_noise)
            next_q = self.target_critic(next_observations, next_actions).min(dim=0).values
            target_q = rewards + self.discount * (1.0 - dones.float()) * next_q

        q = self.critic(observations, actions)
        q_loss = ((q - target_q) ** 2).mean()

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": float(q_loss.item()),
            "q_mean": float(q.mean().item()),
            "target_q_mean": float(target_q.mean().item()),
        }

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, float]:
        """QSM actor update: denoising target plus Q-gradient guidance."""
        batch_size = actions.shape[0]

        # Pick a random diffusion time for each dataset action.
        t = torch.randint(0, self.flow_steps, (batch_size,), device=DEVICE)
        noise = torch.randn_like(actions)

        # Forward diffusion: create noisy action a_t from clean dataset action a_0.
        alpha_hat = self.alpha_hats[t].view(batch_size, 1)
        a_t = alpha_hat.sqrt() * actions + (1 - alpha_hat).sqrt() * noise
        t_input = t.view(batch_size, 1).to(dtype=actions.dtype) / self.flow_steps

        # Actor predicts the noise that produced a_t.
        eps_pred = self.actor(observations, a_t, t_input)

        # Compute grad_a Q(s, a_t).  This is the "Q guidance" direction.
        a_t_for_grad = a_t.detach().requires_grad_(True)
        q_for_grad = self.target_critic(observations, a_t_for_grad).mean(dim=0)
        q_grad = torch.autograd.grad(q_for_grad.sum(), a_t_for_grad)[0].detach()

        # Scale Q guidance by the current diffusion noise level sigma_t.
        sigma_t = (1 - alpha_hat).sqrt()
        guidance_scale = self.alpha * self.inv_temp
        eps_target = noise - guidance_scale * sigma_t * q_grad

        # bc_loss tells how close we are to plain behavior cloning diffusion.
        # qsm_loss is the actual optimized objective.
        bc_loss = ((noise - eps_pred) ** 2).mean()
        qsm_loss = ((eps_pred - eps_target.detach()) ** 2).mean()

        self.actor_optimizer.zero_grad()
        qsm_loss.backward()
        self.actor_optimizer.step()

        return {
            "actor_loss": float(qsm_loss.item()),
            "bc_loss": float(bc_loss.item()),
            "q_grad_norm": float(q_grad.norm(dim=-1).mean().item()),
            "eps_pred_norm": float(eps_pred.norm(dim=-1).mean().item()),
            "sigma_t_mean": float(sigma_t.mean().item()),
        }

    def update_target_critic(self) -> None:
        """Polyak averaging: target <- (1-tau) target + tau critic."""
        tau = self.target_update_rate
        with torch.no_grad():
            for critic_param, target_param in zip(
                self.critic.parameters(),
                self.target_critic.parameters(),
            ):
                target_param.data.mul_(1 - tau).add_(critic_param.data, alpha=tau)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        critic_metrics = self.update_critic(
            batch["observations"],
            batch["actions"],
            batch["rewards"],
            batch["next_observations"],
            batch["dones"],
        )
        actor_metrics = self.update_actor(batch["observations"], batch["actions"])
        self.update_target_critic()
        return {**critic_metrics, **actor_metrics}


agent = QSMAgent(OBS_DIM, ACT_DIM)
print("actor parameters:", sum(p.numel() for p in agent.actor.parameters()))
print("critic parameters:", sum(p.numel() for p in agent.critic.parameters()))
print("betas:", agent.betas.detach().cpu().numpy().round(4))


# %%
# 5) One-batch debug update.
# This block lets you inspect all key tensors before doing long training.

debug_batch = batch_to_torch(dataset.sample(BATCH_SIZE))
print("batch observations:", tuple(debug_batch["observations"].shape))
print("batch actions:", tuple(debug_batch["actions"].shape))
print("batch rewards:", tuple(debug_batch["rewards"].shape), debug_batch["rewards"][:5])

debug_metrics = agent.update(debug_batch)
print("one update metrics:")
for key, value in debug_metrics.items():
    print(f"  {key}: {value:.6f}")


# %%
# 6) Offline training loop.
# No environment interaction happens here.  Every update samples only from the
# fixed OGBench replay buffer.  Evaluation uses env only at EVAL_INTERVAL.


def evaluate_policy(
    eval_env: gymnasium.Env,
    policy: QSMAgent,
    n_episodes: int,
    render_best_gif_path: Path | None = None,
) -> dict[str, float]:
    max_steps = eval_env.spec.max_episode_steps or eval_env.max_episode_steps
    returns: list[float] = []
    successes: list[float] = []
    lengths: list[int] = []
    best_frames: list[np.ndarray] | None = None
    best_key: tuple[float, float] | None = None

    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=SEED + ep)
        total_return = 0.0
        success = 0.0
        frames: list[np.ndarray] = []

        for step in range(max_steps):
            if render_best_gif_path is not None:
                frame = eval_env.render()
                if frame is not None:
                    frames.append(np.asarray(frame))

            action = policy.get_action(obs)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_return += float(reward)
            success = float(info.get("success", 0.0))
            if terminated or truncated:
                break

        ep_length = step + 1
        returns.append(total_return)
        successes.append(success)
        lengths.append(ep_length)

        # Pick successful/high-return rollout for GIF.
        current_key = (success, total_return)
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_frames = frames

    if render_best_gif_path is not None and best_frames:
        render_best_gif_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(render_best_gif_path, best_frames, duration=1000 / 20)

    return {
        "success_rate": float(np.mean(successes)),
        "return_mean": float(np.mean(returns)),
        "length_mean": float(np.mean(lengths)),
    }


def train_qsm(policy: QSMAgent) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    best_success = -1.0
    best_path = OUT_DIR / "qsm_best_agent.pt"

    for step in range(1, TRAINING_STEPS + 1):
        batch = batch_to_torch(dataset.sample(BATCH_SIZE))
        metrics = policy.update(batch)

        if step % LOG_INTERVAL == 0:
            row = {"step": step, **metrics}
            history.append(row)
            print(
                f"step={step:>7} "
                f"q_loss={metrics['q_loss']:.4f} "
                f"actor_loss={metrics['actor_loss']:.4f} "
                f"bc_loss={metrics['bc_loss']:.4f} "
                f"q_grad={metrics['q_grad_norm']:.4f}"
            )

        if step % EVAL_INTERVAL == 0:
            eval_stats = evaluate_policy(env, policy, NUM_EVAL_EPISODES)
            print(
                f"[eval step={step}] "
                f"success={eval_stats['success_rate']:.2f} "
                f"return={eval_stats['return_mean']:.2f} "
                f"len={eval_stats['length_mean']:.1f}"
            )
            if eval_stats["success_rate"] > best_success:
                best_success = eval_stats["success_rate"]
                torch.save(policy.state_dict(), best_path)
                (OUT_DIR / "qsm_best_eval.json").write_text(
                    json.dumps({"step": step, **eval_stats}, indent=2)
                )
                print("  saved best checkpoint:", best_path)

    torch.save(policy.state_dict(), OUT_DIR / "qsm_final_agent.pt")
    return history


if PRETRAINED_CHECKPOINT is None:
    history = train_qsm(agent)
else:
    print("skip training because PRETRAINED_CHECKPOINT is set")


# %%
# 7) Final evaluation and GIF.
# This block can be run after training, or after loading a checkpoint below.

gif_path = OUT_DIR / "qsm_rollout.gif"
final_eval = evaluate_policy(
    env,
    agent,
    n_episodes=20,
    render_best_gif_path=gif_path,
)
print("final eval:", final_eval)
print("gif saved to:", gif_path.resolve())


# %%
# 8) Optional: load an existing checkpoint.
# Use this cell when you want to inspect a trained model without retraining.

if PRETRAINED_CHECKPOINT is not None:
    state_dict = torch.load(PRETRAINED_CHECKPOINT, map_location=DEVICE)
    agent.load_state_dict(state_dict)
    agent.eval()
    print("loaded:", PRETRAINED_CHECKPOINT)

    ckpt_eval = evaluate_policy(
        env,
        agent,
        n_episodes=20,
        render_best_gif_path=OUT_DIR / "qsm_checkpoint_rollout.gif",
    )
    print("checkpoint eval:", ckpt_eval)


# %%
# 9) Clean up environment when done.

env.close()
print("done")
