# %% [markdown]
# # HW2 minimal walkthrough: Policy Gradient
#
# This is a notebook-style single file. Open it in VS Code/Jupyter and run the
# `# %%` cells from top to bottom.
#
# It does not import any homework source files. The full flow is here:
# environment -> trajectory data -> returns/Q-values -> advantages ->
# policy update -> optional baseline update -> evaluation -> visualization.

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as D
from torch import nn

try:
    import gym
except ImportError:  # gymnasium uses the newer reset/step API; wrappers below handle both.
    import gymnasium as gym

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*CartPole-v0 is out of date.*")

try:
    from IPython.display import Image, display
except ImportError:
    Image = None
    display = None


# %% [markdown]
# ## 0. Hard-coded config
#
# Change values directly in this cell. There is no command-line interface.

# %%
ROOT = Path.cwd()
if not (ROOT / "pyproject.toml").exists() and (Path.cwd() / "hw2").exists():
    ROOT = Path.cwd() / "hw2"

OUT_DIR = ROOT / "minimal_outputs_pg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENV_ID = "CartPole-v0"
SEED = 1
N_ITER = 10                  # increase to 30-50 if you want a near-solved policy
BATCH_SIZE = 1000          # collect at least this many env steps per update
EVAL_EPISODES = 3
DISCOUNT = 1.0             # CartPole reward is short-horizon; HW2 default uses 1.0
LEARNING_RATE = 5e-3
N_LAYERS = 2
LAYER_SIZE = 64

USE_REWARD_TO_GO = True
USE_BASELINE = True
BASELINE_LR = 5e-3
BASELINE_STEPS = 5
GAE_LAMBDA = None          # set to 0.95 to inspect GAE
NORMALIZE_ADVANTAGES = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)
torch.manual_seed(SEED)

print("ROOT =", ROOT)
print("DEVICE =", DEVICE)
print("OUT_DIR =", OUT_DIR)


# %% [markdown]
# ## 1. Small Gym compatibility helpers
#
# Gym 0.25 and Gymnasium return slightly different values from `reset` and
# `step`. These wrappers keep the rest of the file clean.

# %%
def reset_env(env, seed: int | None = None) -> np.ndarray:
    out = env.reset(seed=seed) if seed is not None else env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    return np.asarray(obs, dtype=np.float32)


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        next_obs, reward, terminated, truncated, info = out
        done = terminated or truncated
    else:
        next_obs, reward, done, info = out
    return np.asarray(next_obs, dtype=np.float32), float(reward), bool(done), info


def render_frame(env) -> np.ndarray:
    frame = env.render()
    if frame is None:
        frame = env.render(mode="rgb_array")
    if isinstance(frame, (list, tuple)):
        frame = frame[-1]
    frame = np.asarray(frame)
    while frame.ndim > 3:
        frame = frame[0] if frame.shape[0] == 1 else frame[-1]
    if frame.ndim == 3 and frame.shape[0] in (3, 4) and frame.shape[-1] not in (3, 4):
        frame = np.moveaxis(frame, 0, -1)
    return frame.astype(np.uint8)


def make_env(render: bool = False):
    if render:
        try:
            return gym.make(ENV_ID, render_mode="rgb_array")
        except TypeError:
            return gym.make(ENV_ID)
    return gym.make(ENV_ID)


env = make_env(render=False)
ob_dim = int(env.observation_space.shape[0])
discrete = isinstance(env.action_space, gym.spaces.Discrete)
ac_dim = int(env.action_space.n if discrete else env.action_space.shape[0])
max_ep_len = int(env.spec.max_episode_steps)
env.close()

print("env =", ENV_ID)
print("ob_dim =", ob_dim)
print("ac_dim =", ac_dim)
print("discrete =", discrete)
print("max_ep_len =", max_ep_len)


# %% [markdown]
# ## 2. Define policy and value networks
#
# Policy gradient needs a stochastic policy:
# - discrete action space: `Categorical(logits)`
# - continuous action space: `Normal(mean, std)`
#
# The critic/baseline predicts `V(s)` and is trained by MSE against Monte Carlo
# Q-values. The baseline reduces variance; it is not required for correctness.

# %%
def build_mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for _ in range(N_LAYERS):
        layers += [nn.Linear(last, LAYER_SIZE), nn.Tanh()]
        last = LAYER_SIZE
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class PolicyNetwork(nn.Module):
    def __init__(self, ob_dim: int, ac_dim: int, discrete: bool):
        super().__init__()
        self.discrete = discrete
        if discrete:
            self.logits_net = build_mlp(ob_dim, ac_dim)
        else:
            self.mean_net = build_mlp(ob_dim, ac_dim)
            self.logstd = nn.Parameter(torch.zeros(ac_dim))

    def distribution(self, obs: torch.Tensor):
        if self.discrete:
            logits = self.logits_net(obs)
            return D.Categorical(logits=logits)
        mean = self.mean_net(obs)
        std = torch.exp(self.logstd)
        return D.Independent(D.Normal(mean, std), 1)

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray | int:
        obs_t = torch.from_numpy(obs).float().to(DEVICE).unsqueeze(0)
        action = self.distribution(obs_t).sample()[0]
        action_np = action.cpu().numpy()
        return int(action_np) if self.discrete else action_np.astype(np.float32)


class ValueNetwork(nn.Module):
    def __init__(self, ob_dim: int):
        super().__init__()
        self.net = build_mlp(ob_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


policy = PolicyNetwork(ob_dim, ac_dim, discrete).to(DEVICE)
critic = ValueNetwork(ob_dim).to(DEVICE) if USE_BASELINE else None
policy_optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
critic_optimizer = (
    torch.optim.Adam(critic.parameters(), lr=BASELINE_LR) if critic is not None else None
)

print(policy)
print(critic)


# %% [markdown]
# ## 3. Roll out trajectories
#
# A trajectory stores every transition generated by the current policy:
# `obs_t, action_t, reward_t, next_obs_t, terminal_t`.
# This is the "data loading" step for policy gradient: data is collected online
# from the environment instead of read from a dataset file.

# %%
@dataclass
class Trajectory:
    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_observation: np.ndarray
    terminal: np.ndarray
    image_obs: np.ndarray | None = None


def sample_trajectory(env, policy: PolicyNetwork, max_length: int, render: bool = False) -> Trajectory:
    obs = reset_env(env)
    observations, actions, rewards, next_observations, terminals, frames = [], [], [], [], [], []

    for step in range(max_length):
        if render:
            frames.append(render_frame(env))

        action = policy.get_action(obs)
        next_obs, reward, done, _ = step_env(env, action)
        rollout_done = done or (step + 1 >= max_length)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        next_observations.append(next_obs)
        terminals.append(rollout_done)

        obs = next_obs
        if rollout_done:
            break

    return Trajectory(
        observation=np.asarray(observations, dtype=np.float32),
        action=np.asarray(actions, dtype=np.int64 if policy.discrete else np.float32),
        reward=np.asarray(rewards, dtype=np.float32),
        next_observation=np.asarray(next_observations, dtype=np.float32),
        terminal=np.asarray(terminals, dtype=np.float32),
        image_obs=np.asarray(frames, dtype=np.uint8) if render else None,
    )


def sample_trajectories(env, policy: PolicyNetwork, min_steps: int, max_length: int) -> tuple[list[Trajectory], int]:
    trajs: list[Trajectory] = []
    steps = 0
    while steps < min_steps:
        traj = sample_trajectory(env, policy, max_length)
        trajs.append(traj)
        steps += len(traj.reward)
    return trajs, steps


env = make_env(render=False)
first_traj = sample_trajectory(env, policy, max_ep_len)
env.close()

print("one trajectory length =", len(first_traj.reward))
print("first 5 observations:\n", first_traj.observation[:5])
print("first 5 actions:", first_traj.action[:5])
print("first 5 rewards:", first_traj.reward[:5])
print("first 5 terminals:", first_traj.terminal[:5])


# %% [markdown]
# ## 4. Returns, reward-to-go, and advantages
#
# Policy gradient uses:
# `loss = - mean(log pi(a_t | s_t) * advantage_t)`.
#
# Three important choices:
# - vanilla return: every timestep in one episode receives the same total return
# - reward-to-go: timestep `t` only receives future discounted rewards
# - baseline: subtract `V(s_t)` from Q-values to reduce gradient variance

# %%
def discounted_return(rewards: np.ndarray, gamma: float) -> np.ndarray:
    total = 0.0
    for r in reversed(rewards):
        total = float(r) + gamma * total
    return np.full_like(rewards, total, dtype=np.float32)


def discounted_reward_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    out = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = float(rewards[i]) + gamma * running
        out[i] = running
    return out


def calculate_q_values(trajectories: list[Trajectory], use_reward_to_go: bool) -> list[np.ndarray]:
    q_values = []
    for traj in trajectories:
        if use_reward_to_go:
            q_values.append(discounted_reward_to_go(traj.reward, DISCOUNT))
        else:
            q_values.append(discounted_return(traj.reward, DISCOUNT))
    return q_values


def flatten_batch(trajectories: list[Trajectory], q_values: list[np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "obs": np.concatenate([t.observation for t in trajectories], axis=0),
        "actions": np.concatenate([t.action for t in trajectories], axis=0),
        "rewards": np.concatenate([t.reward for t in trajectories], axis=0),
        "terminals": np.concatenate([t.terminal for t in trajectories], axis=0),
        "q_values": np.concatenate(q_values, axis=0),
    }


def estimate_advantages(batch: dict[str, np.ndarray]) -> np.ndarray:
    q_values = batch["q_values"]
    if critic is None:
        advantages = q_values.copy()
    else:
        obs_t = torch.from_numpy(batch["obs"]).float().to(DEVICE)
        with torch.no_grad():
            values = critic(obs_t).cpu().numpy()

        if GAE_LAMBDA is None:
            advantages = q_values - values
        else:
            # GAE recursion: A_t = delta_t + gamma * lambda * A_{t+1}
            rewards = batch["rewards"]
            terminals = batch["terminals"]
            advantages = np.zeros_like(q_values, dtype=np.float32)
            next_advantage = 0.0
            next_value = 0.0
            for i in reversed(range(len(rewards))):
                not_done = 1.0 - terminals[i]
                delta = rewards[i] + DISCOUNT * next_value * not_done - values[i]
                next_advantage = delta + DISCOUNT * GAE_LAMBDA * not_done * next_advantage
                advantages[i] = next_advantage
                next_value = values[i]

    if NORMALIZE_ADVANTAGES:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages.astype(np.float32)


demo_q_full = discounted_return(first_traj.reward, DISCOUNT)
demo_q_rtg = discounted_reward_to_go(first_traj.reward, DISCOUNT)
print("trajectory rewards first 10:", first_traj.reward[:10])
print("vanilla return first 10:", demo_q_full[:10])
print("reward-to-go first 10:", demo_q_rtg[:10])


# %% [markdown]
# ## 5. One policy update, step by step
#
# This cell shows exactly what one training update does:
# collect trajectories -> compute Q-values -> flatten -> compute advantages ->
# update actor -> update critic.

# %%
def update_policy(obs: np.ndarray, actions: np.ndarray, advantages: np.ndarray) -> float:
    obs_t = torch.from_numpy(obs).float().to(DEVICE)
    actions_t = torch.from_numpy(actions).to(DEVICE)
    advantages_t = torch.from_numpy(advantages).float().to(DEVICE)

    dist = policy.distribution(obs_t)
    log_prob = dist.log_prob(actions_t)
    loss = -(log_prob * advantages_t).mean()

    policy_optimizer.zero_grad()
    loss.backward()
    policy_optimizer.step()
    return float(loss.item())


def update_critic(obs: np.ndarray, q_values: np.ndarray) -> float | None:
    if critic is None or critic_optimizer is None:
        return None

    obs_t = torch.from_numpy(obs).float().to(DEVICE)
    targets_t = torch.from_numpy(q_values).float().to(DEVICE)
    last_loss = None
    for _ in range(BASELINE_STEPS):
        values = critic(obs_t)
        loss = ((values - targets_t) ** 2).mean()
        critic_optimizer.zero_grad()
        loss.backward()
        critic_optimizer.step()
        last_loss = float(loss.item())
    return last_loss


env = make_env(render=False)
trajs, env_steps = sample_trajectories(env, policy, BATCH_SIZE, max_ep_len)
env.close()

q_values = calculate_q_values(trajs, USE_REWARD_TO_GO)
batch = flatten_batch(trajs, q_values)
advantages = estimate_advantages(batch)

actor_loss = update_policy(batch["obs"], batch["actions"], advantages)
baseline_loss = update_critic(batch["obs"], batch["q_values"])

print("num trajectories =", len(trajs))
print("env steps =", env_steps)
print("flat obs shape =", batch["obs"].shape)
print("flat actions shape =", batch["actions"].shape)
print("q mean/std =", float(batch["q_values"].mean()), float(batch["q_values"].std()))
print("adv mean/std =", float(advantages.mean()), float(advantages.std()))
print("actor_loss =", actor_loss)
print("baseline_loss =", baseline_loss)


# %% [markdown]
# ## 6. Full training loop
#
# Each iteration repeats the same five operations:
# collect data, estimate Q-values, estimate advantages, update policy, evaluate.

# %%
def evaluate(policy: PolicyNetwork, episodes: int = EVAL_EPISODES, render: bool = False):
    env = make_env(render=render)
    returns, lengths = [], []
    video_traj = None
    for ep in range(episodes):
        traj = sample_trajectory(env, policy, max_ep_len, render=(render and ep == 0))
        returns.append(float(traj.reward.sum()))
        lengths.append(len(traj.reward))
        if render and ep == 0:
            video_traj = traj
    env.close()
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_max": float(np.max(returns)),
        "length_mean": float(np.mean(lengths)),
        "video_traj": video_traj,
    }


history: list[dict[str, float]] = []
total_steps = 0

for itr in range(N_ITER):
    env = make_env(render=False)
    trajs, env_steps = sample_trajectories(env, policy, BATCH_SIZE, max_ep_len)
    env.close()
    total_steps += env_steps

    q_values = calculate_q_values(trajs, USE_REWARD_TO_GO)
    batch = flatten_batch(trajs, q_values)
    advantages = estimate_advantages(batch)
    actor_loss = update_policy(batch["obs"], batch["actions"], advantages)
    baseline_loss = update_critic(batch["obs"], batch["q_values"])

    eval_info = evaluate(policy, EVAL_EPISODES)
    row = {
        "itr": itr,
        "steps": total_steps,
        "actor_loss": actor_loss,
        "baseline_loss": float("nan") if baseline_loss is None else baseline_loss,
        "train_return_mean": float(np.mean([t.reward.sum() for t in trajs])),
        "eval_return_mean": eval_info["return_mean"],
        "eval_return_std": eval_info["return_std"],
    }
    history.append(row)
    print(
        f"itr={itr:03d} steps={total_steps:5d} "
        f"train_return={row['train_return_mean']:.1f} "
        f"eval_return={row['eval_return_mean']:.1f} "
        f"actor_loss={actor_loss:.3f}"
    )


# %% [markdown]
# ## 7. Plot learning curve and save model

# %%
iters = [row["itr"] for row in history]
train_returns = [row["train_return_mean"] for row in history]
eval_returns = [row["eval_return_mean"] for row in history]

plt.figure(figsize=(8, 4))
plt.plot(iters, train_returns, label="train sampled return")
plt.plot(iters, eval_returns, label="eval return")
plt.xlabel("policy update iteration")
plt.ylabel("episode return")
plt.title("Policy Gradient on CartPole")
plt.legend()
plt.grid(alpha=0.3)
curve_path = OUT_DIR / "minimal_pg_learning_curve.png"
plt.savefig(curve_path, dpi=160, bbox_inches="tight")
print("saved learning curve:", curve_path)
if display is not None:
    plt.show()
else:
    plt.close()

checkpoint_path = OUT_DIR / "minimal_pg_cartpole.pt"
torch.save(
    {
        "policy": policy.state_dict(),
        "critic": None if critic is None else critic.state_dict(),
        "history": history,
        "config": {
            "env_id": ENV_ID,
            "use_reward_to_go": USE_REWARD_TO_GO,
            "use_baseline": USE_BASELINE,
            "gae_lambda": GAE_LAMBDA,
            "normalize_advantages": NORMALIZE_ADVANTAGES,
        },
    },
    checkpoint_path,
)
print("saved checkpoint:", checkpoint_path)


# %% [markdown]
# ## 8. Visualize one rollout
#
# This saves a GIF so you can inspect the trained policy behavior.

# %%
eval_with_video = evaluate(policy, episodes=1, render=True)
video_traj = eval_with_video["video_traj"]
gif_path = OUT_DIR / "minimal_pg_cartpole_rollout.gif"

if video_traj is not None and video_traj.image_obs is not None and len(video_traj.image_obs) > 0:
    imageio.mimsave(gif_path, list(video_traj.image_obs), duration=1000 / 30)
    print("rollout return =", eval_with_video["return_mean"])
    print("saved gif:", gif_path)
    if Image is not None and display is not None:
        display(Image(filename=str(gif_path)))
else:
    print("No rendered frames were returned by the environment.")
