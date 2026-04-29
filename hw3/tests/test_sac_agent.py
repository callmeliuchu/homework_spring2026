import os
import sys
import unittest

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from agents.sac_agent import SoftActorCritic
from infrastructure import pytorch_util as ptu


class DummyActor(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(action_dim))
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor):
        batch_size = obs.shape[0]
        mean = self.mean.unsqueeze(0).expand(batch_size, -1)
        std = self.log_std.exp().unsqueeze(0).expand(batch_size, -1)
        base_dist = torch.distributions.Normal(mean, std)
        transformed = torch.distributions.TransformedDistribution(
            base_dist,
            [torch.distributions.TanhTransform(cache_size=1)],
        )
        return torch.distributions.Independent(
            transformed, reinterpreted_batch_ndims=1
        )


class DummyCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.linear = nn.Linear(obs_dim + action_dim, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        return self.linear(torch.cat([obs, action], dim=-1)).squeeze(-1)


def make_agent(**kwargs):
    ptu.init_gpu(use_gpu=False)

    def make_actor(observation_shape, action_dim):
        return DummyActor(action_dim)

    def make_critic(observation_shape, action_dim):
        return DummyCritic(observation_shape[0], action_dim)

    def make_optimizer(params):
        return torch.optim.Adam(params, lr=1e-3)

    def make_scheduler(optimizer):
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    return SoftActorCritic(
        observation_shape=(3,),
        action_dim=2,
        make_actor=make_actor,
        make_actor_optimizer=make_optimizer,
        make_actor_schedule=make_scheduler,
        make_critic=make_critic,
        make_critic_optimizer=make_optimizer,
        make_critic_schedule=make_scheduler,
        discount=0.99,
        target_update_period=1,
        **kwargs,
    )


class SoftActorCriticTest(unittest.TestCase):
    def test_min_backup_strategy_returns_tensor_targets(self):
        agent = make_agent(num_critic_networks=2, target_critic_backup_type="min")

        next_qs = torch.tensor([[1.0, 5.0], [2.0, 4.0]])

        backup = agent.q_backup_strategy(next_qs)

        expected = torch.tensor([[1.0, 4.0], [1.0, 4.0]])
        self.assertTrue(torch.equal(backup, expected))

    def test_entropy_uses_log_prob_estimate_for_tanh_policy(self):
        agent = make_agent()
        obs = torch.zeros(4, 3)
        action_distribution = agent.actor(obs)

        entropy = agent.entropy(action_distribution)

        self.assertEqual(entropy.shape, (4,))
        self.assertTrue(torch.isfinite(entropy).all())

    def test_auto_temperature_update_changes_log_alpha(self):
        agent = make_agent(auto_tune_temperature=True, temperature=0.1)
        initial_log_alpha = agent.log_alpha.detach().clone()

        info = agent.update_alpha(torch.tensor([-0.5, -1.0]))

        self.assertIn("alpha", info)
        self.assertIn("alpha_loss", info)
        self.assertIsInstance(info["alpha"], float)
        self.assertIsInstance(info["alpha_loss"], float)
        self.assertFalse(torch.equal(initial_log_alpha, agent.log_alpha.detach()))


if __name__ == "__main__":
    unittest.main()
