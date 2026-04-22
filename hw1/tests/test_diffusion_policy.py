import unittest

import torch
from torch import nn

from hw1_imitation.model import DiffusionPolicy, build_policy


class CountingMLP(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.num_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.num_calls += 1
        return torch.zeros(x.shape[0], self.output_dim, device=x.device)


class DiffusionPolicyTest(unittest.TestCase):
    def test_build_policy_creates_diffusion_policy(self) -> None:
        model = build_policy(
            "diffusion",
            state_dim=5,
            action_dim=2,
            chunk_size=8,
            hidden_dims=(32, 32),
        )

        self.assertIsInstance(model, DiffusionPolicy)

    def test_compute_loss_returns_scalar(self) -> None:
        model = build_policy(
            "diffusion",
            state_dim=5,
            action_dim=2,
            chunk_size=8,
            hidden_dims=(32, 32),
        )
        state = torch.randn(4, 5)
        action_chunk = torch.randn(4, 8, 2)

        loss = model.compute_loss(state, action_chunk)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_sample_actions_returns_action_chunk(self) -> None:
        model = build_policy(
            "diffusion",
            state_dim=5,
            action_dim=2,
            chunk_size=8,
            hidden_dims=(32, 32),
        )
        state = torch.randn(3, 5)

        action_chunk = model.sample_actions(state, num_steps=5)

        self.assertEqual(action_chunk.shape, (3, 8, 2))
        self.assertTrue(torch.isfinite(action_chunk).all())

    def test_sample_actions_uses_full_diffusion_schedule(self) -> None:
        model = DiffusionPolicy(
            state_dim=5,
            action_dim=2,
            chunk_size=8,
            hidden_dims=(32, 32),
            num_train_steps=7,
        )
        counter = CountingMLP(output_dim=16)
        model.mlp = counter
        state = torch.randn(2, 5)

        _ = model.sample_actions(state, num_steps=3)

        self.assertEqual(counter.num_calls, 7)


if __name__ == "__main__":
    unittest.main()
