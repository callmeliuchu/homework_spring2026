import os
import sys
import unittest

try:
    import torch
    from hw4.train import compute_group_advantages
except ModuleNotFoundError:  # pragma: no cover - depends on optional deps
    torch = None
    compute_group_advantages = None

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@unittest.skipIf(torch is None or compute_group_advantages is None, "train dependencies unavailable")
class TestComputeGroupAdvantages(unittest.TestCase):
    def test_group_size_one_returns_zeros(self):
        rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        advantages = compute_group_advantages(rewards, group_size=1)

        self.assertTrue(torch.equal(advantages, torch.zeros_like(rewards)))

    def test_invalid_group_size_raises(self):
        rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        with self.assertRaises(ValueError):
            compute_group_advantages(rewards, group_size=2)

    def test_zero_variance_group_falls_back_to_zeros(self):
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)

        advantages = compute_group_advantages(rewards, group_size=4)

        self.assertTrue(torch.allclose(advantages, torch.zeros_like(rewards)))


if __name__ == "__main__":
    unittest.main()
