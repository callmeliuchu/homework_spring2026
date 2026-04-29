import os
import sys
import tempfile
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from agents.sac_agent import SoftActorCritic
from configs import sac_config


class PlayTrainedSACTest(unittest.TestCase):
    def test_human_mode_uses_rgb_array_backend_for_mujoco(self):
        from scripts.play_trained_sac import resolve_render_mode

        self.assertEqual(resolve_render_mode("HalfCheetah-v4", "human"), "rgb_array")
        self.assertEqual(
            resolve_render_mode("InvertedPendulum-v4", "human"), "rgb_array"
        )
        self.assertEqual(resolve_render_mode("HalfCheetah-v4", "rgb_array"), "rgb_array")

    def test_load_agent_restores_sac_checkpoint(self):
        from scripts.play_trained_sac import load_agent

        config_file = os.path.join(
            ROOT, "experiments", "sac", "sanity_invertedpendulum.yaml"
        )
        config = self._make_config(config_file)

        env = config["make_env"](eval=True)
        try:
            agent = SoftActorCritic(
                env.observation_space.shape,
                env.action_space.shape[0],
                **config["agent_kwargs"],
            )
        finally:
            env.close()

        with tempfile.TemporaryDirectory() as run_dir:
            torch.save(agent.state_dict(), os.path.join(run_dir, "agent.pt"))
            loaded_agent, loaded_config = load_agent(
                run_dir, config_file, torch.device("cpu")
            )

        self.assertIsInstance(loaded_agent, SoftActorCritic)
        self.assertEqual(loaded_config["log_name"], config["log_name"])

        for key, value in agent.state_dict().items():
            self.assertTrue(torch.equal(value, loaded_agent.state_dict()[key]))

    @staticmethod
    def _make_config(config_file):
        import yaml

        with open(config_file, "r") as f:
            config_kwargs = yaml.safe_load(f)

        base_config_name = config_kwargs.pop("base_config")
        return sac_config.configs[base_config_name](**config_kwargs)


if __name__ == "__main__":
    unittest.main()
