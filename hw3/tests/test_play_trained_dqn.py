import os
import sys
import tempfile
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from agents.dqn_agent import DQNAgent
from configs import dqn_config


class PlayTrainedDQNTest(unittest.TestCase):
    def test_load_agent_restores_dqn_checkpoint(self):
        from scripts.play_trained_dqn import load_agent

        config_file = os.path.join(ROOT, "experiments", "dqn", "cartpole.yaml")
        config = self._make_config(config_file)

        env = config["make_env"](eval=True)
        try:
            agent = DQNAgent(
                env.observation_space.shape,
                env.action_space.n,
                **config["agent_kwargs"],
            )
        finally:
            env.close()

        with tempfile.TemporaryDirectory() as run_dir:
            torch.save(agent.state_dict(), os.path.join(run_dir, "agent.pt"))
            loaded_agent, loaded_config = load_agent(run_dir, config_file, torch.device("cpu"))

        self.assertIsInstance(loaded_agent, DQNAgent)
        self.assertEqual(loaded_config["log_name"], config["log_name"])

        for key, value in agent.state_dict().items():
            self.assertTrue(torch.equal(value, loaded_agent.state_dict()[key]))

    @staticmethod
    def _make_config(config_file):
        import yaml

        with open(config_file, "r") as f:
            config_kwargs = yaml.safe_load(f)

        base_config_name = config_kwargs.pop("base_config")
        return dqn_config.configs[base_config_name](**config_kwargs)


if __name__ == "__main__":
    unittest.main()
