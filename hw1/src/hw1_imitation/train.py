"""Train and evaluate a Push-T imitation policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
import wandb
from torch.utils.data import DataLoader

from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from hw1_imitation.model import build_policy, DiffusionScheduleType, PolicyType
from hw1_imitation.evaluation import Logger, evaluate_policy

LOGDIR_PREFIX = "exp"


@dataclass
class TrainConfig:
    # The path to download the Push-T dataset to.
    data_dir: Path = Path("data")

    # The policy type -- either MSE or flow.
    policy_type: PolicyType = "mse"
    # The number of denoising steps to use for the flow policy (has no effect for the MSE policy).
    flow_num_steps: int = 10
    diffusion_schedule: DiffusionScheduleType = "linear"
    # The action chunk size.
    chunk_size: int = 8

    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.0
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    # The number of epochs to train for.
    num_epochs: int = 400
    # How often to run evaluation, measured in training steps.
    eval_interval: int = 10_000
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)
    # How often to log training metrics, measured in training steps.
    log_interval: int = 100
    # Random seed.
    seed: int = 42
    # WandB project name.
    wandb_project: str = "hw1-imitation"
    # Whether to log to WandB.
    use_wandb: bool = False
    # Experiment name suffix for logging and WandB.
    exp_name: str | None = None


def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train a Push-T MLP policy.",
) -> TrainConfig:
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")

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

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = build_policy(
        config.policy_type,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
        diffusion_schedule=config.diffusion_schedule,
    ).to(device)

    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project, config=config_to_dict(config), name=exp_name
        )
    logger = Logger(log_dir, use_wandb=config.use_wandb)

    ### TODO: PUT YOUR MAIN TRAINING LOOP HERE ###
    # raise NotImplementedError
    optim = torch.optim.Adam(model.parameters(),lr=config.lr)
    best_mean_reward = float("-inf")
    best_ckpt_path = Path(f"best_{config.policy_type}.pt")
    for ep in range(config.num_epochs):

        total_loss = 0.0
        for state,chunk_action in loader:
            state = state.to(device)
            chunk_action = chunk_action.to(device)
            loss = model.compute_loss(state,chunk_action)

            optim.zero_grad()
            loss.backward()
            optim.step()
            print('one step loss',loss)
            total_loss += loss
        total_loss = total_loss / len(loader)

        print('average loss',ep,total_loss)
        # model: BasePolicy,
        # normalizer: Normalizer,
        # device: torch.device,
        # chunk_size: int,
        # video_size: tuple[int, int],
        # num_video_episodes: int,
        # flow_num_steps: int,
        # step: int,
        # logger: Logger,
        # if ep % 20 == 0:
        #     evaluate_policy(model,normalizer,device,config.chunk_size,config.video_size,config.num_video_episodes,
        #     config.flow_num_steps,ep,logger)
        #     if logger.rows:
        #         mean_reward = logger.rows[-1].get("eval/mean_reward")
        #         if isinstance(mean_reward, (float, int)) and mean_reward > best_mean_reward:
        #             best_mean_reward = float(mean_reward)
        #             torch.save(model.state_dict(), best_ckpt_path)
        #             print(
        #                 f"new best eval/mean_reward={best_mean_reward:.4f}, "
        #                 f"saved to {best_ckpt_path}"
        #             )
    

    torch.save(model.state_dict(), f'{config.policy_type}.pt')



def main() -> None:
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
