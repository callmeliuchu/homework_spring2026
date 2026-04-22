"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # applicable for iterative samplers like flow or diffusion
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        dim = state_dim
        for i in  range(0,len(hidden_dims)):
            layers.append(nn.Linear(dim,hidden_dims[i]))
            layers.append(nn.ReLU())
            dim = hidden_dims[i]
        layers.append(nn.Linear(dim,chunk_size * action_dim))
        self.mlp = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred_chunk = self.mlp(state).reshape(
            -1, self.chunk_size, self.action_dim
        )
        return torch.mean((pred_chunk - action_chunk) ** 2)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # raise NotImplementedError
        with torch.no_grad():
            chunks = self.mlp(state) # B K C
            return chunks.reshape(-1,self.chunk_size,self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        dim = state_dim + chunk_size * action_dim + 1 ## [ state chunk_size * action_dim t]
        for i in  range(0,len(hidden_dims)):
            layers.append(nn.Linear(dim,hidden_dims[i]))
            layers.append(nn.ReLU())
            dim = hidden_dims[i]
        layers.append(nn.Linear(dim,chunk_size * action_dim))
        self.mlp = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # raise NotImplementedError
        # v = self.mlp(state) # B K C
        # v = v.reshape(-1,self.chunk_size,self.action_dim)
        B = state.shape[0]
        t = torch.rand(B, 1,1, device=action_chunk.device)
        noise = torch.randn(B,self.chunk_size,self.action_dim,device=state.device)
        chunks = (1-t) * noise +   t * action_chunk
        flaten_chunks = chunks.reshape(-1,self.chunk_size*self.action_dim)
        input_state = torch.concat([state,flaten_chunks,t.reshape(B,1)],dim=-1)
        v  = self.mlp(input_state).reshape(-1,self.chunk_size*self.action_dim)
        target = -noise.reshape(-1,self.chunk_size*self.action_dim) + action_chunk.reshape(-1,self.chunk_size*self.action_dim)
        loss = ((v-target)**2).mean()
        return loss


    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # raise NotImplementedError
        with torch.no_grad():
            batch_size = state.shape[0]
            chunks = torch.randn(batch_size,self.chunk_size,self.action_dim,device=state.device)
            dt =  1 / num_steps
            for i in range(num_steps):
                t = torch.ones(batch_size,1,device=state.device) * i / num_steps
                flaten_chunks = chunks.reshape(-1,self.chunk_size*self.action_dim)
                input_state = torch.concat([state,flaten_chunks,t],dim=-1)
                v  = self.mlp(input_state).reshape(-1,self.chunk_size,self.action_dim)
                chunks += dt * v
            return chunks


class DiffusionPolicy(BasePolicy):
    """Predicts action chunks with a velocity diffusion loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        num_train_steps: int = 50,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.num_train_steps = num_train_steps
        betas = torch.linspace(1e-4, 0.02, num_train_steps)
        alpha_bars = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alpha_bars", alpha_bars)

        layers = []
        dim = state_dim + chunk_size * action_dim + 1
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, chunk_size * action_dim))
        self.mlp = nn.Sequential(*layers)

    def _alpha_sigma(self, timesteps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1)
        alpha = torch.sqrt(alpha_bar)
        sigma = torch.sqrt(1.0 - alpha_bar)
        return alpha, sigma

    def _predict_velocity(
        self,
        state: torch.Tensor,
        noisy_chunk: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        model_input = torch.cat(
            [
                state,
                noisy_chunk.reshape(batch_size, self.chunk_size * self.action_dim),
                timesteps.float().unsqueeze(-1) / max(self.num_train_steps - 1, 1),
            ],
            dim=-1,
        )
        return self.mlp(model_input).reshape(
            batch_size, self.chunk_size, self.action_dim
        )

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        timesteps = torch.randint(
            0,
            self.num_train_steps,
            (batch_size,),
            device=state.device,
        )
        noise = torch.randn_like(action_chunk)
        alpha, sigma = self._alpha_sigma(timesteps)
        noisy_chunk = alpha * action_chunk + sigma * noise
        v_target = alpha * noise - sigma * action_chunk
        v_pred = self._predict_velocity(state, noisy_chunk, timesteps)
        return torch.mean((v_pred - v_target) ** 2)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        with torch.no_grad():
            batch_size = state.shape[0]
            chunk = torch.randn(
                batch_size,
                self.chunk_size,
                self.action_dim,
                device=state.device,
            )
            for timestep in range(self.num_train_steps - 1, -1, -1):
                timestep_batch = torch.full(
                    (batch_size,),
                    timestep,
                    device=state.device,
                    dtype=torch.long,
                )
                alpha, sigma = self._alpha_sigma(timestep_batch)
                v_pred = self._predict_velocity(state, chunk, timestep_batch)
                x0_pred = alpha * chunk - sigma * v_pred
                x0_pred = torch.clamp(x0_pred, -3.0, 3.0)
                noise_pred = sigma * chunk + alpha * v_pred

                if timestep == 0:
                    chunk = x0_pred
                    continue

                prev_timestep = torch.full(
                    (batch_size,),
                    timestep - 1,
                    device=state.device,
                    dtype=torch.long,
                )
                alpha_prev, sigma_prev = self._alpha_sigma(prev_timestep)
                chunk = alpha_prev * x0_pred + sigma_prev * noise_pred
            return chunk


PolicyType: TypeAlias = Literal["mse", "flow", "diffusion"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "diffusion":
        return DiffusionPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
