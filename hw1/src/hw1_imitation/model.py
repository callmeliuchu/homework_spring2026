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
        num_steps: int = 10,  # only applicable for flow policy
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
        start = state_dim
        for hidden_dim in hidden_dims:    
            layers.append(nn.Linear(start,hidden_dim))
            layers.append(nn.ReLU())
            start = hidden_dim
        layers.append(nn.Linear(start,chunk_size * action_dim))
        self.mlp = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # raise NotImplementedError
        B = state.shape[0]
        outputs = self.mlp(state) # B,S
        loss = ((outputs.reshape(-1,self.chunk_size,self.action_dim) - action_chunk) ** 2).mean()
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # raise NotImplementedError
        with torch.no_grad():
            outputs = self.mlp(state)
            return outputs.reshape(-1, self.chunk_size, self.action_dim)

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
        start = state_dim + chunk_size * action_dim + 1
        for hidden_dim in hidden_dims:    
            layers.append(nn.Linear(start,hidden_dim))
            layers.append(nn.ReLU())
            start = hidden_dim
        layers.append(nn.Linear(start,chunk_size * action_dim))
        self.mlp = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # raise NotImplementedError
        B = state.shape[0]
        noise = torch.randn(B,self.chunk_size * self.action_dim)
        t = torch.rand(B,1)
        xt =  noise * (1 - t) + t * action_chunk.reshape(B,-1)
        xt = torch.concat([state,xt,t],dim=-1)
        vt = self.mlp(xt) # B, self.chunk_size * self.action_dim
        labels = action_chunk.reshape(B,-1) - noise
        loss = ((labels - vt) ** 2).mean()
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # raise NotImplementedError
        with torch.no_grad():
            B = state.shape[0]
            action = torch.randn(B,self.chunk_size * self.action_dim)
            for t in range(num_steps):
                xt = torch.concat([state,action,torch.full((B,1),t / num_steps)],dim=-1)
                vt = self.mlp(xt)
                action += vt * 1 / num_steps
            return action.reshape(B,self.chunk_size,self.action_dim)

PolicyType: TypeAlias = Literal["mse", "flow"]


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
    raise ValueError(f"Unknown policy type: {policy_type}")
