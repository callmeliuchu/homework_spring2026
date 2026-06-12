from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class IFQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor_flow,
        make_actor_flow_optimizer,
        make_critic,
        make_critic_optimizer,
        make_value,
        make_value_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        online_training: bool = False,
        num_samples: int = 32,
        expectile: float = 0.9,
        rho: float = 0.5,
    ):
        super().__init__()

        self.action_dim = action_dim
        
        # TODO(student): Create flow actor
        self.flow_actor = make_actor_flow(observation_shape, action_dim)

        # TODO(student): Create critic (ensemble of Q-functions), target critic (ensemble of Q-functions), and value function
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.value = make_value(observation_shape)

        # TODO(student): Create optimizers for all the above models
        self.actor_flow_optimizer = make_actor_flow_optimizer(self.flow_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.value_optimizer = make_value_optimizer(self.value.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.num_samples = num_samples
        self.expectile = expectile

    @staticmethod
    def expectile_loss(adv: torch.Tensor, expectile: float) -> torch.Tensor:
        """
        Compute the expectile loss for IFQL
        """
        # TODO(student): Implement the expectile loss
        # return ...
        weights = torch.where(adv > 0,expectile,1-expectile)
        return (weights * adv ** 2).mean()

    @torch.compile
    def update_value(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """
        Update value function
        """
        # TODO(student): Implement the value function update
        with torch.no_grad():
            q = self.target_critic(observations, actions).min(dim=0).values
        adv =  q - self.value(observations)
        loss = self.expectile_loss(adv,self.expectile)
        # TODO(student): Update value function
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        return {
            'value_loss':loss
        }

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Rejection / best-of-n sampling using the flow policy and critic.

        We:
          1. Sample multiple candidate actions via the BC flow.
          2. Evaluate them with the critic.
          3. Pick the action with the highest Q-value.
        """
        # TODO(student): Implement rejection sampling
        B,_ = observations.shape
        actions_candates = []
        observations_candates = []
        for _ in range(self.num_samples):
            noise = torch.randn(B, self.action_dim, device=observations.device, dtype=observations.dtype)
            actions = self.get_flow_action(observations,noise) # B,A
            actions = torch.clamp(actions,-1,1)
            actions_candates.append(actions) 
            observations_candates.append(observations)
        
        actions_candates = torch.stack(actions_candates,dim=1) # B,N,A
        observations_candates = torch.stack(observations_candates,dim=1) # B,N,A

        actions_candates1 = actions_candates.reshape(B*self.num_samples,-1) # B*N,A
        observations_candates1 = observations_candates.reshape(B*self.num_samples,-1) # B*N,A

        q = self.critic(observations_candates1,actions_candates1).mean(dim=0).reshape(B,-1) #  B,N
        idxs = q.argmax(dim=-1)
        batch_idxs = torch.arange(B, device=observations.device)
        actions = actions_candates[batch_idxs,idxs] # B,A
        return actions

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        # TODO(student): Implement get action
        observations = ptu.from_numpy(observation)[None]
        action = ptu.to_numpy(self.sample_actions(observations)[0])
        return action

    @torch.compile
    def get_flow_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Compute the flow action using Euler integration for `self.flow_steps` steps.
        """
        # TODO(student): Implement euler integration to get flow action
        actions = noise
        B,_ = observation.shape
        for i in range(self.flow_steps):
            t = torch.full((B, 1), i / self.flow_steps, device=observation.device, dtype=observation.dtype)
            vs = self.flow_actor(observation,actions,t)
            actions += vs * 1 / self.flow_steps
        return actions

    @torch.compile
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a) using the learned value function for bootstrapping,
        as in IFQL / IQL-style critic training.
        """
        # TODO(student): Implement Q-function update
        with torch.no_grad():
            target = rewards + self.discount * self.value(next_observations) * (1-dones.float())
        # TODO(student): Update Q-function
        q = self.critic(observations,actions)
        loss = ((q - target) ** 2).mean()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        return {
            'q_loss':loss
        }


    @torch.compile
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the flow actor using the velocity matching loss.
        """
        # TODO(student): Implement flow actor update
        B, _ = observations.shape
        t = torch.rand(B, 1, device=actions.device, dtype=actions.dtype)
        noise = torch.randn_like(actions)
        xt = noise * (1-t) + actions * t
        vs = self.flow_actor(observations,xt,t)
        preds = actions - noise
        loss = ((vs - preds) ** 2).mean()
        # TODO(student): Update flow actor
        self.actor_flow_optimizer.zero_grad()
        loss.backward()
        self.actor_flow_optimizer.step()
        return {
            'actor_loss': loss
        }


    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_v = self.update_value(observations, actions)
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_actor = self.update_actor(observations, actions)
        metrics = {
            **{f"value/{k}": v.item() for k, v in metrics_v.items()},
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        for data2,data1 in zip(self.critic.parameters(),self.target_critic.parameters()):
            data1.data.copy_(data1.data * (1-self.target_update_rate) + data2.data * self.target_update_rate)
