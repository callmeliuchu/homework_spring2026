from typing import Optional
import torch
from torch import nn
import numpy as np
from torch.xpu import device_of
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class FQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_actor,
        make_bc_actor_optimizer,
        make_onestep_actor,
        make_onestep_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        alpha: float,
    ):
        super().__init__()

        self.action_dim = action_dim

        self.bc_actor = make_bc_actor(observation_shape, action_dim)
        self.onestep_actor = make_onestep_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.bc_actor_optimizer = make_bc_actor_optimizer(self.bc_actor.parameters())
        self.onestep_actor_optimizer = make_onestep_actor_optimizer(self.onestep_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.alpha = alpha

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None] # 1 obervation
        # TODO(student): Compute the action for evaluation
        # Hint: Unlike SAC+BC and IQL, the evaluation action is *sampled* (i.e., not the mode or mean) from the policy
        noise = torch.randn(1,self.action_dim,device=observation.device)
        v = self.onestep_actor(observation,noise)
        action = noise + v
        action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action)[0]

    @torch.compile
    def get_bc_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Used for training.
        """
        # TODO(student): Compute the BC flow action using the Euler method for `self.flow_steps` steps
        # Hint: This function should *only* be used in `update_onestep_actor`
        action = noise
        delta = 1.0 / self.flow_steps
        B = observation.shape[0]
        for i in range(self.flow_steps):
            action += delta * self.bc_actor(observation,action,torch.full((B,1),i/self.flow_steps,device=observation.device))
        return action

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
        Update Q(s, a)
        """
        # TODO(student): Compute the Q loss
        # Hint: Use the one-step actor to compute next actions
        # Hint: Remember to clamp the actions to be in [-1, 1] when feeding them to the critic!
        q = self.critic(observations,actions) # 2 B
        noise = torch.randn_like(actions)
        next_actions = noise + self.onestep_actor(next_observations,noise)
        next_actions = torch.clamp(next_actions,-1,1)
        with torch.no_grad():
            target_q = self.target_critic(next_observations,next_actions) # 2 ,B
            target_q = target_q.min(dim=0).values #B
        
        target_v = rewards + (1-dones.float()) * self.discount * target_q # B
        loss = ((q - target_v) ** 2).mean()

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    @torch.compile
    def update_bc_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the BC actor
        """
        # TODO(student): Compute the BC flow loss
        B = observations.shape[0]
        noises = torch.randn_like(actions)
        t = torch.rand(B,1,device=observations.device) # B
        Xt = (1-t) * noises + t * actions
        Vt = actions - noises
        pred_Vt = self.bc_actor(observations,Xt,t) # B,A
        loss = ((pred_Vt - Vt) ** 2).mean()

        self.bc_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_actor_optimizer.step()

        return {
            "loss": loss,
        }

    @torch.compile
    def update_onestep_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the one-step actor
        """
        # TODO(student): Compute the one-step actor loss
        # Hint: Do *not* clip the one-step actor actions when computing the distillation loss
        noise = torch.randn_like(actions)
        B = actions.shape[0]
        t= torch.zeros(B,1,device=observations.device)
        a_student_v = self.onestep_actor(observations,noise,t) # B,A
        a_student = noise + a_student_v 
        a_teacher = self.get_bc_action(observations,noise)
        distill_loss = ((a_student - a_teacher) ** 2).mean()

        # Hint: *Do* clip the one-step actor actions when feeding them to the critic
        a_student_clamp = torch.clamp(a_student,-1,1)
        q = self.critic(observations,a_student_clamp) #  2,B
        q = q.min(dim=0).values # B
        q_loss = -q.mean()

        # Total loss.
        loss = q_loss + self.alpha * distill_loss

        # Additional metrics for logging.
        mse = ((a_student_clamp - actions) ** 2).mean()

        self.onestep_actor_optimizer.zero_grad()
        loss.backward()
        self.onestep_actor_optimizer.step()

        return {
            "total_loss": loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "mse": mse,
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
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_bc_actor = self.update_bc_actor(observations, actions)
        metrics_onestep_actor = self.update_onestep_actor(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"bc_actor/{k}": v.item() for k, v in metrics_bc_actor.items()},
            **{f"onestep_actor/{k}": v.item() for k, v in metrics_onestep_actor.items()},
        }

        self.update_target_critic()
        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        with torch.no_grad():
            for data1,data2 in zip(self.critic.parameters(),self.target_critic.parameters()):
                data2.data.copy_((1-self.target_update_rate) * data2.data + self.target_update_rate * data1.data)
