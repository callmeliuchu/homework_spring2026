from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu
from networks.rl_networks import LogParam

from typing import Sequence


class DSRLAgent(nn.Module):
    """DSRL agent - https://arxiv.org/abs/2506.15799"""

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_flow_actor,
        make_bc_flow_actor_optimizer,
        make_noise_actor,
        make_noise_actor_optimizer,
        make_critic,
        make_critic_optimizer,
        make_z_critic,
        make_z_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        noise_scale: float = 1.0,
        bc_pretrain_steps: int = 100000,
        fixed_alpha: float = 0.01,

        online_training: bool = False,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.noise_scale = noise_scale
        self.bc_pretrain_steps = bc_pretrain_steps
        self.fixed_alpha = fixed_alpha
        self.target_entropy = -action_dim

        # TODO(student): Create BC flow actor and target BC flow actor
        self.bc_flow_actor = make_bc_flow_actor(observation_shape,action_dim)
        self.target_bc_flow_actor = make_bc_flow_actor(observation_shape,action_dim)
        self.target_bc_flow_actor.load_state_dict(self.bc_flow_actor.state_dict())

        # TODO(student): Create noise policy
        self.noise_actor = make_noise_actor(observation_shape,action_dim)

        # TODO(student): Create critic (ensemble of Q-functions), target critic (ensemble of Q-functions), and z critic (for noise policy)
        self.critic = make_critic(observation_shape,action_dim)
        self.target_critic = make_critic(observation_shape,action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.z_critic = make_z_critic(observation_shape,action_dim)

        # TODO(student): Create learnable entropy coefficient
        self.log_alpha = LogParam()

        # TODO(student): Create optimizers for all the above models
        self.bc_flow_actor_optimizer = make_bc_flow_actor_optimizer(self.bc_flow_actor.parameters())
        self.target_bc_flow_actor_optimizer =  make_bc_flow_actor_optimizer(self.target_bc_flow_actor.parameters())
        self.noise_actor_optimizer = make_noise_actor_optimizer(self.noise_actor.parameters())
        self.crtitic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.target_crtitic_optimizer = make_critic_optimizer(self.target_critic.parameters())
        self.z_critic_optimizer = make_z_critic_optimizer(self.z_critic.parameters())
        self.log_alpha_optimizer = torch.optim.Adam(self.log_alpha.parameters(),lr=1e-3)

        self.to(ptu.device)

    @property
    def alpha(self):
        # TODO(student): Allow access to the learnable entropy coefficient (tip: if you are learning log alpha, as in HW3, then when we want to use alpha, you should return the exponential of the log alpha)
        # return ...
        if self.fixed_alpha > 0:
            return torch.as_tensor(self.fixed_alpha, device=ptu.device, dtype=torch.float32)
        return self.log_alpha()

    @torch.compiler.disable
    def sample_flow_actions(self, observations: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        """Euler integration of BC flow from t=0 to t=1."""
        # TODO(student): Implement Euler integration of BC flow. Keep in mind that the target BC flow actor should be used
        # Also note that we can control what we use as the noise input (could be sampled from a noise policy or from a normal distribution)
        # return ...
        actions = noises.clone()
        B = observations.shape[0]
        for t in range(self.flow_steps):
            t_batch = torch.full((B, 1), t / self.flow_steps, device=observations.device, dtype=observations.dtype)
            vs = self.target_bc_flow_actor(observations, actions, t_batch)
            actions += vs / self.flow_steps
        return torch.clamp(actions, -1, 1)


    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """Sample actions using noise policy for noise input to BC flow policy."""
        # TODO(student): Sample noise from the noise policy and use to sample actions from the BC flow policy
        # return ...
        dist = self.noise_actor(observations)
        z = dist.sample() # noise
        actions = self.sample_flow_actions(observations,self.noise_scale * z)
        return actions

    def sample_noise_and_actions(self, observations: torch.Tensor):
        """Sample latent noise z, its log-probability, and the induced action."""
        dist = self.noise_actor(observations)
        z = dist.rsample()
        log_probs = dist.log_prob(z)
        scaled_z = self.noise_scale * z
        actions = self.sample_flow_actions(observations, scaled_z)
        return z, scaled_z, log_probs, actions
        
    
    def get_action(self, observation: np.ndarray):
        """Used for evaluation."""
        # TODO(student): Implement get action
        observations = ptu.from_numpy(observation)[None]
        actions = self.sample_actions(observations)
        return ptu.to_numpy(actions[0])

    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """Update critic"""
        # TODO(student): Implement critic loss
        with torch.no_grad():
            _, _, next_log_probs, next_actions = self.sample_noise_and_actions(next_observations)
            next_q = self.target_critic(next_observations,next_actions).min(dim=0).values
            next_q = next_q - self.alpha.detach() * next_log_probs
            target = rewards + self.discount * (1-dones.float()) * next_q
        
        q = self.critic(observations,actions)
        loss = ((q - target) ** 2).mean()
        self.crtitic_optimizer.zero_grad()
        loss.backward()
        self.crtitic_optimizer.step()
        
        # TODO(student): Update critic
        
        return {
            'criric_loss':loss
        }
    
    def update_qz(self, 
        observations: torch.Tensor,
        actions: torch.Tensor,
        noises: torch.Tensor,
    ) -> dict:
        """Update z_critic."""
        
        # TODO(student): Implement z_critic loss
        scaled_z = self.noise_scale * noises
        with torch.no_grad():
            actions = self.sample_flow_actions(observations,scaled_z)
            targets = self.critic(observations, actions)
        qz = self.z_critic(observations,scaled_z)
        loss = 0.5 * ((qz - targets.detach()) ** 2).mean()

        # TODO(student): Update z_critic
        self.z_critic_optimizer.zero_grad()
        loss.backward()
        self.z_critic_optimizer.step()

        return {
            'qz_loss': loss
        }

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """Update BC flow actor"""
        # TODO(student): Implement BC flow loss
        B = observations.shape[0]
        t = torch.rand(B, 1, device=actions.device, dtype=actions.dtype)
        noise = torch.randn_like(actions)
        xt = (1-t) * noise + t * actions
        vs = self.bc_flow_actor(observations,xt,t)
        pred = actions - noise
        loss = ((pred -vs) ** 2).mean()
        
        # TODO(student): Update BC flow actor
        self.bc_flow_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_flow_actor_optimizer.step()

        return {
            'bc_loss': loss
        }

    @torch.no_grad()
    def evaluate_actor_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        B = observations.shape[0]
        t = torch.rand(B, 1, device=actions.device, dtype=actions.dtype)
        noise = torch.randn_like(actions)
        xt = (1 - t) * noise + t * actions
        vs = self.bc_flow_actor(observations, xt, t)
        pred = actions - noise
        loss = ((pred - vs) ** 2).mean()
        return {
            'bc_loss': loss
        }
    
    def update_noise_actor(self,
        observations: torch.Tensor,
    ) -> dict:
        """Update noise actor."""
        # TODO(student): Implement noise actor loss
        dist = self.noise_actor(observations)
        z = dist.rsample() # noise
        log_probs = dist.log_prob(z)
        scaled_z = self.noise_scale * z

        qz = self.z_critic(observations, scaled_z).min(dim=0).values
        loss = (self.alpha.detach() * log_probs - qz).mean()
        
        # TODO(student): Update noise actor
        self.noise_actor_optimizer.zero_grad()
        loss.backward()
        self.noise_actor_optimizer.step()
        
        return {
            'noise_actor_loss':loss
        }

    def update_alpha(self,observations: torch.Tensor) -> dict:
        """Update alpha."""
        # TODO(student): Implement alpha loss
        if self.fixed_alpha > 0:
            return {
                'alpha_loss': torch.zeros((), device=observations.device),
            }

        dist = self.noise_actor(observations)
        z = dist.rsample()
        log_probs = dist.log_prob(z)

        loss = -(self.log_alpha.log_param * (log_probs + self.target_entropy).detach()).mean()

        self.log_alpha_optimizer.zero_grad()
        loss.backward()
        self.log_alpha_optimizer.step()
        
        # TODO(student): Update alpha
        return {
            'alpha_loss':loss
        }

    def update_target_critic(self) -> None:
        # TODO(student): Implement target critic update
        for data1,data2 in zip(self.target_critic.parameters(),self.critic.parameters()):
            data1.data.copy_(data1.data * (1-self.target_update_rate) + data2.data * self.target_update_rate)

    def update_target_bc_flow_actor(self) -> None:
        # TODO(student): Implement target BC flow actor update
        # return ...
        for data1,data2 in zip(self.target_bc_flow_actor.parameters(),self.bc_flow_actor.parameters()):
            data1.data.copy_(data1.data * (1-self.target_update_rate) + data2.data * self.target_update_rate)

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        # DSRL assumes a fixed behavior diffusion/flow policy and optimizes only
        # the latent-noise policy on top. Train the flow first, then freeze it.
        if step < self.bc_pretrain_steps:
            metrics_actor = self.update_actor(observations, actions)
            self.update_target_bc_flow_actor()
            return {
                "critic/criric_loss": 0.0,
                "z_critic/qz_loss": 0.0,
                "actor/bc_loss": metrics_actor["bc_loss"].item(),
                "noise_actor/noise_actor_loss": 0.0,
                "alpha/alpha_loss": 0.0,
                "alpha/value": self.alpha.item(),
                "noise/log_prob": 0.0,
                "noise/std": 0.0,
                "phase/bc_pretrain": 1.0,
            }

        metrics_actor = self.evaluate_actor_loss(observations, actions)
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        noises = torch.randn_like(actions)
        metrics_qz = self.update_qz(observations, actions, noises)
        metrics_noise_actor = self.update_noise_actor(observations)
        metrics_alpha = self.update_alpha(observations)

        with torch.no_grad():
            dist = self.noise_actor(observations)
            z = dist.rsample()
            noise_log_prob = dist.log_prob(z).mean()
            noise_std = z.std()
            qz_mean = self.z_critic(observations, self.noise_scale * z).mean()

        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"z_critic/{k}": v.item() for k, v in metrics_qz.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
            **{f"noise_actor/{k}": v.item() for k, v in metrics_noise_actor.items()},
            **{f"alpha/{k}": v.item() for k, v in metrics_alpha.items()},
            "alpha/value": self.alpha.item(),
            "noise/log_prob": noise_log_prob.item(),
            "noise/std": noise_std.item(),
            "z_critic/qz_mean": qz_mean.item(),
            "phase/bc_pretrain": 0.0,
        }

        self.update_target_critic()

        return metrics

