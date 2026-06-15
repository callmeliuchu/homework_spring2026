import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Sequence

class QSMAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor,
        make_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        alpha: float,
        inv_temp: float,
        flow_steps: int,
    ):
        super().__init__()

        self.action_dim = action_dim
        
        # TODO(student): Create actor
        self.actor = make_actor(observation_shape, action_dim)
        
        # TODO(student): Create critic (ensemble of Q-functions), target critic (ensemble of Q-functions)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # TODO(student): Create optimizers for all the above models
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())


        self.discount = discount
        self.target_update_rate = target_update_rate
        self.alpha = alpha
        self.inv_temp = inv_temp
        self.flow_steps = flow_steps

        betas = self.cosine_beta_schedule(flow_steps)
        alphas = 1.0 - betas
        alpha_hats = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas) # TODO(student): Implement betas
        self.register_buffer("alphas", alphas) # TODO(student): Implement alphas
        self.register_buffer("alpha_hats", alpha_hats) # TODO(student): Implement alpha_hats

        self.to(ptu.device)
    
    def cosine_beta_schedule(self, timesteps, s: float = 0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alpha_hats = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alpha_hats = alpha_hats / alpha_hats[0]
        betas = 1 - alpha_hats[1:] / alpha_hats[:-1]
        return torch.clamp(betas, min=1e-5, max=0.999)
        
    
    @torch.compiler.disable
    def ddpm_sampler(self, observations: torch.Tensor, noise: torch.Tensor):
        """
        DDPM sampling
        """
        # TODO(student): Implement DDPM sampling
        x = noise
        for t in reversed(range(self.flow_steps)):
            t_batch = torch.full((x.shape[0], 1), t / self.flow_steps, device=x.device, dtype=x.dtype)
            eps_pred = self.actor(observations,x,t_batch)

            alpha = self.alphas[t]
            alpha_hat = self.alpha_hats[t]
            beta = self.betas[t] 

            x = (1 / alpha.sqrt()) * (x - ((1-alpha) / (1-alpha_hat).sqrt()) * eps_pred)

            if t > 0:
                x = x + beta.sqrt() * torch.randn_like(x)

        return torch.clamp(x,-1,1)

    
    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        # TODO(student): Implement get_action
        # return ...
        observations = ptu.from_numpy(observation)[None]
        noise = torch.randn(1, self.action_dim, device=ptu.device)
        actions = self.ddpm_sampler(observations,noise)
        return ptu.to_numpy(actions[0])

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
        Update Critic
        """
        # TODO(student): Implement critic update
        with torch.no_grad():
            noise = torch.randn_like(actions)
            next_actions = self.ddpm_sampler(next_observations,noise)
            target = rewards + self.target_critic(next_observations,next_actions).min(dim=0).values * (1-dones.float()) * self.discount
        # TODO(student): Update critic
        q = self.critic(observations,actions)
        loss = ((q - target) ** 2).mean()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        return {
            'q_loss':loss
        }
        
    @torch.compiler.disable
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the actor
        """

        # TODO(student): Implement actor update

        # TODO(student): Update actor
        
        # return ...
        B = actions.shape[0]
        t = torch.randint(0, self.flow_steps, (B,), device=actions.device)

        noise = torch.randn_like(actions)

        alpha_hat = self.alpha_hats[t].view(B,1)
        a_t = alpha_hat.sqrt() * actions + (1-alpha_hat).sqrt() * noise

        t_input = t.view(B, 1).to(dtype=actions.dtype) / self.flow_steps

        eps_pred = self.actor(observations, a_t, t_input)

        a_t_for_grad = a_t.detach().requires_grad_(True)
        q = self.target_critic(observations, a_t_for_grad).mean(dim=0)
        q_grad = torch.autograd.grad(q.sum(), a_t_for_grad)[0].detach()

        sigma_t = (1 - alpha_hat).sqrt()
        guidance_scale = self.alpha * self.inv_temp
        eps_target = noise - guidance_scale * sigma_t * q_grad

        bc_loss = ((noise - eps_pred) ** 2).mean()
        qsm_loss = ((eps_pred - eps_target.detach()) ** 2).mean()
        loss = qsm_loss
        
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {
            'total_loss': loss,
            'qsm_loss': qsm_loss,
            'bc_loss': bc_loss,
            'q_grad_norm': q_grad.norm(dim=-1).mean(),
            'eps_norm': eps_pred.norm(dim=-1).mean(),
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
        metrics_actor = self.update_actor(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        for data2, data1 in zip(self.critic.parameters(),self.target_critic.parameters()):
            data1.data.copy_(data1.data * (1 - self.target_update_rate) + data2.data * self.target_update_rate)
