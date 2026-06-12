# Offline/Online Agent Bug Notes

This note records the implementation issues found while checking the final project offline-to-online runs on A100.

## 1. QSM beta schedule can produce NaNs

File: `src/agents/qsm_agent.py`

### Problem

`cosine_beta_schedule()` could return a final beta value equal or very close to `1.0`.

Before the fix:

```python
return 1 - f[1:] / f[:-1]
```

Then the sampler computes:

```python
alpha = self.alphas[t]
x = (1 / alpha.sqrt()) * (x - ((1-alpha) / (1-alpha_hat).sqrt()) * eps_pred)
```

Since `alpha = 1 - beta`, `beta = 1.0` makes `alpha = 0.0`, so `1 / alpha.sqrt()` becomes `inf`. This quickly turns the QSM losses into `nan`.

### Evidence

The remote QSM run under:

```text
/mnt1/mnt1/nlp/lc/final_project_offline_online/exp/fp_oo_qsm/
```

already showed NaNs in `train.csv`:

```text
nan,nan,nan,nan,225000
nan,nan,nan,nan,230000
nan,nan,nan,nan,235000
```

### Fix

Clamp the beta schedule to a numerically valid range:

```python
return torch.clamp(1 - f[1:] / f[:-1], min=1e-5, max=0.999)
```

This keeps `alpha = 1 - beta` positive during DDPM sampling.

## 2. FQL teacher action was clipped before distillation

File: `src/agents/fql_agent.py`

### Problem

`get_bc_action()` integrates the BC flow actor and returns the teacher action for one-step actor distillation.

Before the fix, it clipped the teacher:

```python
action = torch.clamp(action, -1, 1)
return action
```

But the FQL one-step actor should distill toward the raw BC flow policy output. Clipping should only happen when feeding actions to the critic or returning actions to the environment.

The affected loss is:

```python
with torch.no_grad():
    bc_actions = self.get_bc_action(observations, noise)
distill_loss = self.alpha * ((bc_actions - pred_actions) ** 2).mean()
```

With clipping inside `get_bc_action()`, the one-step actor learns a saturated teacher near action bounds instead of the actual BC flow output.

### Fix

Return the raw integrated action from `get_bc_action()`:

```python
return action
```

The existing clips remain in the right places:

```python
action = torch.clamp(action, -1, 1)      # evaluation action
clip_actions = torch.clamp(pred_actions, -1, 1)  # critic input
```

## Current interpretation

`SAC+BC` and `IFQL` learning on the same environment suggests the dataset, environment, replay buffer, and evaluation loop are working.

`QSM` was invalid because its training had already gone to NaN.

`FQL` had a clear distillation-target issue. It may still need hyperparameter tuning after the fix.

## 3. DSRL mutates the z input while integrating the flow

File: `src/agents/dsrl_agent.py`

### Problem

`sample_flow_actions()` used the input noise tensor directly:

```python
actions = noises
...
actions += vs / self.flow_steps
```

This mutates the caller's tensor in place. In `update_qz()`, the same tensor is later used as the `z_critic` input:

```python
scaled_z = self.noise_scale * noises
with torch.no_grad():
    actions = self.sample_flow_actions(observations, scaled_z)
qz = self.z_critic(observations, scaled_z)
```

Because `sample_flow_actions()` modified `scaled_z`, `z_critic` was trained on a flow-integrated action-like tensor instead of the original scaled noise `z`. The noise actor later queries `z_critic` with fresh unmodified noise samples, so the learned `Q_z(s,z)` does not match the space used by the policy.

This explains a plausible failure mode where DSRL losses remain finite but evaluation success stays at zero.

### Fix

Clone the noise before Euler integration:

```python
actions = noises.clone()
```

This preserves the original `z` tensor for `z_critic` training while still producing the flow action for the target.

## 4. QSM used raw integer timesteps as MLP input

File: `src/agents/qsm_agent.py`

### Problem

The QSM actor is implemented with `VectorFieldPolicy`, which simply concatenates observation, action/noisy action, and time:

```python
torch.cat([obs, acs, times], dim=-1)
```

Before the fix, QSM passed raw integer timesteps to this MLP:

```python
t_batch = torch.full((x.shape[0], 1), t, ...)
t_input = t.view(B, 1).float()
```

For `flow_steps=10`, the network saw time values in `[0, 9]`. Other flow-style agents in this project use normalized continuous time in `[0, 1]`, and this network has no timestep embedding to handle raw diffusion indices. This can make the actor hard to train even when losses remain finite.

### Fix

Normalize QSM timesteps before passing them to the actor:

```python
t_batch = torch.full((x.shape[0], 1), t / self.flow_steps, ...)
t_input = t.view(B, 1).to(dtype=actions.dtype) / self.flow_steps
```
