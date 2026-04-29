# Day 2: Critic Update

Goal: understand exactly how `update_critic()` in `src/agents/sac_agent.py` implements the Bellman target.

## The core formula

```python
target_q = r + gamma * (Q_target(s', a') - alpha * log pi(a'|s'))
```

In your implementation, this maps to the following conceptual steps:
1. Build the next-state action distribution from `next_obs`
2. Sample `next_action`
3. Evaluate `Q_target(next_obs, next_action)`
4. Add entropy bonus if enabled
5. Apply backup strategy if there are multiple critics
6. Form the final target value
7. Fit the current critics to that target with MSE

## Tensor shape audit

Assume:
- batch size = `B`
- action dimension = `A`
- number of critics = `K`

Then the most important tensors are:

- `obs`: `(B, ob_dim)`
- `action`: `(B, A)`
- `reward`: `(B,)`
- `done`: `(B,)`
- `next_action`: `(B, A)`
- `log_prob(next_action)`: `(B,)`
- `self.target_critic(next_obs, next_action)`: `(K, B)`
- `next_qs` after backup strategy: `(K, B)`
- `target_values`: `(K, B)`
- `q_values = self.critic(obs, action)`: `(K, B)`

## Why `log_prob` should be `(B,)`

This codebase builds the action distribution using `Independent(...)` in `src/networks/policies.py`.

That means:
- one batch element corresponds to one scalar log-probability
- the action dimensions are already treated as part of the event

So:

```python
log_prob = action_distribution.log_prob(action)
```

should be `(B,)`, not `(B, A)`.

This is one of the easiest SAC implementation bugs to make.

## Pseudocode for `update_critic()`

```python
def update_critic(obs, action, reward, next_obs, done):
    with torch.no_grad():
        next_dist = actor(next_obs)
        next_action = next_dist.sample()
        next_log_prob = next_dist.log_prob(next_action)
        next_qs = target_critic(next_obs, next_action)      # (K, B)

        if use_entropy_bonus and backup_entropy:
            next_qs = next_qs - alpha * next_log_prob[None, :]

        next_qs = q_backup_strategy(next_qs)                # still (K, B)
        target_values = reward + discount * (1 - done) * next_qs

    q_values = critic(obs, action)                          # (K, B)
    loss = ((q_values - target_values) ** 2).mean()
    optimize critic
```

## Why the entropy term is inside the critic target

Without the entropy term, the critic would learn the value of a purely reward-maximizing future policy.

With the entropy term, the critic learns the value of a future policy that is also rewarded for staying stochastic.

This matters because the actor is optimized against that critic. If the critic ignores entropy but the actor does not, the two objectives become inconsistent.

## Multi-critic intuition

### `mean` backup
Use when you want a simpler baseline:

```python
next_q = mean(Q1, Q2, ..., QK)
```

### `min` backup
Use when you want clipped double-Q behavior:

```python
next_q = min(Q1, Q2)
```

This reduces overestimation bias.

In PyTorch, remember:

```python
values = next_qs.min(dim=0).values
```

not:

```python
next_qs.min(dim=0)
```

because the latter returns both values and indices.

## Common bugs in critic code

### Bug 1: forgetting to convert replay buffer output to tensors
Symptom:
- `linear(): input must be Tensor, not numpy.ndarray`

Source:
- batch sampled from replay buffer is still NumPy

### Bug 2: treating `log_prob` as `(B, A)`
Symptom:
- shape mismatches
- accidental `sum(dim=-1)` over batch axis

### Bug 3: wrong sign on entropy term
Correct target form:

```python
Q_target - alpha * log_prob
```

If the sign is wrong, learning can become unstable or collapse.

### Bug 4: broken `min` backup
Symptom:
- tuple-related errors or weird shape failures

### Bug 5: wrong broadcasting with `reward` and `(K, B)` targets
You want:

```python
reward.shape == (B,)
next_qs.shape == (K, B)
```

so broadcasting expands reward across critics.

## Day 2 self-check

You are done with Day 2 if you can answer:
- Why is the critic target built with the target critic, not the online critic?
- Why does `log_prob` belong in the target?
- Why does `min` backup reduce optimism?
- What shape should `target_values` have when there are `K` critics?
