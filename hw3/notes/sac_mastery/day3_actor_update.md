# Day 3: Actor Update

Goal: understand why SAC updates the actor through reparameterization, and how the actor loss maps to code.

## The core actor formula

```python
actor_loss = E[alpha * log pi(a|s) - Q(s, a)]
```

You minimize this loss.

That means:
- make `Q(s, a)` larger
- but also keep the policy stochastic enough through the `alpha * log_prob` term

## The single best intuition

The actor is not learning a single best action directly.

It is learning a distribution that:
- tends to place mass on high-value actions
- but is penalized if it becomes too sharp too early

This is why SAC is usually more stable than "fully deterministic and fully greedy" alternatives.

## Why `rsample()` matters

In actor update code, the action must be differentiable with respect to actor parameters.

### Wrong idea

```python
action = action_distribution.sample()
```

This draws a random action, but the gradient path back into the actor is broken.

### Correct idea

```python
action = action_distribution.rsample()
```

This uses the reparameterization trick, so gradients can flow through the sampled action into the actor network.

## Shape flow

Assume:
- number of critics = `K`
- batch size = `B`

Then:
- `action`: `(B, action_dim)`
- `q_values = self.critic(obs, action)`: `(K, B)`
- `q_pi = q_values.mean(dim=0)`: `(B,)`
- `log_prob = action_distribution.log_prob(action)`: `(B,)`
- final actor loss after `.mean()`: scalar

## Why `q_values.mean(dim=0)` becomes `(B,)`

If:

```python
q_values.shape == (K, B)
```

then dimension `0` is the critic axis.

So:

```python
q_values.mean(dim=0)
```

means:
- average across critics
- keep one scalar Q per batch element

Result:

```python
(K, B) -> (B,)
```

This is not averaging over the batch. It is averaging over the ensemble.

## Pseudocode for `actor_loss_reparametrize()`

```python
def actor_loss_reparametrize(obs):
    dist = actor(obs)
    action = dist.rsample()
    q_values = critic(obs, action)               # (K, B)
    q_pi = q_values.mean(dim=0)                  # (B,)
    log_prob = dist.log_prob(action)             # (B,)
    loss = (alpha * log_prob - q_pi).mean()
    entropy_estimate = (-log_prob).mean()
    return loss, entropy_estimate, log_prob
```

## Why the sign is easy to get wrong

You are minimizing loss.

So if you want the actor to prefer larger Q values, the Q term must appear with a minus sign:

```python
-Q(s, a)
```

The entropy-related term appears as:

```python
alpha * log_prob
```

Because `log_prob` is usually negative, this encourages stochasticity.

## Common actor bugs

### Bug 1: using `sample()` instead of `rsample()`
Symptom:
- actor appears to update, but gradient path is wrong

### Bug 2: averaging over the batch instead of the critic axis too early
Symptom:
- shapes collapse unexpectedly
- you see `(K,)` or `(1,)` instead of `(K, B)` or `(B,)`

### Bug 3: minimizing `Q` directly
Wrong:

```python
loss = q_values.mean()
```

This encourages the actor to make Q smaller.

### Bug 4: duplicating entropy bonus in two places
If `actor_loss_reparametrize()` already includes:

```python
alpha * log_prob - q_pi
```

then `update_actor()` should not add the entropy term again.

### Bug 5: treating `log_prob` as per-action-dimension instead of per-sample
This breaks shapes and often leads to accidental extra reductions.

## Natural language explanation of actor loss

The actor tries to put probability mass on actions that the critic says are valuable, but it is charged a price for becoming too certain too quickly.

## Day 3 self-check

You are done with Day 3 if you can answer:
- Why must actor update use `rsample()`?
- Why does `q_values.mean(dim=0)` reduce over critics instead of over the batch?
- Why is the actor loss not just `-Q`?
- If entropy collapses too fast, which part of the actor update would you inspect first?
