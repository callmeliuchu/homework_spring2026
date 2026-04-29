# Day 4: Temperature and Entropy

Goal: understand fixed temperature versus automatic tuning, and why SAC often uses `log_prob` instead of an analytic entropy method.

## What temperature does

Temperature `alpha` balances:
- exploitation: choose actions with high Q values
- exploration: maintain policy entropy

In SAC, the actor objective includes:

```python
alpha * log pi(a|s)
```

and the critic target includes:

```python
- alpha * log pi(a'|s')
```

So `alpha` affects both policy improvement and Bellman target construction.

## Fixed temperature

In [experiments/sac/halfcheetah.yaml](/Users/liuchu/codes/homework_spring2026/hw3/experiments/sac/halfcheetah.yaml):

```yaml
temperature: 0.1
auto_tune_temperature: false
```

This means:
- use one constant `alpha`
- easier to reason about
- fewer moving parts
- but you must manually choose a good value

## Automatic temperature tuning

In [experiments/sac/halfcheetah_autotune.yaml](/Users/liuchu/codes/homework_spring2026/hw3/experiments/sac/halfcheetah_autotune.yaml):

```yaml
temperature: 0.1
auto_tune_temperature: true
alpha_learning_rate: 0.0001
```

This means:
- start from an initial temperature
- then learn it automatically during training

The usual target is:

```python
target_entropy = -action_dim
```

so the algorithm tries to keep the policy entropy above a useful baseline.

## Why optimize `log_alpha`

We do not optimize `alpha` directly.

We optimize:

```python
log_alpha
```

and use:

```python
alpha = exp(log_alpha)
```

This is useful because:
- `alpha` must remain positive
- exponentiation guarantees positivity
- optimization is usually more stable

## Intuition for `update_alpha()`

If the policy entropy becomes too low:
- the policy is too deterministic
- `alpha` should go up
- this increases exploration pressure

If the policy entropy becomes too high:
- the policy is too random
- `alpha` should go down

## Typical dual update

Conceptually:

```python
alpha = exp(log_alpha)
alpha_loss = -(alpha * (log_prob.detach() + target_entropy)).mean()
```

Important detail:
- `log_prob.detach()` prevents alpha update from changing actor gradients directly
- `alpha` must remain a tensor connected to `log_alpha`, not a Python float

## Why `distribution.entropy()` is often not used here

For tanh-transformed continuous policies, PyTorch often does not provide a clean analytic entropy implementation.

That is why many SAC implementations estimate entropy using:

```python
-log_prob
```

This is the more practical quantity anyway, because the main SAC losses already use `log_prob` directly.

## Mapping to your code

### `get_temperature()`
This is the convenient readout function:
- if auto-tune is off, return fixed `self.temperature`
- if auto-tune is on, return `exp(log_alpha)`

### `entropy()`
For your current implementation, the safest interpretation is:
- sample an action from the current distribution
- estimate entropy as `-log_prob(sampled_action)`

### `update_alpha()`
This function should:
1. compute `alpha = self.log_alpha.exp()`
2. compute alpha loss
3. backprop through `log_alpha`
4. step `self.alpha_optimizer`

## Common alpha / entropy bugs

### Bug 1: using a Python float in `update_alpha()`
Wrong idea:

```python
alpha = self.get_temperature()
```

This may return a float and break gradient flow.

### Bug 2: trying to call `.entropy()` on an unsupported transformed distribution
Symptom:
- `NotImplementedError`

### Bug 3: using `self.log_alpha` even when auto-tune is disabled
Symptom:
- attribute errors during fixed-temperature training

### Bug 4: wrong direction of alpha update
If entropy is too low and alpha moves down instead of up, exploration gets even worse.

## Fixed temperature vs auto-tune

### Fixed temperature is better when:
- you are debugging core SAC logic
- you want fewer moving parts
- you want to isolate sign and shape bugs

### Auto-tune is better when:
- the base implementation already works
- you want less manual sensitivity to a single temperature setting
- you want the entropy target to adapt during training

## Day 4 self-check

You are done with Day 4 if you can answer:
- Why do we optimize `log_alpha` instead of `alpha`?
- Why can `-log_prob` stand in for entropy in practice?
- Why does `update_alpha()` need a tensor connected to `self.log_alpha`?
- If entropy is too low, should alpha go up or down?
