#!/usr/bin/env python3
"""Small hands-on practice for torch.distributions."""

from __future__ import annotations

import torch
import torch.distributions as D


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def categorical_demo() -> None:
    section("1) Categorical demo (discrete actions)")
    torch.manual_seed(0)

    # Batch size B=3, action dim=4
    logits = torch.tensor(
        [
            [2.0, 0.5, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [-1.0, 1.0, 3.0, 0.5],
        ],
        dtype=torch.float32,
    )
    dist = D.Categorical(logits=logits)

    actions = dist.sample()  # (B,)
    log_probs = dist.log_prob(actions)  # (B,)
    entropies = dist.entropy()  # (B,)

    print("logits shape:", tuple(logits.shape))
    print("sampled actions:", actions.tolist())
    print("log_prob(actions):", log_probs.tolist())
    print("entropy:", entropies.tolist())

    # Verify against manual softmax + gather + log
    probs = torch.softmax(logits, dim=-1)
    manual = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1))
    print("manual log_prob:", manual.tolist())
    print("max abs diff:", float((manual - log_probs).abs().max()))


def normal_demo() -> None:
    section("2) Normal demo (continuous actions)")
    torch.manual_seed(1)

    mean = torch.tensor(
        [
            [0.0, 0.5],
            [1.0, -1.0],
            [-0.5, 0.2],
        ],
        dtype=torch.float32,
    )
    logstd = torch.tensor([-0.2, -0.7], dtype=torch.float32)  # (act_dim,)
    std = torch.exp(logstd).expand_as(mean)  # broadcast to (B, act_dim)

    dist = D.Normal(mean, std)
    actions = dist.sample()  # (B, act_dim)
    log_probs_per_dim = dist.log_prob(actions)  # (B, act_dim)
    log_probs = log_probs_per_dim.sum(dim=-1)  # (B,)
    entropies = dist.entropy().sum(dim=-1)  # (B,)

    print("mean shape:", tuple(mean.shape))
    print("std shape:", tuple(std.shape))
    print("sampled actions:\n", actions)
    print("log_prob per dim:\n", log_probs_per_dim)
    print("log_prob sum:", log_probs.tolist())
    print("entropy sum:", entropies.tolist())


def policy_gradient_toy_loss() -> None:
    section("3) Toy PG loss with Categorical")
    torch.manual_seed(2)

    logits = torch.tensor(
        [[1.0, 0.0, -1.0], [0.2, 0.2, 0.2], [2.0, 1.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    actions = torch.tensor([0, 2, 1], dtype=torch.long)
    advantages = torch.tensor([1.2, -0.5, 0.3], dtype=torch.float32)

    dist = D.Categorical(logits=logits)
    log_probs = dist.log_prob(actions)  # (B,)
    loss = -(log_probs * advantages).mean()
    loss.backward()

    print("actions:", actions.tolist())
    print("advantages:", advantages.tolist())
    print("log_probs:", log_probs.tolist())
    print("loss:", float(loss.item()))
    print("logits grad:\n", logits.grad)


def mini_exercises() -> None:
    section("4) Mini exercises (print-only)")
    print("Exercise A:")
    print("  Change logits in categorical_demo and rerun.")
    print("  Observe how sampled action frequency changes.")
    print("")
    print("Exercise B:")
    print("  In normal_demo, increase logstd by +1.0 and rerun.")
    print("  Observe larger action variance and entropy.")
    print("")
    print("Exercise C:")
    print("  In policy_gradient_toy_loss, flip advantage sign.")
    print("  Observe gradient direction change.")


def main() -> None:
    categorical_demo()
    normal_demo()
    policy_gradient_toy_loss()
    mini_exercises()


if __name__ == "__main__":
    main()

