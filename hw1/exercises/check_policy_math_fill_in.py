"""Checks for policy_math_fill_in.py.

Run:
    uv run python exercises/check_policy_math_fill_in.py
"""

from __future__ import annotations

import math

import torch

from policy_math_fill_in import (
    cosine_alpha_bars,
    diffusion_alpha_sigma,
    diffusion_forward_noising,
    diffusion_recover_noise,
    diffusion_recover_x0,
    diffusion_velocity_target,
    explain_schedule_gap,
    flow_euler_step,
    flow_interpolate,
    flow_target_velocity,
    linear_alpha_bars,
    mse_predict,
    sqrt_alpha_bars,
)


def assert_close(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    if not torch.allclose(actual, expected, atol=1e-6, rtol=1e-6):
        raise AssertionError(f"{name} mismatch\nactual={actual}\nexpected={expected}")


def check_mse_predict() -> None:
    state = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weight = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    bias = torch.tensor([1.0, -2.0])
    expected = state @ weight + bias
    actual = mse_predict(state, weight, bias)
    assert_close(actual, expected, "mse_predict")


def check_flow() -> None:
    x0 = torch.tensor([[[4.0]], [[10.0]]])
    noise = torch.tensor([[[2.0]], [[6.0]]])
    t = torch.tensor([0.25, 0.5])
    expected_xt = torch.tensor([[[2.5]], [[8.0]]])
    actual_xt = flow_interpolate(x0, noise, t)
    assert_close(actual_xt, expected_xt, "flow_interpolate")

    expected_v = x0 - noise
    actual_v = flow_target_velocity(x0, noise)
    assert_close(actual_v, expected_v, "flow_target_velocity")

    dt = 0.1
    expected_next = expected_xt + dt * expected_v
    actual_next = flow_euler_step(expected_xt, expected_v, dt)
    assert_close(actual_next, expected_next, "flow_euler_step")


def check_diffusion_core() -> None:
    alpha_bar = torch.tensor([0.81, 0.36])
    alpha, sigma = diffusion_alpha_sigma(alpha_bar)
    assert_close(alpha, torch.tensor([0.9, 0.6]), "diffusion_alpha_sigma.alpha")
    assert_close(
        sigma,
        torch.tensor([math.sqrt(0.19), 0.8]),
        "diffusion_alpha_sigma.sigma",
    )

    x0 = torch.tensor([[[1.0]], [[2.0]]])
    noise = torch.tensor([[[0.5]], [[-1.0]]])
    x_t = diffusion_forward_noising(x0, noise, alpha_bar)

    expected_x_t = torch.tensor(
        [
            [[0.9 * 1.0 + math.sqrt(0.19) * 0.5]],
            [[0.6 * 2.0 + 0.8 * (-1.0)]],
        ]
    )
    assert_close(x_t, expected_x_t, "diffusion_forward_noising")

    v = diffusion_velocity_target(x0, noise, alpha_bar)
    expected_v = torch.tensor(
        [
            [[0.9 * 0.5 - math.sqrt(0.19) * 1.0]],
            [[0.6 * (-1.0) - 0.8 * 2.0]],
        ]
    )
    assert_close(v, expected_v, "diffusion_velocity_target")

    recovered_x0 = diffusion_recover_x0(x_t, v, alpha_bar)
    recovered_noise = diffusion_recover_noise(x_t, v, alpha_bar)
    assert_close(recovered_x0, x0, "diffusion_recover_x0")
    assert_close(recovered_noise, noise, "diffusion_recover_noise")


def check_schedules() -> None:
    linear = linear_alpha_bars(50)
    sqrt = sqrt_alpha_bars(50)
    cosine = cosine_alpha_bars(50)

    if linear.shape != (50,):
        raise AssertionError("linear_alpha_bars shape mismatch")
    if sqrt.shape != (50,):
        raise AssertionError("sqrt_alpha_bars shape mismatch")
    if cosine.shape != (50,):
        raise AssertionError("cosine_alpha_bars shape mismatch")

    if not (linear[0] < 1.0 and linear[-1] > 0.5):
        raise AssertionError("linear schedule should end far from pure noise in this setup")
    if not torch.isclose(sqrt[-1], torch.tensor(0.0)):
        raise AssertionError("sqrt schedule should end at 0")
    if not (cosine[-1] < 1e-4):
        raise AssertionError("cosine schedule should end near 0")


def check_explanation() -> None:
    text = explain_schedule_gap()
    lowered = text.lower()
    required = ["alpha_bar", "pure noise", "mismatch"]
    for word in required:
        if word not in lowered:
            raise AssertionError(f"explain_schedule_gap must mention '{word}'")


def main() -> None:
    check_mse_predict()
    check_flow()
    check_diffusion_core()
    check_schedules()
    check_explanation()
    print("All checks passed.")


if __name__ == "__main__":
    main()
