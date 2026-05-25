#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, cwd=cwd, check=True)


def latest_run_dir(exp_root: Path, prefix: str) -> Path:
    matches = sorted(exp_root.glob(f"{prefix}*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No run directory found for prefix: {prefix}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all HW2 PG variants and print demo commands.")
    parser.add_argument("--n_iter", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--eval_batch_size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env_name", type=str, default="CartPole-v0")
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--episodes", type=int, default=5, help="Episodes for demo command examples")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    exp_root = project_root / "exp"
    exp_root.mkdir(parents=True, exist_ok=True)

    # Common training args
    common = [
        "uv",
        "run",
        "python",
        "src/scripts/run.py",
        "--env_name",
        args.env_name,
        "-n",
        str(args.n_iter),
        "-b",
        str(args.batch_size),
        "-eb",
        str(args.eval_batch_size),
        "--seed",
        str(args.seed),
        "--video_log_freq",
        "-1",
    ]

    cases: list[tuple[str, list[str]]] = [
        ("case1_vanilla", []),
        ("case2_rtg", ["--use_reward_to_go"]),
        ("case3_rtg_baseline", ["--use_reward_to_go", "--use_baseline", "--normalize_advantages"]),
        (
            "case4_rtg_baseline_gae",
            [
                "--use_reward_to_go",
                "--use_baseline",
                "--normalize_advantages",
                "--gae_lambda",
                str(args.gae_lambda),
            ],
        ),
    ]

    run_dirs: list[tuple[str, Path]] = []
    for case_name, extra_args in cases:
        cmd = common + ["--exp_name", case_name] + extra_args
        run_cmd(cmd, project_root)
        prefix = f"{args.env_name}_{case_name}_sd{args.seed}_"
        run_dir = latest_run_dir(exp_root, prefix)
        run_dirs.append((case_name, run_dir))

    print("\nAll cases finished. Demo commands:")
    for case_name, run_dir in run_dirs:
        print(
            f"uv run python src/scripts/play_trained.py --run_dir {run_dir.as_posix()} --episodes {args.episodes}"
        )


if __name__ == "__main__":
    main()

