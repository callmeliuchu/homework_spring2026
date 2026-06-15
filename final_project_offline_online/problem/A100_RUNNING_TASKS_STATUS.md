# A100 Running Tasks Status

Generated: 2026-06-12 18:11:25 CST / 2026-06-12 10:11:26 UTC

Remote project directory:

```bash
/mnt1/mnt1/nlp/lc/final_project_offline_online/problem
```

## Summary

There are currently three offline-to-online training sweeps running on A100:

| Algorithm | Run group | Current first combo | GPU visible id | Status | Best eval so far |
|---|---|---:|---:|---|---:|
| FQL | `fp_sweep_fql_cube_official` | `alpha=100, flow_steps=10` | `CUDA_VISIBLE_DEVICES=1` | Running | `1.00 @ 350k` |
| DSRL | `fp_sweep_dsrl_stable` | `noise_scale=0.5, fixed_alpha=0.01` | `CUDA_VISIBLE_DEVICES=2` | Running online phase / checkpoint exists | `1.00 @ 400k` |
| QSM | `fp_sweep_qsm_minimal_linear` | `alpha=0.01, inv_temp=50, flow_steps=10` | `CUDA_VISIBLE_DEVICES=2` | Running after latest fix | `0.00 @ 0` for new run |

Environment for all listed runs:

```bash
ENV_NAME=cube-single-play-singletask-task1-v0
SEED=0
OFFLINE_TRAINING_STEPS=500000
ONLINE_TRAINING_STEPS=100000
LOG_INTERVAL=5000
EVAL_INTERVAL=50000
NUM_EVAL_TRAJECTORIES=25
WANDB_MODE=disabled
```

The training script runs offline first, saves `agent.pt` after offline training, then runs online training using that checkpoint.

## Active Processes

### DSRL

Process:

```bash
maintain 491284 1      bash ./remote_sweep_dsrl_noise.sh
maintain 491288 491284 ../.venv/bin/python src/scripts/train_offline_online.py --run_group=fp_sweep_dsrl_stable --base_config=dsrl --env_name=cube-single-play-singletask-task1-v0 --seed=0 --offline_training_steps=500000 --online_training_steps=100000 --log_interval=5000 --eval_interval=50000 --num_eval_trajectories=25 --replay_buffer_capacity=1000000 --offline_data=0 --wsrl_steps=0 --noise_scale=0.5 --bc_pretrain_steps=100000 --fixed_alpha=0.01
```

What it is doing:

DSRL trains a behavior flow policy first, then optimizes a latent noise policy on top of the learned flow. This run uses a stable configuration with `noise_scale=0.5`, `bc_pretrain_steps=100000`, and fixed entropy temperature `fixed_alpha=0.01`.

Sweep script:

```bash
./remote_sweep_dsrl_noise.sh
```

Sweep queue:

```bash
run_dsrl 0.5 0.01
run_dsrl 0.8 0.01
run_dsrl 0.5 0.05
run_dsrl 0.8 0.05
```

Current result:

```text
run: sd0_20260612_054329_dsrl_cube-single-play-singletask-task1-v0_n0.5_bp100000_fa0.01_online_offline
evals: 11
best: 1.00 @ 400000
latest eval: 0.84 @ 500000
latest train step: 510000
agent.pt: exists
```

Artifacts:

```bash
exp/fp_sweep_dsrl_stable/sd0_20260612_054329_dsrl_cube-single-play-singletask-task1-v0_n0.5_bp100000_fa0.01_online_offline/agent.pt
visualizations/dsrl_cube_best.gif
logs/fp_sweep_dsrl_stable_gpu2.log
```

Visualization result:

The generated DSRL GIF selected a successful rollout:

```text
selected rollout: success=1.0, return=-48.00, length=49
```

Local synced copy:

```bash
/Users/liuchu/codes/homework_spring2026/final_project_offline_online/problem/visualizations/dsrl_cube_best.gif
```

### FQL

Process:

```bash
maintain 3306087 1       bash ./remote_sweep_fql_alpha.sh
maintain 3306096 3306087 ../.venv/bin/python src/scripts/train_offline_online.py --run_group=fp_sweep_fql_cube_official --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=0 --offline_training_steps=500000 --online_training_steps=100000 --log_interval=5000 --eval_interval=50000 --num_eval_trajectories=25 --replay_buffer_capacity=1000000 --offline_data=0 --wsrl_steps=0 --alpha=100 --flow_steps=10
```

What it is doing:

FQL trains a BC flow policy and a one-step policy. The one-step policy is optimized with a Q objective plus distillation to the BC flow. This run is the first alpha sweep combo, `alpha=100, flow_steps=10`, which has already reached high success.

Sweep script:

```bash
./remote_sweep_fql_alpha.sh
```

Sweep queue:

```bash
run_fql 100 10
run_fql 300 10
run_fql 1000 10
```

Current result:

```text
run: sd0_20260612_062348_fql_cube-single-play-singletask-task1-v0_a100.0_f10_online_offline
evals: 10
best: 1.00 @ 350000
latest eval: 0.96 @ 450000
latest train step: 470000
agent.pt: not yet present
```

Artifacts:

```bash
logs/fp_sweep_fql_cube_official_gpu1.log
exp/fp_sweep_fql_cube_official/sd0_20260612_062348_fql_cube-single-play-singletask-task1-v0_a100.0_f10_online_offline/eval.csv
exp/fp_sweep_fql_cube_official/sd0_20260612_062348_fql_cube-single-play-singletask-task1-v0_a100.0_f10_online_offline/train.csv
```

Visualization status:

No FQL GIF yet because `agent.pt` has not appeared. A local watcher is waiting for the checkpoint and will generate/sync:

```bash
visualizations/fql_cube_best.gif
```

Local watcher log:

```bash
/Users/liuchu/codes/homework_spring2026/final_project_offline_online/problem/logs/wait_sync_fql_only.log
```

### QSM

Process:

```bash
maintain 1766705 1       bash ./remote_sweep_qsm_alpha_temp.sh
maintain 1766712 1766705 ../.venv/bin/python src/scripts/train_offline_online.py --run_group=fp_sweep_qsm_minimal_linear --base_config=qsm --env_name=cube-single-play-singletask-task1-v0 --seed=0 --offline_training_steps=500000 --online_training_steps=100000 --log_interval=5000 --eval_interval=50000 --num_eval_trajectories=25 --replay_buffer_capacity=1000000 --offline_data=0 --wsrl_steps=0 --alpha=0.01 --inv_temp=50 --flow_steps=10
```

What it is doing:

QSM is the DDPM-style policy. The current implementation intentionally keeps the simple structure:

```text
VectorFieldPolicy
linear normalized time t / flow_steps
linear beta schedule
```

The latest fix added the two core missing pieces:

```text
1. alpha * DDPM BC denoising loss
2. Q-gradient from target_critic.mean instead of online critic.mean
```

Current actor objective:

```python
q = target_critic(s, a_t).mean(dim=0)
q_grad = grad(q.sum(), a_t)
qsm_loss = ((eps_pred + inv_temp * q_grad) ** 2).mean()
bc_loss = ((noise - eps_pred) ** 2).mean()
loss = qsm_loss + alpha * bc_loss
```

Sweep script:

```bash
./remote_sweep_qsm_alpha_temp.sh
```

Current sweep queue:

```bash
run_qsm 0.01 50 10
run_qsm 0.03 50 10
run_qsm 0.1 50 10
run_qsm 0.03 100 10
```

Current new-run result:

```text
run: sd0_20260612_095936_qsm_cube-single-play-singletask-task1-v0_a0.01_i50.0_f10_online_offline
evals: 1
best: 0.00 @ 0
latest eval: 0.00 @ 0
latest train step: 25000
agent.pt: not yet present
```

Previous QSM attempts under the same run group:

```text
sd0_20260612_092252_qsm_cube-single-play-singletask-task1-v0_i50.0_f10_online_offline:
  best 0.00 @ 0, latest 0.00 @ 50000, latest train step 95000

sd0_20260612_092112_qsm_cube-single-play-singletask-task1-v0_a0.0_i50.0_f10_online_offline:
  best 0.00 @ 0, latest 0.00 @ 0

sd0_20260612_091531_qsm_cube-single-play-singletask-task1-v0_a0.0_i50.0_f10_tan0.1_bslinear_online_offline:
  best 0.00 @ 0, latest 0.00 @ 0
```

Artifacts:

```bash
logs/fp_sweep_qsm_minimal_linear_gpu2.log
exp/fp_sweep_qsm_minimal_linear/sd0_20260612_095936_qsm_cube-single-play-singletask-task1-v0_a0.01_i50.0_f10_online_offline/eval.csv
exp/fp_sweep_qsm_minimal_linear/sd0_20260612_095936_qsm_cube-single-play-singletask-task1-v0_a0.01_i50.0_f10_online_offline/train.csv
```

Next important checkpoint:

The first meaningful QSM eval for the new fixed version is at `50000` steps. Current latest logged train step is `25000`, so this run has not yet reached the first useful eval.

## Current Remote Visualizations

```bash
visualizations/sacbc_cube_best.gif
visualizations/ifql_cube_completed.gif
visualizations/dsrl_cube_best.gif
```

## Useful Commands

Check active processes:

```bash
ssh a100 'cd /mnt1/mnt1/nlp/lc/final_project_offline_online/problem && ps -ef | grep -E "train_offline_online.py|remote_sweep" | grep -v grep || true'
```

Tail QSM log:

```bash
ssh a100 'tail -f /mnt1/mnt1/nlp/lc/final_project_offline_online/problem/logs/fp_sweep_qsm_minimal_linear_gpu2.log'
```

Tail FQL log:

```bash
ssh a100 'tail -f /mnt1/mnt1/nlp/lc/final_project_offline_online/problem/logs/fp_sweep_fql_cube_official_gpu1.log'
```

Tail DSRL log:

```bash
ssh a100 'tail -f /mnt1/mnt1/nlp/lc/final_project_offline_online/problem/logs/fp_sweep_dsrl_stable_gpu2.log'
```

Summarize eval CSVs:

```bash
ssh a100 'cd /mnt1/mnt1/nlp/lc/final_project_offline_online/problem && ../.venv/bin/python - <<PY
from pathlib import Path
import csv
for group in ["fp_sweep_fql_cube_official", "fp_sweep_dsrl_stable", "fp_sweep_qsm_minimal_linear"]:
    print("GROUP", group)
    for p in sorted((Path("exp") / group).glob("*/eval.csv"), key=lambda x: x.stat().st_mtime, reverse=True)[:6]:
        rows = []
        with p.open() as f:
            for row in csv.DictReader(f):
                if row.get("eval/success_rate"):
                    rows.append((int(float(row["step"])), float(row["eval/success_rate"])))
        if rows:
            best = max(rows, key=lambda x: x[1])
            final = rows[-1]
            print(p.parent.name, "best", best, "final", final)
PY'
```
