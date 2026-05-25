#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -x "${HOME}/.local/bin/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [[ ! -x ".venv/bin/python" ]]; then
  "${HOME}/.local/bin/uv" venv --python 3.11 --seed .venv
fi

.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch
.venv/bin/python -m pip install ogbench matplotlib opencv-python ml_collections wandb tqdm
echo "Environment ready. Expect OGBench datasets under ~/.ogbench/data on the remote host."
