#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  rm -rf .venv
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN=python3.11
  else
    PYTHON_BIN=python3
  fi
  "$PYTHON_BIN" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124
.venv/bin/python -m pip install \
  ogbench \
  matplotlib \
  tqdm \
  opencv-python \
  ml_collections

echo "Environment ready."
echo "Expect OGBench data under ~/.ogbench/data on the remote host."
