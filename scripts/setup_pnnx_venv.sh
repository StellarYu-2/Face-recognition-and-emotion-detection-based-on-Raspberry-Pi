#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${HOME}/.venvs/pnnx"

echo "[1/4] Installing venv prerequisites..."
sudo apt install -y python3-venv python3-pip python3-full

echo "[2/4] Creating virtual environment at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"

echo "[3/4] Installing pnnx in virtual environment"
"${VENV_DIR}/bin/python" -m pip install -U pip
"${VENV_DIR}/bin/pip" install -U pnnx

echo "[4/4] Verifying pnnx"
"${VENV_DIR}/bin/pnnx" --help >/dev/null
echo "pnnx is ready at: ${VENV_DIR}/bin/pnnx"

