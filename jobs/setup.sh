#!/usr/bin/env bash
# One-time setup on the GPU machine (cluster login node, or the GPU PC itself).
#
#   OLLAMA_MODELS=/path/on/big/disk SYSTEMS="mistral-small-24b" bash jobs/setup.sh
#
# 1. Creates .venv with the pipeline's requirements.
# 2. Installs Ollama into ~/.local/ollama if `ollama` is not already available (no root needed).
# 3. Downloads the models the job will need (the systems in SYSTEMS + the judges) into
#    OLLAMA_MODELS, so the job itself does not need internet access.
#
# Model weights are large (Mistral Large ~73 GB): put OLLAMA_MODELS on scratch or
# project storage, not a home directory with a small quota. Use the same OLLAMA_MODELS
# when submitting the job.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${OLLAMA_MODELS:?set OLLAMA_MODELS to a folder on a large disk}"
: "${SYSTEMS:?set SYSTEMS to the system names to run, e.g. SYSTEMS=\"mistral-small-24b\"}"
export OLLAMA_MODELS
mkdir -p "$OLLAMA_MODELS"

# 1. Python environment (Python 3.9 or newer; on a cluster, `module load python` first).
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt
fi
.venv/bin/python --version

# 2. Ollama, installed for this user only.
OLLAMA_BIN="${OLLAMA_BIN:-$(command -v ollama || echo "$HOME/.local/ollama/bin/ollama")}"
if [ ! -x "$OLLAMA_BIN" ]; then
  echo "Installing Ollama into ~/.local/ollama"
  mkdir -p "$HOME/.local/ollama"
  tmp=$(mktemp -d)
  if curl -fsSL -o "$tmp/ollama.tgz" https://ollama.com/download/ollama-linux-amd64.tgz; then
    tar -C "$HOME/.local/ollama" -xzf "$tmp/ollama.tgz"
  else
    curl -fsSL -o "$tmp/ollama.tar.zst" https://ollama.com/download/ollama-linux-amd64.tar.zst
    tar --use-compress-program=unzstd -C "$HOME/.local/ollama" -xf "$tmp/ollama.tar.zst"
  fi
  rm -rf "$tmp"
  OLLAMA_BIN="$HOME/.local/ollama/bin/ollama"
fi
echo "Ollama: $OLLAMA_BIN"

# 3. Models, through a temporary server on a free port (stopped on exit).
PORT=$(.venv/bin/python -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
export OLLAMA_HOST="127.0.0.1:$PORT"
"$OLLAMA_BIN" serve > "$OLLAMA_MODELS/setup-ollama.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 30); do curl -sf "http://$OLLAMA_HOST/api/version" > /dev/null && break; sleep 1; done

for model in $(.venv/bin/python -m pipeline models --systems $SYSTEMS); do
  echo "Pulling $model"
  "$OLLAMA_BIN" pull "$model"
done
"$OLLAMA_BIN" list
echo "Setup done. Submit with: sbatch --export=ALL,SYSTEMS=\"$SYSTEMS\",OLLAMA_MODELS=$OLLAMA_MODELS jobs/pipeline.sbatch"
