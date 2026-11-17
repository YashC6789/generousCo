#!/usr/bin/env bash
# Run the pipeline on a GPU machine with its own Ollama server.
#
# Under Slurm: submitted by jobs/pipeline.sbatch. On a GPU PC without a scheduler:
#   SYSTEMS="mistral-small-24b" OLLAMA_MODELS=/path tmux new -d 'bash jobs/run.sh'
#
# Settings (environment variables):
#   SYSTEMS          system names from config/systems.yaml (required)
#   OLLAMA_MODELS    where jobs/setup.sh put the model weights (required)
#   FIXES            "all" (default), "none", or fix names, e.g. "disagree goal_reminder"
#   RUNS             runs per scenario (default: config)
#   OLLAMA_CONTEXT_LENGTH  context window in tokens (default 16384; 8192 for Mistral Large on one 80 GB GPU)
#   OLLAMA_BIN       the ollama binary (default: on PATH, else ~/.local/ollama/bin/ollama)
#
# Every step skips what is already saved, so if the job hits its time limit, submit it
# again and it carries on. Logs go to results/jobs/<job id>/.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${SYSTEMS:?set SYSTEMS, e.g. SYSTEMS=\"mistral-small-24b\"}"
: "${OLLAMA_MODELS:?set OLLAMA_MODELS to the folder jobs/setup.sh downloaded models into}"
FIXES="${FIXES:-all}"
OLLAMA_BIN="${OLLAMA_BIN:-$(command -v ollama || echo "$HOME/.local/ollama/bin/ollama")}"
PY=".venv/bin/python -m pipeline"

export OLLAMA_MODELS
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-16384}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
# A free port, so this job never talks to another user's Ollama on a shared node.
PORT=$(.venv/bin/python -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
export OLLAMA_HOST="127.0.0.1:$PORT"
export OLLAMA_URL="http://$OLLAMA_HOST"   # config/systems.yaml points local models here

LOG_DIR="results/jobs/${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$LOG_DIR"
{
  echo "host $(hostname)  systems: $SYSTEMS  fixes: $FIXES  context: $OLLAMA_CONTEXT_LENGTH"
  echo "git $(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo ' (uncommitted changes)')"
  "$OLLAMA_BIN" --version 2>&1 || true
  nvidia-smi 2>&1 || echo "WARNING: nvidia-smi failed; is this a GPU node?"
} > "$LOG_DIR/environment.txt"

"$OLLAMA_BIN" serve > "$LOG_DIR/ollama.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do curl -sf "$OLLAMA_URL/api/version" > /dev/null && break; sleep 2; done
curl -sf "$OLLAMA_URL/api/version" > /dev/null || { echo "Ollama did not start; see $LOG_DIR/ollama.log"; exit 1; }

# Pull anything missing (needs internet; otherwise run jobs/setup.sh first).
for model in $($PY models --systems $SYSTEMS); do
  "$OLLAMA_BIN" show "$model" > /dev/null 2>&1 || "$OLLAMA_BIN" pull "$model"
done

# One test reply per system: loads the model and shows it answers.
$PY ping --systems $SYSTEMS | tee "$LOG_DIR/ping.txt"
# PROCESSOR should say 100% GPU; any CPU share means the model did not fit and runs very slowly.
"$OLLAMA_BIN" ps | tee "$LOG_DIR/ollama-ps.txt"

# Judge checks on this machine: a new GPU or Ollama version can change judge outputs.
$PY sanity 2>&1 | tee "$LOG_DIR/sanity.txt" || echo "WARNING: judge sanity check failed; see $LOG_DIR/sanity.txt"
$PY pressure sanity 2>&1 | tee "$LOG_DIR/stance-sanity.txt" || echo "WARNING: stance judge sanity check failed"

args=(harness --systems $SYSTEMS)
case "$FIXES" in
  none) ;;
  all) args+=(--fixes) ;;
  *) args+=(--fixes $FIXES) ;;
esac
[ -n "${RUNS:-}" ] && args+=(--runs "$RUNS")
# A second pass retries any judge parse failures from the first.
$PY "${args[@]}" || $PY "${args[@]}"

$PY analyze
$PY pressure analyze
echo "Done. Copy results/ back and run: python -m pipeline merge <copy>" | tee -a "$LOG_DIR/environment.txt"
