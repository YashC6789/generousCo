#!/bin/bash
# Slurm job: the whole pipeline for SYSTEMS on one GPU. Submit from the repo root:
#
#   sbatch --export=ALL,SYSTEMS="mistral-small-24b",OLLAMA_MODELS=/path/to/models jobs/pipeline.sbatch
#
# Fill in the lines marked CHECK for your cluster (`sinfo` lists partitions and GPU types).
# Rough GPU memory needed at 4-bit with a 16k context: Mistral Small 24B ~17 GB,
# Mixtral 8x7B ~30 GB, Mistral Large 123B ~80 GB (use --gres=gpu:2 with 80 GB GPUs, or
# one 80 GB GPU with OLLAMA_CONTEXT_LENGTH=8192).
#
#SBATCH --job-name=sycophancy
#SBATCH --gres=gpu:1                 # CHECK: e.g. gpu:a100:1, gpu:h100:1, or gpu:2 for Mistral Large
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                    # CHECK: system RAM; at least the model file size helps loading
#SBATCH --time=12:00:00              # CHECK: the partition's limit; resubmitting carries on where it stopped
#SBATCH --output=slurm-%j.out
##SBATCH --partition=CHECK           # remove one '#' to use
##SBATCH --account=CHECK             # remove one '#' to use

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"

# CHECK: load a Python 3.9+ module if the cluster's default python3 is older, e.g.
# module load python/3.11

exec bash jobs/run.sh
