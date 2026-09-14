#!/bin/bash
#SBATCH --job-name=joint_addm_diag
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=logs/joint_diag_%j.out
#SBATCH --error=logs/joint_diag_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"

IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT_DIR_HOST="${OUT_DIR_HOST:-/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives}"
MODEL_DIR_HOST="${MODEL_DIR_HOST:-${OUT_DIR_HOST}/models_joint_phase_specific_final}"
DIAG_DIR_HOST="${DIAG_DIR_HOST:-${OUT_DIR_HOST}/figures/phase_specific_aDDM_JOINT/diagnostics}"
CHAINS="${CHAINS:-3}"

mkdir -p "${DIAG_DIR_HOST}"

apptainer exec --cleanenv \
  --bind "${CODE_DIR}:/workspace" \
  --bind "${MODEL_DIR_HOST}:/joint_models:ro" \
  --bind "${DIAG_DIR_HOST}:/diagnostics" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "${IMAGE}" \
  python /workspace/addm_diagnose_joint_phase_specific.py \
    --model-dir /joint_models \
    --out-dir /diagnostics \
    --chains "${CHAINS}"

echo "Finished joint phase-specific aDDM diagnostics."
