#!/bin/bash
#SBATCH --job-name=diag_all_reg
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=diag_all_reg_%j.out
#SBATCH --error=diag_all_reg_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"

IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

OUT_DIR_HOST="${OUT_DIR_HOST:-/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives}"
MODEL_DIR_HOST="${MODEL_DIR_HOST:-${OUT_DIR_HOST}/models_joint_all_random_slopes_final}"
DIAG_DIR_HOST="${DIAG_DIR_HOST:-${OUT_DIR_HOST}/figures/aDDM_JOINT_ALL_RANDOM_SLOPES/diagnostics}"
CHAINS="${CHAINS:-3}"

mkdir -p "${DIAG_DIR_HOST}"

echo "========================================================================"
echo "ALL-RANDOM-SLOPES JOINT aDDM DIAGNOSTICS"
echo "========================================================================"
echo "IMAGE=${IMAGE}"
echo "CODE_DIR=${CODE_DIR}"
echo "MODEL_DIR_HOST=${MODEL_DIR_HOST}"
echo "DIAG_DIR_HOST=${DIAG_DIR_HOST}"
echo "CHAINS=${CHAINS}"
echo "========================================================================"

apptainer exec --cleanenv \
  --bind "${CODE_DIR}:/workspace" \
  --bind "${MODEL_DIR_HOST}:/random_models:ro" \
  --bind "${DIAG_DIR_HOST}:/diagnostics" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "${IMAGE}" \
  python /workspace/addm_diagnose_joint_all_random_slopes.py \
    --model-dir /random_models \
    --out-dir /diagnostics \
    --chains "${CHAINS}"

echo "Finished all-random-slopes joint aDDM diagnostics."
