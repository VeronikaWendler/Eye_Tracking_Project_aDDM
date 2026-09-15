#!/bin/bash
#SBATCH --job-name=diag_es_id_joint
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=diag_es_id_joint_%j.out
#SBATCH --error=diag_es_id_joint_%j.err
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

OUT_ROOT="/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives"

MODEL_DIR_HOST="${MODEL_DIR_HOST:-${OUT_ROOT}/models_ES_identity_S_upper_joint_contrasts_final}"

DIAG_DIR_HOST="${DIAG_DIR_HOST:-${OUT_ROOT}/figures/aDDM_ES_identity_S_upper_joint_contrasts/diagnostics}"

CHAINS="${CHAINS:-3}"

mkdir -p "${DIAG_DIR_HOST}"

apptainer exec --cleanenv \
  --bind "${CODE_DIR}:/workspace" \
  --bind "${MODEL_DIR_HOST}:/target_models:ro" \
  --bind "${DIAG_DIR_HOST}:/diagnostics" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "${IMAGE}" \
  python /workspace/addm_diagnose_es_identity_S_upper_joint_contrasts.py \
    --model-dir /target_models \
    --out-dir /diagnostics \
    --chains "${CHAINS}"

echo "Finished S-upper identity joint-contrast diagnostics."
