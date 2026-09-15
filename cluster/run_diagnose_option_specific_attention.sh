#!/bin/bash
#SBATCH --job-name=diag_att_ES
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=diag_att_ES_%j.out
#SBATCH --error=diag_att_ES_%j.err
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

MODEL_DIR_HOST="${MODEL_DIR_HOST:-${OUT_DIR_HOST}/models_ES_option_specific_attention_final}"

DIAG_DIR_HOST="${DIAG_DIR_HOST:-${OUT_DIR_HOST}/figures/aDDM_ES_option_specific_attention/diagnostics}"

CHAINS="${CHAINS:-3}"

mkdir -p "${DIAG_DIR_HOST}"

echo "========================================================================"
echo "ES OPTION-SPECIFIC ATTENTION DIAGNOSTICS"
echo "========================================================================"
echo "IMAGE=${IMAGE}"
echo "CODE_DIR=${CODE_DIR}"
echo "MODEL_DIR_HOST=${MODEL_DIR_HOST}"
echo "DIAG_DIR_HOST=${DIAG_DIR_HOST}"
echo "CHAINS=${CHAINS}"
echo ""
echo "The model directory is mounted at /target_models because the saved"
echo "HDDM objects remember their original database path there."
echo "========================================================================"

apptainer exec --cleanenv \
  --bind "${CODE_DIR}:/workspace" \
  --bind "${MODEL_DIR_HOST}:/target_models:ro" \
  --bind "${DIAG_DIR_HOST}:/diagnostics" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "${IMAGE}" \
  python /workspace/addm_diagnose_option_specific_attention.py \
    --model-dir /target_models \
    --out-dir /diagnostics \
    --chains "${CHAINS}"

echo "Finished ES option-specific-attention diagnostics."
