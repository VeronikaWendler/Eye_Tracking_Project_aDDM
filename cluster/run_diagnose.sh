#!/bin/bash
#SBATCH --job-name=addm_diag
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=logs/diagnose_%j.out
#SBATCH --error=logs/diagnose_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail

# Submit from repository root.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"

# ---------------------------------------------------------------------
# Repository / output locations
# ---------------------------------------------------------------------

IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT_DIR_HOST="${OUT_DIR_HOST:-/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives}"

# Override with PHASE=ES or PHASE=EE when submitting.
PHASE="${PHASE:-ES}"

CHAINS="${CHAINS:-3}"
PPC_SAMPLES="${PPC_SAMPLES:-50}"

echo "========================================"
echo "aDDM DIAGNOSTICS"
echo "PHASE=${PHASE}"
echo "CHAINS=${CHAINS}"
echo "PPC_SAMPLES=${PPC_SAMPLES}"
echo "CODE_DIR=${CODE_DIR}"
echo "OUT=${OUT_DIR_HOST}"
echo "========================================"

apptainer exec --cleanenv \
  --bind "${CODE_DIR}:/workspace" \
  --bind "${OUT_DIR_HOST}:/out" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "${IMAGE}" \
  python /workspace/addm_diagnose.py \
    --phase "${PHASE}" \
    --model-dir /out/models \
    --fig-dir /out/figures \
    --chains "${CHAINS}" \
    --ppc-samples "${PPC_SAMPLES}"

echo "Finished diagnostics for ${PHASE}"