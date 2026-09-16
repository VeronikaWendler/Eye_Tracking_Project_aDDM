#!/bin/bash
#SBATCH --job-name=joint_phase_addm
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=180G
#SBATCH --output=logs/joint_phase_addm_%j.out
#SBATCH --error=logs/joint_phase_addm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail

# Submit from the repository root.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"

IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

# Existing successful separate-fit outputs live here.
OUT_DIR_HOST="${OUT_DIR_HOST:-/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives}"

# Keep this new model separate from the existing ES/EE models.
JOINT_MODEL_DIR_HOST="${JOINT_MODEL_DIR_HOST:-${OUT_DIR_HOST}/models_joint_phase_specific}"

CHAINS="${CHAINS:-3}"
SAMPLES="${SAMPLES:-2000}"
BURN="${BURN:-500}"

mkdir -p "${JOINT_MODEL_DIR_HOST}"

echo "========================================================================"
echo "JOINT PHASE-SPECIFIC aDDM"
echo "========================================================================"
echo "IMAGE=${IMAGE}"
echo "CODE_DIR=${CODE_DIR}"
echo "OUT_DIR_HOST=${OUT_DIR_HOST}"
echo "JOINT_MODEL_DIR_HOST=${JOINT_MODEL_DIR_HOST}"
echo "CHAINS=${CHAINS}"
echo "SAMPLES=${SAMPLES}"
echo "BURN=${BURN}"
echo "========================================================================"

apptainer exec --cleanenv \
  --bind "${CODE_DIR}:/workspace" \
  --bind "${OUT_DIR_HOST}:/out" \
  --bind "${JOINT_MODEL_DIR_HOST}:/joint_models" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "${IMAGE}" \
  python /workspace/addm_fit_joint_phase_specific.py \
    --es-input /out/models/basic_aDDM_ES_MODEL_INPUT.csv \
    --ee-input /out/models/basic_aDDM_EE_MODEL_INPUT.csv \
    --model-dir /joint_models \
    --samples "${SAMPLES}" \
    --burn "${BURN}" \
    --chains "${CHAINS}" \
    --jobs "${SLURM_CPUS_PER_TASK}"

echo "Finished joint phase-specific aDDM."
