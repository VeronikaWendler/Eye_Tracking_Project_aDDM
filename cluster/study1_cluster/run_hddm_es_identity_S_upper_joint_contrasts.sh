#!/bin/bash
#SBATCH --job-name=addm_es_id_joint
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=180G
#SBATCH --output=logs/addm_es_id_joint_%j.out
#SBATCH --error=logs/addm_es_id_joint_%j.err
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

OUT_ROOT="/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives"

# IMPORTANT: exactly the same S-upper identity input used by the
# option-specific-theta model, including equal-value trials.
INPUT_HOST="${INPUT_HOST:-${OUT_ROOT}/models/model_input_ES_identity_S_upper.csv}"

MODEL_DIR_HOST="${MODEL_DIR_HOST:-${OUT_ROOT}/models_ES_identity_S_upper_joint_contrasts_final}"

CHAINS="${CHAINS:-3}"
SAMPLES="${SAMPLES:-4000}"
BURN="${BURN:-1000}"

mkdir -p "${MODEL_DIR_HOST}"

echo "========================================================================"
echo "ES S-UPPER IDENTITY JOINT E/S CONTRAST aDDM"
echo "S = UPPER BOUNDARY / E = LOWER BOUNDARY"
echo "ATTENDED + UNATTENDED E/S DIFFERENCES FREE"
echo "TIES RETAINED"
echo "========================================================================"
echo "IMAGE=${IMAGE}"
echo "CODE_DIR=${CODE_DIR}"
echo "INPUT_HOST=${INPUT_HOST}"
echo "MODEL_DIR_HOST=${MODEL_DIR_HOST}"
echo "CHAINS=${CHAINS}"
echo "SAMPLES=${SAMPLES}"
echo "BURN=${BURN}"
echo "========================================================================"

apptainer exec --cleanenv \
  --bind "${CODE_DIR}:/workspace" \
  --bind "${OUT_ROOT}:/out" \
  --bind "${MODEL_DIR_HOST}:/target_models" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "${IMAGE}" \
  python /workspace/addm_fit_es_identity_S_upper_joint_contrasts.py \
    --data /out/models/model_input_ES_identity_S_upper.csv \
    --model-dir /target_models \
    --samples "${SAMPLES}" \
    --burn "${BURN}" \
    --chains "${CHAINS}" \
    --jobs "${SLURM_CPUS_PER_TASK}"

echo "Finished S-upper identity joint E/S contrast aDDM."
