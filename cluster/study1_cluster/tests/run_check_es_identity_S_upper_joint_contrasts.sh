#!/bin/bash
#SBATCH --job-name=check_es_id_joint
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=check_es_id_joint_%j.out
#SBATCH --error=check_es_id_joint_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

OUT_ROOT="/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives"
INPUT_HOST="${INPUT_HOST:-${OUT_ROOT}/models/model_input_ES_identity_S_upper.csv}"
MODEL_DIR_HOST="${MODEL_DIR_HOST:-${OUT_ROOT}/models_ES_identity_S_upper_joint_contrasts_preflight}"

mkdir -p "${MODEL_DIR_HOST}"

echo "========================================================================"
echo "S-UPPER IDENTITY JOINT E/S CONTRAST MODEL: DESIGN CHECK ONLY"
echo "TIES RETAINED; REUSES EXACT S-UPPER IDENTITY INPUT"
echo "========================================================================"
echo "INPUT_HOST=${INPUT_HOST}"
echo "MODEL_DIR_HOST=${MODEL_DIR_HOST}"
echo "========================================================================"

apptainer exec --cleanenv \
  --bind "${CODE_DIR}:/workspace" \
  --bind "${OUT_ROOT}:/out" \
  --bind "${MODEL_DIR_HOST}:/target_models" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  "${IMAGE}" \
  python /workspace/addm_fit_es_identity_S_upper_joint_contrasts.py \
    --data /out/models/model_input_ES_identity_S_upper.csv \
    --model-dir /target_models \
    --design-only \
    --chains 1 \
    --jobs 1 \
    --samples 100 \
    --burn 20

echo "Finished S-upper identity joint-contrast design check."
