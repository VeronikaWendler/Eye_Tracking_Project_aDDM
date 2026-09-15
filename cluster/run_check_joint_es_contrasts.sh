#!/bin/bash
#SBATCH --job-name=check_es_contr
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=check_es_contr_%j.out
#SBATCH --error=check_es_contr_%j.err
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
INPUT_HOST="${INPUT_HOST:-${OUT_DIR_HOST}/models/model_input_ES_joint_es_contrasts.csv}"
MODEL_DIR_HOST="${MODEL_DIR_HOST:-${OUT_DIR_HOST}/models_ES_joint_es_contrasts_preflight}"

mkdir -p "${MODEL_DIR_HOST}"

echo "========================================================================"
echo "JOINT E/S CONTRAST MODEL: DESIGN CHECK ONLY"
echo "========================================================================"
echo "INPUT_HOST=${INPUT_HOST}"
echo "MODEL_DIR_HOST=${MODEL_DIR_HOST}"
echo "========================================================================"

apptainer exec --cleanenv \
  --bind "${CODE_DIR}:/workspace" \
  --bind "${OUT_DIR_HOST}:/out" \
  --bind "${MODEL_DIR_HOST}:/target_models" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  "${IMAGE}" \
  python /workspace/addm_fit_joint_es_contrasts.py \
    --data /out/models/model_input_ES_joint_es_contrasts.csv \
    --model-dir /target_models \
    --design-only \
    --chains 1 \
    --jobs 1 \
    --samples 100 \
    --burn 20

echo "Finished design-matrix preflight."
