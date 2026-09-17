#!/bin/bash
#SBATCH --job-name=s2_check_es_contr
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/preflight/es_joint_es_contrasts_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/preflight/es_joint_es_contrasts_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"

IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"

OUT_ROOT="/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives"
INPUT_HOST="${OUT_ROOT}/prepared_data/model_input_ES_joint_es_contrasts.csv"
MODEL_DIR_HOST="${OUT_ROOT}/models/es_joint_es_contrasts_preflight"

mkdir -p "${MODEL_DIR_HOST}" "${OUT_ROOT}/logs/preflight"

echo "========================================================================"
echo "STUDY 2 ACCURACY-CODED JOINT E/S CONTRAST: DESIGN CHECK ONLY"
echo "========================================================================"
echo "INPUT=${INPUT_HOST}"
echo "PRECHECK_OUTPUT=${MODEL_DIR_HOST}"
echo "========================================================================"

[[ -f "${REPO_ROOT}/py_fit/study2_fit/addm_fit_joint_es_contrasts.py" ]] || {
  echo "ERROR: missing fit script." >&2
  exit 1
}

[[ -f "${INPUT_HOST}" ]] || {
  echo "ERROR: missing prepared joint-contrast input: ${INPUT_HOST}" >&2
  exit 1
}

apptainer exec --cleanenv \
  --bind "${REPO_ROOT}:/workspace" \
  --bind "${OUT_ROOT}:/out" \
  --bind "${MODEL_DIR_HOST}:/target_models" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "${IMAGE}" \
  python /workspace/py_fit/study2_fit/addm_fit_joint_es_contrasts.py \
    --data /out/prepared_data/model_input_ES_joint_es_contrasts.csv \
    --model-dir /target_models \
    --design-only \
    --chains 1 \
    --jobs 1 \
    --samples 100 \
    --burn 20

echo "Finished Study 2 accuracy-coded joint E/S contrast design check."
