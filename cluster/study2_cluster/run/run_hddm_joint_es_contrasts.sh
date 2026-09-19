#!/bin/bash
#SBATCH --job-name=s2_es_contr
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=180G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/final/es_joint_es_contrasts_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/final/es_joint_es_contrasts_%j.err
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
MODEL_DIR_HOST="${OUT_ROOT}/models/es_joint_es_contrasts_final"

CHAINS="${CHAINS:-3}"
SAMPLES="${SAMPLES:-4000}"
BURN="${BURN:-1000}"

mkdir -p "${MODEL_DIR_HOST}" "${OUT_ROOT}/logs/final"

echo "========================================================================"
echo "STUDY 2 ES ACCURACY-CODED JOINT E/S CONTRAST aDDM"
echo "========================================================================"
echo "INPUT=${INPUT_HOST}"
echo "MODEL_DIR=${MODEL_DIR_HOST}"
echo "CHAINS=${CHAINS}"
echo "SAMPLES=${SAMPLES}"
echo "BURN=${BURN}"
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
    --samples "${SAMPLES}" \
    --burn "${BURN}" \
    --chains "${CHAINS}" \
    --jobs "${SLURM_CPUS_PER_TASK}"

echo "Finished Study 2 ES accuracy-coded joint E/S contrast aDDM."
