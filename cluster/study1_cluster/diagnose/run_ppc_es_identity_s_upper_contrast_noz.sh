#!/bin/bash
#SBATCH --job-name=s1_ppc_cnoz
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives/logs/es_identity_s_upper_contrast_noz_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives/logs/es_identity_s_upper_contrast_noz_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "${MPLCONFIGDIR}"

IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
STUDY_ROOT="/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM"
MODEL_DIR_HOST="${STUDY_ROOT}/derivatives/models/es_identity_S_upper_contrast_noz_final"
DATA_DIR_HOST="${STUDY_ROOT}/data"
ANALYSISREADY_HOST="${DATA_DIR_HOST}/Study1_Behaviour_with_Gaze_AnalysisReady.csv"
PPC_DIR_HOST="${STUDY_ROOT}/derivatives/ppc/es_identity_s_upper_contrast_noz_main"
LOG_DIR_HOST="${STUDY_ROOT}/derivatives/logs"

mkdir -p "${PPC_DIR_HOST}" "${LOG_DIR_HOST}"

SCRIPT_HOST="${REPO_ROOT}/py_diagnose/addm_ppc_es_identity_s_upper_contrast_noz.py"

[[ -f "${SCRIPT_HOST}" ]] || {
  echo "ERROR: missing PPC script: ${SCRIPT_HOST}" >&2
  exit 1
}

[[ -f "${ANALYSISREADY_HOST}" ]] || {
  echo "ERROR: missing AnalysisReady file: ${ANALYSISREADY_HOST}" >&2
  exit 1
}

[[ -d "${MODEL_DIR_HOST}" ]] || {
  echo "ERROR: missing model directory: ${MODEL_DIR_HOST}" >&2
  exit 1
}

for c in 0 1 2; do
  mapfile -t MATCHES < <(find "${MODEL_DIR_HOST}" -maxdepth 1 -type f -name "*_${c}.hddm" | sort)
  if [[ "${#MATCHES[@]}" -ne 1 ]]; then
    echo "ERROR: expected exactly one *_${c}.hddm in ${MODEL_DIR_HOST}, found ${#MATCHES[@]}" >&2
    printf '  %s\n' "${MATCHES[@]:-<none>}" >&2
    exit 1
  fi
done

echo "========================================================================"
echo "STUDY1 S-UPPER CONTRAST NO-Z PPC FULL"
echo "========================================================================"
echo "REPO_ROOT=${REPO_ROOT}"
echo "MODEL_DIR_HOST=${MODEL_DIR_HOST}"
echo "ANALYSISREADY_HOST=${ANALYSISREADY_HOST}"
echo "PPC_DIR_HOST=${PPC_DIR_HOST}"
echo "PPC_TOTAL=1000"
echo "========================================================================"

apptainer exec --cleanenv \
  --bind "${REPO_ROOT}:/workspace" \
  --bind "${MODEL_DIR_HOST}:/target_models:ro" \
  --bind "${DATA_DIR_HOST}:/data:ro" \
  --bind "${PPC_DIR_HOST}:/ppc" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "${IMAGE}" \
  python "/workspace/py_diagnose/addm_ppc_es_identity_s_upper_contrast_noz.py" \
    --study "study1" \
    --model-dir /target_models \
    --output-dir /ppc \
    --chains 3 \
    --ppc-total 1000 \
    --bootstrap-samples 5000 \
    --bootstrap-ci 0.95 \
    --seed 20260920 \
    --subject-col subj_idx \
    --analysisready "/data/Study1_Behaviour_with_Gaze_AnalysisReady.csv" \
    --save-replot-data

echo "Finished study1 es_identity_s_upper_contrast_noz full PPC."
