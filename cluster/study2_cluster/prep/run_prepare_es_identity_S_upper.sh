#!/bin/bash
#SBATCH --job-name=s2_prep_es_id
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/prep/es_identity_S_upper_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/prep/es_identity_S_upper_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"

IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"

DATA_ROOT="/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/data"
OUT_ROOT="/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives"
DATA_FILE="Study2_Behaviour_with_Gaze_AnalysisReady.csv"

INPUT_HOST="${DATA_ROOT}/${DATA_FILE}"
PREP_DIR_HOST="${OUT_ROOT}/prepared_data"
OUT_HOST="${PREP_DIR_HOST}/model_input_ES_identity_S_upper.csv"
AUDIT_HOST="${PREP_DIR_HOST}/model_input_ES_identity_S_upper_audit.json"

mkdir -p "${PREP_DIR_HOST}" "${OUT_ROOT}/logs/prep"

echo "========================================================================"
echo "STUDY 2 PREP: ES IDENTITY COORDINATE, S=UPPER / E=LOWER"
echo "========================================================================"
echo "REPO_ROOT=${REPO_ROOT}"
echo "DATA=${INPUT_HOST}"
echo "OUTPUT=${OUT_HOST}"
echo "AUDIT=${AUDIT_HOST}"
echo "========================================================================"

[[ -f "${REPO_ROOT}/py_prep/study2_prep/addm_prepare_es_identity_S_upper.py" ]] || {
  echo "ERROR: missing prep script." >&2
  exit 1
}

[[ -f "${INPUT_HOST}" ]] || {
  echo "ERROR: missing Study 2 AnalysisReady data: ${INPUT_HOST}" >&2
  exit 1
}

apptainer exec --cleanenv \
  --bind "${REPO_ROOT}:/workspace" \
  --bind "${DATA_ROOT}:/data:ro" \
  --bind "${OUT_ROOT}:/out" \
  "${IMAGE}" \
  python /workspace/py_prep/study2_prep/addm_prepare_es_identity_S_upper.py \
    --data "/data/${DATA_FILE}" \
    --out /out/prepared_data/model_input_ES_identity_S_upper.csv \
    --audit /out/prepared_data/model_input_ES_identity_S_upper_audit.json

echo "Finished Study 2 S-upper identity preparation."
