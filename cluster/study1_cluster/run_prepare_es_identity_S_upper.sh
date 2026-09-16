#!/bin/bash
#SBATCH --job-name=prep_es_id
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=prep_es_id_%j.out
#SBATCH --error=prep_es_id_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

DATA_ROOT="/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/data"
OUT_ROOT="/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives"

mkdir -p "${OUT_ROOT}/models"

apptainer exec --cleanenv \
  --bind "${CODE_DIR}:/workspace" \
  --bind "${DATA_ROOT}:/data:ro" \
  --bind "${OUT_ROOT}:/out" \
  "${IMAGE}" \
  python /workspace/addm_prepare_es_identity_S_upper.py \
    --data /data/Study1_Behaviour_with_Gaze_AnalysisReady.csv \
    --out /out/models/model_input_ES_identity_S_upper.csv \
    --audit /out/models/model_input_ES_identity_S_upper_audit.json

echo "Finished ES identity preparation."
