#!/bin/bash
#SBATCH --job-name=s2_ppc_cnoz
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/s2_ppc_cnoz_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/s2_ppc_cnoz_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
DATA_ROOT="/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/data"
OUT_ROOT="/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives"
DATA_FILE="Study2_Behaviour_with_Gaze_AnalysisReady.csv"
MODEL_DIR="${OUT_ROOT}/models/es_identity_S_upper_contrast_noz_final"
PPC_DIR="${OUT_ROOT}/ppc/es_identity_s_upper_contrast_noz_main"

mkdir -p "$PPC_DIR" "${OUT_ROOT}/logs"

[[ -f "${DATA_ROOT}/${DATA_FILE}" ]] || {
  echo "ERROR: missing AnalysisReady file: ${DATA_ROOT}/${DATA_FILE}" >&2
  exit 1
}

for c in 0 1 2; do
  n=$(find "$MODEL_DIR" -maxdepth 1 -type f -name "*_${c}.hddm" | wc -l)
  if [[ "$n" -ne 1 ]]; then
    echo "ERROR: expected exactly one *_${c}.hddm in $MODEL_DIR, found $n" >&2
    exit 1
  fi
done

echo "======================================================================"
echo "STUDY2 s-upper contrast no-z PPC (main)"
echo "model=$MODEL_DIR"
echo "analysisready=${DATA_ROOT}/${DATA_FILE}"
echo "response coordinate: 1=S upper, 0=E lower"
echo "main dwell PPC: DwellTimeAdvantage_ES = Dwell_S - Dwell_E"
echo "ppc_total=1000"
echo "======================================================================"

apptainer exec --cleanenv \
  --bind "$REPO_ROOT:/workspace" \
  --bind "$DATA_ROOT:/data:ro" \
  --bind "$MODEL_DIR:/target_models:ro" \
  --bind "$PPC_DIR:/ppc_output" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  "$IMAGE" \
  python /workspace/py_diagnose/addm_ppc_es_identity_s_upper_contrast_noz.py \
    --study study2 \
    --model-dir /target_models \
    --analysisready "/data/${DATA_FILE}" \
    --output-dir /ppc_output \
    --chains 3 \
    --ppc-total 1000 \
    --bootstrap-samples 5000 \
    --bootstrap-ci 0.95 \
    --seed 20260920

echo "Finished study2 PPC main."
