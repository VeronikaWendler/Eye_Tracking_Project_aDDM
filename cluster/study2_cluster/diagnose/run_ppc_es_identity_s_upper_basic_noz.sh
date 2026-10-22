#!/bin/bash
#SBATCH --job-name=s2_ppc_bn
#SBATCH --time=08:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/es_identity_s_upper_basic_noz_ppc_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/es_identity_s_upper_basic_noz_ppc_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"

IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
STUDY_ROOT="/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM"
MODEL_DIR_HOST="$STUDY_ROOT/derivatives/models/es_identity_S_upper_basic_noz_final"
DATA_DIR_HOST="$STUDY_ROOT/data"
ANALYSISREADY_HOST="$DATA_DIR_HOST/Study2_Behaviour_with_Gaze_AnalysisReady.csv"
PPC_DIR_HOST="$STUDY_ROOT/derivatives/ppc/es_identity_s_upper_basic_noz_main"
LOG_DIR_HOST="$STUDY_ROOT/derivatives/logs"

SCRIPT_HOST="$REPO_ROOT/py_diagnose/addm_ppc_es_identity_s_upper_basic_noz.py"
REPLOT_HOST="$REPO_ROOT/py_diagnose/replot_ppc_talk_ready.py"

mkdir -p "$LOG_DIR_HOST" "$PPC_DIR_HOST/figures/talk_ready"

for f in "$IMAGE" "$SCRIPT_HOST" "$REPLOT_HOST" "$ANALYSISREADY_HOST"; do
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 1; }
done

[[ -d "$MODEL_DIR_HOST" ]] || { echo "MISSING model directory: $MODEL_DIR_HOST" >&2; exit 1; }

for c in 0 1 2; do
  f="$MODEL_DIR_HOST/aDDM_ES_IDENTITY_S_UPPER_BASIC_NOZ_NO_INTERCEPT_${c}.hddm"
  [[ -f "$f" ]] || { echo "MISSING expected chain: $f" >&2; exit 1; }
done

echo "========================================================================"
echo "STUDY 2 BASIC aDDM NO-Z PPC"
echo "========================================================================"
echo "REPO_ROOT=$REPO_ROOT"
echo "MODEL_DIR_HOST=$MODEL_DIR_HOST"
echo "ANALYSISREADY_HOST=$ANALYSISREADY_HOST"
echo "PPC_DIR_HOST=$PPC_DIR_HOST"
echo "PPC_TOTAL=1000"
echo "========================================================================"

apptainer exec --cleanenv \
  --bind "$REPO_ROOT:/workspace" \
  --bind "$MODEL_DIR_HOST:/target_models:ro" \
  --bind "$DATA_DIR_HOST:/data:ro" \
  --bind "$PPC_DIR_HOST:/ppc" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "$IMAGE" \
  python "/workspace/py_diagnose/addm_ppc_es_identity_s_upper_basic_noz.py" \
    --study study2 \
    --model-dir /target_models \
    --output-dir /ppc \
    --chains 3 \
    --ppc-total 1000 \
    --bootstrap-samples 5000 \
    --bootstrap-ci 0.95 \
    --seed 20260920 \
    --subject-col subj_idx \
    --analysisready /data/Study2_Behaviour_with_Gaze_AnalysisReady.csv \
    --save-replot-data

echo "Making Study 2 talk-ready PPC figures for basic_noz"

apptainer exec --cleanenv \
  --bind "$REPO_ROOT:/workspace" \
  --bind "$PPC_DIR_HOST:/ppc" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "$IMAGE" \
  python /workspace/py_diagnose/replot_ppc_talk_ready.py \
    --study study2 \
    --ppc-dir /ppc \
    --out-dir /ppc/figures/talk_ready

echo "Completed: $PPC_DIR_HOST"
