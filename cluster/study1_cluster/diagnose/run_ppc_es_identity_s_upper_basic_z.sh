#!/bin/bash
#SBATCH --job-name=s1_ppc_bz
#SBATCH --time=08:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives/logs/es_identity_s_upper_basic_z_ppc_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives/logs/es_identity_s_upper_basic_z_ppc_%j.err
#SBATCH --mail-type=FAIL,END

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
[[ -d "$REPO_ROOT/py_diagnose/study1_diagnose" && -d "$REPO_ROOT/cluster/study1_cluster/run" ]] || { echo "Submit from Eye_Tracking_Project_aDDM repository root; got $REPO_ROOT" >&2; exit 1; }
IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
STUDY_ROOT=/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM
MODEL_DIR_HOST="$STUDY_ROOT/derivatives/models/es_identity_S_upper_basic_z_final"
DATA_DIR_HOST="$STUDY_ROOT/data"
PPC_DIR_HOST="$STUDY_ROOT/derivatives/ppc/es_identity_s_upper_basic_z_main"
SCRIPT_HOST="$REPO_ROOT/py_diagnose/study1_diagnose/addm_ppc_es_identity_s_upper_basic_z.py"
REPLOT_HOST="$REPO_ROOT/py_diagnose/study1_diagnose/replot_ppc_talk_ready.py"
LOG_DIR_HOST="$STUDY_ROOT/derivatives/logs"
mkdir -p "$LOG_DIR_HOST" "$PPC_DIR_HOST/figures/talk_ready"
for f in "$IMAGE" "$SCRIPT_HOST" "$REPLOT_HOST" "$DATA_DIR_HOST/Study1_Behaviour_with_Gaze_AnalysisReady.csv"; do
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 1; }
done
[[ -d "$MODEL_DIR_HOST" ]] || { echo "MISSING model directory: $MODEL_DIR_HOST" >&2; exit 1; }
for c in 0 1 2; do
  f="$MODEL_DIR_HOST/aDDM_ES_IDENTITY_S_UPPER_BASIC_Z_NO_INTERCEPT_${c}.hddm"
  [[ -f "$f" ]] || { echo "MISSING expected chain: $f" >&2; exit 1; }
done

echo "Study1 basic_z: PPC (1000 datasets / 3 chains)"
apptainer exec --cleanenv \
  --bind "$REPO_ROOT:/workspace" \
  --bind "$MODEL_DIR_HOST:/target_models:ro" \
  --bind "$DATA_DIR_HOST:/data:ro" \
  --bind "$PPC_DIR_HOST:/ppc" \
  --env PYTHONUNBUFFERED=1 --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg --env MPLCONFIGDIR=/tmp/mplcache \
  "$IMAGE" \
  python "/workspace/py_diagnose/study1_diagnose/addm_ppc_es_identity_s_upper_basic_z.py" \
    --study study1 \
    --model-dir /target_models \
    --output-dir /ppc \
    --chains 3 --ppc-total 1000 \
    --bootstrap-samples 5000 --bootstrap-ci 0.95 \
    --seed 20260920 --subject-col subj_idx \
    --analysisready /data/Study1_Behaviour_with_Gaze_AnalysisReady.csv \
    --save-replot-data

echo "Making talk-ready PDF + PNG figures for basic_z"
apptainer exec --cleanenv \
  --bind "$REPO_ROOT:/workspace" \
  --bind "$PPC_DIR_HOST:/ppc" \
  --env PYTHONUNBUFFERED=1 --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg --env MPLCONFIGDIR=/tmp/mplcache \
  "$IMAGE" \
  python /workspace/py_diagnose/study1_diagnose/replot_ppc_talk_ready.py \
    --study study1 --ppc-dir /ppc --out-dir /ppc/figures/talk_ready

echo "Completed: $PPC_DIR_HOST"
