#!/bin/bash
#SBATCH --job-name=s1_basic_z
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=180G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives/logs/final/es_identity_S_upper_basic_z_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives/logs/final/es_identity_S_upper_basic_z_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"
IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
DATA_ROOT="/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/data"
OUT_ROOT="/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives"
DATA_FILE="Study1_Behaviour_with_Gaze_AnalysisReady.csv"
INPUT_HOST="$OUT_ROOT/models/model_input_ES_identity_S_upper.csv"
SOURCE_HOST="$DATA_ROOT/$DATA_FILE"
MODEL_DIR_HOST="$OUT_ROOT/models/es_identity_S_upper_basic_z_final"
CHAINS="${CHAINS:-3}"
SAMPLES="${SAMPLES:-4000}"
BURN="${BURN:-1000}"
mkdir -p "$MODEL_DIR_HOST" "$OUT_ROOT/logs/final"

echo "============================================================================"
echo "STUDY 1 ES S-UPPER / E-LOWER BASIC aDDM FULL FIT"
echo "DRIFT INTERCEPT: NONE | Z ESTIMATED"
echo "============================================================================"
echo "INPUT=$INPUT_HOST"
echo "SOURCE=$SOURCE_HOST"
echo "MODEL_DIR=$MODEL_DIR_HOST"
echo "CHAINS=$CHAINS SAMPLES=$SAMPLES BURN=$BURN"

test -f "$REPO_ROOT/py_fit/study1_fit/addm_fit_es_identity_S_upper_family.py"
test -f "$INPUT_HOST"
test -f "$SOURCE_HOST"

apptainer exec --cleanenv \
  --bind "$REPO_ROOT:/workspace" \
  --bind "$DATA_ROOT:/source:ro" \
  --bind "$OUT_ROOT:/out" \
  --bind "$MODEL_DIR_HOST:/target_models" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/mplcache \
  "$IMAGE" \
  python /workspace/py_fit/study1_fit/addm_fit_es_identity_S_upper_family.py \
    --data /out/models/model_input_ES_identity_S_upper.csv \
    --source-data "/source/$DATA_FILE" \
    --model-dir /target_models \
    --model basic \
    --include-z \
    --samples "$SAMPLES" \
    --burn "$BURN" \
    --chains "$CHAINS" \
    --jobs "$SLURM_CPUS_PER_TASK"

echo "Finished STUDY 1 es_identity_S_upper_basic_z final."
