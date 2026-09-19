#!/bin/bash
#SBATCH --job-name=s2_basic_noz
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/smoke/es_identity_S_upper_basic_noz_%j.out
#SBATCH --error=/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives/logs/smoke/es_identity_S_upper_basic_noz_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"
IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
DATA_ROOT="/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/data"
OUT_ROOT="/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM/derivatives"
DATA_FILE="Study2_Behaviour_with_Gaze_AnalysisReady.csv"
INPUT_HOST="$OUT_ROOT/prepared_data/model_input_ES_identity_S_upper.csv"
SOURCE_HOST="$DATA_ROOT/$DATA_FILE"
MODEL_DIR_HOST="$OUT_ROOT/models/es_identity_S_upper_basic_noz_smoke"
CHAINS="${CHAINS:-1}"
SAMPLES="${SAMPLES:-100}"
BURN="${BURN:-20}"
mkdir -p "$MODEL_DIR_HOST" "$OUT_ROOT/logs/smoke"

echo "============================================================================"
echo "STUDY 2 ES S-UPPER / E-LOWER BASIC aDDM SMOKE"
echo "DRIFT INTERCEPT: NONE | Z FIXED AT 0.5"
echo "============================================================================"
echo "INPUT=$INPUT_HOST"
echo "SOURCE=$SOURCE_HOST"
echo "MODEL_DIR=$MODEL_DIR_HOST"
echo "CHAINS=$CHAINS SAMPLES=$SAMPLES BURN=$BURN"

test -f "$REPO_ROOT/py_fit/study2_fit/addm_fit_es_identity_S_upper_family.py"
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
  python /workspace/py_fit/study2_fit/addm_fit_es_identity_S_upper_family.py \
    --data /out/prepared_data/model_input_ES_identity_S_upper.csv \
    --source-data "/source/$DATA_FILE" \
    --model-dir /target_models \
    --model basic \
    --samples "$SAMPLES" \
    --burn "$BURN" \
    --chains "$CHAINS" \
    --jobs "$SLURM_CPUS_PER_TASK"

echo "Finished STUDY 2 es_identity_S_upper_basic_noz smoke."
