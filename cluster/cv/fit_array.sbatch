#!/bin/bash
#SBATCH --job-name=ES_CV_fit
#SBATCH --array=0-9%1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=180G
#SBATCH --output=cluster/cv/logs/es_cv_fit_%x_%A_%a.out
#SBATCH --error=cluster/cv/logs/es_cv_fit_%x_%A_%a.err
#SBATCH --mail-type=FAIL,END
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repo root}"
: "${STUDY:?Pass STUDY=study1 or study2 via --export}"
: "${SROOT:?Pass SROOT via --export}"
[[ "$STUDY" == study1 || "$STUDY" == study2 ]] || exit 2
IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
DERIV="$SROOT/derivatives"; CV="$DERIV/cv/es_grouped_5fold"
if [[ "$STUDY" == study1 ]]; then INPUT="$DERIV/models/model_input_ES_identity_S_upper.csv"; else INPUT="$DERIV/prepared_data/model_input_ES_identity_S_upper.csv"; fi
SOURCE="$SROOT/data/$(echo "$STUDY" | sed -E 's/study([12])/Study\1/')_Behaviour_with_Gaze_AnalysisReady.csv"
FIT_SCRIPT="py_fit/${STUDY}_fit/addm_fit_es_identity_S_upper_family.py"
FOLD=$((SLURM_ARRAY_TASK_ID / 2))
if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then MODEL=basic; else MODEL=contrast; fi
OUT="$CV/fold_${FOLD}/${MODEL}/fit"; mkdir -p "$OUT"
test -f "$IMAGE" && test -f "$CV/fold_${FOLD}/train.csv" && test -f "$FIT_SCRIPT" && test -f "$SOURCE"
apptainer exec --cleanenv --bind "$PWD:/workspace:ro" --bind "$SROOT:/study" --env PYTHONUNBUFFERED=1 "$IMAGE" \
 python /workspace/py_cv/02_fit_cv_fold.py \
 --train "/study/derivatives/cv/es_grouped_5fold/fold_${FOLD}/train.csv" --source "/study/data/$(basename "$SOURCE")" \
 --original-fit-script "/workspace/$FIT_SCRIPT" --out "/study/derivatives/cv/es_grouped_5fold/fold_${FOLD}/${MODEL}/fit" \
 --study "$STUDY" --model "$MODEL" --fold "$FOLD" --chains 3 --samples 4000 --burn 1000 --jobs 3
