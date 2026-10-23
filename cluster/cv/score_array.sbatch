#!/bin/bash
#SBATCH --job-name=ES_CV_score
#SBATCH --array=0-9%2
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=cluster/cv/logs/es_cv_score_%x_%A_%a.out
#SBATCH --error=cluster/cv/logs/es_cv_score_%x_%A_%a.err
#SBATCH --mail-type=FAIL,END
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repo root}"
: "${STUDY:?Missing STUDY}"; : "${SROOT:?Missing SROOT}"
IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
FOLD=$((SLURM_ARRAY_TASK_ID / 2))
if (( SLURM_ARRAY_TASK_ID % 2 == 0 ));then MODEL=basic; else MODEL=contrast;fi
[[ -f "$IMAGE" ]] || exit 3
apptainer exec --cleanenv --bind "$PWD:/workspace:ro" --bind "$SROOT:/study" --env PYTHONUNBUFFERED=1 "$IMAGE" \
 python /workspace/py_cv/03_score_heldout_fold.py \
 --train "/study/derivatives/cv/es_grouped_5fold/fold_${FOLD}/train.csv" \
 --test "/study/derivatives/cv/es_grouped_5fold/fold_${FOLD}/test.csv" \
 --fit-dir "/study/derivatives/cv/es_grouped_5fold/fold_${FOLD}/${MODEL}/fit" \
 --out "/study/derivatives/cv/es_grouped_5fold/fold_${FOLD}/${MODEL}/score" \
 --model "$MODEL" --study "$STUDY" --fold "$FOLD" --chains 3 --draws-per-chain 200
