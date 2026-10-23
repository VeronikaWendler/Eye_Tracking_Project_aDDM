#!/bin/bash
#SBATCH --job-name=ES_CV_pilot
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --output=cluster/cv/logs/es_cv_pilot_%j.out
#SBATCH --error=cluster/cv/logs/es_cv_pilot_%j.err
#SBATCH --mail-type=FAIL,END
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repo root}"
IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
SROOT=/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM
for MODEL in basic contrast;do
 mkdir -p "$SROOT/derivatives/cv/es_grouped_5fold/pilot/fold_0/$MODEL/fit"
 apptainer exec --cleanenv --bind "$PWD:/workspace:ro" --bind "$SROOT:/study" --env PYTHONUNBUFFERED=1 "$IMAGE" \
 python /workspace/py_cv/02_fit_cv_fold.py \
 --train /study/derivatives/cv/es_grouped_5fold/fold_0/train.csv \
 --source /study/data/Study1_Behaviour_with_Gaze_AnalysisReady.csv \
 --original-fit-script /workspace/py_fit/study1_fit/addm_fit_es_identity_S_upper_family.py \
 --out "/study/derivatives/cv/es_grouped_5fold/pilot/fold_0/${MODEL}/fit" --study study1 --model "$MODEL" --fold 0 --chains 1 --samples 1200 --burn 300 --jobs 1
 apptainer exec --cleanenv --bind "$PWD:/workspace:ro" --bind "$SROOT:/study" --env PYTHONUNBUFFERED=1 "$IMAGE" \
 python /workspace/py_cv/03_score_heldout_fold.py \
 --train /study/derivatives/cv/es_grouped_5fold/fold_0/train.csv \
 --test /study/derivatives/cv/es_grouped_5fold/fold_0/test.csv \
 --fit-dir "/study/derivatives/cv/es_grouped_5fold/pilot/fold_0/${MODEL}/fit" \
 --out "/study/derivatives/cv/es_grouped_5fold/pilot/fold_0/${MODEL}/score" \
 --study study1 --model "$MODEL" --fold 0 --chains 1 --draws-per-chain 25
done
printf 'Pilot passed both model scorers\n' > "$SROOT/derivatives/cv/es_grouped_5fold/pilot/PILOT_PASSED.flag"
echo 'PILOT PASSED: fit+score for both models, 25 draws/chain, Fold 0 Study 1; not final estimates.'
