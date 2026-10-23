#!/bin/bash
# Submit ONLY after the pilot passes; creates fit->score dependencies separately per study.
set -euo pipefail
cd "$(dirname "$0")/../.."
PILOT=/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM/derivatives/cv/es_grouped_5fold/pilot/PILOT_PASSED.flag
[[ -f "$PILOT" ]] || { echo "Pilot has NOT passed; refusing full submission" >&2; exit 2; }
mkdir -p cluster/cv/logs
for I in 1 2;do
 STUDY="study$I";SROOT="/rds/projects/z/zhanglp-vwendler-core/Study${I}_aDDM"
 test -f "$SROOT/derivatives/cv/es_grouped_5fold/fold_manifest.json"
 fit=$(sbatch --parsable --export=ALL,STUDY="$STUDY",SROOT="$SROOT" cluster/cv/fit_array.sbatch)
 score=$(sbatch --parsable --dependency=afterok:"$fit" --export=ALL,STUDY="$STUDY",SROOT="$SROOT" cluster/cv/score_array.sbatch)
 echo "$STUDY fit array job=$fit; dependent score array=$score"
done
