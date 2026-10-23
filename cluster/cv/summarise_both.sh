#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
for I in 1 2;do
 STUDY="study$I";SROOT="/rds/projects/z/zhanglp-vwendler-core/Study${I}_aDDM"
 apptainer exec --cleanenv --bind "$PWD:/workspace:ro" --bind "$SROOT:/study" "$IMAGE" \
 python /workspace/py_cv/04_summarise_cv.py --study "$STUDY" \
 --root /study/derivatives/cv/es_grouped_5fold --out /study/derivatives/cv/es_grouped_5fold/summary
done
