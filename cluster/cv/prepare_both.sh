#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p cluster/cv/logs
IMAGE="${IMAGE:-$HOME/containers/hddm_latest.sif}"
for I in 1 2;do
 STUDY="study$I";SROOT="/rds/projects/z/zhanglp-vwendler-core/Study${I}_aDDM";D="$SROOT/derivatives"
 if [[ "$I" == 1 ]];then INPUT="$D/models/model_input_ES_identity_S_upper.csv";else INPUT="$D/prepared_data/model_input_ES_identity_S_upper.csv";fi
 test -f "$INPUT";mkdir -p "$D/cv/es_grouped_5fold"
 apptainer exec --cleanenv --bind "$PWD:/workspace:ro" --bind "$SROOT:/study" "$IMAGE" \
 python /workspace/py_cv/01_prepare_grouped_folds.py \
 --input "/study/derivatives/$(realpath --relative-to="$D" "$INPUT")" \
 --out /study/derivatives/cv/es_grouped_5fold --study "$STUDY"
done
