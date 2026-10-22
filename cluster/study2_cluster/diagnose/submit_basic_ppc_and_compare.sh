#!/usr/bin/env bash
# Submit Study-2 basic PPCs, then compare all four models after both succeed.
# Existing Study-2 contrast_noz and contrast_z PPCs must already be complete.
set -euo pipefail

REPO_ROOT="${PWD}"
ROOT="/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM"
RUN="cluster/study2_cluster/diagnose"

for tag in contrast_noz contrast_z; do
  PPC="$ROOT/derivatives/ppc/es_identity_s_upper_${tag}_main"
  [[ -f "$PPC/metadata.json" ]] || {
    echo "Missing existing Study-2 contrast PPC metadata for $tag" >&2
    exit 1
  }
  [[ -f "$PPC/tables/observed_replot_trials.csv.gz" ]] || {
    echo "Missing Study-2 contrast replot data for $tag" >&2
    exit 1
  }
done

J1=$(sbatch --parsable "$RUN/run_ppc_es_identity_s_upper_basic_noz.sh")
J2=$(sbatch --parsable "$RUN/run_ppc_es_identity_s_upper_basic_z.sh")
J3=$(sbatch --parsable --dependency="afterok:${J1}:${J2}" "$RUN/run_compare_es_four_model_ppc.sh")

printf 'Study2 Basic no-z PPC: %s\nStudy2 Basic z PPC: %s\nStudy2 four-model comparison: %s\n' "$J1" "$J2" "$J3"
