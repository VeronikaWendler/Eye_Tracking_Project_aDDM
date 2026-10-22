#!/usr/bin/env bash
# Can be run from anywhere: derive repository root from its location.
# Existing contrast PPCs must already have completed successfully.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
ROOT=/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM
mkdir -p "$ROOT/derivatives/logs"
for tag in contrast_noz contrast_z; do
  [[ -f "$ROOT/derivatives/ppc/es_identity_s_upper_${tag}_main/metadata.json" ]] || {
    echo "Missing existing contrast PPC metadata for $tag" >&2
    exit 1
  }
  [[ -f "$ROOT/derivatives/ppc/es_identity_s_upper_${tag}_main/tables/observed_replot_trials.csv.gz" ]] || {
    echo "Missing existing contrast replot data for $tag; rerun PPC with --save-replot-data" >&2
    exit 1
  }
done
RUN=cluster/study1_cluster/run
J1=$(sbatch --parsable "$RUN/run_ppc_es_identity_s_upper_basic_noz.sh")
J2=$(sbatch --parsable "$RUN/run_ppc_es_identity_s_upper_basic_z.sh")
J3=$(sbatch --parsable --dependency="afterok:${J1}:${J2}" "$RUN/run_compare_es_four_model_ppc.sh")
printf 'Basic no-z PPC: %s\nBasic z PPC: %s\nFour-model comparison: %s\n' "$J1" "$J2" "$J3"
