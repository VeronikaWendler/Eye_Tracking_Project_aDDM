# Study 1 — cleaned basic probability-value aDDM

Use dataset:

    data_sets/Data_Sets_Study1/Study1_Behaviour_with_Gaze_AnalysisReady.csv


## Model

The basic model is fitted separately to ES and EE:

    v ~ 0 + AttentionW + InattentionW

Probability itself is the option value:
- 10% = 0.10
- 80% = 0.80

No `2p-1` EV transform is used.
No 0..100 value scale is used.

The preparation script rebuilds the model regressors from the corrected
physical columns instead of trusting old stored aDDM columns:

    V_high = max(p1, p2)
    V_low  = min(p1, p2)

    AttentionW   = PropDwell_high * V_high - PropDwell_low * V_low
    InattentionW = PropDwell_low  * V_high - PropDwell_high * V_low

`response=1` means the higher-probability option was chosen.

All continuous model inputs are rounded to 3 decimals. Probabilities and gaze
proportions are clipped only to their theoretical [0, 1] bounds. The code does
NOT arbitrarily truncate AttentionW/InattentionW or RT ranges.

## Local preparation

From the repository root:

    python addm_prepare_data.py --phase ES \
      --out data_sets/Data_Sets_Study1/model_input_ES.csv \
      --audit data_sets/Data_Sets_Study1/model_input_ES_audit.json

    python addm_prepare_data.py --phase EE \
      --out data_sets/Data_Sets_Study1/model_input_EE.csv \
      --audit data_sets/Data_Sets_Study1/model_input_EE_audit.json

Because the corrected CSV is the default path, `--data` is not needed locally
if the repository layout is unchanged.

## Safety checks

The preparation fails if:
- p1/p2 are not on the 0..1 scale;
- ES looks canonicalized like the old `_re` data;
- ES does not contain one E and one S;
- physical choice coding disagrees with `chose_right_spatial`;
- the rebuilt aDDM regressors fail their formulas.

Equal-probability trials are removed because "higher" and "lower" are undefined.

## BlueBEAR

The cluster script expects the corrected AnalysisReady CSV in the bound RDS
data folder. Submit:

    sbatch --export=ALL,PHASE=ES cluster/run_hddm.sh
    sbatch --export=ALL,PHASE=EE cluster/run_hddm.sh

## LE

LE remains available structurally, but should not be interpreted until we
decide whether its value signal should use objective probabilities or learned/Q
values.
