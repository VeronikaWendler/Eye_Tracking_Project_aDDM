from pathlib import Path
import argparse
import json
import dill
import numpy as np
import pandas as pd
import hddm
from joblib import Parallel, delayed

STUDY_NAME = 'STUDY 2'
EXCLUDED_SUBJECTS = {2, 6, 9, 14, 18, 20, 26}
EXPECTED_PARTICIPANTS = 20

BASIC = ["AttentionW_SE", "InattentionW_SE"]
CONTRAST = BASIC + ["AttentionContrast_SE", "InattentionContrast_SE"]


def spec(model_kind, include_z):
    predictors = BASIC if model_kind == "basic" else CONTRAST
    formula = "v ~ 0 + " + " + ".join(predictors)
    include = ["a", "t", "v"] + (["z"] if include_z else [])
    tag = f"{model_kind}_{'z' if include_z else 'noz'}"
    name = f"aDDM_ES_IDENTITY_S_UPPER_{tag.upper()}_NO_INTERCEPT"
    return predictors, formula, include, name


def expected_ids_from_source(path):
    src = pd.read_csv(path, low_memory=False)
    need = ["sub_id", "phase", "op1", "op2"]
    miss = [c for c in need if c not in src.columns]
    if miss:
        raise ValueError(f"Source file missing audit columns: {miss}")
    src["sub_id"] = pd.to_numeric(src["sub_id"], errors="coerce")
    es = src.loc[src["phase"].astype(str).str.strip().eq("ES")].copy()
    valid = ((es["op1"].eq("E") & es["op2"].eq("S")) |
             (es["op1"].eq("S") & es["op2"].eq("E")))
    es = es.loc[valid]
    return sorted(int(x) for x in es["sub_id"].dropna().unique()
                  if int(x) not in EXCLUDED_SUBJECTS)


def validate(data, source_data, model_kind, include_z):
    predictors, formula, include, model_name = spec(model_kind, include_z)
    required = [
        "subj_idx", "rt", "response", "phase", "V_E", "V_S",
        "PropDwell_E", "PropDwell_S", "value_tie",
        "Attended_S", "Attended_E", "Unattended_S", "Unattended_E",
        "AttentionW_SE", "InattentionW_SE",
        "AttentionContrast_SE", "InattentionContrast_SE",
    ]
    miss = [c for c in required if c not in data.columns]
    if miss:
        raise ValueError(f"Prepared S-upper input missing columns: {miss}")

    phases = set(data["phase"].dropna().astype(str).str.strip().unique())
    if phases != {"ES"}:
        raise ValueError(f"Expected ES only; found {sorted(phases)}")

    numeric = [c for c in required if c not in ["phase", "value_tie"]]
    for c in numeric:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    if data[numeric].isna().any().any():
        bad = data[numeric].isna().sum()
        raise ValueError(f"Missing/non-numeric model values: {bad[bad>0].to_dict()}")

    data["subj_idx"] = data["subj_idx"].astype(int)
    data["response"] = data["response"].astype(int)
    if not data["response"].isin([0, 1]).all():
        raise ValueError("response must be coded 1=S / 0=E")
    if (data["rt"] <= 0).any():
        raise ValueError("All RT values must be > 0")
    if not np.allclose(data["PropDwell_E"] + data["PropDwell_S"], 1.0, atol=.002, rtol=0):
        raise ValueError("PropDwell_E + PropDwell_S does not sum to ~1")

    # Verify S-upper signed regressors.
    checks = {
        "attention_sum": float(np.max(np.abs((data["Attended_S"] + data["Attended_E"]) - data["AttentionW_SE"]))),
        "inattention_sum": float(np.max(np.abs((data["Unattended_S"] + data["Unattended_E"]) - data["InattentionW_SE"]))),
        "attention_contrast": float(np.max(np.abs(((data["Attended_S"] - data["Attended_E"])/2).round(3) - data["AttentionContrast_SE"]))),
        "inattention_contrast": float(np.max(np.abs(((data["Unattended_S"] - data["Unattended_E"])/2).round(3) - data["InattentionContrast_SE"]))),
    }
    for k, v in checks.items():
        if v > .002000001:
            raise ValueError(f"S-upper regressor reconstruction failed: {k}={v}")

    expected = expected_ids_from_source(source_data)
    actual = sorted(data["subj_idx"].unique().tolist())
    if len(expected) != EXPECTED_PARTICIPANTS:
        raise ValueError(
            f"{STUDY_NAME} source-derived expected N={len(expected)} but expected "
            f"{EXPECTED_PARTICIPANTS}. IDs={expected}"
        )
    if actual != expected:
        missing = sorted(set(expected)-set(actual))
        extra = sorted(set(actual)-set(expected))
        raise ValueError(
            f"Participant audit failed. Missing={missing}, extra={extra}, "
            f"expected={expected}, actual={actual}"
        )

    if "chosen_identity" in data.columns:
        chosen = data["chosen_identity"].astype(str).str.strip()
        if chosen.isin(["E", "S"]).all():
            resp = chosen.eq("S").astype(int)
            if not np.array_equal(resp.to_numpy(), data["response"].to_numpy()):
                raise ValueError("Response check failed: response is not exactly 1=S / 0=E")

    part = data.groupby("subj_idx").agg(
        rows=("response", "size"),
        E_response_0=("response", lambda x: int((x==0).sum())),
        S_response_1=("response", lambda x: int((x==1).sum())),
    ).reset_index()

    # Rank check for the actual no-intercept design.
    X = data[predictors].astype(float)
    rank = int(np.linalg.matrix_rank(X.to_numpy()))
    if rank != len(predictors):
        raise ValueError(
            f"No-intercept drift design is rank deficient: rank={rank}, p={len(predictors)}"
        )
    corr = X.corr()

    audit = {
        "study": STUDY_NAME,
        "model_name": model_name,
        "model_kind": model_kind,
        "coordinate": "S upper / E lower",
        "upper_boundary_response_1": "S",
        "lower_boundary_response_0": "E",
        "drift_intercept": "ABSENT",
        "drift_formula": formula,
        "estimated_parameters": include,
        "starting_point": "estimated z" if include_z else "fixed at z=0.5 (not estimated)",
        "rows": int(len(data)),
        "participants": int(data["subj_idx"].nunique()),
        "expected_participants": EXPECTED_PARTICIPANTS,
        "expected_ids": expected,
        "actual_ids": actual,
        "excluded_ids": sorted(EXCLUDED_SUBJECTS),
        "response_0_E_n": int((data["response"]==0).sum()),
        "response_1_S_n": int((data["response"]==1).sum()),
        "predictors": predictors,
        "design_rank": rank,
        "regressor_reconstruction": checks,
    }
    return audit, part, corr


def build_model(data, model_kind, include_z):
    _, formula, include, _ = spec(model_kind, include_z)
    v_reg = {"model": formula, "link_func": lambda x: x}
    return hddm.HDDMRegressor(
        data, v_reg, include=include, p_outlier=.05,
        group_only_regressors=False, keep_regressor_trace=True,
    )


def fit_chain(chain, data, model_dir, model_kind, include_z, model_name, samples, burn, seed):
    np.random.seed(seed + chain)
    print("\n" + "="*78, flush=True)
    print(f"STARTING {STUDY_NAME} {model_name} CHAIN {chain}", flush=True)
    print("="*78, flush=True)
    model = build_model(data, model_kind, include_z)
    model.find_starting_values()
    db = model_dir / f"{model_name}_db{chain}"
    model.sample(samples, burn=burn, dbname=str(db), db="pickle")
    hp = model_dir / f"{model_name}_{chain}.hddm"
    pp = model_dir / f"{model_name}_{chain}.pkl"
    model.save(str(hp))
    with open(pp, "wb") as f:
        dill.dump(model, f, recurse=True)
    print(f"Finished chain {chain}: {hp}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--source-data", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--model", choices=["basic", "contrast"], required=True)
    p.add_argument("--include-z", action="store_true")
    p.add_argument("--samples", type=int, default=4000)
    p.add_argument("--burn", type=int, default=1000)
    p.add_argument("--chains", type=int, default=3)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260919)
    p.add_argument("--design-only", action="store_true")
    a = p.parse_args()
    if a.burn >= a.samples:
        raise ValueError("--burn must be smaller than --samples")
    a.model_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(a.data, low_memory=False)
    audit, part, corr = validate(data, a.source_data, a.model, a.include_z)
    _, _, _, model_name = spec(a.model, a.include_z)
    data.to_csv(a.model_dir/f"{model_name}_MODEL_INPUT.csv", index=False, float_format="%.3f")
    part.to_csv(a.model_dir/f"{model_name}_PARTICIPANT_AUDIT.csv", index=False)
    corr.to_csv(a.model_dir/f"{model_name}_PREDICTOR_CORRELATIONS.csv", float_format="%.8f")
    (a.model_dir/f"{model_name}_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("\nMODEL AUDIT", flush=True)
    print(json.dumps(audit, indent=2), flush=True)
    print("\nPARTICIPANT AUDIT", flush=True)
    print(part.to_string(index=False), flush=True)
    if a.design_only:
        print("\nAll checks passed; design-only requested, so no MCMC started.", flush=True)
        return
    Parallel(n_jobs=min(a.jobs, a.chains))(
        delayed(fit_chain)(i, data, a.model_dir, a.model, a.include_z, model_name, a.samples, a.burn, a.seed)
        for i in range(a.chains)
    )
    print(f"\nAll {STUDY_NAME} {model_name} chains finished.", flush=True)


if __name__ == "__main__":
    main()
