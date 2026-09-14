from __future__ import annotations

from pathlib import Path
import argparse
import json
import pickle

import numpy as np
import pandas as pd
import hddm
from joblib import Parallel, delayed


MODEL_NAME = "phase_specific_aDDM_JOINT"


def identity_link(x):
    return x


def _standardize_subject_column(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return a copy with HDDM's required subject column named subj_idx."""
    df = df.copy()

    if "subj_idx" in df.columns:
        return df

    if "sub_id" in df.columns:
        return df.rename(columns={"sub_id": "subj_idx"})

    raise ValueError(
        f"{label}: expected either 'subj_idx' or 'sub_id', but neither was found."
    )


def load_joint_data(es_path: Path, ee_path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Load the already-prepared ES and EE model-input files.

    This script DOES NOT rebuild AttentionW/InattentionW. It uses the exact
    model inputs that were already used successfully in the separate fits.
    """
    es = _standardize_subject_column(pd.read_csv(es_path), "ES")
    ee = _standardize_subject_column(pd.read_csv(ee_path), "EE")

    es["phase"] = "ES"
    ee["phase"] = "EE"

    required = {
        "subj_idx",
        "rt",
        "response",
        "AttentionW",
        "InattentionW",
    }

    for df, label in [(es, "ES"), (ee, "EE")]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{label}: missing required columns: {sorted(missing)}"
            )

        if df[list(required)].isna().any().any():
            bad = df[list(required)].isna().sum()
            bad = bad[bad > 0].to_dict()
            raise ValueError(f"{label}: missing values in model columns: {bad}")

        responses = set(pd.unique(df["response"]))
        if not responses.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"{label}: response must be coded 0/1. Found: {sorted(responses)}"
            )

        if (df["rt"] <= 0).any():
            raise ValueError(f"{label}: all RTs must be > 0 seconds.")

    es_subjects = set(pd.unique(es["subj_idx"]))
    ee_subjects = set(pd.unique(ee["subj_idx"]))

    if es_subjects != ee_subjects:
        raise ValueError(
            "ES and EE do not contain exactly the same participants.\n"
            f"Only in ES: {sorted(es_subjects - ee_subjects)}\n"
            f"Only in EE: {sorted(ee_subjects - es_subjects)}"
        )

    data = pd.concat([ee, es], ignore_index=True)

    data["phase"] = pd.Categorical(
        data["phase"],
        categories=["EE", "ES"],
        ordered=True,
    )

    # Cell-means coding for fully separate phase-specific drift parameters.
    is_ee = (data["phase"] == "EE").astype(float)
    is_es = (data["phase"] == "ES").astype(float)

    data["drift0_EE"] = is_ee
    data["drift0_ES"] = is_es

    data["AttentionW_EE"] = data["AttentionW"] * is_ee
    data["AttentionW_ES"] = data["AttentionW"] * is_es

    data["InattentionW_EE"] = data["InattentionW"] * is_ee
    data["InattentionW_ES"] = data["InattentionW"] * is_es

    model_cols = [
        "subj_idx",
        "phase",
        "rt",
        "response",
        "drift0_EE",
        "drift0_ES",
        "AttentionW_EE",
        "AttentionW_ES",
        "InattentionW_EE",
        "InattentionW_ES",
    ]

    if data[model_cols].isna().any().any():
        raise ValueError("Joint model data contain missing values after construction.")

    audit = {
        "model_name": MODEL_NAME,
        "es_rows": int(len(es)),
        "ee_rows": int(len(ee)),
        "total_rows": int(len(data)),
        "participants": int(data["subj_idx"].nunique()),
        "participant_ids": [str(x) for x in sorted(es_subjects)],
        "response_1_ES": int(es["response"].sum()),
        "response_0_ES": int(len(es) - es["response"].sum()),
        "response_1_EE": int(ee["response"].sum()),
        "response_0_EE": int(len(ee) - ee["response"].sum()),
        "mean_rt_ES": float(es["rt"].mean()),
        "mean_rt_EE": float(ee["rt"].mean()),
        "drift_formula": (
            "v ~ 0 + drift0_EE + drift0_ES "
            "+ AttentionW_EE + AttentionW_ES "
            "+ InattentionW_EE + InattentionW_ES"
        ),
        "depends_on": {
            "a": "phase",
            "t": "phase",
            "z": "phase",
        },
        "interpretation": {
            "v_drift0_EE": "b0_EE",
            "v_drift0_ES": "b0_ES",
            "v_AttentionW_EE": "b1_EE",
            "v_AttentionW_ES": "b1_ES",
            "v_InattentionW_EE": "b2_EE",
            "v_InattentionW_ES": "b2_ES",
            "theta_EE": "b2_EE / b1_EE",
            "theta_ES": "b2_ES / b1_ES",
        },
    }

    return data, audit


def build_model(data: pd.DataFrame):
    """
    Joint phase-specific aDDM.

    a, t, z:
        estimated separately for EE and ES via depends_on.

    v:
        six regression coefficients:
          b0_EE, b0_ES
          b1_EE, b1_ES
          b2_EE, b2_ES

    Therefore theta is also phase-specific:
        theta_EE = b2_EE / b1_EE
        theta_ES = b2_ES / b1_ES
    """
    v_reg = {
        "model": (
            "v ~ 0 "
            "+ drift0_EE + drift0_ES "
            "+ AttentionW_EE + AttentionW_ES "
            "+ InattentionW_EE + InattentionW_ES"
        ),
        "link_func": identity_link,
    }

    model = hddm.HDDMRegressor(
        data,
        v_reg,
        depends_on={
            "a": "phase",
            "t": "phase",
            "z": "phase",
        },
        include=["a", "t", "v", "z"],
        p_outlier=0.05,
        group_only_regressors=False,
        keep_regressor_trace=True,
    )

    return model


def fit_one_chain(
    chain: int,
    data: pd.DataFrame,
    model_dir: Path,
    samples: int,
    burn: int,
    seed: int,
):
    np.random.seed(seed + chain)

    print("\n" + "=" * 72, flush=True)
    print(f"STARTING JOINT PHASE-SPECIFIC CHAIN {chain}", flush=True)
    print("=" * 72, flush=True)

    model = build_model(data)

    print(f"Finding starting values for chain {chain}...", flush=True)
    model.find_starting_values()

    db_path = model_dir / f"{MODEL_NAME}_db{chain}"

    print(
        f"Sampling chain {chain}: samples={samples}, burn={burn}",
        flush=True,
    )

    model.sample(
        samples,
        burn=burn,
        dbname=str(db_path),
        db="pickle",
    )

    hddm_path = model_dir / f"{MODEL_NAME}_{chain}.hddm"
    pkl_path = model_dir / f"{MODEL_NAME}_{chain}.pkl"

    model.save(str(hddm_path))

    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Finished chain {chain}", flush=True)
    print(f"Saved HDDM model: {hddm_path}", flush=True)
    print(f"Saved pickle:     {pkl_path}", flush=True)

    return str(hddm_path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit one joint hierarchical aDDM to ES + EE, while allowing "
            "a, t, z, b0, b1, b2, and therefore theta to differ by phase."
        )
    )

    parser.add_argument("--es-input", type=Path, required=True)
    parser.add_argument("--ee-input", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)

    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--burn", type=int, default=500)
    parser.add_argument("--chains", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260914)

    args = parser.parse_args()

    if args.burn >= args.samples:
        raise ValueError("--burn must be smaller than --samples.")

    if args.chains < 1:
        raise ValueError("--chains must be at least 1.")

    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1.")

    args.model_dir.mkdir(parents=True, exist_ok=True)

    data, audit = load_joint_data(
        es_path=args.es_input,
        ee_path=args.ee_input,
    )

    joint_input_path = args.model_dir / f"{MODEL_NAME}_MODEL_INPUT.csv"
    audit_path = args.model_dir / f"{MODEL_NAME}_AUDIT.json"

    data.to_csv(joint_input_path, index=False)

    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print("=" * 72, flush=True)
    print("JOINT PHASE-SPECIFIC aDDM", flush=True)
    print("=" * 72, flush=True)
    print(json.dumps(audit, indent=2), flush=True)
    print(f"\nSaved exact joint model input:\n{joint_input_path}", flush=True)
    print(f"\nSaved audit:\n{audit_path}", flush=True)

    n_jobs = min(args.jobs, args.chains)

    Parallel(n_jobs=n_jobs)(
        delayed(fit_one_chain)(
            chain=chain,
            data=data,
            model_dir=args.model_dir,
            samples=args.samples,
            burn=args.burn,
            seed=args.seed,
        )
        for chain in range(args.chains)
    )

    print("\nAll joint phase-specific chains finished.", flush=True)


if __name__ == "__main__":
    main()
