from __future__ import annotations

from pathlib import Path
import argparse
import json
import dill

import numpy as np
import pandas as pd
import hddm
from joblib import Parallel, delayed


MODEL_NAME = "aDDM_JOINT_ALL_RANDOM_SLOPES"


def _standardize_subject_column(
    df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Return a copy with HDDM's required subject column named subj_idx."""
    df = df.copy()

    if "subj_idx" in df.columns:
        return df

    if "sub_id" in df.columns:
        return df.rename(columns={"sub_id": "subj_idx"})

    raise ValueError(
        f"{label}: expected either 'subj_idx' or 'sub_id', "
        "but neither was found."
    )


def load_joint_data(
    es_path: Path,
    ee_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Load the exact already-prepared ES and EE model-input files.

    EE is coded 0 and ES is coded 1. Therefore every phase coefficient
    in the regression is directly an ES - EE effect.
    """

    es = _standardize_subject_column(
        pd.read_csv(es_path),
        "ES",
    )

    ee = _standardize_subject_column(
        pd.read_csv(ee_path),
        "EE",
    )

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

            raise ValueError(
                f"{label}: missing values in model columns: {bad}"
            )

        responses = set(pd.unique(df["response"]))

        if not responses.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"{label}: response must be coded 0/1. "
                f"Found: {sorted(responses)}"
            )

        if (df["rt"] <= 0).any():
            raise ValueError(
                f"{label}: all RTs must be > 0 seconds."
            )

    es_subjects = set(pd.unique(es["subj_idx"]))
    ee_subjects = set(pd.unique(ee["subj_idx"]))

    if es_subjects != ee_subjects:
        raise ValueError(
            "ES and EE do not contain exactly the same participants.\n"
            f"Only in ES: {sorted(es_subjects - ee_subjects)}\n"
            f"Only in EE: {sorted(ee_subjects - es_subjects)}"
        )

    data = pd.concat(
        [ee, es],
        ignore_index=True,
    )

    data["phase"] = pd.Categorical(
        data["phase"],
        categories=["EE", "ES"],
        ordered=True,
    )

    # EE = 0, ES = 1.
    # Thus each *_phase_ES coefficient is directly ES - EE.
    data["phase_ES"] = (
        data["phase"] == "ES"
    ).astype(float)

    model_cols = [
        "subj_idx",
        "phase",
        "phase_ES",
        "rt",
        "response",
        "AttentionW",
        "InattentionW",
    ]

    if data[model_cols].isna().any().any():
        raise ValueError(
            "Joint model data contain missing values after construction."
        )

    audit = {
        "model_name": MODEL_NAME,

        "es_rows": int(len(es)),
        "ee_rows": int(len(ee)),
        "total_rows": int(len(data)),

        "participants": int(
            data["subj_idx"].nunique()
        ),

        "participant_ids": [
            str(x)
            for x in sorted(es_subjects)
        ],

        "phase_coding": {
            "EE": 0,
            "ES": 1,
        },

        "regressions": {
            "a": "a ~ 1 + phase_ES",
            "t": "t ~ 1 + phase_ES",
            "z": "z ~ 1 + phase_ES",
            "v": (
                "v ~ 1 + phase_ES "
                "+ AttentionW + InattentionW "
                "+ phase_ES:AttentionW "
                "+ phase_ES:InattentionW"
            ),
        },

        "interpretation": {
            "a_Intercept": "EE boundary separation",
            "a_phase_ES": "ES - EE boundary difference",

            "t_Intercept": "EE non-decision time",
            "t_phase_ES": "ES - EE non-decision-time difference",

            "z_Intercept": "EE starting point",
            "z_phase_ES": "ES - EE starting-point difference",

            "v_Intercept": "b0_EE",
            "v_phase_ES": "b0_ES - b0_EE",

            "v_AttentionW": "b1_EE",
            "v_phase_ES:AttentionW": "b1_ES - b1_EE",

            "v_InattentionW": "b2_EE",
            "v_phase_ES:InattentionW": "b2_ES - b2_EE",

            "b0_ES": "v_Intercept + v_phase_ES",
            "b1_ES": "v_AttentionW + v_phase_ES:AttentionW",
            "b2_ES": "v_InattentionW + v_phase_ES:InattentionW",

            "theta_EE": "v_InattentionW / v_AttentionW",
            "theta_ES": (
                "(v_InattentionW + v_phase_ES:InattentionW) / "
                "(v_AttentionW + v_phase_ES:AttentionW)"
            ),
        },

        "hierarchical_structure": (
            "group_only_regressors=False: HDDM estimates participant-level "
            "regression coefficients hierarchically for all regressors."
        ),

        "link_functions": (
            "Identity links (lambda x: x) for v, a, t, and z. "
            "No logistic z link is used."
        ),
    }

    return data, audit


def build_model(
    data: pd.DataFrame,
):
    """
    Full joint hierarchical regression model.

    Every core DDM parameter is modeled as a regression:

        a ~ 1 + phase_ES
        t ~ 1 + phase_ES
        z ~ 1 + phase_ES

    Drift is also fully regression-based:

        v ~ 1
            + phase_ES
            + AttentionW
            + InattentionW
            + phase_ES:AttentionW
            + phase_ES:InattentionW

    Since EE=0 and ES=1:
        Intercepts/main gaze effects = EE
        phase terms/interactions     = ES - EE

    group_only_regressors=False gives participant-level hierarchical
    regression coefficients for all regressors.

    All link functions are identity links, exactly lambda x: x.
    """

    v_reg = {
        "model": (
            "v ~ 1 "
            "+ phase_ES "
            "+ AttentionW "
            "+ InattentionW "
            "+ phase_ES:AttentionW "
            "+ phase_ES:InattentionW"
        ),
        "link_func": lambda x: x,
    }

    a_reg = {
        "model": "a ~ 1 + phase_ES",
        "link_func": lambda x: x,
    }

    t_reg = {
        "model": "t ~ 1 + phase_ES",
        "link_func": lambda x: x,
    }

    z_reg = {
        "model": "z ~ 1 + phase_ES",
        "link_func": lambda x: x,
    }

    model = hddm.HDDMRegressor(
        data,
        [
            v_reg,
            a_reg,
            t_reg,
            z_reg,
        ],

        include=[
            "a",
            "t",
            "v",
            "z",
        ],

        p_outlier=0.05,

        # Crucial: estimate participant-level hierarchical
        # regression coefficients instead of group-only regressors.
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
    """Fit and save one independent MCMC chain."""

    np.random.seed(
        seed + chain
    )

    print(
        "\n" + "=" * 72,
        flush=True,
    )

    print(
        f"STARTING ALL-REGRESSION CHAIN {chain}",
        flush=True,
    )

    print(
        "=" * 72,
        flush=True,
    )

    model = build_model(
        data
    )

    print(
        f"Finding starting values for chain {chain}...",
        flush=True,
    )

    model.find_starting_values()

    db_path = (
        model_dir
        / f"{MODEL_NAME}_db{chain}"
    )

    print(
        f"Sampling chain {chain}: "
        f"samples={samples}, burn={burn}",
        flush=True,
    )

    model.sample(
        samples,
        burn=burn,
        dbname=str(db_path),
        db="pickle",
    )

    hddm_path = (
        model_dir
        / f"{MODEL_NAME}_{chain}.hddm"
    )

    pkl_path = (
        model_dir
        / f"{MODEL_NAME}_{chain}.pkl"
    )

    model.save(
        str(hddm_path)
    )

    print(
        f"Saved HDDM model: {hddm_path}",
        flush=True,
    )

    with open(
        pkl_path,
        "wb",
    ) as f:

        dill.dump(
            model,
            f,
            recurse=True,
        )

    print(
        f"Saved dill pickle: {pkl_path}",
        flush=True,
    )

    print(
        f"Finished chain {chain}",
        flush=True,
    )

    return str(
        hddm_path
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Fit a joint ES+EE hierarchical aDDM in which "
            "v, a, t, and z are all regression outcomes and "
            "phase effects are estimated within participants."
        )
    )

    parser.add_argument(
        "--es-input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--ee-input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=4000,
    )

    parser.add_argument(
        "--burn",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--chains",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260915,
    )

    args = parser.parse_args()

    if args.burn >= args.samples:
        raise ValueError(
            "--burn must be smaller than --samples."
        )

    if args.chains < 1:
        raise ValueError(
            "--chains must be at least 1."
        )

    if args.jobs < 1:
        raise ValueError(
            "--jobs must be at least 1."
        )

    args.model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data, audit = load_joint_data(
        es_path=args.es_input,
        ee_path=args.ee_input,
    )

    input_path = (
        args.model_dir
        / f"{MODEL_NAME}_MODEL_INPUT.csv"
    )

    audit_path = (
        args.model_dir
        / f"{MODEL_NAME}_AUDIT.json"
    )

    data.to_csv(
        input_path,
        index=False,
    )

    with open(
        audit_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            audit,
            f,
            indent=2,
        )

    print(
        "=" * 72,
        flush=True,
    )

    print(
        "JOINT aDDM: ALL CORE PARAMETERS AS HIERARCHICAL REGRESSIONS",
        flush=True,
    )

    print(
        "=" * 72,
        flush=True,
    )

    print(
        json.dumps(
            audit,
            indent=2,
        ),
        flush=True,
    )

    print(
        f"\nSaved exact model input:\n{input_path}",
        flush=True,
    )

    print(
        f"\nSaved audit:\n{audit_path}",
        flush=True,
    )

    n_jobs = min(
        args.jobs,
        args.chains,
    )

    Parallel(
        n_jobs=n_jobs,
    )(
        delayed(
            fit_one_chain
        )(
            chain=chain,
            data=data,
            model_dir=args.model_dir,
            samples=args.samples,
            burn=args.burn,
            seed=args.seed,
        )
        for chain in range(
            args.chains
        )
    )

    print(
        "\nAll all-regression chains finished.",
        flush=True,
    )


if __name__ == "__main__":
    main()
