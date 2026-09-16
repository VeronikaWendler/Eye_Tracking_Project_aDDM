from pathlib import Path
from typing import Optional
import argparse
import json
import dill

import numpy as np
import pandas as pd
import hddm
from joblib import Parallel, delayed


MODEL_NAME = "aDDM_ES_OPTION_SPECIFIC_THETA"


def validate_input(data: pd.DataFrame) -> dict:
    """
    Validate the already-prepared ES option-specific-theta input.

    The response coordinate remains the same as in the conventional aDDM:
        response = 1 -> higher-valued option chosen
        response = 0 -> lower-valued option chosen

    The option-specific model changes ONLY the unattended-value term:

        conventional:
            v ~ 1 + AttentionW + InattentionW

        option-specific:
            v ~ 1 + AttentionW + InattentionW_E + InattentionW_S

    with:
        theta_E = b_E / b_A
        theta_S = b_S / b_A
    """

    required = [
        "subj_idx",
        "rt",
        "response",
        "phase",
        "AttentionW",
        "InattentionW",
        "InattentionW_E",
        "InattentionW_S",
        "V_E",
        "V_S",
        "PropDwell_E",
        "PropDwell_S",
    ]

    missing = [
        c for c in required
        if c not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required option-specific-theta columns: "
            f"{missing}"
        )

    # ES only.
    phases = set(
        data["phase"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    if phases != {"ES"}:
        raise ValueError(
            "This model is defined for ES only. "
            f"Found phases: {sorted(phases)}"
        )

    # Numeric safety.
    numeric_cols = [
        "subj_idx",
        "rt",
        "response",
        "AttentionW",
        "InattentionW",
        "InattentionW_E",
        "InattentionW_S",
        "V_E",
        "V_S",
        "PropDwell_E",
        "PropDwell_S",
    ]

    for col in numeric_cols:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce",
        )

    if data[numeric_cols].isna().any().any():
        bad = (
            data[numeric_cols]
            .isna()
            .sum()
        )
        bad = bad[
            bad > 0
        ].to_dict()

        raise ValueError(
            f"Missing/non-numeric model values: {bad}"
        )

    if not data["response"].isin(
        [0, 1]
    ).all():
        raise ValueError(
            "response must contain only 0/1."
        )

    if (
        data["rt"] <= 0
    ).any():
        raise ValueError(
            "All RT values must be > 0."
        )

    # Values remain probabilities on 0..1 scale.
    for col in [
        "V_E",
        "V_S",
    ]:
        if not (
            (
                data[col] >= 0
            )
            &
            (
                data[col] <= 1
            )
        ).all():
            raise ValueError(
                f"{col} contains values outside 0..1."
            )

    # Identity gaze should sum approximately to one.
    gaze_sum = (
        data["PropDwell_E"]
        +
        data["PropDwell_S"]
    )

    if not np.allclose(
        gaze_sum,
        1.0,
        atol=0.002,
        rtol=0,
    ):
        raise ValueError(
            "PropDwell_E + PropDwell_S does not "
            "sum approximately to 1."
        )

    # The new two unattended components must reconstruct
    # the original single InattentionW term.
    split_inattention = (
        data["InattentionW_E"]
        +
        data["InattentionW_S"]
    )

    split_diff = np.abs(
        split_inattention
        -
        data["InattentionW"]
    )

    max_split_diff = float(
        split_diff.max()
    )

    if max_split_diff > 0.001 + 1e-9:
        raise ValueError(
            "InattentionW_E + InattentionW_S does not "
            "reconstruct InattentionW within rounding tolerance. "
            f"Max absolute difference={max_split_diff:.6f}"
        )

    audit = {
        "model_name": MODEL_NAME,
        "phase": "ES",
        "rows": int(
            len(data)
        ),
        "participants": int(
            data["subj_idx"].nunique()
        ),
        "response_0_n": int(
            (
                data["response"] == 0
            ).sum()
        ),
        "response_1_n": int(
            (
                data["response"] == 1
            ).sum()
        ),
        "max_abs_split_inattention_difference": (
            max_split_diff
        ),
        "drift_formula": (
            "v ~ 1 + AttentionW "
            "+ InattentionW_E + InattentionW_S"
        ),
        "parameter_interpretation": {
            "v_Intercept": "b0",
            "v_AttentionW": "bA",
            "v_InattentionW_E": "bE",
            "v_InattentionW_S": "bS",
            "theta_E": "bE / bA",
            "theta_S": "bS / bA",
            "theta_difference": "theta_E - theta_S",
        },
        "important_interpretation": (
            "theta_E is the discount applied to E when E is unattended; "
            "theta_S is the discount applied to S when S is unattended."
        ),
    }

    return audit


def build_model(
    data: pd.DataFrame,
):
    """
    ES option-specific-theta aDDM.

    Conventional model:
        v = b0
            + bA * AttentionW
            + bI * InattentionW

    Current extension:
        v = b0
            + bA * AttentionW
            + bE * InattentionW_E
            + bS * InattentionW_S

    Derived parameters:
        theta_E = bE / bA
        theta_S = bS / bA

    a, t and z are treated exactly as ordinary HDDM parameters,
    matching the conventional ES model as closely as possible.
    """

    v_reg = {
        "model": (
            "v ~ 1 "
            "+ AttentionW "
            "+ InattentionW_E "
            "+ InattentionW_S"
        ),
        "link_func": lambda x: x,
    }

    model = hddm.HDDMRegressor(
        data,
        v_reg,

        include=[
            "a",
            "t",
            "v",
            "z",
        ],

        p_outlier=0.05,

        # Match the conventional model: hierarchical
        # participant-level regression coefficients.
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
        f"STARTING OPTION-SPECIFIC THETA CHAIN {chain}",
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

    # Native HDDM save.
    model.save(
        str(hddm_path)
    )

    print(
        f"Saved HDDM model: {hddm_path}",
        flush=True,
    )

    # Full object save using dill because HDDMRegressor
    # contains a lambda link function.
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
            "Fit the ES option-specific-theta aDDM with "
            "separate unattended-value coefficients for E and S."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help=(
            "Prepared model_input_ES_option_specific_theta.csv"
        ),
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

    data = pd.read_csv(
        args.data,
        low_memory=False,
    )

    audit = validate_input(
        data
    )

    # HDDM needs integer IDs / responses.
    data["subj_idx"] = (
        data["subj_idx"]
        .astype(int)
    )

    data["response"] = (
        data["response"]
        .astype(int)
    )

    # Save the exact model input again in the model folder so the
    # fitted object is permanently tied to the exact rows/columns used.
    exact_input_path = (
        args.model_dir
        / f"{MODEL_NAME}_MODEL_INPUT.csv"
    )

    audit_path = (
        args.model_dir
        / f"{MODEL_NAME}_AUDIT.json"
    )

    data.to_csv(
        exact_input_path,
        index=False,
        float_format="%.3f",
    )

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=" * 72,
        flush=True,
    )

    print(
        "ES OPTION-SPECIFIC THETA aDDM",
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
        f"\nExact model input saved to:\n"
        f"{exact_input_path}",
        flush=True,
    )

    print(
        f"\nAudit saved to:\n"
        f"{audit_path}",
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
        "\nAll option-specific-theta chains finished.",
        flush=True,
    )


if __name__ == "__main__":
    main()
