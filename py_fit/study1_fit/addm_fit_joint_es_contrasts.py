from pathlib import Path
import argparse
import json
import dill

import numpy as np
import pandas as pd
import hddm
from joblib import Parallel, delayed

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except Exception:
    variance_inflation_factor = None


MODEL_NAME = "aDDM_ES_JOINT_ES_CONTRASTS"

PREDICTORS = [
    "AttentionW",
    "InattentionW",
    "AttentionContrast",
    "InattentionContrast",
]


def validate_input(data: pd.DataFrame) -> dict:
    """
    Validate the prepared ES joint-contrast model input.

    Model:

        v = b0
            + bA * AttentionW
            + bI * InattentionW
            + deltaA * AttentionContrast
            + deltaI * InattentionContrast

    where:

        AttentionContrast
            = (AttentionW_E - AttentionW_S) / 2

        InattentionContrast
            = (InattentionW_E - InattentionW_S) / 2

    Hence:

        deltaA = bAE - bAS
        deltaI = bIE - bIS

    and:

        bAE = bA + deltaA/2
        bAS = bA - deltaA/2

        bIE = bI + deltaI/2
        bIS = bI - deltaI/2
    """

    required = [
        "subj_idx",
        "rt",
        "response",
        "phase",
        "AttentionW",
        "InattentionW",
        "AttentionW_E",
        "AttentionW_S",
        "InattentionW_E",
        "InattentionW_S",
        "AttentionContrast",
        "InattentionContrast",
    ]

    missing = [
        col
        for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required joint-contrast columns: "
            f"{missing}"
        )

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

    numeric_cols = [
        "subj_idx",
        "rt",
        "response",
        "AttentionW",
        "InattentionW",
        "AttentionW_E",
        "AttentionW_S",
        "InattentionW_E",
        "InattentionW_S",
        "AttentionContrast",
        "InattentionContrast",
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

    att_sum_diff = np.abs(
        (
            data["AttentionW_E"]
            + data["AttentionW_S"]
        )
        - data["AttentionW"]
    )

    inatt_sum_diff = np.abs(
        (
            data["InattentionW_E"]
            + data["InattentionW_S"]
        )
        - data["InattentionW"]
    )

    att_contrast_expected = (
        (
            data["AttentionW_E"]
            - data["AttentionW_S"]
        )
        / 2.0
    ).round(3)

    inatt_contrast_expected = (
        (
            data["InattentionW_E"]
            - data["InattentionW_S"]
        )
        / 2.0
    ).round(3)

    att_contrast_diff = np.abs(
        att_contrast_expected
        - data["AttentionContrast"]
    )

    inatt_contrast_diff = np.abs(
        inatt_contrast_expected
        - data["InattentionContrast"]
    )

    tolerance = 0.002 + 1e-9

    checks = {
        "max_abs_Attention_component_sum_difference": float(
            att_sum_diff.max()
        ),
        "max_abs_Inattention_component_sum_difference": float(
            inatt_sum_diff.max()
        ),
        "max_abs_AttentionContrast_formula_difference": float(
            att_contrast_diff.max()
        ),
        "max_abs_InattentionContrast_formula_difference": float(
            inatt_contrast_diff.max()
        ),
    }

    if (
        checks["max_abs_Attention_component_sum_difference"]
        > tolerance
    ):
        raise ValueError(
            "Attention component reconstruction failed."
        )

    if (
        checks["max_abs_Inattention_component_sum_difference"]
        > tolerance
    ):
        raise ValueError(
            "Inattention component reconstruction failed."
        )

    if (
        checks["max_abs_AttentionContrast_formula_difference"]
        > tolerance
    ):
        raise ValueError(
            "AttentionContrast formula validation failed."
        )

    if (
        checks["max_abs_InattentionContrast_formula_difference"]
        > tolerance
    ):
        raise ValueError(
            "InattentionContrast formula validation failed."
        )

    audit = {
        "model_name": MODEL_NAME,
        "phase": "ES",
        "rows": int(len(data)),
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
        "drift_formula": (
            "v ~ 1 + AttentionW + InattentionW "
            "+ AttentionContrast + InattentionContrast"
        ),
        "contrast_definitions": {
            "AttentionContrast": (
                "(AttentionW_E - AttentionW_S) / 2"
            ),
            "InattentionContrast": (
                "(InattentionW_E - InattentionW_S) / 2"
            ),
        },
        "coefficient_interpretation": {
            "v_Intercept": "b0",
            "v_AttentionW": "bA: midpoint/common attended value weight",
            "v_InattentionW": "bI: midpoint/common unattended value weight",
            "v_AttentionContrast": "deltaA = bAE - bAS",
            "v_InattentionContrast": "deltaI = bIE - bIS",
            "bAE": "bA + deltaA/2",
            "bAS": "bA - deltaA/2",
            "bIE": "bI + deltaI/2",
            "bIS": "bI - deltaI/2",
            "theta_E": "bIE / bAE",
            "theta_S": "bIS / bAS",
        },
        "validation": checks,
    }

    return audit


def compute_design_diagnostics(
    data: pd.DataFrame,
    out_dir: Path,
) -> dict:
    """
    Check whether the four regression predictors are separately identifiable.

    Saves:
        design_predictor_correlations.csv
        design_vif.csv
        design_diagnostics.json

    Condition number is calculated after z-standardizing predictors so it is
    not driven merely by different predictor scales.
    """

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    X = (
        data[PREDICTORS]
        .astype(float)
        .copy()
    )

    corr = X.corr()

    corr.to_csv(
        out_dir
        / "design_predictor_correlations.csv",
        float_format="%.8f",
    )

    std = X.std(
        axis=0,
        ddof=0,
    )

    if (
        std <= 0
    ).any():
        bad = std[
            std <= 0
        ].index.tolist()

        raise ValueError(
            "Zero-variance design predictors: "
            f"{bad}"
        )

    Xz = (
        X
        - X.mean(
            axis=0
        )
    ) / std

    Xz_with_intercept = np.column_stack(
        [
            np.ones(
                len(Xz)
            ),
            Xz.to_numpy(),
        ]
    )

    matrix_rank = int(
        np.linalg.matrix_rank(
            Xz_with_intercept
        )
    )

    number_columns = int(
        Xz_with_intercept.shape[1]
    )

    condition_number = float(
        np.linalg.cond(
            Xz_with_intercept
        )
    )

    # VIF is scale invariant. Use the established statsmodels implementation
    # when available in the HDDM environment.
    vif_rows = []

    if variance_inflation_factor is not None:

        X_with_intercept = np.column_stack(
            [
                np.ones(
                    len(X)
                ),
                X.to_numpy(),
            ]
        )

        for idx, name in enumerate(
            PREDICTORS,
            start=1,
        ):

            vif_value = float(
                variance_inflation_factor(
                    X_with_intercept,
                    idx,
                )
            )

            vif_rows.append(
                {
                    "Predictor": name,
                    "VIF": vif_value,
                }
            )

    else:

        # Fallback using the defining VIF = 1 / (1 - R^2).
        # This is only used if statsmodels is unavailable.
        for target in PREDICTORS:

            others = [
                name
                for name in PREDICTORS
                if name != target
            ]

            y = X[target].to_numpy()
            Z = np.column_stack(
                [
                    np.ones(
                        len(X)
                    ),
                    X[others].to_numpy(),
                ]
            )

            beta, _, _, _ = np.linalg.lstsq(
                Z,
                y,
                rcond=None,
            )

            fitted = Z @ beta

            ss_res = float(
                np.sum(
                    (
                        y
                        - fitted
                    ) ** 2
                )
            )

            ss_tot = float(
                np.sum(
                    (
                        y
                        - np.mean(y)
                    ) ** 2
                )
            )

            r2 = (
                1.0
                - ss_res / ss_tot
                if ss_tot > 0
                else 1.0
            )

            vif_value = (
                float("inf")
                if r2 >= 1.0
                else float(
                    1.0
                    / (
                        1.0
                        - r2
                    )
                )
            )

            vif_rows.append(
                {
                    "Predictor": target,
                    "VIF": vif_value,
                }
            )

    vif_df = pd.DataFrame(
        vif_rows
    )

    vif_df.to_csv(
        out_dir
        / "design_vif.csv",
        index=False,
        float_format="%.8f",
    )

    corr_abs = np.abs(
        corr.to_numpy()
    )

    np.fill_diagonal(
        corr_abs,
        np.nan,
    )

    max_abs_correlation = float(
        np.nanmax(
            corr_abs
        )
    )

    max_vif = float(
        vif_df["VIF"].max()
    )

    warnings = []

    if matrix_rank < number_columns:
        warnings.append(
            "DESIGN MATRIX IS RANK DEFICIENT."
        )

    if max_abs_correlation >= 0.90:
        warnings.append(
            "At least one absolute pairwise predictor correlation is >= 0.90."
        )

    if max_vif >= 10.0:
        warnings.append(
            "At least one predictor has VIF >= 10."
        )

    if condition_number >= 30.0:
        warnings.append(
            "Standardized design condition number is >= 30."
        )

    diagnostics = {
        "predictors": PREDICTORS,
        "n_rows": int(
            len(X)
        ),
        "matrix_rank_with_intercept": matrix_rank,
        "number_columns_with_intercept": number_columns,
        "full_rank": bool(
            matrix_rank
            == number_columns
        ),
        "standardized_condition_number": condition_number,
        "max_absolute_pairwise_correlation": max_abs_correlation,
        "max_VIF": max_vif,
        "VIF_method": (
            "statsmodels"
            if variance_inflation_factor is not None
            else "numpy fallback"
        ),
        "warnings": warnings,
    }

    (
        out_dir
        / "design_diagnostics.json"
    ).write_text(
        json.dumps(
            diagnostics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nDESIGN MATRIX DIAGNOSTICS",
        flush=True,
    )

    print(
        json.dumps(
            diagnostics,
            indent=2,
        ),
        flush=True,
    )

    print(
        "\nPredictor correlations:",
        flush=True,
    )

    print(
        corr.to_string(),
        flush=True,
    )

    print(
        "\nVIF:",
        flush=True,
    )

    print(
        vif_df.to_string(
            index=False
        ),
        flush=True,
    )

    if matrix_rank < number_columns:
        raise ValueError(
            "Design matrix is exactly rank deficient; "
            "do not fit the model."
        )

    if not np.isfinite(
        max_vif
    ):
        raise ValueError(
            "At least one VIF is infinite; "
            "do not fit the model."
        )

    return diagnostics


def build_model(
    data: pd.DataFrame,
):
    """
    Joint direct-contrast ES model.

        v = b0
            + bA * AttentionW
            + bI * InattentionW
            + deltaA * AttentionContrast
            + deltaI * InattentionContrast

    Primary parameters:
        deltaA = attended E - attended S weight
        deltaI = unattended E - unattended S weight
    """

    v_reg = {
        "model": (
            "v ~ 1 "
            "+ AttentionW "
            "+ InattentionW "
            "+ AttentionContrast "
            "+ InattentionContrast"
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
    np.random.seed(
        seed + chain
    )

    print(
        "\n" + "=" * 72,
        flush=True,
    )

    print(
        f"STARTING JOINT E/S CONTRAST CHAIN {chain}",
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

    model.sample(
        samples,
        burn=burn,
        dbname=str(
            db_path
        ),
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
        str(
            hddm_path
        )
    )

    with open(
        pkl_path,
        "wb",
    ) as handle:

        dill.dump(
            model,
            handle,
            recurse=True,
        )

    print(
        f"Finished chain {chain}: {hddm_path}",
        flush=True,
    )

    return str(
        hddm_path
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit the ES joint direct-contrast aDDM, estimating attended and "
            "unattended E-vs-S effects simultaneously."
        )
    )

    parser.add_argument(
        "--data",
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

    parser.add_argument(
        "--design-only",
        action="store_true",
        help=(
            "Run input validation and design-matrix diagnostics, "
            "save them, then stop before MCMC."
        ),
    )

    args = parser.parse_args()

    if args.burn >= args.samples:
        raise ValueError(
            "--burn must be smaller than --samples."
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

    data["subj_idx"] = (
        data["subj_idx"]
        .astype(int)
    )

    data["response"] = (
        data["response"]
        .astype(int)
    )

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

    design_dir = (
        args.model_dir
        / "design_diagnostics"
    )

    compute_design_diagnostics(
        data,
        design_dir,
    )

    if args.design_only:
        print(
            "\nDesign-only requested: no HDDM sampling started.",
            flush=True,
        )
        return

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
        "\nAll joint E/S contrast chains finished.",
        flush=True,
    )


if __name__ == "__main__":
    main()
