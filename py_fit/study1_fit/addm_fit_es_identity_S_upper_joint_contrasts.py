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


MODEL_NAME = "aDDM_ES_IDENTITY_S_UPPER_JOINT_CONTRASTS"

PREDICTORS = [
    "AttentionW_SE",
    "InattentionW_SE",
    "AttentionContrast_SE",
    "InattentionContrast_SE",
]


def validate_input(data: pd.DataFrame) -> dict:
    """
    ES identity-coordinate joint E/S contrast aDDM.

    Response coordinate:
        response = 1 -> S -> upper boundary
        response = 0 -> E -> lower boundary

    Equal-value trials are retained.

    Signed identity components:
        Attended_S   = +g_S * V_S
        Attended_E   = -g_E * V_E
        Unattended_S = +g_E * V_S
        Unattended_E = -g_S * V_E

    Regressors:
        AttentionW_SE
            = Attended_S + Attended_E

        InattentionW_SE
            = Unattended_S + Unattended_E

        AttentionContrast_SE
            = (Attended_S - Attended_E) / 2

        InattentionContrast_SE
            = (Unattended_S - Unattended_E) / 2

    Model:
        v = b0
            + bA     * AttentionW_SE
            + bI     * InattentionW_SE
            + deltaA * AttentionContrast_SE
            + deltaI * InattentionContrast_SE

    Therefore:
        bAS = bA + deltaA/2
        bAE = bA - deltaA/2
        bIS = bI + deltaI/2
        bIE = bI - deltaI/2

        deltaA = bAS - bAE
        deltaI = bIS - bIE

        theta_S = bIS / bAS
        theta_E = bIE / bAE

    Positive deltaA:
        stronger attended S than attended E value sensitivity.

    Positive deltaI:
        stronger unattended S than unattended E value sensitivity.
    """

    required = [
        "subj_idx",
        "rt",
        "response",
        "phase",
        "V_E",
        "V_S",
        "PropDwell_E",
        "PropDwell_S",
        "value_tie",
        "Attended_S",
        "Attended_E",
        "Unattended_S",
        "Unattended_E",
        "AttentionW_SE",
        "InattentionW_SE",
        "AttentionContrast_SE",
        "InattentionContrast_SE",
    ]

    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(
            "Missing required columns for S-upper identity joint-contrast model: "
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
        "V_E",
        "V_S",
        "PropDwell_E",
        "PropDwell_S",
        "Attended_S",
        "Attended_E",
        "Unattended_S",
        "Unattended_E",
        "AttentionW_SE",
        "InattentionW_SE",
        "AttentionContrast_SE",
        "InattentionContrast_SE",
    ]

    for col in numeric_cols:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce",
        )

    if data[numeric_cols].isna().any().any():
        bad = data[numeric_cols].isna().sum()
        bad = bad[bad > 0].to_dict()
        raise ValueError(
            f"Missing/non-numeric required values: {bad}"
        )

    if not data["response"].isin([0, 1]).all():
        raise ValueError(
            "response must be coded 0/1 with 1=S and 0=E."
        )

    if (data["rt"] <= 0).any():
        raise ValueError("All RT values must be > 0.")

    if not np.allclose(
        data["PropDwell_E"] + data["PropDwell_S"],
        1.0,
        atol=0.002,
        rtol=0,
    ):
        raise ValueError(
            "PropDwell_E + PropDwell_S does not sum approximately to 1."
        )

    tolerance = 0.002 + 1e-9

    checks = {
        "max_abs_attention_sum_reconstruction_difference": float(
            np.max(
                np.abs(
                    (
                        data["Attended_S"]
                        + data["Attended_E"]
                    )
                    - data["AttentionW_SE"]
                )
            )
        ),
        "max_abs_inattention_sum_reconstruction_difference": float(
            np.max(
                np.abs(
                    (
                        data["Unattended_S"]
                        + data["Unattended_E"]
                    )
                    - data["InattentionW_SE"]
                )
            )
        ),
        "max_abs_attention_contrast_formula_difference": float(
            np.max(
                np.abs(
                    (
                        (
                            data["Attended_S"]
                            - data["Attended_E"]
                        )
                        / 2.0
                    ).round(3)
                    - data["AttentionContrast_SE"]
                )
            )
        ),
        "max_abs_inattention_contrast_formula_difference": float(
            np.max(
                np.abs(
                    (
                        (
                            data["Unattended_S"]
                            - data["Unattended_E"]
                        )
                        / 2.0
                    ).round(3)
                    - data["InattentionContrast_SE"]
                )
            )
        ),
    }

    for key, value in checks.items():
        if value > tolerance:
            raise ValueError(
                f"Identity regressor reconstruction failed for {key}: "
                f"{value:.6f}"
            )

    value_tie_bool = (
        data["value_tie"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "1.0"])
    )

    audit = {
        "model_name": MODEL_NAME,
        "phase": "ES",
        "coordinate": "identity",
        "upper_boundary_response_1": "S",
        "lower_boundary_response_0": "E",
        "equal_value_trials": "retained",
        "rows": int(len(data)),
        "participants": int(data["subj_idx"].nunique()),
        "response_0_E_n": int((data["response"] == 0).sum()),
        "response_1_S_n": int((data["response"] == 1).sum()),
        "value_tie_rows_retained": int(value_tie_bool.sum()),
        "drift_formula": (
            "v ~ 1 + AttentionW_SE + InattentionW_SE "
            "+ AttentionContrast_SE + InattentionContrast_SE"
        ),
        "parameter_interpretation": {
            "v_Intercept": (
                "b0: residual drift bias; positive toward S, negative toward E"
            ),
            "v_AttentionW_SE": (
                "bA: midpoint/common attended-value sensitivity"
            ),
            "v_InattentionW_SE": (
                "bI: midpoint/common unattended-value sensitivity"
            ),
            "v_AttentionContrast_SE": (
                "deltaA = bAS - bAE; positive means stronger attended S weighting"
            ),
            "v_InattentionContrast_SE": (
                "deltaI = bIS - bIE; positive means stronger unattended S weighting"
            ),
            "bAS": "bA + deltaA/2",
            "bAE": "bA - deltaA/2",
            "bIS": "bI + deltaI/2",
            "bIE": "bI - deltaI/2",
            "theta_S": "bIS / bAS",
            "theta_E": "bIE / bAE",
            "delta_theta_S_minus_E": "theta_S - theta_E",
        },
        "validation": checks,
    }

    return audit


def compute_design_diagnostics(
    data: pd.DataFrame,
    out_dir: Path,
) -> dict:
    """
    Check whether the four drift predictors are separately identifiable.
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
        out_dir / "design_predictor_correlations.csv",
        float_format="%.8f",
    )

    std = X.std(
        axis=0,
        ddof=0,
    )

    if (std <= 0).any():
        bad = std[std <= 0].index.tolist()
        raise ValueError(
            f"Zero-variance design predictor(s): {bad}"
        )

    Xz = (
        X - X.mean(axis=0)
    ) / std

    Xz_i = np.column_stack(
        [
            np.ones(len(Xz)),
            Xz.to_numpy(),
        ]
    )

    rank = int(
        np.linalg.matrix_rank(Xz_i)
    )

    ncols = int(
        Xz_i.shape[1]
    )

    cond = float(
        np.linalg.cond(Xz_i)
    )

    vif_rows = []

    if variance_inflation_factor is not None:
        X_i = np.column_stack(
            [
                np.ones(len(X)),
                X.to_numpy(),
            ]
        )

        for idx, name in enumerate(
            PREDICTORS,
            start=1,
        ):
            vif_rows.append(
                {
                    "Predictor": name,
                    "VIF": float(
                        variance_inflation_factor(
                            X_i,
                            idx,
                        )
                    ),
                }
            )

        vif_method = "statsmodels"

    else:
        for target in PREDICTORS:
            others = [
                c
                for c in PREDICTORS
                if c != target
            ]

            y = X[target].to_numpy()

            Z = np.column_stack(
                [
                    np.ones(len(X)),
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
                np.sum((y - fitted) ** 2)
            )

            ss_tot = float(
                np.sum((y - np.mean(y)) ** 2)
            )

            r2 = (
                1.0 - ss_res / ss_tot
                if ss_tot > 0
                else 1.0
            )

            vif = (
                float("inf")
                if r2 >= 1.0
                else float(
                    1.0 / (1.0 - r2)
                )
            )

            vif_rows.append(
                {
                    "Predictor": target,
                    "VIF": vif,
                }
            )

        vif_method = "numpy fallback"

    vif_df = pd.DataFrame(vif_rows)

    vif_df.to_csv(
        out_dir / "design_vif.csv",
        index=False,
        float_format="%.8f",
    )

    abs_corr = np.abs(
        corr.to_numpy()
    )

    np.fill_diagonal(
        abs_corr,
        np.nan,
    )

    max_corr = float(
        np.nanmax(abs_corr)
    )

    max_vif = float(
        vif_df["VIF"].max()
    )

    warnings = []

    if rank < ncols:
        warnings.append(
            "DESIGN MATRIX IS RANK DEFICIENT."
        )

    if max_corr >= 0.90:
        warnings.append(
            "At least one absolute pairwise predictor correlation is >= 0.90."
        )

    if max_vif >= 10.0:
        warnings.append(
            "At least one predictor has VIF >= 10."
        )

    if cond >= 30.0:
        warnings.append(
            "Standardized design condition number is >= 30."
        )

    diagnostics = {
        "predictors": PREDICTORS,
        "n_rows": int(len(X)),
        "matrix_rank_with_intercept": rank,
        "number_columns_with_intercept": ncols,
        "full_rank": bool(rank == ncols),
        "standardized_condition_number": cond,
        "max_absolute_pairwise_correlation": max_corr,
        "max_VIF": max_vif,
        "VIF_method": vif_method,
        "warnings": warnings,
    }

    (
        out_dir / "design_diagnostics.json"
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
        vif_df.to_string(index=False),
        flush=True,
    )

    if rank < ncols:
        raise ValueError(
            "Design matrix is rank deficient. Do not fit."
        )

    if not np.isfinite(max_vif):
        raise ValueError(
            "At least one VIF is infinite. Do not fit."
        )

    return diagnostics


def build_model(
    data: pd.DataFrame,
):
    """
    S is upper boundary; E is lower boundary.

    v = b0
        + bA     * AttentionW_SE
        + bI     * InattentionW_SE
        + deltaA * AttentionContrast_SE
        + deltaI * InattentionContrast_SE
    """

    v_reg = {
        "model": (
            "v ~ 1 "
            "+ AttentionW_SE "
            "+ InattentionW_SE "
            "+ AttentionContrast_SE "
            "+ InattentionContrast_SE"
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
        "\n" + "=" * 76,
        flush=True,
    )

    print(
        f"STARTING S-UPPER IDENTITY JOINT-CONTRAST CHAIN {chain}",
        flush=True,
    )

    print(
        "=" * 76,
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

    return str(hddm_path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit ES S-upper/E-lower identity joint E/S contrast aDDM "
            "while retaining equal-value trials."
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
        default=20260916,
    )

    parser.add_argument(
        "--design-only",
        action="store_true",
        help=(
            "Validate input and design matrix, then stop before MCMC."
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

    data.to_csv(
        args.model_dir
        / f"{MODEL_NAME}_MODEL_INPUT.csv",
        index=False,
        float_format="%.3f",
    )

    (
        args.model_dir
        / f"{MODEL_NAME}_AUDIT.json"
    ).write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    compute_design_diagnostics(
        data,
        args.model_dir
        / "design_diagnostics",
    )

    print(
        "\nMODEL AUDIT",
        flush=True,
    )

    print(
        json.dumps(
            audit,
            indent=2,
        ),
        flush=True,
    )

    if args.design_only:
        print(
            "\nDesign-only requested. No MCMC started.",
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
        "\nAll S-upper identity joint-contrast chains finished.",
        flush=True,
    )


if __name__ == "__main__":
    main()
