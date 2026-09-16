from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


EXCLUDED_SUBJECTS = {1, 4, 5, 6, 14, 99}
ROUND_DECIMALS = 3


def numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def assert_probability_scale(df):
    p = pd.concat(
        [
            pd.to_numeric(df["p1"], errors="coerce"),
            pd.to_numeric(df["p2"], errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()

    if p.empty:
        raise ValueError("p1/p2 contain no usable probabilities.")

    pmin = float(p.min())
    pmax = float(p.max())

    if pmin < -1e-9 or pmax > 1.0 + 1e-9:
        raise ValueError(
            "Expected probabilities on the 0..1 scale, "
            f"but found min={pmin:.6f}, max={pmax:.6f}."
        )


def validate_analysisready(df):
    required = {
        "sub_id", "phase", "trial", "cho",
        "op1", "op2", "p1", "p2",
        "DwellLeft", "DwellRight", "DwellTotal",
        "rtime",
    }

    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            "Missing required AnalysisReady columns: "
            f"{sorted(missing)}"
        )

    es = df.loc[df["phase"].eq("ES")].copy()

    if es.empty:
        raise ValueError("No ES trials found.")

    valid_identity = (
        (es["op1"].eq("E") & es["op2"].eq("S"))
        |
        (es["op1"].eq("S") & es["op2"].eq("E"))
    )

    if not valid_identity.all():
        raise ValueError(
            "Some ES rows do not contain exactly one E and one S."
        )

    assert_probability_scale(es)


def prepare_es_identity(
    data_path: Path,
    min_rt: float = 0.250,
):
    """
    Prepare an ES-only aDDM in a stable IDENTITY response coordinate:

        upper boundary / response=1 = choose S
        lower boundary / response=0 = choose E

    IMPORTANT:
        Equal-value trials are RETAINED.

    Identity-aligned values and gaze:
        V_S, V_E
        g_S = PropDwell_S
        g_E = PropDwell_E

    Conventional identity-coordinate aDDM regressors:

        AttentionW_SE
            = g_S * V_S - g_E * V_E

        InattentionW_SE
            = g_E * V_S - g_S * V_E

    With positive coefficients, positive drift points toward S.

    Identity-specific signed components:

        Attended_S   = + g_S * V_S
        Attended_E   = - g_E * V_E

        Unattended_S = + g_E * V_S
        Unattended_E = - g_S * V_E

    so:

        AttentionW_SE   = Attended_S + Attended_E
        InattentionW_SE = Unattended_S + Unattended_E

    Direct identity contrasts:

        AttentionContrast_SE
            = (Attended_S - Attended_E) / 2

        InattentionContrast_SE
            = (Unattended_S - Unattended_E) / 2

    In the fitted model:

        deltaA = bAS - bAE
        deltaI = bIS - bIE

    Therefore:
        deltaA > 0 -> attended S value receives stronger weight than attended E
        deltaA < 0 -> attended E value receives stronger weight than attended S

        deltaI > 0 -> unattended S value receives stronger weight than unattended E
        deltaI < 0 -> unattended E value receives stronger weight than unattended S
    """

    df = pd.read_csv(data_path, low_memory=False)
    validate_analysisready(df)

    z = df.loc[df["phase"].eq("ES")].copy()
    n_phase = len(z)

    numeric(
        z,
        [
            "sub_id", "trial", "cho",
            "p1", "p2",
            "DwellLeft", "DwellRight", "DwellTotal",
            "rtime",
        ],
    )

    z = z.loc[
        ~z["sub_id"].isin(EXCLUDED_SUBJECTS)
    ].copy()
    n_after_excl = len(z)

    # --------------------------------------------------------------
    # Physical values and gaze.
    # Physical coordinates are NOT reordered.
    # --------------------------------------------------------------

    z["p_left_model"] = (
        z["p1"].clip(0.0, 1.0).round(ROUND_DECIMALS)
    )
    z["p_right_model"] = (
        z["p2"].clip(0.0, 1.0).round(ROUND_DECIMALS)
    )

    valid_total = z["DwellTotal"] > 0

    z["PropDwell_Left_model"] = np.nan
    z["PropDwell_Right_model"] = np.nan

    z.loc[
        valid_total,
        "PropDwell_Left_model",
    ] = (
        z.loc[valid_total, "DwellLeft"]
        / z.loc[valid_total, "DwellTotal"]
    )

    z.loc[
        valid_total,
        "PropDwell_Right_model",
    ] = (
        z.loc[valid_total, "DwellRight"]
        / z.loc[valid_total, "DwellTotal"]
    )

    z["PropDwell_Left_model"] = (
        z["PropDwell_Left_model"]
        .clip(0.0, 1.0)
        .round(ROUND_DECIMALS)
    )

    z["PropDwell_Right_model"] = (
        z["PropDwell_Right_model"]
        .clip(0.0, 1.0)
        .round(ROUND_DECIMALS)
    )

    # --------------------------------------------------------------
    # Map physical left/right into E/S identity coordinates.
    # --------------------------------------------------------------

    e_left = z["op1"].eq("E")
    s_left = z["op1"].eq("S")

    z["V_E"] = np.where(
        e_left,
        z["p_left_model"],
        z["p_right_model"],
    )

    z["V_S"] = np.where(
        s_left,
        z["p_left_model"],
        z["p_right_model"],
    )

    z["PropDwell_E"] = np.where(
        e_left,
        z["PropDwell_Left_model"],
        z["PropDwell_Right_model"],
    )

    z["PropDwell_S"] = np.where(
        s_left,
        z["PropDwell_Left_model"],
        z["PropDwell_Right_model"],
    )

    for col in [
        "V_E", "V_S",
        "PropDwell_E", "PropDwell_S",
    ]:
        z[col] = (
            pd.to_numeric(z[col], errors="coerce")
            .round(ROUND_DECIMALS)
        )

    # --------------------------------------------------------------
    # Response coordinate:
    #   response=1 -> chose S -> upper boundary
    #   response=0 -> chose E -> lower boundary
    # --------------------------------------------------------------

    z["chosen_identity"] = np.nan

    z.loc[z["cho"].eq(1), "chosen_identity"] = (
        z.loc[z["cho"].eq(1), "op1"]
    )

    z.loc[z["cho"].eq(2), "chosen_identity"] = (
        z.loc[z["cho"].eq(2), "op2"]
    )

    z["response"] = np.nan

    valid_choice = z["chosen_identity"].isin(["E", "S"])

    z.loc[
        valid_choice,
        "response",
    ] = (
        z.loc[valid_choice, "chosen_identity"].eq("S")
    ).astype(int)

    z["rt"] = z["rtime"].round(ROUND_DECIMALS)
    z["subj_idx"] = z["sub_id"]

    # --------------------------------------------------------------
    # Signed E/S components. Positive drift points toward S.
    # --------------------------------------------------------------

    z["Attended_S"] = (
        z["PropDwell_S"] * z["V_S"]
    ).round(ROUND_DECIMALS)

    z["Attended_E"] = (
        -z["PropDwell_E"] * z["V_E"]
    ).round(ROUND_DECIMALS)

    z["Unattended_S"] = (
        z["PropDwell_E"] * z["V_S"]
    ).round(ROUND_DECIMALS)

    z["Unattended_E"] = (
        -z["PropDwell_S"] * z["V_E"]
    ).round(ROUND_DECIMALS)

    z["AttentionW_SE"] = (
        z["Attended_S"] + z["Attended_E"]
    ).round(ROUND_DECIMALS)

    z["InattentionW_SE"] = (
        z["Unattended_S"] + z["Unattended_E"]
    ).round(ROUND_DECIMALS)

    z["AttentionContrast_SE"] = (
        (
            z["Attended_S"] - z["Attended_E"]
        )
        / 2.0
    ).round(ROUND_DECIMALS)

    z["InattentionContrast_SE"] = (
        (
            z["Unattended_S"] - z["Unattended_E"]
        )
        / 2.0
    ).round(ROUND_DECIMALS)

    # --------------------------------------------------------------
    # Ties are explicitly retained.
    # --------------------------------------------------------------

    z["value_tie"] = (
        np.isclose(
            z["V_E"],
            z["V_S"],
            atol=1e-12,
            rtol=0,
        )
    )

    ties_after_exclusions = int(
        z["value_tie"].sum()
    )

    # --------------------------------------------------------------
    # RT filter.
    # --------------------------------------------------------------

    before_rt = len(z)

    z = z.loc[
        z["rt"].notna()
        & (z["rt"] >= float(min_rt))
    ].copy()

    rows_removed_rt = before_rt - len(z)

    # --------------------------------------------------------------
    # Required model information.
    # NO value-tie exclusion occurs here.
    # --------------------------------------------------------------

    required_model_cols = [
        "subj_idx",
        "rt",
        "response",
        "V_E",
        "V_S",
        "PropDwell_E",
        "PropDwell_S",
        "AttentionW_SE",
        "InattentionW_SE",
        "AttentionContrast_SE",
        "InattentionContrast_SE",
    ]

    before_missing = len(z)

    z = z.dropna(
        subset=required_model_cols
    ).copy()

    rows_removed_missing = before_missing - len(z)

    if not z["response"].isin([0, 1]).all():
        raise ValueError(
            "Final response coding must contain only 0/1."
        )

    if not np.allclose(
        z["PropDwell_E"] + z["PropDwell_S"],
        1.0,
        atol=0.002,
        rtol=0,
    ):
        raise ValueError(
            "PropDwell_E + PropDwell_S does not sum approximately to 1."
        )

    # Validate component reconstruction.
    att_sum_diff = np.abs(
        (
            z["Attended_S"] + z["Attended_E"]
        )
        - z["AttentionW_SE"]
    )

    inatt_sum_diff = np.abs(
        (
            z["Unattended_S"] + z["Unattended_E"]
        )
        - z["InattentionW_SE"]
    )

    att_contrast_diff = np.abs(
        (
            (
                z["Attended_S"] - z["Attended_E"]
            )
            / 2.0
        ).round(ROUND_DECIMALS)
        - z["AttentionContrast_SE"]
    )

    inatt_contrast_diff = np.abs(
        (
            (
                z["Unattended_S"] - z["Unattended_E"]
            )
            / 2.0
        ).round(ROUND_DECIMALS)
        - z["InattentionContrast_SE"]
    )

    ties_final = z.loc[z["value_tie"]].copy()

    audit = {
        "phase": "ES",
        "response_coordinate": {
            "upper_boundary_response_1": "S",
            "lower_boundary_response_0": "E",
        },
        "phase_rows_before_exclusions": int(n_phase),
        "rows_after_subject_exclusions": int(n_after_excl),

        "value_ties_removed": 0,
        "value_ties_retained_before_other_filters": ties_after_exclusions,
        "value_ties_retained_final": int(
            ties_final.shape[0]
        ),

        "rows_removed_rt": int(rows_removed_rt),
        "rows_removed_missing_model_columns": int(
            rows_removed_missing
        ),

        "final_rows": int(len(z)),
        "final_participants": int(
            z["subj_idx"].nunique()
        ),

        "response_0_E_n": int(
            (z["response"] == 0).sum()
        ),
        "response_1_S_n": int(
            (z["response"] == 1).sum()
        ),

        "overall_S_choice_fraction": float(
            z["response"].mean()
        ),

        "tie_S_choice_fraction": (
            float(
                ties_final["response"].mean()
            )
            if len(ties_final)
            else None
        ),

        "tie_mean_PropDwell_S": (
            float(
                ties_final["PropDwell_S"].mean()
            )
            if len(ties_final)
            else None
        ),

        "S_left_fraction": float(
            z["op1"].eq("S").mean()
        ),

        "probability_scale_min": float(
            min(
                z["V_E"].min(),
                z["V_S"].min(),
            )
        ),

        "probability_scale_max": float(
            max(
                z["V_E"].max(),
                z["V_S"].max(),
            )
        ),

        "max_abs_attention_component_reconstruction_diff": float(
            att_sum_diff.max()
        ),

        "max_abs_inattention_component_reconstruction_diff": float(
            inatt_sum_diff.max()
        ),

        "max_abs_attention_contrast_formula_diff": float(
            att_contrast_diff.max()
        ),

        "max_abs_inattention_contrast_formula_diff": float(
            inatt_contrast_diff.max()
        ),

        "round_decimals": ROUND_DECIMALS,
    }

    keep = [
        "subj_idx", "rt", "response",
        "phase", "trial", "sub_id",
        "cho", "op1", "op2",
        "chosen_identity",

        "p_left_model", "p_right_model",
        "PropDwell_Left_model", "PropDwell_Right_model",

        "V_E", "V_S",
        "PropDwell_E", "PropDwell_S",
        "value_tie",

        "Attended_S", "Attended_E",
        "Unattended_S", "Unattended_E",

        "AttentionW_SE",
        "InattentionW_SE",
        "AttentionContrast_SE",
        "InattentionContrast_SE",
    ]

    return z[keep].copy(), audit


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare ES aDDM input using a stable identity coordinate: "
            "S=upper boundary, E=lower boundary. Equal-value trials are retained."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--audit",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--min-rt",
        type=float,
        default=0.250,
    )

    args = parser.parse_args()

    clean, audit = prepare_es_identity(
        data_path=args.data,
        min_rt=args.min_rt,
    )

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.audit.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean.to_csv(
        args.out,
        index=False,
        float_format="%.3f",
    )

    args.audit.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            audit,
            indent=2,
        )
    )

    print(
        "\nFirst rows:"
    )

    print(
        clean[
            [
                "subj_idx",
                "trial",
                "response",
                "chosen_identity",
                "V_E",
                "V_S",
                "PropDwell_E",
                "PropDwell_S",
                "value_tie",
                "AttentionW_SE",
                "InattentionW_SE",
                "AttentionContrast_SE",
                "InattentionContrast_SE",
            ]
        ].head(12)
    )

    print(
        f"\nSaved model input to: {args.out}"
    )

    print(
        f"Saved audit to: {args.audit}"
    )


if __name__ == "__main__":
    main()
