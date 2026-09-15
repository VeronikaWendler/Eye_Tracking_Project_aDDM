

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import argparse
import json

import numpy as np
import pandas as pd


EXCLUDED_SUBJECTS = {1, 4, 5, 6, 14, 99}
ALLOWED_PHASES = {"ES", "EE", "LE"}
ROUND_DECIMALS = 3


# ---------------------------------------------------------------------
# Local defaults
# ---------------------------------------------------------------------

DEFAULT_DATA = (
    Path(__file__).resolve().parent
    / "data_sets"
    / "Data_Sets_Study1"
    / "Study1_Behaviour_with_Gaze_AnalysisReady.csv"
)


# ---------------------------------------------------------------------
# Output columns
# ---------------------------------------------------------------------

BASE_MODEL_COLUMNS = [
    "subj_idx", "rt", "response",
    "AttentionW", "InattentionW",
    "phase", "trial", "sub_id",
    "cho", "BetterOption_model",
    "p_left_model", "p_right_model",
    "V_high", "V_low",
    "PropDwell_high", "PropDwell_low",
    "OVcate_2", "Abscate_2",
]

# Extra columns needed for Sebastian's option-specific-theta extension.
#
# These are created only for ES because ES contains one E and one S option.
OPTION_SPECIFIC_THETA_COLUMNS = [
    "V_E", "V_S",
    "PropDwell_E", "PropDwell_S",
    "E_sign_highlow", "S_sign_highlow",
    "InattentionW_E", "InattentionW_S",
]


@dataclass
class PrepAudit:
    phase: str
    phase_rows_before_exclusions: int
    rows_after_subject_exclusions: int
    value_ties_removed: int
    rows_removed_rt: int
    rows_removed_missing_model_columns: int
    final_rows: int
    final_participants: int
    response_0_n: int
    response_1_n: int

    es_E_left_fraction: float | None = None
    probability_scale_min: float | None = None
    probability_scale_max: float | None = None

    option_specific_theta: bool = False
    option_specific_attention_max_abs_diff: float | None = None
    option_specific_inattention_max_abs_diff: float | None = None

    round_decimals: int = ROUND_DECIMALS


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def _numeric(
    df: pd.DataFrame,
    cols: Iterable[str],
) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )


def _assert_probability_scale(
    df: pd.DataFrame,
) -> None:
    """
    This pipeline uses probability itself as value on the 0..1 scale:

        10% -> 0.10
        80% -> 0.80

    We deliberately FAIL rather than silently divide by 100.
    """

    p = pd.concat(
        [
            pd.to_numeric(
                df["p1"],
                errors="coerce",
            ),
            pd.to_numeric(
                df["p2"],
                errors="coerce",
            ),
        ],
        ignore_index=True,
    ).dropna()

    if p.empty:
        raise ValueError(
            "p1/p2 contain no usable probability values."
        )

    pmin = float(p.min())
    pmax = float(p.max())

    if pmin < -1e-9 or pmax > 1.0 + 1e-9:
        raise ValueError(
            "Expected probabilities on 0..1 scale, "
            f"but found min={pmin:.6f}, max={pmax:.6f}. "
            "For this pipeline 10% must be stored as 0.10, not 10."
        )


# ---------------------------------------------------------------------
# Coordinate validation
# ---------------------------------------------------------------------

def validate_analysisready_coordinates(
    df: pd.DataFrame,
) -> None:
    """
    Fail loudly if the input resembles the old `_re` organization.

    Correct current file:
        p1 / op1 / gaze-left  = physical LEFT
        p2 / op2 / gaze-right = physical RIGHT

    E/S identity variables are derived separately.
    Physical rows are never reordered.
    """

    required = {
        "sub_id",
        "phase",
        "trial",
        "cho",
        "op1",
        "op2",
        "p1",
        "p2",
        "DwellLeft",
        "DwellRight",
        "DwellTotal",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise KeyError(
            "Missing required AnalysisReady columns: "
            f"{sorted(missing)}"
        )

    if "coordinate_system" in df.columns:

        labels = (
            df.loc[
                df["phase"].isin(
                    ["LE", "ES", "EE"]
                ),
                "coordinate_system",
            ]
            .dropna()
            .astype(str)
            .str.lower()
            .unique()
        )

        bad = [
            x for x in labels
            if "physical" not in x
        ]

        if bad:
            raise ValueError(
                "Unexpected non-physical coordinate labels: "
                f"{bad}"
            )

    es = df[
        df["phase"].eq("ES")
    ].copy()

    if len(es):

        valid_identity = (
            (
                es["op1"].eq("E")
                & es["op2"].eq("S")
            )
            |
            (
                es["op1"].eq("S")
                & es["op2"].eq("E")
            )
        )

        if not valid_identity.all():
            raise ValueError(
                "Some ES trials do not contain exactly one E and one S."
            )

        e_left_fraction = float(
            es["op1"].eq("E").mean()
        )

        if (
            e_left_fraction < 0.02
            or e_left_fraction > 0.98
        ):
            raise ValueError(
                "ES positions look canonicalized "
                f"(E-left fraction={e_left_fraction:.4f}). "
                "This resembles the old `_re` organization."
            )

    if "chose_right_spatial" in df.columns:

        cho = pd.to_numeric(
            df["cho"],
            errors="coerce",
        )

        right = pd.to_numeric(
            df["chose_right_spatial"],
            errors="coerce",
        )

        mask = (
            df["phase"].isin(
                ["LE", "ES", "EE"]
            )
            & cho.isin([1, 2])
            & right.notna()
        )

        expected = (
            cho.loc[mask]
            .eq(2)
            .astype(int)
        )

        if not np.array_equal(
            expected.to_numpy(),
            right.loc[mask]
            .astype(int)
            .to_numpy(),
        ):
            raise ValueError(
                "chose_right_spatial does not agree "
                "with physical cho coding."
            )


# ---------------------------------------------------------------------
# Basic probability-value aDDM features
# ---------------------------------------------------------------------

def build_probability_addm_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the basic aDDM regressors directly from corrected physical data.

    IMPORTANT:
        - probability is the value (0..1)
        - no EV transform (2p-1)
        - no 0..100 values
        - no `_re`
        - high/low is derived trial-by-trial from physical p1/p2

    Basic regressors:

        AttentionW
            = g_high * V_high
            - g_low  * V_low

        InattentionW
            = g_low  * V_high
            - g_high * V_low
    """

    z = df.copy()

    _numeric(
        z,
        [
            "p1",
            "p2",
            "cho",
            "DwellLeft",
            "DwellRight",
            "DwellTotal",
            "rtime",
        ],
    )

    _assert_probability_scale(
        z
    )

    # --------------------------------------------------------------
    # Physical values
    # --------------------------------------------------------------

    z["p_left_model"] = (
        z["p1"]
        .clip(0.0, 1.0)
        .round(ROUND_DECIMALS)
    )

    z["p_right_model"] = (
        z["p2"]
        .clip(0.0, 1.0)
        .round(ROUND_DECIMALS)
    )

    # --------------------------------------------------------------
    # Recompute physical gaze proportions from audited dwell columns
    # --------------------------------------------------------------

    valid_total = (
        z["DwellTotal"] > 0
    )

    z["PropDwell_Left_model"] = np.nan
    z["PropDwell_Right_model"] = np.nan

    z.loc[
        valid_total,
        "PropDwell_Left_model",
    ] = (
        z.loc[
            valid_total,
            "DwellLeft",
        ]
        /
        z.loc[
            valid_total,
            "DwellTotal",
        ]
    )

    z.loc[
        valid_total,
        "PropDwell_Right_model",
    ] = (
        z.loc[
            valid_total,
            "DwellRight",
        ]
        /
        z.loc[
            valid_total,
            "DwellTotal",
        ]
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
    # Higher / lower value option
    # --------------------------------------------------------------

    z["BetterOption_model"] = np.nan

    z.loc[
        z["p_left_model"]
        > z["p_right_model"],
        "BetterOption_model",
    ] = 1

    z.loc[
        z["p_right_model"]
        > z["p_left_model"],
        "BetterOption_model",
    ] = 2

    left_high = (
        z["BetterOption_model"].eq(1)
    )

    right_high = (
        z["BetterOption_model"].eq(2)
    )

    z["V_high"] = np.nan
    z["V_low"] = np.nan
    z["PropDwell_high"] = np.nan
    z["PropDwell_low"] = np.nan

    z.loc[
        left_high,
        "V_high",
    ] = z.loc[
        left_high,
        "p_left_model",
    ]

    z.loc[
        left_high,
        "V_low",
    ] = z.loc[
        left_high,
        "p_right_model",
    ]

    z.loc[
        left_high,
        "PropDwell_high",
    ] = z.loc[
        left_high,
        "PropDwell_Left_model",
    ]

    z.loc[
        left_high,
        "PropDwell_low",
    ] = z.loc[
        left_high,
        "PropDwell_Right_model",
    ]

    z.loc[
        right_high,
        "V_high",
    ] = z.loc[
        right_high,
        "p_right_model",
    ]

    z.loc[
        right_high,
        "V_low",
    ] = z.loc[
        right_high,
        "p_left_model",
    ]

    z.loc[
        right_high,
        "PropDwell_high",
    ] = z.loc[
        right_high,
        "PropDwell_Right_model",
    ]

    z.loc[
        right_high,
        "PropDwell_low",
    ] = z.loc[
        right_high,
        "PropDwell_Left_model",
    ]

    for col in [
        "V_high",
        "V_low",
        "PropDwell_high",
        "PropDwell_low",
    ]:
        z[col] = (
            z[col]
            .round(ROUND_DECIMALS)
        )

    # --------------------------------------------------------------
    # Basic aDDM regressors
    # --------------------------------------------------------------

    z["AttentionW"] = (
        z["PropDwell_high"]
        * z["V_high"]
        -
        z["PropDwell_low"]
        * z["V_low"]
    ).round(
        ROUND_DECIMALS
    )

    z["InattentionW"] = (
        z["PropDwell_low"]
        * z["V_high"]
        -
        z["PropDwell_high"]
        * z["V_low"]
    ).round(
        ROUND_DECIMALS
    )

    # --------------------------------------------------------------
    # RT and response
    # --------------------------------------------------------------

    z["rt"] = (
        z["rtime"]
        .round(ROUND_DECIMALS)
    )

    # response=1 means higher-valued option chosen.
    z["response"] = np.nan

    valid_choice = (
        z["cho"].isin([1, 2])
        & z["BetterOption_model"].isin(
            [1, 2]
        )
    )

    z.loc[
        valid_choice,
        "response",
    ] = (
        z.loc[
            valid_choice,
            "cho",
        ]
        .astype(float)
        ==
        z.loc[
            valid_choice,
            "BetterOption_model",
        ]
        .astype(float)
    ).astype(int)

    z["subj_idx"] = pd.to_numeric(
        z["sub_id"],
        errors="coerce",
    )

    return z


# ---------------------------------------------------------------------
# Sebastian / Smith option-specific-theta extension
# ---------------------------------------------------------------------

def build_option_specific_theta_features(
    z: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add ES option-identity regressors for an option-specific theta model.

    This is defined only for ES trials, where every trial contains
    exactly one E option and one S option.

    The basic model is:

        v = b0
            + bA * AttentionW
            + bI * InattentionW

        theta = bI / bA

    The option-specific extension keeps ONE attended-value coefficient
    bA, but splits the unattended contribution by option identity:

        v = b0
            + bA * AttentionW
            + bE * InattentionW_E
            + bS * InattentionW_S

    giving:

        theta_E = bE / bA
        theta_S = bS / bA

    IMPORTANT INTERPRETATION:
        InattentionW_E is E's signed value contribution specifically
        when E is unattended (i.e. gaze is on S).

        InattentionW_S is S's signed value contribution specifically
        when S is unattended (i.e. gaze is on E).

    The signs preserve the current high-vs-low response coordinate:
        response = 1 means the higher-valued option was chosen.
    """

    z = z.copy()

    phases = set(
        z["phase"]
        .dropna()
        .astype(str)
        .unique()
    )

    if phases != {"ES"}:
        raise ValueError(
            "Option-specific theta features are currently defined only "
            f"for ES. Found phases: {sorted(phases)}"
        )

    valid_identity = (
        (
            z["op1"].eq("E")
            & z["op2"].eq("S")
        )
        |
        (
            z["op1"].eq("S")
            & z["op2"].eq("E")
        )
    )

    if not valid_identity.all():
        raise ValueError(
            "Option-specific theta preparation requires exactly "
            "one E and one S on every ES trial."
        )

    e_left = (
        z["op1"].eq("E")
    )

    s_left = (
        z["op1"].eq("S")
    )

    # --------------------------------------------------------------
    # Identity-aligned values
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Identity-aligned gaze
    # --------------------------------------------------------------

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
        "V_E",
        "V_S",
        "PropDwell_E",
        "PropDwell_S",
    ]:
        z[col] = (
            pd.to_numeric(
                z[col],
                errors="coerce",
            )
            .round(ROUND_DECIMALS)
        )

    # --------------------------------------------------------------
    # High / low sign in the SAME response coordinate as the basic model
    #
    # +1 = this identity is the higher-valued option
    # -1 = this identity is the lower-valued option
    # --------------------------------------------------------------

    z["E_sign_highlow"] = np.where(
        z["V_E"] > z["V_S"],
        1.0,
        np.where(
            z["V_E"] < z["V_S"],
            -1.0,
            np.nan,
        ),
    )

    z["S_sign_highlow"] = (
        -z["E_sign_highlow"]
    )

    # --------------------------------------------------------------
    # Split unattended contributions
    #
    # E is unattended when gaze is on S:
    #
    #     sign_E * g_S * V_E
    #
    # S is unattended when gaze is on E:
    #
    #     sign_S * g_E * V_S
    # --------------------------------------------------------------

    z["InattentionW_E"] = (
        z["E_sign_highlow"]
        * z["PropDwell_S"]
        * z["V_E"]
    ).round(
        ROUND_DECIMALS
    )

    z["InattentionW_S"] = (
        z["S_sign_highlow"]
        * z["PropDwell_E"]
        * z["V_S"]
    ).round(
        ROUND_DECIMALS
    )

    return z


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_model_features(
    z: pd.DataFrame,
) -> None:
    """
    Validate the existing basic aDDM features.
    """

    needed = [
        "V_high",
        "V_low",
        "PropDwell_high",
        "PropDwell_low",
        "AttentionW",
        "InattentionW",
        "response",
    ]

    complete = (
        z.dropna(
            subset=needed
        )
        .copy()
    )

    if complete.empty:
        raise ValueError(
            "No complete aDDM model rows."
        )

    if not (
        (
            complete["V_high"] >= 0
        )
        &
        (
            complete["V_high"] <= 1
        )
    ).all():
        raise ValueError(
            "V_high left the 0..1 probability range."
        )

    if not (
        (
            complete["V_low"] >= 0
        )
        &
        (
            complete["V_low"] <= 1
        )
    ).all():
        raise ValueError(
            "V_low left the 0..1 probability range."
        )

    gaze_sum = (
        complete["PropDwell_high"]
        +
        complete["PropDwell_low"]
    )

    if not np.allclose(
        gaze_sum,
        1.0,
        atol=0.002,
        rtol=0,
    ):
        raise ValueError(
            "Rounded high+low gaze proportions "
            "do not sum approximately to 1."
        )

    att = (
        complete["PropDwell_high"]
        * complete["V_high"]
        -
        complete["PropDwell_low"]
        * complete["V_low"]
    ).round(
        ROUND_DECIMALS
    )

    inatt = (
        complete["PropDwell_low"]
        * complete["V_high"]
        -
        complete["PropDwell_high"]
        * complete["V_low"]
    ).round(
        ROUND_DECIMALS
    )

    if not np.array_equal(
        complete["AttentionW"].to_numpy(),
        att.to_numpy(),
    ):
        raise ValueError(
            "AttentionW formula validation failed."
        )

    if not np.array_equal(
        complete["InattentionW"].to_numpy(),
        inatt.to_numpy(),
    ):
        raise ValueError(
            "InattentionW formula validation failed."
        )


def validate_option_specific_theta_features(
    z: pd.DataFrame,
) -> tuple[float, float]:
    """
    Validate that the option-specific representation is only a decomposition
    of the already-audited basic aDDM regressors.

    Checks:
        1. Identity-based attended term reproduces AttentionW.
        2. InattentionW_E + InattentionW_S reproduces InattentionW
           up to rounding tolerance.

    Returns:
        max absolute AttentionW discrepancy,
        max absolute InattentionW discrepancy.
    """

    needed = [
        "V_E",
        "V_S",
        "PropDwell_E",
        "PropDwell_S",
        "E_sign_highlow",
        "S_sign_highlow",
        "AttentionW",
        "InattentionW",
        "InattentionW_E",
        "InattentionW_S",
    ]

    complete = (
        z.dropna(
            subset=needed
        )
        .copy()
    )

    if complete.empty:
        raise ValueError(
            "No complete option-specific-theta rows."
        )

    # Gaze should remain a partition of total dwell.
    gaze_sum = (
        complete["PropDwell_E"]
        +
        complete["PropDwell_S"]
    )

    if not np.allclose(
        gaze_sum,
        1.0,
        atol=0.002,
        rtol=0,
    ):
        raise ValueError(
            "Rounded E+S gaze proportions "
            "do not sum approximately to 1."
        )

    identity_attention = (
        complete["E_sign_highlow"]
        * complete["PropDwell_E"]
        * complete["V_E"]
        +
        complete["S_sign_highlow"]
        * complete["PropDwell_S"]
        * complete["V_S"]
    ).round(
        ROUND_DECIMALS
    )

    attention_diff = np.abs(
        identity_attention.to_numpy()
        -
        complete["AttentionW"].to_numpy()
    )

    max_attention_diff = float(
        np.max(attention_diff)
    )

    # Attention should reproduce exactly after the same 3-decimal rounding.
    if max_attention_diff > 1e-12:
        raise ValueError(
            "Option-specific identity coding does not reproduce "
            f"AttentionW exactly. Max difference={max_attention_diff:.6f}"
        )

    split_inattention = (
        complete["InattentionW_E"]
        +
        complete["InattentionW_S"]
    )

    inattention_diff = np.abs(
        split_inattention.to_numpy()
        -
        complete["InattentionW"].to_numpy()
    )

    max_inattention_diff = float(
        np.max(inattention_diff)
    )

    # Each split component is rounded separately, so allow 0.001.
    if max_inattention_diff > 0.001 + 1e-12:
        raise ValueError(
            "Option-specific unattended components do not reproduce "
            "the original InattentionW within rounding tolerance. "
            f"Max difference={max_inattention_diff:.6f}"
        )

    return (
        max_attention_diff,
        max_inattention_diff,
    )


# ---------------------------------------------------------------------
# Main preparation
# ---------------------------------------------------------------------

def prepare_addm_data(
    source: str | Path | pd.DataFrame,
    phase: str,
    min_rt: float = 0.250,
    excluded_subjects: set[int] | None = None,
    option_specific_theta: bool = False,
) -> tuple[pd.DataFrame, PrepAudit]:

    phase = phase.upper()

    if phase not in ALLOWED_PHASES:
        raise ValueError(
            f"phase must be one of {sorted(ALLOWED_PHASES)}"
        )

    if option_specific_theta and phase != "ES":
        raise ValueError(
            "--option-specific-theta is currently valid only for phase ES."
        )

    excluded_subjects = (
        EXCLUDED_SUBJECTS
        if excluded_subjects is None
        else set(excluded_subjects)
    )

    df = (
        source.copy()
        if isinstance(
            source,
            pd.DataFrame,
        )
        else pd.read_csv(
            source,
            low_memory=False,
        )
    )

    df["phase"] = (
        df["phase"]
        .astype(str)
        .str.strip()
    )

    validate_analysisready_coordinates(
        df
    )

    _assert_probability_scale(
        df
    )

    # --------------------------------------------------------------
    # Select phase and exclusions
    # --------------------------------------------------------------

    z = (
        df[
            df["phase"].eq(phase)
        ]
        .copy()
    )

    n_phase = len(
        z
    )

    z = (
        z[
            ~pd.to_numeric(
                z["sub_id"],
                errors="coerce",
            )
            .isin(
                excluded_subjects
            )
        ]
        .copy()
    )

    n_after_excl = len(
        z
    )

    # --------------------------------------------------------------
    # Build existing basic aDDM features
    # --------------------------------------------------------------

    z = build_probability_addm_features(
        z
    )

    # --------------------------------------------------------------
    # Remove equal-value trials
    # --------------------------------------------------------------

    tie = (
        z["p_left_model"].notna()
        &
        z["p_right_model"].notna()
        &
        np.isclose(
            z["p_left_model"],
            z["p_right_model"],
        )
    )

    n_ties = int(
        tie.sum()
    )

    z = (
        z[
            ~tie
        ]
        .copy()
    )

    # --------------------------------------------------------------
    # Option-specific theta extension, if requested
    # --------------------------------------------------------------

    max_attention_diff = None
    max_inattention_diff = None

    if option_specific_theta:

        z = build_option_specific_theta_features(
            z
        )

        (
            max_attention_diff,
            max_inattention_diff,
        ) = validate_option_specific_theta_features(
            z
        )

    # --------------------------------------------------------------
    # RT filter
    # --------------------------------------------------------------

    before_rt = len(
        z
    )

    z = (
        z[
            z["rt"] > min_rt
        ]
        .copy()
    )

    n_removed_rt = (
        before_rt
        -
        len(z)
    )

    # --------------------------------------------------------------
    # Drop missing model rows
    # --------------------------------------------------------------

    model_required = [
        "subj_idx",
        "rt",
        "response",
        "AttentionW",
        "InattentionW",
    ]

    if option_specific_theta:
        model_required.extend(
            [
                "V_E",
                "V_S",
                "PropDwell_E",
                "PropDwell_S",
                "InattentionW_E",
                "InattentionW_S",
            ]
        )

    before_missing = len(
        z
    )

    z = (
        z.dropna(
            subset=model_required
        )
        .copy()
    )

    n_removed_missing = (
        before_missing
        -
        len(z)
    )

    z["subj_idx"] = (
        z["subj_idx"]
        .astype(int)
    )

    z["response"] = (
        z["response"]
        .astype(int)
    )

    # --------------------------------------------------------------
    # Final validations
    # --------------------------------------------------------------

    validate_model_features(
        z
    )

    if option_specific_theta:
        (
            max_attention_diff,
            max_inattention_diff,
        ) = validate_option_specific_theta_features(
            z
        )

    if not z["response"].isin(
        [0, 1]
    ).all():
        raise ValueError(
            f"{phase}: response contains values outside 0/1."
        )

    # --------------------------------------------------------------
    # Keep concise model/audit columns
    # --------------------------------------------------------------

    columns = list(
        BASE_MODEL_COLUMNS
    )

    if option_specific_theta:
        columns.extend(
            OPTION_SPECIFIC_THETA_COLUMNS
        )

    keep = [
        c for c in columns
        if c in z.columns
    ]

    z = (
        z[
            keep
        ]
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------------
    # Audit
    # --------------------------------------------------------------

    all_p = pd.concat(
        [
            pd.to_numeric(
                df["p1"],
                errors="coerce",
            ),
            pd.to_numeric(
                df["p2"],
                errors="coerce",
            ),
        ],
        ignore_index=True,
    ).dropna()

    e_left = None

    if phase == "ES":
        es_all = (
            df[
                df["phase"].eq("ES")
            ]
        )

        e_left = float(
            es_all["op1"]
            .eq("E")
            .mean()
        )

    audit = PrepAudit(
        phase=phase,
        phase_rows_before_exclusions=n_phase,
        rows_after_subject_exclusions=n_after_excl,
        value_ties_removed=n_ties,
        rows_removed_rt=n_removed_rt,
        rows_removed_missing_model_columns=n_removed_missing,
        final_rows=len(z),
        final_participants=z["subj_idx"].nunique(),
        response_0_n=int(
            (
                z["response"] == 0
            ).sum()
        ),
        response_1_n=int(
            (
                z["response"] == 1
            ).sum()
        ),
        es_E_left_fraction=e_left,
        probability_scale_min=float(
            all_p.min()
        ),
        probability_scale_max=float(
            all_p.max()
        ),
        option_specific_theta=option_specific_theta,
        option_specific_attention_max_abs_diff=max_attention_diff,
        option_specific_inattention_max_abs_diff=max_inattention_diff,
    )

    return (
        z,
        audit,
    )


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

def save_prepared_data(
    source: str | Path,
    phase: str,
    out_csv: str | Path,
    audit_json: str | Path | None = None,
    min_rt: float = 0.250,
    option_specific_theta: bool = False,
) -> pd.DataFrame:

    clean, audit = prepare_addm_data(
        source,
        phase=phase,
        min_rt=min_rt,
        option_specific_theta=option_specific_theta,
    )

    out_csv = Path(
        out_csv
    )

    out_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean.to_csv(
        out_csv,
        index=False,
        float_format=f"%.{ROUND_DECIMALS}f",
    )

    if audit_json is not None:

        audit_json = Path(
            audit_json
        )

        audit_json.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        audit_json.write_text(
            json.dumps(
                asdict(audit),
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        json.dumps(
            asdict(audit),
            indent=2,
        )
    )

    print(
        "\nCore model columns:"
    )

    print(
        clean[
            [
                "subj_idx",
                "rt",
                "response",
                "V_high",
                "V_low",
                "PropDwell_high",
                "PropDwell_low",
                "AttentionW",
                "InattentionW",
            ]
        ]
        .head(10)
    )

    if option_specific_theta:

        print(
            "\nOption-specific theta columns:"
        )

        print(
            clean[
                [
                    "subj_idx",
                    "trial",
                    "V_E",
                    "V_S",
                    "PropDwell_E",
                    "PropDwell_S",
                    "E_sign_highlow",
                    "S_sign_highlow",
                    "InattentionW_E",
                    "InattentionW_S",
                ]
            ]
            .head(10)
        )

    print(
        f"\nSaved exact HDDM input to: {out_csv}"
    )

    return clean


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Prepare corrected Study 1 data for the probability-value aDDM, "
            "with an optional ES option-specific-theta extension."
        )
    )

    parser.add_argument(
        "--data",
        default=str(
            DEFAULT_DATA
        ),
        help=(
            "Correct Study1_Behaviour_with_Gaze_AnalysisReady.csv"
        ),
    )

    parser.add_argument(
        "--phase",
        default="ES",
        choices=[
            "ES",
            "EE",
            "LE",
        ],
    )

    parser.add_argument(
        "--out",
        default=None,
    )

    parser.add_argument(
        "--audit",
        default=None,
    )

    parser.add_argument(
        "--min-rt",
        type=float,
        default=0.250,
    )

    parser.add_argument(
        "--option-specific-theta",
        action="store_true",
        help=(
            "For ES only: add E/S identity-specific unattended-value "
            "regressors for theta_E versus theta_S modeling."
        ),
    )

    # Allows use inside VS Code / Jupyter Interactive.
    args, unknown = (
        parser.parse_known_args()
    )

    output_dir = (
        Path(__file__).resolve().parent
        / "data_sets"
        / "Data_Sets_Study1"
    )

    if args.option_specific_theta:

        default_out_name = (
            f"model_input_{args.phase}_option_specific_theta.csv"
        )

        default_audit_name = (
            f"model_input_{args.phase}_option_specific_theta_audit.json"
        )

    else:

        default_out_name = (
            f"model_input_{args.phase}.csv"
        )

        default_audit_name = (
            f"model_input_{args.phase}_audit.json"
        )

    out_file = (
        args.out
        or str(
            output_dir
            / default_out_name
        )
    )

    audit_file = (
        args.audit
        or str(
            output_dir
            / default_audit_name
        )
    )

    if unknown:
        print(
            "Ignoring Interactive/Jupyter arguments:",
            unknown,
        )

    save_prepared_data(
        args.data,
        args.phase,
        out_file,
        audit_file,
        args.min_rt,
        option_specific_theta=args.option_specific_theta,
    )
