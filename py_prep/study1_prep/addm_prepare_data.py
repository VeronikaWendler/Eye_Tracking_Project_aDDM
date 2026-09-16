from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional, Union, Set, Tuple
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


# Extra columns for the complementary hypothesis:
# attended value sensitivity differs for E versus S, while the
# unattended-value coefficient remains shared.
OPTION_SPECIFIC_ATTENTION_COLUMNS = [
    "V_E", "V_S",
    "PropDwell_E", "PropDwell_S",
    "E_sign_highlow", "S_sign_highlow",
    "AttentionW_E", "AttentionW_S",
]


# Joint E/S contrast model:
# estimate attended and unattended E-vs-S differences simultaneously.
JOINT_ES_CONTRAST_COLUMNS = [
    "V_E", "V_S",
    "PropDwell_E", "PropDwell_S",
    "E_sign_highlow", "S_sign_highlow",
    "AttentionW_E", "AttentionW_S",
    "InattentionW_E", "InattentionW_S",
    "AttentionContrast", "InattentionContrast",
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

    es_E_left_fraction: Optional[float] = None
    probability_scale_min: Optional[float] = None
    probability_scale_max: Optional[float] = None

    option_specific_theta: bool = False
    option_specific_attention_max_abs_diff: Optional[float] = None
    option_specific_inattention_max_abs_diff: Optional[float] = None

    option_specific_attention: bool = False
    option_specific_attended_sum_max_abs_diff: Optional[float] = None

    joint_es_contrasts: bool = False
    joint_attention_sum_max_abs_diff: Optional[float] = None
    joint_inattention_sum_max_abs_diff: Optional[float] = None
    joint_attention_reconstruction_max_abs_diff: Optional[float] = None
    joint_inattention_reconstruction_max_abs_diff: Optional[float] = None

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
# Complementary E/S attended-weight extension
# ---------------------------------------------------------------------

def build_option_specific_attention_features(
    z: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add ES identity-specific ATTENDED-value regressors while keeping one
    shared unattended-value regressor.

    Conventional aDDM:

        v = b0
            + bA * AttentionW
            + bI * InattentionW

    Complementary extension:

        v = b0
            + bAE * AttentionW_E
            + bAS * AttentionW_S
            + bI  * InattentionW

    where:

        AttentionW_E = sign_E * g_E * V_E
        AttentionW_S = sign_S * g_S * V_S

    and therefore:

        AttentionW_E + AttentionW_S = AttentionW

    up to the deliberate 3-decimal rounding used throughout preprocessing.

    This model directly tests whether attended value sensitivity differs
    between E and S:

        delta_bA = bAE - bAS

    The unattended contribution remains the SAME single InattentionW term
    as in the conventional aDDM.
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
            "Option-specific attention features are currently defined only "
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
            "Option-specific attention preparation requires exactly "
            "one E and one S on every ES trial."
        )

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

    z["E_sign_highlow"] = np.where(
        z["V_E"] > z["V_S"],
        1.0,
        np.where(
            z["V_E"] < z["V_S"],
            -1.0,
            np.nan,
        ),
    )

    z["S_sign_highlow"] = -z["E_sign_highlow"]

    # Identity-specific ATTENDED contributions.
    z["AttentionW_E"] = (
        z["E_sign_highlow"]
        * z["PropDwell_E"]
        * z["V_E"]
    ).round(ROUND_DECIMALS)

    z["AttentionW_S"] = (
        z["S_sign_highlow"]
        * z["PropDwell_S"]
        * z["V_S"]
    ).round(ROUND_DECIMALS)

    return z



# ---------------------------------------------------------------------
# Joint E/S contrast extension
# ---------------------------------------------------------------------

def build_joint_es_contrast_features(
    z: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build BOTH E/S decompositions and parameterize them as direct contrasts.

    Start from the four identity-specific components:

        AttentionW_E
        AttentionW_S
        InattentionW_E
        InattentionW_S

    The conventional regressors are:

        AttentionW   = AttentionW_E + AttentionW_S
        InattentionW = InattentionW_E + InattentionW_S

    Define direct E-S contrasts:

        AttentionContrast
            = (AttentionW_E - AttentionW_S) / 2

        InattentionContrast
            = (InattentionW_E - InattentionW_S) / 2

    Therefore the model

        v = b0
            + bA * AttentionW
            + bI * InattentionW
            + deltaA * AttentionContrast
            + deltaI * InattentionContrast

    is algebraically equivalent to separate identity-specific coefficients:

        bAE = bA + deltaA/2
        bAS = bA - deltaA/2

        bIE = bI + deltaI/2
        bIS = bI - deltaI/2

    so:

        deltaA = bAE - bAS
        deltaI = bIE - bIS

    This lets the attended and unattended E-vs-S asymmetries be estimated
    simultaneously rather than forcing one of them to be common.
    """

    z = build_option_specific_attention_features(z)
    z = build_option_specific_theta_features(z)

    z["AttentionContrast"] = (
        (
            z["AttentionW_E"]
            - z["AttentionW_S"]
        )
        / 2.0
    ).round(ROUND_DECIMALS)

    z["InattentionContrast"] = (
        (
            z["InattentionW_E"]
            - z["InattentionW_S"]
        )
        / 2.0
    ).round(ROUND_DECIMALS)

    return z


def validate_joint_es_contrast_features(
    z: pd.DataFrame,
) -> Tuple[float, float, float, float]:
    """
    Validate the joint contrast parameterization.

    Returns:
        max |AttentionW_E + AttentionW_S - AttentionW|
        max |InattentionW_E + InattentionW_S - InattentionW|
        max |(AttentionW/2 + AttentionContrast) - AttentionW_E|
        max |(InattentionW/2 + InattentionContrast) - InattentionW_E|

    Small discrepancies up to ~0.001-0.002 are expected only because the
    pipeline deliberately writes regressors to 3 decimal places.
    """

    needed = [
        "AttentionW",
        "InattentionW",
        "AttentionW_E",
        "AttentionW_S",
        "InattentionW_E",
        "InattentionW_S",
        "AttentionContrast",
        "InattentionContrast",
    ]

    complete = z.dropna(
        subset=needed
    ).copy()

    if complete.empty:
        raise ValueError(
            "No complete rows for the joint E/S contrast model."
        )

    att_sum_diff = np.abs(
        (
            complete["AttentionW_E"]
            + complete["AttentionW_S"]
        ).to_numpy()
        - complete["AttentionW"].to_numpy()
    )

    inatt_sum_diff = np.abs(
        (
            complete["InattentionW_E"]
            + complete["InattentionW_S"]
        ).to_numpy()
        - complete["InattentionW"].to_numpy()
    )

    att_reconstructed_E = (
        complete["AttentionW"] / 2.0
        + complete["AttentionContrast"]
    )

    inatt_reconstructed_E = (
        complete["InattentionW"] / 2.0
        + complete["InattentionContrast"]
    )

    att_recon_diff = np.abs(
        att_reconstructed_E.to_numpy()
        - complete["AttentionW_E"].to_numpy()
    )

    inatt_recon_diff = np.abs(
        inatt_reconstructed_E.to_numpy()
        - complete["InattentionW_E"].to_numpy()
    )

    max_att_sum = float(np.max(att_sum_diff))
    max_inatt_sum = float(np.max(inatt_sum_diff))
    max_att_recon = float(np.max(att_recon_diff))
    max_inatt_recon = float(np.max(inatt_recon_diff))

    # Components are rounded separately, and contrasts are then rounded again.
    tolerance = 0.002 + 1e-12

    if max_att_sum > tolerance:
        raise ValueError(
            "Joint contrast validation failed: "
            "AttentionW_E + AttentionW_S does not reproduce AttentionW. "
            f"Max difference={max_att_sum:.6f}"
        )

    if max_inatt_sum > tolerance:
        raise ValueError(
            "Joint contrast validation failed: "
            "InattentionW_E + InattentionW_S does not reproduce InattentionW. "
            f"Max difference={max_inatt_sum:.6f}"
        )

    if max_att_recon > tolerance:
        raise ValueError(
            "Joint contrast validation failed for AttentionContrast. "
            f"Max E-component reconstruction difference={max_att_recon:.6f}"
        )

    if max_inatt_recon > tolerance:
        raise ValueError(
            "Joint contrast validation failed for InattentionContrast. "
            f"Max E-component reconstruction difference={max_inatt_recon:.6f}"
        )

    return (
        max_att_sum,
        max_inatt_sum,
        max_att_recon,
        max_inatt_recon,
    )


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
) -> Tuple[float, float]:
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



def validate_option_specific_attention_features(
    z: pd.DataFrame,
) -> float:
    """
    Validate that the two identity-specific attended components reproduce
    the already-audited conventional AttentionW term:

        AttentionW_E + AttentionW_S = AttentionW

    Because each split component is rounded separately to 3 decimals,
    a maximum discrepancy of 0.001 is allowed.

    The shared InattentionW column is deliberately left unchanged.
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
        "AttentionW_E",
        "AttentionW_S",
    ]

    complete = (
        z.dropna(
            subset=needed
        )
        .copy()
    )

    if complete.empty:
        raise ValueError(
            "No complete option-specific-attention rows."
        )

    gaze_sum = (
        complete["PropDwell_E"]
        + complete["PropDwell_S"]
    )

    if not np.allclose(
        gaze_sum,
        1.0,
        atol=0.002,
        rtol=0,
    ):
        raise ValueError(
            "Rounded E+S gaze proportions do not sum approximately to 1."
        )

    split_attention = (
        complete["AttentionW_E"]
        + complete["AttentionW_S"]
    )

    attention_diff = np.abs(
        split_attention.to_numpy()
        - complete["AttentionW"].to_numpy()
    )

    max_attention_diff = float(
        np.max(attention_diff)
    )

    if max_attention_diff > 0.001 + 1e-12:
        raise ValueError(
            "AttentionW_E + AttentionW_S does not reproduce the original "
            "AttentionW within rounding tolerance. "
            f"Max difference={max_attention_diff:.6f}"
        )

    return max_attention_diff


# ---------------------------------------------------------------------
# Main preparation
# ---------------------------------------------------------------------

def prepare_addm_data(
    source: Union[str, Path, pd.DataFrame],
    phase: str,
    min_rt: float = 0.250,
    excluded_subjects: Optional[Set[int]] = None,
    option_specific_theta: bool = False,
    option_specific_attention: bool = False,
    joint_es_contrasts: bool = False,
) -> Tuple[pd.DataFrame, PrepAudit]:

    phase = phase.upper()

    if phase not in ALLOWED_PHASES:
        raise ValueError(
            f"phase must be one of {sorted(ALLOWED_PHASES)}"
        )

    if option_specific_theta and phase != "ES":
        raise ValueError(
            "--option-specific-theta is currently valid only for phase ES."
        )

    if option_specific_attention and phase != "ES":
        raise ValueError(
            "--option-specific-attention is currently valid only for phase ES."
        )

    if joint_es_contrasts and phase != "ES":
        raise ValueError(
            "--joint-es-contrasts is currently valid only for phase ES."
        )

    extension_count = sum(
        [
            bool(option_specific_theta),
            bool(option_specific_attention),
            bool(joint_es_contrasts),
        ]
    )

    if extension_count > 1:
        raise ValueError(
            "Choose only one extension at a time: "
            "--option-specific-theta, --option-specific-attention, "
            "or --joint-es-contrasts."
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
    max_attended_sum_diff = None

    joint_att_sum_diff = None
    joint_inatt_sum_diff = None
    joint_att_recon_diff = None
    joint_inatt_recon_diff = None

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

    if option_specific_attention:

        z = build_option_specific_attention_features(
            z
        )

        max_attended_sum_diff = (
            validate_option_specific_attention_features(
                z
            )
        )

    if joint_es_contrasts:

        z = build_joint_es_contrast_features(
            z
        )

        (
            joint_att_sum_diff,
            joint_inatt_sum_diff,
            joint_att_recon_diff,
            joint_inatt_recon_diff,
        ) = validate_joint_es_contrast_features(
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

    if option_specific_attention:
        model_required.extend(
            [
                "V_E",
                "V_S",
                "PropDwell_E",
                "PropDwell_S",
                "AttentionW_E",
                "AttentionW_S",
            ]
        )

    if joint_es_contrasts:
        model_required.extend(
            [
                "V_E",
                "V_S",
                "PropDwell_E",
                "PropDwell_S",
                "AttentionW_E",
                "AttentionW_S",
                "InattentionW_E",
                "InattentionW_S",
                "AttentionContrast",
                "InattentionContrast",
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

    if option_specific_attention:
        max_attended_sum_diff = (
            validate_option_specific_attention_features(
                z
            )
        )

    if joint_es_contrasts:
        (
            joint_att_sum_diff,
            joint_inatt_sum_diff,
            joint_att_recon_diff,
            joint_inatt_recon_diff,
        ) = validate_joint_es_contrast_features(
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

    if option_specific_attention:
        columns.extend(
            OPTION_SPECIFIC_ATTENTION_COLUMNS
        )

    if joint_es_contrasts:
        columns.extend(
            JOINT_ES_CONTRAST_COLUMNS
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
        option_specific_attention=option_specific_attention,
        option_specific_attended_sum_max_abs_diff=max_attended_sum_diff,

        joint_es_contrasts=joint_es_contrasts,
        joint_attention_sum_max_abs_diff=joint_att_sum_diff,
        joint_inattention_sum_max_abs_diff=joint_inatt_sum_diff,
        joint_attention_reconstruction_max_abs_diff=joint_att_recon_diff,
        joint_inattention_reconstruction_max_abs_diff=joint_inatt_recon_diff,
    )

    return (
        z,
        audit,
    )


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

def save_prepared_data(
    source: Union[str, Path],
    phase: str,
    out_csv: Union[str, Path],
    audit_json: Optional[Union[str, Path]] = None,
    min_rt: float = 0.250,
    option_specific_theta: bool = False,
    option_specific_attention: bool = False,
    joint_es_contrasts: bool = False,
) -> pd.DataFrame:

    clean, audit = prepare_addm_data(
        source,
        phase=phase,
        min_rt=min_rt,
        option_specific_theta=option_specific_theta,
        option_specific_attention=option_specific_attention,
        joint_es_contrasts=joint_es_contrasts,
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

    if option_specific_attention:

        print(
            "\nOption-specific attention columns:"
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
                    "AttentionW_E",
                    "AttentionW_S",
                    "InattentionW",
                ]
            ]
            .head(10)
        )

    if joint_es_contrasts:

        print(
            "\nJoint E/S contrast columns:"
        )

        print(
            clean[
                [
                    "subj_idx",
                    "trial",
                    "AttentionW",
                    "InattentionW",
                    "AttentionW_E",
                    "AttentionW_S",
                    "InattentionW_E",
                    "InattentionW_S",
                    "AttentionContrast",
                    "InattentionContrast",
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
            "with optional ES option-specific unattended- or attended-weight extensions."
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
            "For ES only: split the unattended-value term by E/S identity."
        ),
    )

    parser.add_argument(
        "--option-specific-attention",
        action="store_true",
        help=(
            "For ES only: split the attended-value term by E/S identity "
            "while keeping one shared InattentionW term."
        ),
    )

    parser.add_argument(
        "--joint-es-contrasts",
        action="store_true",
        help=(
            "For ES only: estimate direct E-S contrasts for attended and "
            "unattended value contributions simultaneously."
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

    extension_count = sum(
        [
            bool(args.option_specific_theta),
            bool(args.option_specific_attention),
            bool(args.joint_es_contrasts),
        ]
    )

    if extension_count > 1:
        raise ValueError(
            "Choose only one extension: --option-specific-theta, "
            "--option-specific-attention, or --joint-es-contrasts."
        )

    if args.option_specific_theta:

        default_out_name = (
            f"model_input_{args.phase}_option_specific_theta.csv"
        )

        default_audit_name = (
            f"model_input_{args.phase}_option_specific_theta_audit.json"
        )

    elif args.option_specific_attention:

        default_out_name = (
            f"model_input_{args.phase}_option_specific_attention.csv"
        )

        default_audit_name = (
            f"model_input_{args.phase}_option_specific_attention_audit.json"
        )

    elif args.joint_es_contrasts:

        default_out_name = (
            f"model_input_{args.phase}_joint_es_contrasts.csv"
        )

        default_audit_name = (
            f"model_input_{args.phase}_joint_es_contrasts_audit.json"
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
        option_specific_attention=args.option_specific_attention,
        joint_es_contrasts=args.joint_es_contrasts,
    )
