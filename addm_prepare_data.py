from __future__ import annotations

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

# Local default when run from the repository root.
DEFAULT_DATA = (
    Path(__file__).resolve().parent
    / "data_sets"
    / "Data_Sets_Study1"
    / "Study1_Behaviour_with_Gaze_AnalysisReady.csv"
)

# Compact model input plus audit columns.
MODEL_COLUMNS = [
    "subj_idx", "rt", "response",
    "AttentionW", "InattentionW",
    "phase", "trial", "sub_id",
    "cho", "BetterOption_model",
    "p_left_model", "p_right_model",
    "V_high", "V_low",
    "PropDwell_high", "PropDwell_low",
    "OVcate_2", "Abscate_2",
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
    round_decimals: int = ROUND_DECIMALS


def _numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _assert_probability_scale(df: pd.DataFrame) -> None:
    """
    The file stores probability as 0..1:
      10% -> 0.10
      80% -> 0.80

    We deliberately FAIL instead of silently dividing by 100. This prevents an
    old/incorrect file with 10, 20, ..., 80 from entering the model unnoticed.
    """
    p = pd.concat(
        [
            pd.to_numeric(df["p1"], errors="coerce"),
            pd.to_numeric(df["p2"], errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()

    if p.empty:
        raise ValueError("p1/p2 contain no usable probability values.")

    pmin, pmax = float(p.min()), float(p.max())
    if pmin < -1e-9 or pmax > 1.0 + 1e-9:
        raise ValueError(
            f"Expected probabilities on 0..1 scale, but found min={pmin:.6f}, max={pmax:.6f}. "
            "For this pipeline 10% must be stored as 0.10, not 10."
        )


def validate_analysisready_coordinates(df: pd.DataFrame) -> None:
    """
    Fail loudly if ES resembles the old `_re` organization.

    Correct current file:
      p1/ev1/op1/gaze-left  = physical LEFT
      p2/ev2/op2/gaze-right = physical RIGHT
    E/S variables are derived separately and physical rows are never reordered.
    """
    required = {
        "sub_id", "phase", "trial", "cho", "op1", "op2", "p1", "p2",
        "DwellLeft", "DwellRight", "DwellTotal",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required AnalysisReady columns: {sorted(missing)}")

    if "coordinate_system" in df.columns:
        labels = (
            df.loc[df["phase"].isin(["LE", "ES", "EE"]), "coordinate_system"]
            .dropna().astype(str).str.lower().unique()
        )
        bad = [x for x in labels if "physical" not in x]
        if bad:
            raise ValueError(f"Unexpected non-physical coordinate labels: {bad}")

    es = df[df["phase"].eq("ES")].copy()
    if len(es):
        valid_identity = (
            (es["op1"].eq("E") & es["op2"].eq("S"))
            | (es["op1"].eq("S") & es["op2"].eq("E"))
        )
        if not valid_identity.all():
            raise ValueError("Some ES trials do not contain exactly one E and one S.")

        e_left_fraction = float(es["op1"].eq("E").mean())
        if e_left_fraction < 0.02 or e_left_fraction > 0.98:
            raise ValueError(
                f"ES positions look canonicalized (E-left fraction={e_left_fraction:.4f}). "
                "This resembles the old `_re` organization."
            )

    if "chose_right_spatial" in df.columns:
        cho = pd.to_numeric(df["cho"], errors="coerce")
        right = pd.to_numeric(df["chose_right_spatial"], errors="coerce")
        mask = df["phase"].isin(["LE", "ES", "EE"]) & cho.isin([1, 2]) & right.notna()
        expected = cho.loc[mask].eq(2).astype(int)
        if not np.array_equal(expected.to_numpy(), right.loc[mask].astype(int).to_numpy()):
            raise ValueError("chose_right_spatial does not agree with physical cho coding.")


def build_probability_addm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build model-specific aDDM regressors DIRECTLY from corrected physical data.

    IMPORTANT:
      - probability is the value: 0.10 means 10%
      - no EV transform (2p-1)
      - no 0..100 percentage values
      - no `_re`
      - high/low is derived trial-by-trial from physical p1/p2

    Basic aDDM regressors:
      AttentionW   = g_high * V_high - g_low * V_low
      InattentionW = g_low  * V_high - g_high * V_low
    """
    z = df.copy()

    _numeric(z, ["p1", "p2", "cho", "DwellLeft", "DwellRight", "DwellTotal", "rtime"])
    _assert_probability_scale(z)

    # Clip only to theoretically valid ranges, then round to stable precision.
    z["p_left_model"] = z["p1"].clip(0.0, 1.0).round(ROUND_DECIMALS)
    z["p_right_model"] = z["p2"].clip(0.0, 1.0).round(ROUND_DECIMALS)

    # Recompute gaze proportions from audited physical dwell columns.
    valid_total = z["DwellTotal"] > 0
    z["PropDwell_Left_model"] = np.nan
    z["PropDwell_Right_model"] = np.nan
    z.loc[valid_total, "PropDwell_Left_model"] = (
        z.loc[valid_total, "DwellLeft"] / z.loc[valid_total, "DwellTotal"]
    )
    z.loc[valid_total, "PropDwell_Right_model"] = (
        z.loc[valid_total, "DwellRight"] / z.loc[valid_total, "DwellTotal"]
    )

    z["PropDwell_Left_model"] = z["PropDwell_Left_model"].clip(0.0, 1.0).round(ROUND_DECIMALS)
    z["PropDwell_Right_model"] = z["PropDwell_Right_model"].clip(0.0, 1.0).round(ROUND_DECIMALS)

    # Ranking is PHYSICAL and trial-specific.
    z["BetterOption_model"] = np.nan
    z.loc[z["p_left_model"] > z["p_right_model"], "BetterOption_model"] = 1
    z.loc[z["p_right_model"] > z["p_left_model"], "BetterOption_model"] = 2

    left_high = z["BetterOption_model"].eq(1)
    right_high = z["BetterOption_model"].eq(2)

    z["V_high"] = np.nan
    z["V_low"] = np.nan
    z["PropDwell_high"] = np.nan
    z["PropDwell_low"] = np.nan

    z.loc[left_high, "V_high"] = z.loc[left_high, "p_left_model"]
    z.loc[left_high, "V_low"] = z.loc[left_high, "p_right_model"]
    z.loc[left_high, "PropDwell_high"] = z.loc[left_high, "PropDwell_Left_model"]
    z.loc[left_high, "PropDwell_low"] = z.loc[left_high, "PropDwell_Right_model"]

    z.loc[right_high, "V_high"] = z.loc[right_high, "p_right_model"]
    z.loc[right_high, "V_low"] = z.loc[right_high, "p_left_model"]
    z.loc[right_high, "PropDwell_high"] = z.loc[right_high, "PropDwell_Right_model"]
    z.loc[right_high, "PropDwell_low"] = z.loc[right_high, "PropDwell_Left_model"]

    # Round inputs first, then compute and round regressors.
    for col in ["V_high", "V_low", "PropDwell_high", "PropDwell_low"]:
        z[col] = z[col].round(ROUND_DECIMALS)

    z["AttentionW"] = (
        z["PropDwell_high"] * z["V_high"]
        - z["PropDwell_low"] * z["V_low"]
    ).round(ROUND_DECIMALS)

    z["InattentionW"] = (
        z["PropDwell_low"] * z["V_high"]
        - z["PropDwell_high"] * z["V_low"]
    ).round(ROUND_DECIMALS)

    # RT in seconds, to millisecond precision.
    z["rt"] = z["rtime"].round(ROUND_DECIMALS)

    # response=1 means higher-valued option chosen.
    z["response"] = np.nan
    valid_choice = z["cho"].isin([1, 2]) & z["BetterOption_model"].isin([1, 2])
    z.loc[valid_choice, "response"] = (
        z.loc[valid_choice, "cho"].astype(float)
        == z.loc[valid_choice, "BetterOption_model"].astype(float)
    ).astype(int)

    z["subj_idx"] = pd.to_numeric(z["sub_id"], errors="coerce")

    return z


def validate_model_features(z: pd.DataFrame) -> None:
    needed = [
        "V_high", "V_low", "PropDwell_high", "PropDwell_low",
        "AttentionW", "InattentionW", "response",
    ]
    complete = z.dropna(subset=needed).copy()
    if complete.empty:
        raise ValueError("No complete aDDM model rows.")

    if not ((complete["V_high"] >= 0) & (complete["V_high"] <= 1)).all():
        raise ValueError("V_high left the 0..1 probability range.")
    if not ((complete["V_low"] >= 0) & (complete["V_low"] <= 1)).all():
        raise ValueError("V_low left the 0..1 probability range.")

    # Allow tiny deviation because gaze proportions are intentionally rounded.
    gaze_sum = complete["PropDwell_high"] + complete["PropDwell_low"]
    if not np.allclose(gaze_sum, 1.0, atol=0.002, rtol=0):
        raise ValueError("Rounded high+low gaze proportions do not sum approximately to 1.")

    att = (
        complete["PropDwell_high"] * complete["V_high"]
        - complete["PropDwell_low"] * complete["V_low"]
    ).round(ROUND_DECIMALS)
    inatt = (
        complete["PropDwell_low"] * complete["V_high"]
        - complete["PropDwell_high"] * complete["V_low"]
    ).round(ROUND_DECIMALS)

    if not np.array_equal(complete["AttentionW"].to_numpy(), att.to_numpy()):
        raise ValueError("AttentionW formula validation failed.")
    if not np.array_equal(complete["InattentionW"].to_numpy(), inatt.to_numpy()):
        raise ValueError("InattentionW formula validation failed.")


def prepare_addm_data(
    source: str | Path | pd.DataFrame,
    phase: str,
    min_rt: float = 0.250,
    excluded_subjects: set[int] | None = None,
) -> tuple[pd.DataFrame, PrepAudit]:
    phase = phase.upper()
    if phase not in ALLOWED_PHASES:
        raise ValueError(f"phase must be one of {sorted(ALLOWED_PHASES)}")

    excluded_subjects = EXCLUDED_SUBJECTS if excluded_subjects is None else set(excluded_subjects)
    df = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source, low_memory=False)
    df["phase"] = df["phase"].astype(str).str.strip()

    validate_analysisready_coordinates(df)
    _assert_probability_scale(df)

    z = df[df["phase"].eq(phase)].copy()
    n_phase = len(z)

    z = z[~pd.to_numeric(z["sub_id"], errors="coerce").isin(excluded_subjects)].copy()
    n_after_excl = len(z)

    z = build_probability_addm_features(z)

    # Equal-probability trials have no higher/lower boundary.
    tie = (
        z["p_left_model"].notna()
        & z["p_right_model"].notna()
        & np.isclose(z["p_left_model"], z["p_right_model"])
    )
    n_ties = int(tie.sum())
    z = z[~tie].copy()

    before_rt = len(z)
    z = z[z["rt"] > min_rt].copy()
    n_removed_rt = before_rt - len(z)

    model_required = ["subj_idx", "rt", "response", "AttentionW", "InattentionW"]
    before_missing = len(z)
    z = z.dropna(subset=model_required).copy()
    n_removed_missing = before_missing - len(z)

    z["subj_idx"] = z["subj_idx"].astype(int)
    z["response"] = z["response"].astype(int)

    validate_model_features(z)

    if not z["response"].isin([0, 1]).all():
        raise ValueError(f"{phase}: response contains values outside 0/1.")

    keep = [c for c in MODEL_COLUMNS if c in z.columns]
    z = z[keep].reset_index(drop=True)

    all_p = pd.concat(
        [pd.to_numeric(df["p1"], errors="coerce"), pd.to_numeric(df["p2"], errors="coerce")],
        ignore_index=True,
    ).dropna()

    e_left = None
    if phase == "ES":
        es = df[df["phase"].eq("ES")]
        e_left = float(es["op1"].eq("E").mean())

    audit = PrepAudit(
        phase=phase,
        phase_rows_before_exclusions=n_phase,
        rows_after_subject_exclusions=n_after_excl,
        value_ties_removed=n_ties,
        rows_removed_rt=n_removed_rt,
        rows_removed_missing_model_columns=n_removed_missing,
        final_rows=len(z),
        final_participants=z["subj_idx"].nunique(),
        response_0_n=int((z["response"] == 0).sum()),
        response_1_n=int((z["response"] == 1).sum()),
        es_E_left_fraction=e_left,
        probability_scale_min=float(all_p.min()),
        probability_scale_max=float(all_p.max()),
    )

    return z, audit


def save_prepared_data(
    source: str | Path,
    phase: str,
    out_csv: str | Path,
    audit_json: str | Path | None = None,
    min_rt: float = 0.250,
) -> pd.DataFrame:
    clean, audit = prepare_addm_data(source, phase=phase, min_rt=min_rt)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_csv, index=False, float_format=f"%.{ROUND_DECIMALS}f")

    if audit_json is not None:
        audit_json = Path(audit_json)
        audit_json.parent.mkdir(parents=True, exist_ok=True)
        audit_json.write_text(json.dumps(asdict(audit), indent=2))

    print(json.dumps(asdict(audit), indent=2))
    print("\nModel columns:")
    print(clean[["subj_idx", "rt", "response", "V_high", "V_low",
                 "PropDwell_high", "PropDwell_low", "AttentionW", "InattentionW"]].head(10))
    print(f"\nSaved exact HDDM input to: {out_csv}")
    return clean



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare corrected Study 1 data for the basic probability-value aDDM."
    )

    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="Correct Study1_Behaviour_with_Gaze_AnalysisReady.csv"
    )

    # ES is our current default phase
    parser.add_argument(
        "--phase",
        default="ES",
        choices=["ES", "EE", "LE"]
    )

    # Leave these as None so filenames are created automatically from the phase
    parser.add_argument("--out", default=None)
    parser.add_argument("--audit", default=None)

    parser.add_argument("--min-rt", type=float, default=0.250)

    # parse_known_args() allows the script to run inside Jupyter/VS Code Interactive
    args, unknown = parser.parse_known_args()

    # default output folder
    output_dir = (
        Path(__file__).resolve().parent
        / "data_sets"
        / "Data_Sets_Study1"
    )

    # automatically name output according to phase
    out_file = args.out or str(
        output_dir / f"model_input_{args.phase}.csv"
    )

    audit_file = args.audit or str(
        output_dir / f"model_input_{args.phase}_audit.json"
    )

    if unknown:
        print("Ignoring Interactive/Jupyter arguments:", unknown)

    save_prepared_data(
        args.data,
        args.phase,
        out_file,
        audit_file,
        args.min_rt
    )
