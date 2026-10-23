#!/usr/bin/env python3
"""
Targeted source-specific posterior-predictive checks for the ES phase.

This script DOES NOT fit HDDM and DOES NOT generate new PPC draws.
It reuses the trial-level posterior-predictive files already created by the
four model-specific PPC jobs:

    basic_noz
    basic_z
    contrast_noz  (ESaDDM)
    contrast_z    (ESaDDM + z)

Two targeted empirical signatures are tested:

1) Unattended-source value sensitivity
   ------------------------------------
   "Low gaze" for E or S is defined as the participant-specific bottom
   quantile of PropDwell_E or PropDwell_S (default: lowest 20%).

   Within low-E-gaze trials:
       response_S ~ V_E + V_S + participant fixed effects
   E sensitivity is reported as -beta(V_E), so larger positive values mean
   stronger influence of E value on choosing E.

   Within low-S-gaze trials:
       response_S ~ V_E + V_S + participant fixed effects
   S sensitivity is reported as +beta(V_S), so larger positive values mean
   stronger influence of S value on choosing S.

   Values are standardized using ONE pooled SD calculated from V_E and V_S
   together, preserving direct comparability of E and S sensitivities.
   The reported coefficient is a linear-probability PPC summary in
   percentage points per pooled value SD.

   The empirical statistic and exactly the same statistic from every PPC draw
   are compared for all four models. The E-minus-S asymmetry is also saved.

2) Value psychometric curves conditional on gaze state
   ---------------------------------------------------
   Gaze state is defined participant-by-participant from DwellPropAdvantage_ES
   (= PropDwell_S - PropDwell_E):
       lower third  -> E-heavy gaze
       middle third -> middle gaze
       upper third  -> S-heavy gaze

   Relative value is DeltaV = V_S - V_E. DeltaV is split into five global
   quantile bins based only on the observed fitted trials. Gaze state and
   DeltaV bins are then held FIXED for every PPC draw.

   For each gaze state x DeltaV bin, the script compares observed P(choose S)
   with the four models' posterior-predictive distributions.

This is an in-sample posterior-predictive diagnostic, NOT an out-of-sample
model-selection criterion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = [
    ("basic_noz", "Basic aDDM"),
    ("basic_z", "Basic aDDM + z"),
    ("contrast_noz", "ESaDDM"),
    ("contrast_z", "ESaDDM + z"),
]

MODEL_FOLDER = "es_identity_s_upper_{}_main"

MODEL_COLORS = {
    "basic_noz": "C0",
    "basic_z": "C1",
    "contrast_noz": "C2",
    "contrast_z": "C3",
}

GAZE_STATES = ["E-heavy", "Middle", "S-heavy"]
N_VALUE_BINS = 5


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def numeric(s):
    return pd.to_numeric(s, errors="coerce")


def quantile_interval(x, q=(0.025, 0.975)):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return (np.nan, np.nan)
    return tuple(np.quantile(a, q))


def load_metadata(root: Path) -> dict:
    p = root / "metadata.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing PPC metadata: {p}")
    return json.loads(p.read_text())


def load_observed_replot(root: Path) -> pd.DataFrame:
    p = root / "tables" / "observed_replot_trials.csv.gz"
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing {p}. The PPC must have been run with --save-replot-data."
        )
    cols = [
        "subj_idx", "sub_id", "phase", "trial", "response", "rt",
        "DwellPropAdvantage_ES",
    ]
    d = pd.read_csv(p, usecols=cols, low_memory=False)
    d["phase"] = d["phase"].astype(str).str.strip()
    for c in ["subj_idx", "sub_id", "trial", "response", "rt",
              "DwellPropAdvantage_ES"]:
        d[c] = numeric(d[c])
    if d.duplicated(["sub_id", "phase", "trial"]).any():
        raise ValueError(f"Duplicate observed trial identity in {p}")
    if set(d["phase"].dropna().unique()) != {"ES"}:
        raise ValueError(f"Expected ES only in {p}")
    return d.sort_values(["sub_id", "phase", "trial"]).reset_index(drop=True)


def audit_four_models(ppc_root: Path, study: str):
    roots = {k: ppc_root / MODEL_FOLDER.format(k) for k, _ in MODELS}
    ref = None
    rows = []

    for key, label in MODELS:
        root = roots[key]
        meta = load_metadata(root)

        if meta.get("study") != study:
            raise ValueError(
                f"{key}: expected metadata study={study}, got {meta.get('study')}"
            )
        if meta.get("response_coordinate") != \
                "response=1 S upper; response=0 E lower":
            raise ValueError(f"{key}: inconsistent response coordinate")
        if int(meta.get("detected_total_ppc_draws", -1)) != 1000:
            raise ValueError(
                f"{key}: expected 1000 PPC draws, "
                f"got {meta.get('detected_total_ppc_draws')}"
            )
        if meta.get("participant_mapping_audit", {}).get("status") != "passed":
            raise ValueError(f"{key}: participant mapping audit did not pass")

        obs = load_observed_replot(root)
        if ref is None:
            ref = obs
        else:
            cols = [
                "subj_idx", "sub_id", "phase", "trial", "response", "rt",
                "DwellPropAdvantage_ES",
            ]
            a = ref[cols].copy()
            b = obs[cols].copy()
            if len(a) != len(b):
                raise ValueError(f"{key}: observed row count differs across models")
            for c in cols:
                if c == "phase":
                    good = a[c].astype(str).equals(b[c].astype(str))
                else:
                    good = np.allclose(
                        numeric(a[c]).to_numpy(float),
                        numeric(b[c]).to_numpy(float),
                        equal_nan=True, atol=1e-8, rtol=0,
                    )
                if not good:
                    raise ValueError(
                        f"{key}: observed fitted trials differ in column {c}"
                    )

        rows.append({
            "model": key,
            "label": label,
            "n_observed_trials": len(obs),
            "n_participants": obs["subj_idx"].nunique(),
            "n_ppc_draws": int(meta["detected_total_ppc_draws"]),
        })

    return roots, ref, pd.DataFrame(rows)


def merge_exact_model_features(obs: pd.DataFrame, model_input: Path):
    if not model_input.is_file():
        raise FileNotFoundError(f"Missing exact S-upper model input: {model_input}")

    need = [
        "subj_idx", "sub_id", "phase", "trial",
        "V_E", "V_S", "PropDwell_E", "PropDwell_S",
    ]
    mi = pd.read_csv(model_input, usecols=need, low_memory=False)
    mi["phase"] = mi["phase"].astype(str).str.strip()
    for c in [x for x in need if x != "phase"]:
        mi[c] = numeric(mi[c])

    if mi.duplicated(["sub_id", "phase", "trial"]).any():
        raise ValueError("Exact model input has duplicate sub_id+phase+trial keys")

    keys = ["sub_id", "phase", "trial"]
    keep = keys + ["V_E", "V_S", "PropDwell_E", "PropDwell_S"]

    out = obs.merge(
        mi[keep],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    required = ["V_E", "V_S", "PropDwell_E", "PropDwell_S"]
    if out[required].isna().any().any():
        n = out[required].isna().any(axis=1).sum()
        raise ValueError(f"{n} observed fitted rows did not match exact model input")

    gaze_sum = out["PropDwell_E"] + out["PropDwell_S"]
    if not np.allclose(gaze_sum, 1.0, atol=0.003, rtol=0):
        raise ValueError("PropDwell_E + PropDwell_S does not equal ~1")

    recon = out["PropDwell_S"] - out["PropDwell_E"]
    if not np.allclose(
        recon,
        out["DwellPropAdvantage_ES"],
        atol=0.003,
        rtol=0,
        equal_nan=False,
    ):
        raise ValueError(
            "Exact model-input gaze does not reproduce DwellPropAdvantage_ES"
        )

    out["deltaV"] = out["V_S"] - out["V_E"]

    # One common location/scale for BOTH sources, preserving E-vs-S comparison.
    pooled = pd.concat([out["V_E"], out["V_S"]], ignore_index=True).to_numpy(float)
    pooled_mean = float(np.mean(pooled))
    pooled_sd = float(np.std(pooled, ddof=0))
    if not np.isfinite(pooled_sd) or pooled_sd <= 0:
        raise ValueError("Invalid pooled value SD")

    out["V_E_z"] = (out["V_E"] - pooled_mean) / pooled_sd
    out["V_S_z"] = (out["V_S"] - pooled_mean) / pooled_sd

    return out, pooled_mean, pooled_sd


# ---------------------------------------------------------------------
# Fixed empirical gaze definitions
# ---------------------------------------------------------------------

def add_low_gaze_flags(d: pd.DataFrame, q: float):
    out = d.copy()
    out["low_E_gaze"] = False
    out["low_S_gaze"] = False

    for _, ix in out.groupby("subj_idx").groups.items():
        ix = np.asarray(list(ix))
        e_thr = float(out.loc[ix, "PropDwell_E"].quantile(q))
        s_thr = float(out.loc[ix, "PropDwell_S"].quantile(q))
        out.loc[ix, "low_E_gaze"] = out.loc[ix, "PropDwell_E"] <= e_thr
        out.loc[ix, "low_S_gaze"] = out.loc[ix, "PropDwell_S"] <= s_thr

    return out


def add_gaze_states(d: pd.DataFrame):
    out = d.copy()
    state = pd.Series(index=out.index, dtype="object")

    for _, ix in out.groupby("subj_idx").groups.items():
        ix = np.asarray(list(ix))
        x = out.loc[ix, "DwellPropAdvantage_ES"]
        q1 = float(x.quantile(1 / 3))
        q2 = float(x.quantile(2 / 3))
        state.loc[ix] = np.where(
            x <= q1,
            "E-heavy",
            np.where(x >= q2, "S-heavy", "Middle"),
        )

    out["gaze_state"] = pd.Categorical(
        state, categories=GAZE_STATES, ordered=True
    )
    return out


def add_delta_value_bins(d: pd.DataFrame, n_bins: int = N_VALUE_BINS):
    out = d.copy()

    # Quantile edges are defined ONCE from observed fitted trials.
    _, edges = pd.qcut(
        out["deltaV"],
        q=n_bins,
        retbins=True,
        duplicates="drop",
    )
    edges = np.unique(edges)
    if len(edges) - 1 != n_bins:
        raise ValueError(
            f"Expected {n_bins} distinct DeltaV quantile bins, "
            f"but observed values permit only {len(edges)-1}. Edges={edges}"
        )

    # Expand endpoints slightly so min/max are included deterministically.
    edges[0] = -np.inf
    edges[-1] = np.inf

    labels = [str(i) for i in range(1, n_bins + 1)]
    out["deltaV_bin"] = pd.cut(
        out["deltaV"], bins=edges, labels=labels, include_lowest=True
    )

    centers = (
        out.groupby("deltaV_bin", observed=False)["deltaV"]
        .mean()
        .reindex(labels)
    )
    if centers.isna().any():
        raise ValueError("At least one DeltaV bin is empty")

    return out, edges, centers


# ---------------------------------------------------------------------
# Sensitivity statistic: fixed-effects linear probability coefficient
# ---------------------------------------------------------------------

def design_matrix(d: pd.DataFrame):
    # participant fixed intercepts; first participant is reference
    subj = pd.Categorical(d["subj_idx"])
    dum = pd.get_dummies(subj, drop_first=True, dtype=float)

    X = np.column_stack([
        np.ones(len(d), dtype=float),
        d["V_E_z"].to_numpy(float),
        d["V_S_z"].to_numpy(float),
        dum.to_numpy(float),
    ])
    names = ["intercept", "V_E_z", "V_S_z"] + [
        f"subj_{x}" for x in dum.columns
    ]

    # Moore-Penrose inverse handles any empty/redundant participant dummy safely.
    pinv = np.linalg.pinv(X)
    return X, pinv, names


def sensitivity_from_y(d: pd.DataFrame, y, source: str):
    X, pinv, names = design_matrix(d)
    y = np.asarray(y, dtype=float)
    beta = pinv @ y
    b_e = float(beta[names.index("V_E_z")])
    b_s = float(beta[names.index("V_S_z")])
    if source == "E":
        return -100.0 * b_e
    if source == "S":
        return +100.0 * b_s
    raise ValueError(source)


def ppc_sensitivity_distribution(ppc: pd.DataFrame, feature_ref: pd.DataFrame, source: str):
    flag = "low_E_gaze" if source == "E" else "low_S_gaze"
    ref = feature_ref.loc[feature_ref[flag]].copy()

    keycols = ["sub_id", "phase", "trial"]
    p = ppc.merge(
        ref[keycols + ["V_E_z", "V_S_z", flag]],
        on=keycols,
        how="inner",
        validate="many_to_one",
    )

    # Stable trial row index shared across PPC draws.
    trial_keys = ref[keycols].copy()
    trial_keys["trial_row"] = np.arange(len(trial_keys), dtype=int)
    p = p.merge(trial_keys, on=keycols, how="left", validate="many_to_one")

    # Build fixed design once, then calculate OLS slopes for all draws at once.
    ref2 = ref.reset_index(drop=True)
    X, pinv, names = design_matrix(ref2)

    mat = p.pivot(
        index="trial_row",
        columns="ppc_draw",
        values="response_sampled",
    ).sort_index()

    if mat.shape[0] != len(ref2):
        raise ValueError(
            f"{source}: PPC pivot has {mat.shape[0]} trial rows, expected {len(ref2)}"
        )
    if mat.isna().any().any():
        raise ValueError(f"{source}: missing response_sampled values in PPC matrix")

    B = pinv @ mat.to_numpy(float)
    b_e = B[names.index("V_E_z"), :]
    b_s = B[names.index("V_S_z"), :]

    if source == "E":
        sens = -100.0 * b_e
    else:
        sens = +100.0 * b_s

    return pd.Series(sens, index=mat.columns.astype(str), name=f"{source}_sensitivity",)

# ---------------------------------------------------------------------
# PPC loading and fixed-feature attachment
# ---------------------------------------------------------------------

def load_ppc_trials(root: Path):
    p = root / "tables" / "ppc_replot_trials.csv.gz"
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing {p}. The PPC must have been run with --save-replot-data."
        )

    use = ["ppc_draw", "sub_id", "phase", "trial", "response_sampled"]
    d = pd.read_csv(
        p,
        usecols=use,
        dtype={
            "ppc_draw": "string",
            "sub_id": "float64",
            "trial": "float64",
            "response_sampled": "float32",
        },
        low_memory=False,
    )
    d["phase"] = d["phase"].astype(str).str.strip()
    d["response_sampled"] = numeric(d["response_sampled"])
    if not d["response_sampled"].isin([0, 1]).all():
        bad = sorted(d.loc[~d["response_sampled"].isin([0, 1]), "response_sampled"].unique())
        raise ValueError(f"Simulated response is not 0/1: {bad[:20]}")
    return d


# ---------------------------------------------------------------------
# Psychometric PPC
# ---------------------------------------------------------------------

def observed_psychometric(d: pd.DataFrame):
    rows = []
    for state in GAZE_STATES:
        z = d.loc[d["gaze_state"].astype(str).eq(state)]
        for b in [str(i) for i in range(1, N_VALUE_BINS + 1)]:
            x = z.loc[z["deltaV_bin"].astype(str).eq(b)]
            rows.append({
                "gaze_state": state,
                "deltaV_bin": b,
                "n_trials": len(x),
                "n_participants": x["subj_idx"].nunique(),
                "deltaV_mean": x["deltaV"].mean(),
                "p_choose_s": x["response"].mean(),
            })
    out = pd.DataFrame(rows)
    if (out["n_trials"] == 0).any():
        raise ValueError("At least one observed gaze-state x DeltaV cell is empty")
    return out


def ppc_psychometric(ppc: pd.DataFrame, features: pd.DataFrame):
    keycols = ["sub_id", "phase", "trial"]
    feat = features[keycols + ["gaze_state", "deltaV_bin"]].copy()

    p = ppc.merge(feat, on=keycols, how="left", validate="many_to_one")
    if p[["gaze_state", "deltaV_bin"]].isna().any().any():
        raise ValueError("PPC trial rows failed to match fixed empirical features")

    cell = (
        p.groupby(
            ["ppc_draw", "gaze_state", "deltaV_bin"],
            observed=False,
        )["response_sampled"]
        .mean()
        .rename("p_choose_s")
        .reset_index()
    )

    expected = len(pd.unique(p["ppc_draw"])) * len(GAZE_STATES) * N_VALUE_BINS
    if len(cell) != expected:
        raise ValueError(
            f"Psychometric PPC has {len(cell)} draw/cells, expected {expected}"
        )

    summ = (
        cell.groupby(["gaze_state", "deltaV_bin"], observed=False)["p_choose_s"]
        .agg(
            predictive_mean="mean",
            predictive_median="median",
            predictive_pi95_low=lambda x: x.quantile(0.025),
            predictive_pi95_high=lambda x: x.quantile(0.975),
        )
        .reset_index()
    )
    return cell, summ


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(fig, stem: Path):
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_unattended_sensitivity(obs_stats, model_stats, outdir: Path):
    cats = ["E", "S"]
    x = np.arange(2, dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 6.5))

    # Slight offsets avoid overlapping error bars.
    offsets = np.linspace(-0.18, 0.18, len(MODELS))

    for off, (key, label) in zip(offsets, MODELS):
        z = model_stats.loc[model_stats["model"].eq(key)].set_index("source").reindex(cats)
        m = z["predictive_mean"].to_numpy(float)
        lo = z["predictive_pi95_low"].to_numpy(float)
        hi = z["predictive_pi95_high"].to_numpy(float)
        ax.errorbar(
            x + off,
            m,
            yerr=np.vstack([m - lo, hi - m]),
            marker="o",
            linestyle="none",
            capsize=4,
            linewidth=1.8,
            label=label,
            color=MODEL_COLORS[key],
        )

    obs = (
        obs_stats.loc[obs_stats["source"].isin(cats)]
        .set_index("source")
        .reindex(cats)["observed_sensitivity"]
        .to_numpy(float)
    )
    ax.scatter(
        x,
        obs,
        marker="s",
        s=75,
        color="black",
        label="Observed",
        zorder=10,
    )

    ax.axhline(0, color="0.6", linewidth=1, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([
        "E value\nwhen E gaze is low",
        "S value\nwhen S gaze is low",
    ])
    ax.set_ylabel("Value sensitivity\n(percentage-point choice change per pooled value SD)")
    ax.set_title("Value sensitivity when the source receives little gaze")
    ax.legend(frameon=False, ncol=2)
    style(ax)
    savefig(fig, outdir / "unattended_value_sensitivity")


def plot_unattended_asymmetry(obs_asym, asym_stats, outdir: Path):
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    x = np.arange(len(MODELS), dtype=float)

    m = asym_stats["predictive_mean"].to_numpy(float)
    lo = asym_stats["predictive_pi95_low"].to_numpy(float)
    hi = asym_stats["predictive_pi95_high"].to_numpy(float)

    for i, (key, label) in enumerate(MODELS):
        ax.errorbar(
            [x[i]],
            [m[i]],
            yerr=[[m[i] - lo[i]], [hi[i] - m[i]]],
            marker="o",
            capsize=5,
            linewidth=1.8,
            color=MODEL_COLORS[key],
        )

    ax.axhline(obs_asym, color="black", linestyle="--", linewidth=2.2,
               label=f"Observed = {obs_asym:.2f}")
    ax.axhline(0, color="0.65", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in MODELS], rotation=15, ha="right")
    ax.set_ylabel("E minus S unattended-value sensitivity\n(percentage points per pooled value SD)")
    ax.set_title("Asymmetry in value influence under low gaze")
    ax.legend(frameon=False)
    style(ax)
    savefig(fig, outdir / "unattended_value_asymmetry")


def plot_psychometric_state(state, observed, model_summaries, outdir: Path):
    obs = (
        observed.loc[observed["gaze_state"].eq(state)]
        .sort_values("deltaV_bin")
        .copy()
    )
    x = obs["deltaV_mean"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(8.8, 6.4))

    for key, label in MODELS:
        s = (
            model_summaries.loc[
                (model_summaries["model"].eq(key))
                & (model_summaries["gaze_state"].eq(state))
            ]
            .sort_values("deltaV_bin")
        )
        m = 100 * s["predictive_mean"].to_numpy(float)
        lo = 100 * s["predictive_pi95_low"].to_numpy(float)
        hi = 100 * s["predictive_pi95_high"].to_numpy(float)

        ax.plot(
            x, m, marker="o", linewidth=2, label=label,
            color=MODEL_COLORS[key]
        )
        ax.fill_between(x, lo, hi, alpha=0.10, color=MODEL_COLORS[key])

    ax.plot(
        x,
        100 * obs["p_choose_s"].to_numpy(float),
        color="black",
        linewidth=2.5,
        linestyle="--",
        marker="s",
        label="Observed",
    )

    ax.axvline(0, color="0.7", linestyle=":", linewidth=1)
    ax.set_xlabel("Relative value ΔV = V_S − V_E")
    ax.set_ylabel("P(choose S) (%)")
    ax.set_title(f"Value-to-choice function under {state.lower()} gaze")
    ax.legend(frameon=False, ncol=2)
    style(ax)

    stem = state.lower().replace("-", "_")
    savefig(fig, outdir / f"psychometric_{stem}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["study1", "study2"], required=True)
    ap.add_argument("--ppc-root", type=Path, required=True)
    ap.add_argument("--model-input", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--low-gaze-quantile", type=float, default=0.20)
    args = ap.parse_args()

    if not (0 < args.low_gaze_quantile < 0.5):
        raise ValueError("--low-gaze-quantile must lie between 0 and 0.5")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figdir = args.out_dir / "figures"
    tabdir = args.out_dir / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)

    roots, obs0, audit = audit_four_models(args.ppc_root, args.study)
    audit.to_csv(tabdir / "four_model_input_audit.csv", index=False)

    obs, pooled_mean, pooled_sd = merge_exact_model_features(
        obs0, args.model_input
    )
    obs = add_low_gaze_flags(obs, args.low_gaze_quantile)
    obs = add_gaze_states(obs)
    obs, delta_edges, delta_centers = add_delta_value_bins(obs)

    # Save empirical binning/audit table.
    gaze_counts = (
        obs.groupby(["gaze_state"], observed=False)
        .agg(n_trials=("response", "size"), n_participants=("subj_idx", "nunique"))
        .reset_index()
    )
    low_counts = pd.DataFrame([
        {
            "source": "E",
            "n_low_gaze_trials": int(obs["low_E_gaze"].sum()),
            "n_participants": int(obs.loc[obs["low_E_gaze"], "subj_idx"].nunique()),
            "mean_prop_dwell_source": float(
                obs.loc[obs["low_E_gaze"], "PropDwell_E"].mean()
            ),
        },
        {
            "source": "S",
            "n_low_gaze_trials": int(obs["low_S_gaze"].sum()),
            "n_participants": int(obs.loc[obs["low_S_gaze"], "subj_idx"].nunique()),
            "mean_prop_dwell_source": float(
                obs.loc[obs["low_S_gaze"], "PropDwell_S"].mean()
            ),
        },
    ])
    gaze_counts.to_csv(tabdir / "gaze_state_counts.csv", index=False)
    low_counts.to_csv(tabdir / "low_gaze_counts.csv", index=False)

    obs_psy = observed_psychometric(obs)
    obs_psy.to_csv(tabdir / "observed_psychometric_cells.csv", index=False)

    # Empirical low-gaze sensitivities.
    obs_e_data = obs.loc[obs["low_E_gaze"]].copy()
    obs_s_data = obs.loc[obs["low_S_gaze"]].copy()

    obs_e = sensitivity_from_y(
        obs_e_data, obs_e_data["response"].to_numpy(float), "E"
    )
    obs_s = sensitivity_from_y(
        obs_s_data, obs_s_data["response"].to_numpy(float), "S"
    )
    obs_asym = obs_e - obs_s

    observed_sensitivity = pd.DataFrame([
        {"source": "E", "observed_sensitivity": obs_e},
        {"source": "S", "observed_sensitivity": obs_s},
    ])
    observed_sensitivity.to_csv(
        tabdir / "observed_unattended_value_sensitivity.csv", index=False
    )

    sensitivity_rows = []
    asym_rows = []
    psychometric_model_rows = []
    psychometric_error_rows = []

    for key, label in MODELS:
        print(f"\nLoading targeted PPC trial data: {label}", flush=True)
        ppc = load_ppc_trials(roots[key])

        n_draws = ppc["ppc_draw"].nunique()
        if n_draws != 1000:
            raise ValueError(f"{key}: expected 1000 PPC draws, found {n_draws}")

        e_dist = ppc_sensitivity_distribution(ppc, obs, "E")
        s_dist = ppc_sensitivity_distribution(ppc, obs, "S")

        common = sorted(set(e_dist.index) & set(s_dist.index))
        if len(common) != 1000:
            raise ValueError(f"{key}: E/S sensitivity draws do not align")
        asym = e_dist.loc[common].to_numpy() - s_dist.loc[common].to_numpy()

        for source, dist, obs_val in [
            ("E", e_dist.to_numpy(), obs_e),
            ("S", s_dist.to_numpy(), obs_s),
        ]:
            lo, hi = quantile_interval(dist)
            sensitivity_rows.append({
                "model": key,
                "label": label,
                "source": source,
                "observed_sensitivity": obs_val,
                "predictive_mean": float(np.mean(dist)),
                "predictive_median": float(np.median(dist)),
                "predictive_pi95_low": lo,
                "predictive_pi95_high": hi,
                "absolute_error": float(abs(np.mean(dist) - obs_val)),
                "observed_inside_95pi": bool(lo <= obs_val <= hi),
            })

        alo, ahi = quantile_interval(asym)
        asym_rows.append({
            "model": key,
            "label": label,
            "observed_asymmetry_E_minus_S": obs_asym,
            "predictive_mean": float(np.mean(asym)),
            "predictive_median": float(np.median(asym)),
            "predictive_pi95_low": alo,
            "predictive_pi95_high": ahi,
            "absolute_error": float(abs(np.mean(asym) - obs_asym)),
            "observed_inside_95pi": bool(alo <= obs_asym <= ahi),
        })

        _, psy_summ = ppc_psychometric(ppc, obs)
        psy_summ.insert(0, "model", key)
        psy_summ.insert(1, "label", label)
        psychometric_model_rows.append(psy_summ)

        cmp = obs_psy.merge(
            psy_summ,
            on=["gaze_state", "deltaV_bin"],
            how="left",
            validate="one_to_one"
        )
        err = cmp["predictive_mean"] - cmp["p_choose_s"]
        psychometric_error_rows.append({
            "model": key,
            "label": label,
            "rmse_all_15_cells_pct": float(
                100 * np.sqrt(np.mean(err.to_numpy(float) ** 2))
            ),
            "mae_all_15_cells_pct": float(
                100 * np.mean(np.abs(err.to_numpy(float)))
            ),
            "covered_cells_95pi": int(
                (
                    (cmp["p_choose_s"] >= cmp["predictive_pi95_low"])
                    & (cmp["p_choose_s"] <= cmp["predictive_pi95_high"])
                ).sum()
            ),
            "n_cells": int(len(cmp)),
        })

        del ppc

    sens_tab = pd.DataFrame(sensitivity_rows)
    asym_tab = pd.DataFrame(asym_rows)
    psy_tab = pd.concat(psychometric_model_rows, ignore_index=True)
    psy_err = pd.DataFrame(psychometric_error_rows)

    sens_tab.to_csv(
        tabdir / "unattended_value_sensitivity_model_comparison.csv",
        index=False,
    )
    asym_tab.to_csv(
        tabdir / "unattended_value_asymmetry_model_comparison.csv",
        index=False,
    )
    psy_tab.to_csv(
        tabdir / "psychometric_model_predictions.csv",
        index=False,
    )
    psy_err.to_csv(
        tabdir / "psychometric_model_error_summary.csv",
        index=False,
    )

    plot_unattended_sensitivity(observed_sensitivity, sens_tab, figdir)
    plot_unattended_asymmetry(obs_asym, asym_tab, figdir)

    for state in GAZE_STATES:
        plot_psychometric_state(state, obs_psy, psy_tab, figdir)

    manifest = {
        "study": args.study,
        "analysis_type": "targeted source-specific PPC",
        "ppc_draws_per_model": 1000,
        "models": [k for k, _ in MODELS],
        "response_coordinate": "response=1 S; response=0 E",
        "model_input": str(args.model_input),
        "low_gaze_definition": (
            f"participant-specific bottom {100*args.low_gaze_quantile:.1f}% "
            "of source-specific proportional dwell; ties at threshold included"
        ),
        "gaze_state_definition": (
            "participant-specific lower/middle/upper thirds of "
            "DwellPropAdvantage_ES = PropDwell_S - PropDwell_E"
        ),
        "relative_value": "deltaV = V_S - V_E",
        "deltaV_bin_edges": [
            None if not np.isfinite(x) else float(x) for x in delta_edges
        ],
        "pooled_value_mean": pooled_mean,
        "pooled_value_sd": pooled_sd,
        "sensitivity_statistic": (
            "participant-fixed-effect linear probability coefficient; "
            "E sensitivity=-beta(V_E_z), S sensitivity=+beta(V_S_z), "
            "reported in percentage points per pooled value SD"
        ),
        "important_note": (
            "This is an in-sample posterior-predictive diagnostic and must not "
            "be interpreted as held-out predictive model selection."
        ),
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    readme = f"""TARGETED SOURCE-SPECIFIC PPC — {args.study}

Primary questions
1. Does E value remain influential when E receives little gaze?
2. Does S value remain influential when S receives little gaze?
3. Does the full value-to-choice function change differently under E-heavy,
   middle, and S-heavy gaze?

Main figures
- figures/unattended_value_sensitivity.png/pdf
- figures/unattended_value_asymmetry.png/pdf
- figures/psychometric_e_heavy.png/pdf
- figures/psychometric_middle.png/pdf
- figures/psychometric_s_heavy.png/pdf

Main numerical summaries
- tables/unattended_value_sensitivity_model_comparison.csv
- tables/unattended_value_asymmetry_model_comparison.csv
- tables/psychometric_model_error_summary.csv

Interpretation
These are targeted IN-SAMPLE posterior-predictive checks. They are intended
to reveal which empirical signatures each fitted architecture can reproduce,
not to replace DIC or held-out predictive comparison.
"""
    (args.out_dir / "README.txt").write_text(readme, encoding="utf-8")

    print("\nFinished targeted source-specific PPC.", flush=True)
    print("Observed unattended E sensitivity:", obs_e, flush=True)
    print("Observed unattended S sensitivity:", obs_s, flush=True)
    print("Observed E-S asymmetry:", obs_asym, flush=True)
    print("\nPsychometric model error summary:", flush=True)
    print(psy_err.to_string(index=False), flush=True)
    print(f"\nSaved to: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
