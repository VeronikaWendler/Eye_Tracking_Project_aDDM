#!/usr/bin/env python3
"""
05 — INDIVIDUAL DIFFERENCES: GARCIA-STYLE VALUE SENSITIVITY x ESaDDM+z

Relates participant-level Garcia-style ES/EE/SP slopes to participant-level
posterior parameters from the exact best-fitting model:

A_DDM_ES_IDENTITY_S_UPPER_CONTRAST_Z_NO_INTERCEPT
response=1 -> S (upper), response=0 -> E (lower)

v ~ 0 + AttentionW_SE + InattentionW_SE
      + AttentionContrast_SE + InattentionContrast_SE

Reconstruction:
    b_AS = b_A + delta_A/2
    b_AE = b_A - delta_A/2
    b_IS = b_I + delta_I/2
    b_IE = b_I - delta_I/2
    theta_S = b_IS / b_AS
    theta_E = b_IE / b_AE

Primary behavioural individual-difference score:
    selective_vs_EE = EE slope - ES slope

Positive values mean E-value sensitivity is lower specifically in the hybrid
ES context than in the choice-based EE context. This is the cleanest continuous
individual-level analogue of Garcia-style selective experiential neglect.

Secondary behavioural scores:
    selective_vs_SP = SP - ES
    selective_composite = mean(EE, SP) - ES
    ES_raw_compression = 1 - ES

Primary mechanistic test:
    selective_vs_EE ~ theta_S - theta_E

Also planned:
    selective_vs_EE ~ delta_I
    selective_vs_EE ~ delta_A
    selective_vs_EE ~ z_S_bias

Full exploratory inspection:
    selective_vs_EE is additionally related to EVERY substantive participant-
    level parameter reconstructed from the final model (b_A, b_I, delta_A,
    delta_I, b_AS, b_AE, b_IS, b_IE, theta_S, theta_E, theta_S-theta_E,
    z, a, t). A large old-code-style correlation grid plus individual plots
    are saved for quick inspection. Secondary grids are also made for SP-ES,
    mean(EE,SP)-ES, raw ES compression, and the raw ES slope.

Statistics:
    Pearson r, Spearman rho, simple regression slope + 95% CI, R²,
    leave-one-out Pearson range, and Holm correction across planned tests.

Example:
python 05_ES_selective_neglect_addm_individual_differences.py \
    --study 1 \
    --slopes /workspace/individual_differences_input/Study1_phase_slopes_full.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


ROOTS = {
    1: Path("/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM"),
    2: Path("/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM"),
}
FIT_REL = Path("derivatives/models/es_identity_S_upper_contrast_z_final")
OUT_REL = Path("derivatives/figures/es_identity_S_upper_contrast_z_final/individual_differences")

plt.rcParams.update({
    "font.size": 17,
    "axes.titlesize": 21,
    "axes.labelsize": 19,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.titlesize": 23,
})


def norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def expit(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def p_text(p: float) -> str:
    if not np.isfinite(p):
        return "p = NA"
    return "p < .001" if p < 0.001 else f"p = {p:.3f}"


def save_figure(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adjusted_sorted[rank] = min(running, 1.0)
    out = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        out[idx] = adjusted_sorted[rank]
    return out


def load_chain_traces(model_dir: Path) -> pd.DataFrame:
    pkls = sorted(model_dir.glob("*CONTRAST_Z_NO_INTERCEPT_*.pkl"))
    if not pkls:
        pkls = sorted(model_dir.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No HDDM chain .pkl files found in:\n{model_dir}")

    traces = []
    for chain_i, p in enumerate(pkls):
        print(f"Loading chain: {p.name}")
        with open(p, "rb") as f:
            model = dill.load(f)
        if not hasattr(model, "get_traces"):
            raise AttributeError(f"{p.name}: loaded model has no get_traces()")
        tr = model.get_traces().copy()
        tr["__chain__"] = chain_i
        tr["__chain_file__"] = p.name
        traces.append(tr)

    out = pd.concat(traces, ignore_index=True)
    print(f"\nLoaded {len(pkls)} chains")
    print(f"Combined posterior draws: {len(out):,}")
    return out


SUBJ_PATTERN = re.compile(
    r"^(?P<base>.+)_subj(?:\((?P<mod>.+?)\))?\.(?P<sid>\d+)$"
)


def index_subject_trace_columns(traces: pd.DataFrame):
    out = {}
    for col in traces.columns:
        m = SUBJ_PATTERN.match(str(col))
        if not m:
            continue
        sid = int(m.group("sid"))
        base = m.group("base")
        mod = m.group("mod")
        full_base = f"{base}({mod})" if mod else base
        out.setdefault(sid, {})[norm(full_base)] = str(col)
    return out


def find_subject_col(index, sid, aliases, label, required=True):
    available = index.get(int(sid), {})
    aliases_n = [norm(a) for a in aliases]

    for a in aliases_n:
        if a in available:
            return available[a]

    hits = []
    for base_n, col in available.items():
        if any(a in base_n for a in aliases_n):
            hits.append(col)
    hits = list(dict.fromkeys(hits))

    if len(hits) == 1:
        return hits[0]
    if not required:
        return None

    raise KeyError(
        f"\nParticipant {sid}: could not identify {label}.\n"
        f"Aliases tried: {aliases}\n"
        f"Candidate hits: {hits}\n"
        f"Available participant-level bases:\n  "
        + "\n  ".join(sorted(available.keys()))
    )


def summarize_draws(x, prefix):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q025": np.nan,
            f"{prefix}_q975": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_q025": float(np.quantile(x, 0.025)),
        f"{prefix}_q975": float(np.quantile(x, 0.975)),
    }


def extract_individual_addm_parameters(traces, subject_ids):
    """Reconstruct theta draw-by-draw, then summarize each participant."""
    idx = index_subject_trace_columns(traces)
    rows = []

    missing = sorted(set(map(int, subject_ids)) - set(idx))
    if missing:
        raise KeyError(
            "Behavioural IDs with no participant-level trace columns: "
            f"{missing}"
        )

    for sid in sorted(map(int, subject_ids)):
        b_a_col = find_subject_col(
            idx, sid, ["v_AttentionW_SE", "AttentionW_SE"], "b_A"
        )
        b_i_col = find_subject_col(
            idx, sid, ["v_InattentionW_SE", "InattentionW_SE"], "b_I"
        )
        da_col = find_subject_col(
            idx, sid, ["v_AttentionContrast_SE", "AttentionContrast_SE"], "delta_A"
        )
        di_col = find_subject_col(
            idx, sid, ["v_InattentionContrast_SE", "InattentionContrast_SE"], "delta_I"
        )

        a_col = find_subject_col(idx, sid, ["a"], "a", required=False)
        t_col = find_subject_col(idx, sid, ["t"], "t", required=False)
        z_trans_col = find_subject_col(
            idx, sid, ["z_trans"], "z_trans", required=False
        )
        z_raw_col = None
        if z_trans_col is None:
            z_raw_col = find_subject_col(idx, sid, ["z"], "z", required=False)

        b_A = pd.to_numeric(traces[b_a_col], errors="coerce").to_numpy(float)
        b_I = pd.to_numeric(traces[b_i_col], errors="coerce").to_numpy(float)
        delta_A = pd.to_numeric(traces[da_col], errors="coerce").to_numpy(float)
        delta_I = pd.to_numeric(traces[di_col], errors="coerce").to_numpy(float)

        b_AS = b_A + delta_A / 2.0
        b_AE = b_A - delta_A / 2.0
        b_IS = b_I + delta_I / 2.0
        b_IE = b_I - delta_I / 2.0

        with np.errstate(divide="ignore", invalid="ignore"):
            theta_S = b_IS / b_AS
            theta_E = b_IE / b_AE

        if z_trans_col is not None:
            z0 = pd.to_numeric(traces[z_trans_col], errors="coerce").to_numpy(float)
            z = expit(z0)
            z_source = z_trans_col
        elif z_raw_col is not None:
            z0 = pd.to_numeric(traces[z_raw_col], errors="coerce").to_numpy(float)
            finite = z0[np.isfinite(z0)]
            z = expit(z0) if len(finite) and (finite.min() < 0 or finite.max() > 1) else z0
            z_source = z_raw_col
        else:
            z = np.repeat(np.nan, len(b_A))
            z_source = "NOT FOUND"

        a = (
            pd.to_numeric(traces[a_col], errors="coerce").to_numpy(float)
            if a_col is not None else np.repeat(np.nan, len(b_A))
        )
        t = (
            pd.to_numeric(traces[t_col], errors="coerce").to_numpy(float)
            if t_col is not None else np.repeat(np.nan, len(b_A))
        )

        row = {
            "sub_id": sid,
            "trace_b_A": b_a_col,
            "trace_b_I": b_i_col,
            "trace_delta_A": da_col,
            "trace_delta_I": di_col,
            "trace_z": z_source,
            "n_posterior_draws": int(len(b_A)),
            "p_abs_b_AS_lt_0p05": float(np.nanmean(np.abs(b_AS) < 0.05)),
            "p_abs_b_AE_lt_0p05": float(np.nanmean(np.abs(b_AE) < 0.05)),
        }

        quantities = {
            "b_A": b_A,
            "b_I": b_I,
            "delta_A": delta_A,
            "delta_I": delta_I,
            "b_AS": b_AS,
            "b_AE": b_AE,
            "b_IS": b_IS,
            "b_IE": b_IE,
            "theta_S": theta_S,
            "theta_E": theta_E,
            "theta_S_minus_E": theta_S - theta_E,
            "theta_E_minus_S": theta_E - theta_S,
            "z": z,
            "z_S_bias": z - 0.5,
            "z_E_bias": 0.5 - z,
            "a": a,
            "t": t,
        }

        for name, values in quantities.items():
            row.update(summarize_draws(values, name))

        row["theta_ratio_warning"] = bool(
            row["p_abs_b_AS_lt_0p05"] > 0.05
            or row["p_abs_b_AE_lt_0p05"] > 0.05
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("sub_id").reset_index(drop=True)


def read_behavioural_slopes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    if "sub_id" not in df.columns:
        raise KeyError(f"{path}: no sub_id column")
    df["sub_id"] = pd.to_numeric(df["sub_id"], errors="raise").astype(int)

    if {"ES", "EE", "SP"}.issubset(df.columns):
        df["ES"] = pd.to_numeric(df["ES"], errors="coerce")
        df["EE"] = pd.to_numeric(df["EE"], errors="coerce")
        df["SP"] = pd.to_numeric(df["SP"], errors="coerce")
    else:
        needed = {
            "ES_slope_all_trials",
            "EE_slope_all_trials",
            "SP_slope_E_only",
        }
        missing = needed - set(df.columns)
        if missing:
            raise KeyError(
                f"{path}: expected ES/EE/SP or canonical full-table columns. "
                f"Missing {sorted(missing)}"
            )
        df["ES"] = pd.to_numeric(df["ES_slope_all_trials"], errors="coerce")
        df["EE"] = pd.to_numeric(df["EE_slope_all_trials"], errors="coerce")
        df["SP"] = pd.to_numeric(df["SP_slope_E_only"], errors="coerce")

    df["ES_raw_compression"] = 1.0 - df["ES"]
    df["selective_vs_EE"] = df["EE"] - df["ES"]
    df["selective_vs_SP"] = df["SP"] - df["ES"]
    df["nonES_mean"] = df[["EE", "SP"]].mean(axis=1)
    df["selective_composite"] = df["nonES_mean"] - df["ES"]

    df["ES_lower_than_EE"] = df["ES"] < df["EE"]
    df["ES_lower_than_SP"] = df["ES"] < df["SP"]
    df["ES_lower_than_both"] = (
        df["ES_lower_than_EE"] & df["ES_lower_than_SP"]
    )
    return df


def leave_one_out_pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    vals = []
    for i in range(len(x)):
        xx = np.delete(x, i)
        yy = np.delete(y, i)
        if len(xx) >= 3 and np.std(xx) > 0 and np.std(yy) > 0:
            r_loo, _ = stats.pearsonr(xx, yy)
            vals.append(r_loo)
    return (float(np.min(vals)), float(np.max(vals))) if vals else (np.nan, np.nan)


def association_stats(df, x_col, y_col):
    d = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    x = d[x_col].to_numpy(float)
    y = d[y_col].to_numpy(float)
    n = len(d)
    if n < 5 or np.std(x) == 0 or np.std(y) == 0:
        return None

    # Tuple unpacking is compatible with both old and new SciPy versions.
    pear_r, pear_p = stats.pearsonr(x, y)
    spear_rho, spear_p = stats.spearmanr(x, y)
    reg = stats.linregress(x, y)
    tcrit = stats.t.ppf(0.975, df=n - 2)
    slope_lo = reg.slope - tcrit * reg.stderr
    slope_hi = reg.slope + tcrit * reg.stderr
    loo_lo, loo_hi = leave_one_out_pearson(x, y)

    return {
        "x": x_col, "y": y_col, "n": n,
        "pearson_r": float(pear_r),
        "pearson_p": float(pear_p),
        "spearman_rho": float(spear_rho),
        "spearman_p": float(spear_p),
        "regression_slope": float(reg.slope),
        "regression_slope_ci_low": float(slope_lo),
        "regression_slope_ci_high": float(slope_hi),
        "regression_intercept": float(reg.intercept),
        "r2": float(reg.rvalue ** 2),
        "loo_pearson_r_min": loo_lo,
        "loo_pearson_r_max": loo_hi,
    }


PARAM_LABELS = {
    "b_A_mean": r"$b_A$ (overall attended value weight)",
    "b_I_mean": r"$b_I$ (overall unattended value weight)",
    "theta_S_minus_E_mean": r"$\theta_S-\theta_E$",
    "theta_E_mean": r"$\theta_E$",
    "theta_S_mean": r"$\theta_S$",
    "delta_I_mean": r"$\delta_I$ (unattended S−E contrast)",
    "delta_A_mean": r"$\delta_A$ (attended S−E contrast)",
    "b_AS_mean": r"$b_{AS}$",
    "b_AE_mean": r"$b_{AE}$",
    "b_IS_mean": r"$b_{IS}$",
    "b_IE_mean": r"$b_{IE}$",
    "z_S_bias_mean": r"$z-.5$ (S starting bias)",
    "z_mean": r"$z$",
    "a_mean": r"$a$",
    "t_mean": r"$t$",
}

OUTCOME_LABELS = {
    "selective_vs_EE": "Selective ES compression (EE slope − ES slope)",
    "selective_vs_SP": "Selective ES compression (SP slope − ES slope)",
    "selective_composite": "Selective ES compression (mean[EE,SP] − ES)",
    "ES_raw_compression": "Raw ES compression (1 − ES slope)",
    "ES": "ES value-sensitivity slope",
}


def plot_relationship(df, x_col, y_col, stats_row, out_base, title):
    d = df[["sub_id", x_col, y_col]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    x = d[x_col].to_numpy(float)
    y = d[y_col].to_numpy(float)
    n = len(d)

    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    pts = ax.scatter(x, y, s=82, alpha=0.85)
    face = pts.get_facecolor()
    line_colour = face[0] if len(face) else None

    slope = stats_row["regression_slope"]
    intercept = stats_row["regression_intercept"]
    x_line = np.linspace(np.min(x), np.max(x), 200)
    y_line = intercept + slope * x_line
    ax.plot(x_line, y_line, linewidth=2.7, color=line_colour)

    y_hat = intercept + slope * x
    resid = y - y_hat
    s_err = np.sqrt(np.sum(resid ** 2) / max(n - 2, 1))
    denom = np.sum((x - np.mean(x)) ** 2)
    if denom > 0 and n > 2:
        tcrit = stats.t.ppf(0.975, df=n - 2)
        ci = tcrit * s_err * np.sqrt(
            1.0 / n + (x_line - np.mean(x)) ** 2 / denom
        )
        ax.fill_between(
            x_line, y_line - ci, y_line + ci,
            alpha=0.20, color=line_colour, linewidth=0,
        )

    for sid, xi, yi in d.itertuples(index=False):
        ax.annotate(
            str(int(sid)), (xi, yi),
            xytext=(4, 4), textcoords="offset points",
            fontsize=9, alpha=0.65,
        )

    if y_col.startswith("selective_"):
        ax.axhline(0, linestyle="--", linewidth=1.5, alpha=0.55)

    ax.set_xlabel(PARAM_LABELS.get(x_col, x_col))
    ax.set_ylabel(OUTCOME_LABELS.get(y_col, y_col))
    ax.set_title(title, pad=14)
    text = (
        f"Pearson r = {stats_row['pearson_r']:.2f}, "
        f"{p_text(stats_row['pearson_p'])}\n"
        f"Spearman ρ = {stats_row['spearman_rho']:.2f}, "
        f"{p_text(stats_row['spearman_p'])}\n"
        f"$R^2$ = {stats_row['r2']:.2f}, N = {stats_row['n']}"
    )
    ax.text(
        0.03, 0.97, text, transform=ax.transAxes,
        ha="left", va="top", fontsize=13,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.9),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1.5)
    ax.tick_params(width=1.3, length=6)
    fig.tight_layout()
    save_figure(fig, out_base)



def plot_correlation_grid(
    df,
    outcome_col,
    parameter_cols,
    out_base,
    study,
    title_prefix="",
):
    """
    Old-code-style inspection figure:
    one panel per participant-level aDDM parameter, all against the SAME
    behavioural individual-difference score.

    Each panel shows:
        participants
        least-squares line
        95% CI around fitted mean
        Pearson r and p
        Spearman rho and p
    """
    panels = []

    for x_col in parameter_cols:
        if x_col not in df.columns:
            continue

        s = association_stats(df, x_col, outcome_col)
        if s is None:
            continue

        d = df[[x_col, outcome_col]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        panels.append((x_col, d, s))

    if not panels:
        print(f"WARNING: no usable panels for {outcome_col}")
        return

    ncols = 4 if len(panels) > 9 else 3
    nrows = math.ceil(len(panels) / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.1 * ncols, 4.6 * nrows),
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, (x_col, d, s) in zip(axes, panels):
        x = d[x_col].to_numpy(float)
        y = d[outcome_col].to_numpy(float)
        n = len(d)

        pts = ax.scatter(x, y, s=48, alpha=0.82)
        face = pts.get_facecolor()
        line_colour = face[0] if len(face) else None

        slope = s["regression_slope"]
        intercept = s["regression_intercept"]

        x_line = np.linspace(np.min(x), np.max(x), 150)
        y_line = intercept + slope * x_line
        ax.plot(x_line, y_line, linewidth=2.0, color=line_colour)

        # 95% CI around fitted mean
        y_hat = intercept + slope * x
        resid = y - y_hat
        denom = np.sum((x - np.mean(x)) ** 2)

        if n > 2 and denom > 0:
            s_err = np.sqrt(np.sum(resid ** 2) / (n - 2))
            tcrit = stats.t.ppf(0.975, df=n - 2)
            ci = tcrit * s_err * np.sqrt(
                1.0 / n + (x_line - np.mean(x)) ** 2 / denom
            )
            ax.fill_between(
                x_line,
                y_line - ci,
                y_line + ci,
                alpha=0.18,
                color=line_colour,
                linewidth=0,
            )

        if outcome_col.startswith("selective_"):
            ax.axhline(
                0,
                linestyle="--",
                linewidth=1.2,
                alpha=0.45,
            )

        ax.set_xlabel(PARAM_LABELS.get(x_col, x_col), fontsize=13)
        ax.set_ylabel(
            OUTCOME_LABELS.get(outcome_col, outcome_col),
            fontsize=12,
        )
        ax.set_title(PARAM_LABELS.get(x_col, x_col), fontsize=15)

        txt = (
            f"r = {s['pearson_r']:.2f}, {p_text(s['pearson_p'])}\n"
            f"ρ = {s['spearman_rho']:.2f}, {p_text(s['spearman_p'])}\n"
            f"$R^2$ = {s['r2']:.2f}"
        )
        ax.text(
            0.03,
            0.97,
            txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec="none",
                alpha=0.90,
            ),
        )

        ax.tick_params(labelsize=11)
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(len(panels), len(axes)):
        axes[j].axis("off")

    title = (
        f"Study {study}: {title_prefix}"
        f"{OUTCOME_LABELS.get(outcome_col, outcome_col)} vs all aDDM parameters"
    )
    fig.suptitle(title, y=0.995, fontsize=22)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save_figure(fig, out_base)


def plot_garcia_profile(df, out_base, study):
    d = df[["sub_id", "ES", "nonES_mean"]].dropna()
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    ax.scatter(d["ES"], d["nonES_mean"], s=90, alpha=0.85)

    lo = float(min(d["ES"].min(), d["nonES_mean"].min()))
    hi = float(max(d["ES"].max(), d["nonES_mean"].max()))
    pad = 0.08 * max(hi - lo, 0.2)
    lo -= pad
    hi += pad

    ax.plot(
        [lo, hi], [lo, hi],
        linestyle="--", linewidth=1.8, alpha=0.6,
        label="Equal ES and non-ES sensitivity",
    )
    for sid, es, nones in d.itertuples(index=False):
        ax.annotate(
            str(int(sid)), (es, nones),
            xytext=(4, 4), textcoords="offset points",
            fontsize=10, alpha=0.7,
        )

    ax.set_xlabel("ES value-sensitivity slope")
    ax.set_ylabel("Mean non-ES sensitivity (EE + SP) / 2")
    ax.set_title(f"Study {study}: who shows selective ES compression?", pad=14)
    ax.legend(frameon=False)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, out_base)


def write_summary(path, study, merged, planned_df):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"STUDY {study} — GARCIA-LIKE INDIVIDUAL DIFFERENCES x ESaDDM+z\n")
        f.write("=" * 86 + "\n\n")
        f.write("PRIMARY behavioural score: selective_vs_EE = EE - ES\n")
        f.write(
            "Positive = E sensitivity is lower specifically in ES than in EE.\n\n"
        )

        for col in [
            "ES", "EE", "SP", "ES_raw_compression",
            "selective_vs_EE", "selective_vs_SP", "selective_composite",
        ]:
            vals = merged[col].dropna().to_numpy(float)
            if len(vals):
                f.write(
                    f"{col}: N={len(vals)}, M={np.mean(vals):.4f}, "
                    f"SD={np.std(vals, ddof=1):.4f}, "
                    f"range=[{np.min(vals):.4f}, {np.max(vals):.4f}]\n"
                )

        f.write("\nINDIVIDUAL GARCIA-LIKE DIRECTION\n")
        f.write(f"ES < EE: {int(merged['ES_lower_than_EE'].sum())}/{len(merged)}\n")
        f.write(f"ES < SP: {int(merged['ES_lower_than_SP'].sum())}/{len(merged)}\n")
        f.write(
            f"ES < BOTH EE and SP: "
            f"{int(merged['ES_lower_than_both'].sum())}/{len(merged)}\n\n"
        )

        warnings = merged.loc[
            merged["theta_ratio_warning"].fillna(False),
            ["sub_id", "p_abs_b_AS_lt_0p05", "p_abs_b_AE_lt_0p05"],
        ]
        f.write("THETA RATIO DIAGNOSTIC\n")
        f.write(
            f"Participants flagged for possible denominator instability: "
            f"{len(warnings)}\n"
        )
        if len(warnings):
            f.write(warnings.to_string(index=False) + "\n")

        f.write("\nPLANNED ASSOCIATIONS — HOLM CORRECTED\n")
        if len(planned_df):
            cols = [
                "x", "y", "n", "pearson_r", "pearson_p", "pearson_p_holm",
                "spearman_rho", "spearman_p", "regression_slope",
                "regression_slope_ci_low", "regression_slope_ci_high",
                "r2", "loo_pearson_r_min", "loo_pearson_r_max",
            ]
            f.write(planned_df[cols].to_string(index=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=int, choices=[1, 2], required=True)
    parser.add_argument(
        "--slopes", type=Path, required=True,
        help="Prefer StudyX_phase_slopes_full.csv; primary.csv also works.",
    )
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    root = ROOTS[args.study]
    model_dir = args.model_dir or (root / FIT_REL)
    out_dir = args.out_dir or (root / OUT_REL)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 86)
    print(f"STUDY {args.study}")
    print("=" * 86)
    print("Slope file:", args.slopes)
    print("Model dir:", model_dir)
    print("Output dir:", out_dir)

    beh = read_behavioural_slopes(args.slopes)
    print("\nBehavioural participant IDs:", beh["sub_id"].tolist())

    traces = load_chain_traces(model_dir)
    addm = extract_individual_addm_parameters(traces, beh["sub_id"].tolist())
    addm_file = out_dir / f"Study{args.study}_addm_individual_params.csv"
    addm.to_csv(addm_file, index=False)

    merged = beh.merge(addm, on="sub_id", how="inner", validate="one_to_one")
    merged = merged.sort_values("sub_id").reset_index(drop=True)
    if len(merged) != len(beh):
        raise ValueError(
            f"Behaviour N={len(beh)} but merged N={len(merged)}; ID mismatch."
        )

    merged_file = out_dir / f"Study{args.study}_slopes_addm_merged.csv"
    merged.to_csv(merged_file, index=False)

    plot_garcia_profile(
        merged,
        fig_dir / f"Study{args.study}_Garcia_like_profile",
        args.study,
    )

    planned = [
        ("theta_S_minus_E_mean", "selective_vs_EE"),
        ("delta_I_mean", "selective_vs_EE"),
        ("delta_A_mean", "selective_vs_EE"),
        ("z_S_bias_mean", "selective_vs_EE"),
    ]

    planned_rows = []
    for x_col, y_col in planned:
        if x_col not in merged.columns:
            print(f"WARNING: missing planned parameter {x_col}")
            continue
        s = association_stats(merged, x_col, y_col)
        if s is None:
            continue
        planned_rows.append(s)
        plot_relationship(
            merged, x_col, y_col, s,
            fig_dir / f"Study{args.study}_PRIMARY_{y_col}_vs_{x_col}",
            f"Study {args.study}: {OUTCOME_LABELS[y_col]} vs {PARAM_LABELS[x_col]}",
        )

    planned_df = pd.DataFrame(planned_rows)
    if len(planned_df):
        planned_df["pearson_p_holm"] = holm_adjust(
            planned_df["pearson_p"].to_numpy(float)
        )
        planned_df.to_csv(
            out_dir / f"Study{args.study}_planned_associations.csv",
            index=False,
        )

    outcomes = [
        "selective_vs_EE", "selective_vs_SP",
        "selective_composite", "ES_raw_compression", "ES",
    ]
    # ALL substantive participant-level quantities from the final model.
    # We deliberately do not include redundant sign/shift duplicates such as
    # theta_E_minus_S or both z and z-.5 in the same inspection grid.
    parameters = [
        # Direct no-intercept regression coefficients
        "b_A_mean",
        "b_I_mean",
        "delta_A_mean",
        "delta_I_mean",

        # Reconstructed option-specific attended/unattended weights
        "b_AS_mean",
        "b_AE_mean",
        "b_IS_mean",
        "b_IE_mean",

        # Attentional discounting parameters
        "theta_S_mean",
        "theta_E_mean",
        "theta_S_minus_E_mean",

        # Standard DDM parameters
        "z_mean",
        "a_mean",
        "t_mean",
    ]

    exploratory_rows = []
    for y_col in outcomes:
        for x_col in parameters:
            if x_col not in merged.columns:
                continue
            s = association_stats(merged, x_col, y_col)
            if s is not None:
                exploratory_rows.append(s)

    exploratory_df = pd.DataFrame(exploratory_rows)
    if len(exploratory_df):
        exploratory_df.to_csv(
            out_dir / f"Study{args.study}_all_associations_exploratory.csv",
            index=False,
        )

        # ---------------------------------------------------------------
        # OLD-CODE-STYLE FULL INSPECTION
        # ---------------------------------------------------------------
        # PRIMARY: one grid showing EE-ES against EVERY substantive aDDM
        # parameter. This is meant as the first visual inspection.
        plot_correlation_grid(
            merged,
            outcome_col="selective_vs_EE",
            parameter_cols=parameters,
            out_base=(
                fig_dir
                / f"Study{args.study}_GRID_PRIMARY_selective_vs_EE_all_addm_params"
            ),
            study=args.study,
            title_prefix="PRIMARY — ",
        )

        # Also save a full-size individual figure for EVERY parameter vs
        # the primary EE-ES score, matching the spirit of the older
        # correlation scripts.
        for x_col in parameters:
            row = exploratory_df.loc[
                exploratory_df["x"].eq(x_col)
                & exploratory_df["y"].eq("selective_vs_EE")
            ]
            if len(row) == 1:
                s = row.iloc[0].to_dict()
                plot_relationship(
                    merged,
                    x_col,
                    "selective_vs_EE",
                    s,
                    fig_dir
                    / f"Study{args.study}_ALLPARAMS_selective_vs_EE_vs_{x_col}",
                    (
                        f"Study {args.study}: "
                        f"{OUTCOME_LABELS['selective_vs_EE']} vs "
                        f"{PARAM_LABELS.get(x_col, x_col)}"
                    ),
                )

        # Secondary inspection grids. These are useful for interpretation,
        # but keep them explicitly secondary to the EE-ES analysis.
        for outcome_col, tag in [
            ("selective_vs_SP", "SECONDARY_SP_MINUS_ES"),
            ("selective_composite", "SECONDARY_COMPOSITE"),
            ("ES_raw_compression", "SECONDARY_RAW_ES_COMPRESSION"),
            ("ES", "SECONDARY_RAW_ES_SLOPE"),
        ]:
            plot_correlation_grid(
                merged,
                outcome_col=outcome_col,
                parameter_cols=parameters,
                out_base=(
                    fig_dir
                    / f"Study{args.study}_GRID_{tag}_all_addm_params"
                ),
                study=args.study,
                title_prefix="SECONDARY — ",
            )

    beh_rows = []
    for col in [
        "selective_vs_EE", "selective_vs_SP",
        "selective_composite", "ES_raw_compression",
    ]:
        vals = merged[col].dropna().to_numpy(float)
        one = stats.ttest_1samp(vals, popmean=0.0)
        try:
            wil = stats.wilcoxon(vals)
            wil_stat, wil_p = float(wil.statistic), float(wil.pvalue)
        except ValueError:
            wil_stat, wil_p = np.nan, np.nan
        beh_rows.append({
            "measure": col, "n": len(vals),
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)),
            "t_vs_0": float(one.statistic),
            "p_t_vs_0": float(one.pvalue),
            "wilcoxon_stat": wil_stat,
            "wilcoxon_p": wil_p,
        })

    pd.DataFrame(beh_rows).to_csv(
        out_dir / f"Study{args.study}_behavioural_selective_scores_stats.csv",
        index=False,
    )

    write_summary(
        out_dir / f"Study{args.study}_individual_differences_summary.txt",
        args.study, merged, planned_df,
    )

    print("\nDONE")
    print("Participant aDDM params:", addm_file)
    print("Merged table:", merged_file)
    print("Figures:", fig_dir)
    if len(planned_df):
        print("\nPLANNED ASSOCIATIONS")
        print(
            planned_df[
                ["x", "y", "n", "pearson_r", "pearson_p",
                 "pearson_p_holm", "spearman_rho", "spearman_p", "r2"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
