"""
    model_kind = contrast
    include_z  = True
    drift      = v ~ 0 + AttentionW_SE + InattentionW_SE
                    + AttentionContrast_SE + InattentionContrast_SE

So:
- S upper / E lower
- NO drift intercept
- contrast/source-sensitive weights
- starting point z estimated

This script reads the actual HDDM chain .pkl files produced by:
    addm_fit_es_identity_S_upper_family.py

Examples:
    python py_diagnose/plot_esaddm_contrast_z_nointercept_talk_ready.py --study 1
    python py_diagnose/plot_esaddm_contrast_z_nointercept_talk_ready.py --study 2
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOTS = {
    1: Path("/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM"),
    2: Path("/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM"),
}

FIT_REL = Path("derivatives/models/es_identity_S_upper_contrast_z_final")
OUT_REL = Path("derivatives/figures/es_identity_S_upper_contrast_z_final/talk_ready")

MODEL_TAG = "A_DDM_ES_IDENTITY_S_UPPER_CONTRAST_Z_NO_INTERCEPT"

FILL = "#8FA6B8"      # muted blue-grey
EDGE = "#587286"      # darker blue-grey
REF = "#D95F5F"       # soft red
TEXT = "#222222"


def norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def hdi(x: np.ndarray, mass: float = 0.95):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x.sort()
    if len(x) == 0:
        return np.nan, np.nan

    m = int(np.floor(mass * len(x)))
    if m < 1 or m >= len(x):
        q = (1.0 - mass) / 2.0
        return tuple(np.quantile(x, [q, 1 - q]))

    widths = x[m:] - x[:-m]
    i = int(np.argmin(widths))
    return float(x[i]), float(x[i + m])


def load_chain_traces(model_dir: Path) -> pd.DataFrame:
    """
    Load the actual chain .pkl files written by fit_chain():

        A_DDM_ES_IDENTITY_S_UPPER_CONTRAST_Z_NO_INTERCEPT_0.pkl
        A_DDM_ES_IDENTITY_S_UPPER_CONTRAST_Z_NO_INTERCEPT_1.pkl
        A_DDM_ES_IDENTITY_S_UPPER_CONTRAST_Z_NO_INTERCEPT_2.pkl

    and concatenate their posterior traces.
    """
    pkls = sorted(model_dir.glob("*CONTRAST_Z_NO_INTERCEPT_*.pkl"))

    if not pkls:
        pkls = sorted(model_dir.glob("*.pkl"))

    if not pkls:
        raise FileNotFoundError(
            f"No HDDM chain .pkl files found in:\n{model_dir}"
        )

    traces = []

    for p in pkls:
        print(f"Loading {p.name}")
        with open(p, "rb") as f:
            model = dill.load(f)

        if hasattr(model, "get_traces"):
            tr = model.get_traces()
        else:
            raise AttributeError(
                f"{p.name}: loaded model has no get_traces() method."
            )

        tr = tr.copy()
        tr["__chain_file__"] = p.name
        traces.append(tr)

    out = pd.concat(traces, ignore_index=True)

    print(f"\nLoaded {len(pkls)} chains")
    print(f"Combined posterior draws: {len(out):,}")
    print("\nAvailable trace columns:")
    for c in out.columns:
        print(" ", c)

    return out


def find_trace_col(df: pd.DataFrame, aliases, label):
    """
    Match a group-level trace column robustly.

    Exact normalized matches are preferred.
    Subject-specific columns such as ...subj... are excluded.
    """
    candidates = [
        c for c in df.columns
        if c != "__chain_file__"
        and "subj" not in str(c).lower()
    ]

    normalized = {norm(c): c for c in candidates}

    for a in aliases:
        na = norm(a)
        if na in normalized:
            return normalized[na]

    hits = []
    for c in candidates:
        nc = norm(c)
        if any(norm(a) in nc for a in aliases):
            hits.append(c)

    hits = list(dict.fromkeys(hits))

    if len(hits) == 1:
        return hits[0]

    raise KeyError(
        f"\nCould not uniquely identify {label}.\n"
        f"Aliases tried: {aliases}\n"
        f"Candidate hits: {hits}\n\n"
        "Available non-subject trace columns:\n"
        + "\n".join(map(str, candidates))
    )


def extract_parameters(traces: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the four NO-INTERCEPT contrast-model drift coefficients + z.

    For:
      v = b_A * AttentionW_SE
        + b_I * InattentionW_SE
        + delta_A * AttentionContrast_SE
        + delta_I * InattentionContrast_SE

    with:
      AttentionContrast_SE   = (A_S - A_E)/2
      InattentionContrast_SE = (I_S - I_E)/2

    Therefore:
      b_AS = b_A + delta_A/2
      b_AE = b_A - delta_A/2
      b_IS = b_I + delta_I/2
      b_IE = b_I - delta_I/2

    and:
      theta_S = b_IS / b_AS
      theta_E = b_IE / b_AE
    """

    b_a_col = find_trace_col(
        traces,
        [
            "v_AttentionW_SE",
            "v_AttentionW_SE_reg",
            "AttentionW_SE",
        ],
        "b_A (AttentionW_SE coefficient)",
    )

    b_i_col = find_trace_col(
        traces,
        [
            "v_InattentionW_SE",
            "v_InattentionW_SE_reg",
            "InattentionW_SE",
        ],
        "b_I (InattentionW_SE coefficient)",
    )

    delta_a_col = find_trace_col(
        traces,
        [
            "v_AttentionContrast_SE",
            "v_AttentionContrast_SE_reg",
            "AttentionContrast_SE",
        ],
        "delta_A (AttentionContrast_SE coefficient)",
    )

    delta_i_col = find_trace_col(
        traces,
        [
            "v_InattentionContrast_SE",
            "v_InattentionContrast_SE_reg",
            "InattentionContrast_SE",
        ],
        "delta_I (InattentionContrast_SE coefficient)",
    )

    z_col = find_trace_col(
        traces,
        [
            "z",
            "z_Intercept",
            "zIntercept",
        ],
        "group-level starting point z",
    )

    d = pd.DataFrame({
        "b_A": pd.to_numeric(traces[b_a_col], errors="coerce"),
        "b_I": pd.to_numeric(traces[b_i_col], errors="coerce"),
        "delta_a": pd.to_numeric(traces[delta_a_col], errors="coerce"),
        "delta_i": pd.to_numeric(traces[delta_i_col], errors="coerce"),
        "z": pd.to_numeric(traces[z_col], errors="coerce"),
    }).dropna()

    # Exact algebra for the contrast parameterisation used in the family code.
    d["b_AS"] = d["b_A"] + d["delta_a"] / 2.0
    d["b_AE"] = d["b_A"] - d["delta_a"] / 2.0
    d["b_IS"] = d["b_I"] + d["delta_i"] / 2.0
    d["b_IE"] = d["b_I"] - d["delta_i"] / 2.0

    d["theta_s"] = d["b_IS"] / d["b_AS"]
    d["theta_e"] = d["b_IE"] / d["b_AE"]
    d["theta_s_minus_e"] = d["theta_s"] - d["theta_e"]

    d = d.replace([np.inf, -np.inf], np.nan).dropna()

    print("\nMatched EXACT model coefficients:")
    print(f"  b_A      <- {b_a_col}")
    print(f"  b_I      <- {b_i_col}")
    print(f"  delta_A  <- {delta_a_col}")
    print(f"  delta_I  <- {delta_i_col}")
    print(f"  z        <- {z_col}")
    print(f"\nUsable draws after reconstruction: {len(d):,}")

    return d


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(labelsize=16, width=1.1)
    ax.set_ylabel("Posterior density", fontsize=18, color=TEXT)
    ax.grid(False)


def posterior_panel(
    ax,
    x,
    title,
    xlabel,
    reference=None,
    reference_label=None,
    bins=45,
):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    mean = float(np.mean(x))
    lo, hi = hdi(x, 0.95)

    ax.hist(
        x,
        bins=bins,
        density=True,
        color=FILL,
        edgecolor="none",
        alpha=0.82,
    )

    ax.axvline(
        mean,
        color=EDGE,
        linewidth=2.2,
    )

    if reference is not None:
        ax.axvline(
            reference,
            color=REF,
            linestyle=":",
            linewidth=2.6,
            alpha=0.72,
        )

    ax.set_title(
        f"{title}\nmean = {mean:.3f}, 95% HDI [{lo:.3f}, {hi:.3f}]",
        fontsize=18,
        color=TEXT,
        pad=12,
    )
    ax.set_xlabel(
        xlabel,
        fontsize=18,
        color=TEXT,
    )

    style(ax)

    if reference_label and reference is not None:
        ymax = ax.get_ylim()[1]
        ax.text(
            reference,
            ymax * 0.95,
            reference_label,
            color=REF,
            fontsize=13,
            ha="center",
            va="top",
        )

    return mean, lo, hi


def save_single(d, key, title, xlabel, out_dir, study, reference=None, reference_label=None):
    fig, ax = plt.subplots(figsize=(8.2, 5.3))

    vals = posterior_panel(
        ax,
        d[key],
        title,
        xlabel,
        reference=reference,
        reference_label=reference_label,
    )

    fig.suptitle(
        f"ESaDDM + z — Study {study}",
        fontsize=23,
        y=1.01,
    )

    fig.tight_layout()

    stem = out_dir / f"study{study}_{key}_talk"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return vals


def save_horizontal_row(d, specs, out_dir, study, stem_name):
    fig, axes = plt.subplots(
        1,
        len(specs),
        figsize=(7.2 * len(specs), 5.4),
        squeeze=False,
    )

    for ax, spec in zip(axes[0], specs):
        posterior_panel(
            ax,
            d[spec["key"]],
            spec["title"],
            spec["xlabel"],
            reference=spec.get("reference"),
            reference_label=spec.get("reference_label"),
        )

    fig.suptitle(
        f"ESaDDM + z — Study {study}",
        fontsize=25,
        y=1.02,
    )

    fig.tight_layout()

    stem = out_dir / stem_name
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--study",
        type=int,
        choices=[1, 2],
        required=True,
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Optional override for the fitted contrast_z model directory.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory.",
    )

    args = parser.parse_args()

    root = ROOTS[args.study]

    model_dir = (
        args.model_dir
        if args.model_dir is not None
        else root / FIT_REL
    )

    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else root / OUT_REL
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("TALK-READY PLOTS FOR EXACT MODEL:")
    print("  contrast + z")
    print("  S upper / E lower")
    print("  NO drift intercept")
    print("")
    print("Expected drift formula:")
    print(
        "  v ~ 0 + AttentionW_SE + InattentionW_SE "
        "+ AttentionContrast_SE + InattentionContrast_SE"
    )
    print("=" * 78)

    print(f"\nStudy: {args.study}")
    print(f"Model directory: {model_dir}")
    print(f"Output directory: {out_dir}")

    traces = load_chain_traces(model_dir)
    d = extract_parameters(traces)

    specs = {
        "delta_a": {
            "key": "delta_a",
            "title": "Attended source difference (S − E)",
            "xlabel": "Attended slope difference",
            "reference": 0.0,
            "reference_label": "0 = no S–E difference",
        },
        "delta_i": {
            "key": "delta_i",
            "title": "Unattended source difference (S − E)",
            "xlabel": "Unattended slope difference",
            "reference": 0.0,
            "reference_label": "0 = no S–E difference",
        },
        "theta_e": {
            "key": "theta_e",
            "title": "Relative unattended weight — E",
            "xlabel": r"$\theta_E$",
            "reference": 0.0,
        },
        "theta_s": {
            "key": "theta_s",
            "title": "Relative unattended weight — S",
            "xlabel": r"$\theta_S$",
            "reference": 0.0,
        },
        "theta_s_minus_e": {
            "key": "theta_s_minus_e",
            "title": "Relative unattended weight difference (S − E)",
            "xlabel": r"$\theta_S - \theta_E$",
            "reference": 0.0,
            "reference_label": "0 = no S–E difference",
        },
        "z": {
            "key": "z",
            "title": "Starting point",
            "xlabel": "z",
            "reference": 0.5,
            "reference_label": "0.5 = no starting bias",
        },
    }

    summary_rows = []

    for key, spec in specs.items():
        mean, lo, hi = save_single(
            d,
            key=spec["key"],
            title=spec["title"],
            xlabel=spec["xlabel"],
            out_dir=out_dir,
            study=args.study,
            reference=spec.get("reference"),
            reference_label=spec.get("reference_label"),
        )

        summary_rows.append({
            "parameter": key,
            "mean": mean,
            "hdi_lower": lo,
            "hdi_upper": hi,
        })

    # Horizontal, PowerPoint-ready main row.
    save_horizontal_row(
        d,
        [
            specs["delta_a"],
            specs["delta_i"],
            specs["z"],
        ],
        out_dir,
        args.study,
        f"study{args.study}_ESaDDM_contrast_z_MAIN_horizontal",
    )

    # Horizontal support row with the theta quantities.
    save_horizontal_row(
        d,
        [
            specs["theta_e"],
            specs["theta_s"],
            specs["theta_s_minus_e"],
        ],
        out_dir,
        args.study,
        f"study{args.study}_ESaDDM_contrast_z_THETA_horizontal",
    )

    pd.DataFrame(summary_rows).to_csv(
        out_dir / f"study{args.study}_ESaDDM_contrast_z_talk_summary.csv",
        index=False,
    )

    print("\nDone.")
    print("\nMain horizontal figure:")
    print(
        out_dir
        / f"study{args.study}_ESaDDM_contrast_z_MAIN_horizontal.png"
    )
    print("\nTheta horizontal figure:")
    print(
        out_dir
        / f"study{args.study}_ESaDDM_contrast_z_THETA_horizontal.png"
    )


if __name__ == "__main__":
    main()
