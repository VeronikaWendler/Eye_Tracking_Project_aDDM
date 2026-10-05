#!/usr/bin/env python3
"""
Create four talk-ready cross-task marker plots from an already-generated
StudyX_behaviour_addm_merged.csv.

Plots:
  1) theta_E -> EE choice accuracy
  2) b_IE    -> EE choice accuracy
  3) theta_E -> SP E estimation accuracy (1 - MAE)
  4) b_IE    -> SP E estimation accuracy (1 - MAE)

This script deliberately does NOT reload HDDM chains. It uses the participant-level
model summaries and behavioural measures that are already present in the merged CSV
produced by 05_ES_selective_neglect_addm_individual_differences_ALLPARAMS.py.

For each relationship it saves:
  - PNG (300 dpi)
  - SVG
  - one row in StudyX_marker_plot_stats.csv

The shaded band is the 95% confidence interval around the fitted mean regression line.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


PLOTS = [
    {
        "slug": "thetaE_vs_EE_accuracy",
        "x": "theta_E_mean",
        "y": "accuracy_EE",
        "xlabel": r"Relative unattended E influence ($\theta_E$)",
        "ylabel": "EE choice accuracy",
    },
    {
        "slug": "bIE_vs_EE_accuracy",
        "x": "b_IE_mean",
        "y": "accuracy_EE",
        "xlabel": r"Unattended E-value slope ($b_{IE}$)",
        "ylabel": "EE choice accuracy",
    },
    {
        "slug": "thetaE_vs_SP_E_estimation_accuracy",
        "x": "theta_E_mean",
        "y": "SP_E_estimation_accuracy",
        "xlabel": r"Relative unattended E influence ($\theta_E$)",
        "ylabel": "Accuracy of E probability estimates (1 − MAE)",
    },
    {
        "slug": "bIE_vs_SP_E_estimation_accuracy",
        "x": "b_IE_mean",
        "y": "SP_E_estimation_accuracy",
        "xlabel": r"Unattended E-value slope ($b_{IE}$)",
        "ylabel": "Accuracy of E probability estimates (1 − MAE)",
    },
]


def p_text(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    return "< .001" if p < .001 else f"= {p:.3f}"


def regression_details(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    d = (
        df[["sub_id", x_col, y_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    x = d[x_col].to_numpy(float)
    y = d[y_col].to_numpy(float)
    n = len(d)

    if n < 5:
        raise ValueError(f"Not enough complete cases: {x_col} vs {y_col}; N={n}")
    if np.std(x) == 0 or np.std(y) == 0:
        raise ValueError(f"Zero variance: {x_col} vs {y_col}")

    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    reg = stats.linregress(x, y)

    tcrit = stats.t.ppf(0.975, df=n - 2)
    slope_ci_low = reg.slope - tcrit * reg.stderr
    slope_ci_high = reg.slope + tcrit * reg.stderr

    beta_std = reg.slope * np.std(x, ddof=1) / np.std(y, ddof=1)

    return {
        "data": d,
        "x": x,
        "y": y,
        "n": int(n),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "regression_b": float(reg.slope),
        "regression_b_ci_low": float(slope_ci_low),
        "regression_b_ci_high": float(slope_ci_high),
        "regression_intercept": float(reg.intercept),
        "regression_p": float(reg.pvalue),
        "regression_stderr": float(reg.stderr),
        "standardized_beta": float(beta_std),
        "r2": float(reg.rvalue ** 2),
    }


def regression_curve(s: dict):
    x = s["x"]
    y = s["y"]
    n = s["n"]
    slope = s["regression_b"]
    intercept = s["regression_intercept"]

    x_grid = np.linspace(np.min(x), np.max(x), 250)
    y_fit = intercept + slope * x_grid

    residuals = y - (intercept + slope * x)
    residual_se = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    x_bar = np.mean(x)
    sxx = np.sum((x - x_bar) ** 2)
    tcrit = stats.t.ppf(0.975, df=n - 2)

    mean_fit_se = residual_se * np.sqrt(
        (1.0 / n) + ((x_grid - x_bar) ** 2 / sxx)
    )
    lower = y_fit - tcrit * mean_fit_se
    upper = y_fit + tcrit * mean_fit_se
    return x_grid, y_fit, lower, upper


def save_plot(fig, base: Path):
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")


def make_plot(df: pd.DataFrame, spec: dict, study: int, out_dir: Path) -> dict:
    s = regression_details(df, spec["x"], spec["y"])
    x_grid, y_fit, lower, upper = regression_curve(s)

    rc = {
        "font.size": 20,
        "axes.titlesize": 28,
        "axes.labelsize": 23,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(9.5, 7.5))

        # Match the existing blue talk-ready plots.
        blue = "#1f77b4"
        ax.scatter(s["x"], s["y"], s=150, alpha=0.88, linewidth=0.8, color=blue)
        ax.plot(x_grid, y_fit, linewidth=3.0, color=blue)
        ax.fill_between(x_grid, lower, upper, alpha=0.18, color=blue)

        annotation = (
            f"b = {s['regression_b']:.2f}, "
            f"95% CI [{s['regression_b_ci_low']:.2f}, "
            f"{s['regression_b_ci_high']:.2f}]\n"
            f"p {p_text(s['regression_p'])}, "
            f"$R^2$ = {s['r2']:.2f}, N = {s['n']}"
        )

        ax.text(
            0.035, 0.955, annotation,
            transform=ax.transAxes,
            ha="left", va="top", fontsize=18,
        )

        ax.set_title(f"Study {study}")
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylabel(spec["ylabel"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        base = out_dir / f"Study{study}_{spec['slug']}"
        save_plot(fig, base)
        plt.close(fig)

    return {
        "study": study,
        "plot": spec["slug"],
        "x": spec["x"],
        "y": spec["y"],
        "n": s["n"],
        "pearson_r": s["pearson_r"],
        "pearson_p": s["pearson_p"],
        "spearman_rho": s["spearman_rho"],
        "spearman_p": s["spearman_p"],
        "regression_b": s["regression_b"],
        "regression_b_ci_low": s["regression_b_ci_low"],
        "regression_b_ci_high": s["regression_b_ci_high"],
        "regression_intercept": s["regression_intercept"],
        "regression_p": s["regression_p"],
        "standardized_beta": s["standardized_beta"],
        "r2": s["r2"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=int, choices=[1, 2], required=True)
    parser.add_argument("--merged", type=Path, required=True,
                        help="StudyX_behaviour_addm_merged.csv")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.merged)

    required = {
        "sub_id", "theta_E_mean", "b_IE_mean",
        "accuracy_EE", "SP_E_estimation_accuracy",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns in {args.merged}: {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in PLOTS:
        rows.append(make_plot(df, spec, args.study, args.out_dir))

    stats_df = pd.DataFrame(rows)
    stats_file = args.out_dir / f"Study{args.study}_marker_plot_stats.csv"
    stats_df.to_csv(stats_file, index=False)

    print("\nDONE")
    print("Input:", args.merged)
    print("Output directory:", args.out_dir)
    print("Stats:", stats_file)
    print("\nKey results:")
    print(
        stats_df[[
            "plot", "n", "pearson_r", "pearson_p",
            "regression_b", "regression_p", "r2",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
