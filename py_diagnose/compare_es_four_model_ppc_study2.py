#!/usr/bin/env python3
"""Compare Study-2 ES PPC summaries from the four existing fitted model families.

No HDDM is imported and no new posterior-predictive draws are generated.
The four model-specific Study-2 PPC jobs must already have completed.
All panels compare the same observed Study-2 ES trial records and response coordinate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = [
    ("basic_noz", "Basic aDDM"),
    ("basic_z", "Basic aDDM + z"),
    ("contrast_noz", "ESaDDM"),
    ("contrast_z", "ESaDDM + z"),
]
BASE = "es_identity_s_upper_{}_main"
EXPECTED_STUDY = "study2"
EXPECTED_OBS = [
    "subj_idx", "sub_id", "phase", "trial", "response", "rt",
    "DwellTimeAdvantage_ES", "DwellPropAdvantage_ES",
]
ORDER_DWELL = ["E>>S", "E>S", "S~E", "S>E", "S>>E"]
ORDER_RT = ["1", "2", "3", "4", "5"]


def read_table(root: Path, fname: str) -> pd.DataFrame:
    p = root / "tables" / fname
    if not p.is_file():
        raise FileNotFoundError(f"Missing completed PPC table: {p}")
    return pd.read_csv(p)


def prepare_observed(root: Path) -> pd.DataFrame:
    p = root / "tables" / "observed_replot_trials.csv.gz"
    if not p.is_file():
        raise FileNotFoundError(
            f"Comparison requires --save-replot-data in PPC: {p}"
        )
    tab = pd.read_csv(p)
    missing = sorted(set(EXPECTED_OBS) - set(tab.columns))
    if missing:
        raise ValueError(f"{p}: missing observed columns: {missing}")
    if tab.duplicated(["sub_id", "phase", "trial"]).any():
        raise ValueError(f"{p}: duplicate observed trial identity")
    if set(tab["phase"].astype(str).unique()) != {"ES"}:
        raise ValueError(f"{p}: expected only ES phase")
    return (
        tab[EXPECTED_OBS]
        .sort_values(["sub_id", "phase", "trial"])
        .reset_index(drop=True)
    )


def audit(roots: dict[str, Path]) -> pd.DataFrame:
    reference = None
    metadata = []

    for key, label in MODELS:
        root = roots[key]
        if not root.is_dir():
            raise FileNotFoundError(f"Missing model PPC output: {root}")

        mp = root / "metadata.json"
        if not mp.is_file():
            raise FileNotFoundError(f"Missing completed PPC metadata: {mp}")

        meta = json.loads(mp.read_text())

        if meta.get("study") != EXPECTED_STUDY:
            raise ValueError(
                f"{key}: expected {EXPECTED_STUDY}, got {meta.get('study')}"
            )
        if meta.get("response_coordinate") != "response=1 S upper; response=0 E lower":
            raise ValueError(f"{key}: inconsistent response coordinate")
        if meta.get("detected_total_ppc_draws") != 1000:
            raise ValueError(
                f"{key}: expected 1000 detected PPC draws, "
                f"got {meta.get('detected_total_ppc_draws')}"
            )

        ppcaudit = meta.get("participant_mapping_audit", {})
        if ppcaudit.get("status") != "passed":
            raise ValueError(f"{key}: participant mapping audit not passed")

        if key.startswith("basic"):
            if (
                meta.get("model_family") != "basic"
                or meta.get("estimated_z") != key.endswith("_z")
            ):
                raise ValueError(
                    f"{key}: basic PPC metadata inconsistent with model selection"
                )
        else:
            names = [Path(s).name for s in meta["model_files"]]
            expected = f'_CONTRAST_{"Z" if key.endswith("_z") else "NOZ"}_'
            if any(expected not in n for n in names):
                raise ValueError(
                    f"{key}: model file names inconsistent with contrast fit: {names}"
                )

        obs = prepare_observed(root)
        if reference is None:
            reference = obs
        else:
            for c in EXPECTED_OBS:
                x = reference[c]
                y = obs[c]
                if c == "phase":
                    good = x.astype(str).equals(y.astype(str))
                else:
                    good = np.allclose(
                        pd.to_numeric(x, errors="coerce"),
                        pd.to_numeric(y, errors="coerce"),
                        equal_nan=True,
                        atol=1e-8,
                        rtol=0,
                    )
                if not good:
                    raise ValueError(
                        f"Observed fitted trials differ across PPC models in {c} "
                        f"({key}); do not compare them as matched models"
                    )

        metadata.append(
            {
                "model": key,
                "label": label,
                "n_observed_trials": len(obs),
                "n_participants": obs.subj_idx.nunique(),
                "n_draws": meta["detected_total_ppc_draws"],
                "seed": meta.get("seed"),
                "model_dir": meta.get("model_dir"),
            }
        )

    return pd.DataFrame(metadata)


def plot_metric(
    roots,
    out,
    filename,
    title,
    table_name,
    select,
    order,
    observed_col,
    pred_col,
    low_col,
    high_col,
    yscale=1,
    ylabel="",
):
    fig, ax = plt.subplots(figsize=(10.3, 6.4))
    x = np.arange(len(order))
    observed_ref = None
    rows = []

    for idx, (key, label) in enumerate(MODELS):
        t = read_table(roots[key], table_name)
        t = select(t).copy()
        t["bin"] = t["bin"].astype(str)
        t = t.set_index("bin").reindex(order)

        if t[observed_col].isna().any() or t[pred_col].isna().any():
            raise ValueError(f"{key}, {filename}: missing data in one or more bins")

        obs = t[observed_col].to_numpy(float) * yscale
        pred = t[pred_col].to_numpy(float) * yscale
        low = t[low_col].to_numpy(float) * yscale
        high = t[high_col].to_numpy(float) * yscale

        if observed_ref is None:
            observed_ref = obs
        elif not np.allclose(observed_ref, obs, atol=1e-8, rtol=0):
            raise ValueError(
                f"{filename}: inconsistent observed summaries among models"
            )

        color = f"C{idx}"
        ax.plot(
            x, pred, marker="o", linewidth=2, markersize=6,
            label=label, color=color
        )
        ax.fill_between(x, low, high, alpha=0.12, color=color)

        rows.append(
            {
                "model": key,
                "metric": filename,
                "mae": float(np.mean(np.abs(pred - obs))),
                "rmse": float(np.sqrt(np.mean((pred - obs) ** 2))),
                "covered_bins_95": int(((obs >= low) & (obs <= high)).sum()),
                "n_bins": len(order),
            }
        )

    ax.plot(
        x, observed_ref, color="black", linewidth=2.5,
        linestyle="--", marker="s", markersize=6, label="Observed"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(out / f"{filename}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ppc-root", type=Path, required=True,
        help="Study2 derivatives/ppc directory"
    )
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    roots = {key: a.ppc_root / BASE.format(key) for key, _ in MODELS}
    metadata = audit(roots)
    metadata.to_csv(a.out_dir / "four_model_input_audit.csv", index=False)

    metrics = []

    metrics += plot_metric(
        roots,
        a.out_dir,
        "choice_by_dwell",
        "P(choose S) across identity-coded dwell quintiles",
        "dwell_prop_model_vs_observed_comparison_identity.csv",
        lambda t: t.rename(columns={"bin": "bin"}),
        ORDER_DWELL,
        "mean",
        "predictive_mean",
        "predictive_pi95_low",
        "predictive_pi95_high",
        100,
        "P(choose S) (%)",
    )

    metrics += plot_metric(
        roots,
        a.out_dir,
        "choice_by_rt_quintile",
        "P(choose S) across within-draw RT quintiles",
        "choice_e_s_by_rt_quintile_ppc.csv",
        lambda t: t[t["choice"] == "S"],
        ORDER_RT,
        "observed_mean",
        "ppc_mean",
        "ppc_pi95_low",
        "ppc_pi95_high",
        100,
        "P(choose S) (%)",
    )

    metrics += plot_metric(
        roots,
        a.out_dir,
        "median_rt_by_choice",
        "Median RT conditional on choice",
        "rt_by_response_group_plot.csv",
        lambda t: (
            t.assign(bin=t["choice"])
            .query('source == "Posterior predictive"')
            .merge(
                read_table(
                    roots["basic_noz"],
                    "rt_by_response_group_plot.csv"
                )
                .query('source == "Observed"')[["choice", "mean"]]
                .rename(columns={"mean": "observed_truth"}),
                on="choice",
            )
        ),
        ["E", "S"],
        "observed_truth",
        "mean",
        "low",
        "high",
        1,
        "Median RT (s)",
    )

    metrics += plot_metric(
        roots,
        a.out_dir,
        "overall_choice",
        "Overall probability of choosing S",
        "overall_choice_es_group_plot.csv",
        lambda t: (
            t.assign(bin=t["choice"])
            .query('source == "Posterior predictive"')
            .merge(
                read_table(
                    roots["basic_noz"],
                    "overall_choice_es_group_plot.csv"
                )
                .query('source == "Observed"')[["choice", "mean"]]
                .rename(columns={"mean": "observed_truth"}),
                on="choice",
            )
        ),
        ["E", "S"],
        "observed_truth",
        "mean",
        "low",
        "high",
        100,
        "Choice probability (%)",
    )

    pd.DataFrame(metrics).to_csv(
        a.out_dir / "four_model_ppc_error_summary.csv", index=False
    )

    (a.out_dir / "README.txt").write_text(
        "These are Study-2 posterior-predictive IN-SAMPLE descriptive "
        "comparisons; smaller RMSE is not an out-of-sample model-selection "
        "criterion. Each shaded band is a 95% posterior predictive interval; "
        "the dashed black line is the observed group summary. The choice-by-RT "
        "panel uses model-specific simulated RT quintiles, recomputed in each "
        "draw. Use matching fitted trial records and inspect participant-level "
        "PPCs before inferring one model is adequate.\n"
    )

    print(
        "All four Study-2 input-model trial and response-coding audits passed. "
        "Saved plots and error summary to",
        a.out_dir,
        flush=True,
    )


if __name__ == "__main__":
    main()
