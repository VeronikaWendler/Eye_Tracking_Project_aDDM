from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIRECT = ["a", "t", "z", "b_a", "b_i", "delta_a", "delta_i"]
DERIVED = ["b_as", "b_ae", "b_is", "b_ie", "theta_s", "theta_e", "delta_theta_s_minus_e"]
LABELS = {
    "a": "a", "t": "t", "z": "z", "b_a": "b_a", "b_i": "b_i",
    "delta_a": "delta_a", "delta_i": "delta_i", "b_as": "b_as", "b_ae": "b_ae",
    "b_is": "b_is", "b_ie": "b_ie", "theta_s": "theta_s", "theta_e": "theta_e",
    "delta_theta_s_minus_e": "theta_s - theta_e",
}


def collect_reps(recovery_dir):
    group, indiv = [], []
    for rep_dir in sorted(recovery_dir.glob("rep_*")):
        if not (rep_dir / "complete.flag").exists():
            continue
        gf = rep_dir / "recovery_group.csv"
        inf = rep_dir / "recovery_individual.csv"
        if gf.exists():
            group.append(pd.read_csv(gf))
        if inf.exists():
            indiv.append(pd.read_csv(inf))
    if not group:
        raise RuntimeError(f"no completed recovery reps in {recovery_dir}")
    return pd.concat(group, ignore_index=True), (pd.concat(indiv, ignore_index=True) if indiv else pd.DataFrame())


def metric_row(df, parameter):
    z = df.loc[df["parameter"].eq(parameter)].replace([np.inf, -np.inf], np.nan).dropna(subset=["true", "recovered_mean"]).copy()
    x = z["true"].to_numpy(float)
    y = z["recovered_mean"].to_numpy(float)
    n = len(z)
    if n >= 3 and np.std(x) > 0:
        corr = float(np.corrcoef(x, y)[0, 1])
        slope, intercept = np.polyfit(x, y, 1)
    else:
        corr = slope = intercept = np.nan
    err = y - x
    return {
        "parameter": parameter,
        "kind": z["kind"].iloc[0] if n else ("direct" if parameter in DIRECT else "derived"),
        "n": n,
        "correlation": corr,
        "slope": float(slope),
        "intercept": float(intercept),
        "bias_mean": float(np.mean(err)) if n else np.nan,
        "mae": float(np.mean(np.abs(err))) if n else np.nan,
        "rmse": float(np.sqrt(np.mean(err ** 2))) if n else np.nan,
        "coverage_95": float(z["covered_95"].astype(float).mean()) if n else np.nan,
        "mean_hdi_width": float(z["hdi_width"].mean()) if n else np.nan,
        "median_r_hat": float(z["r_hat"].median()) if n else np.nan,
        "max_r_hat": float(z["r_hat"].max()) if n else np.nan,
        "median_ess_bulk": float(z["ess_bulk"].median()) if n else np.nan,
        "min_ess_bulk": float(z["ess_bulk"].min()) if n else np.nan,
    }


def rep_quality(group_df):
    rows = []
    z = group_df[group_df["parameter"].isin(DIRECT)]
    for rep, g in z.groupby("rep"):
        mr = float(g["r_hat"].max())
        me = float(g["ess_bulk"].min())
        mt = float(g["ess_tail"].min())
        rhat_ok = bool(np.isfinite(mr) and mr <= 1.01)
        ess_bulk_ok = bool(np.isfinite(me) and me >= 400)
        ess_tail_ok = bool(np.isfinite(mt) and mt >= 400)
        rows.append({
            "rep": int(rep),
            "max_direct_r_hat": mr,
            "min_direct_ess_bulk": me,
            "min_direct_ess_tail": mt,
            "rhat_ok_1p01": rhat_ok,
            "ess_bulk_ok_400": ess_bulk_ok,
            "ess_tail_ok_400": ess_tail_ok,
            "convergence_ok": bool(rhat_ok and ess_bulk_ok and ess_tail_ok),
        })
    return pd.DataFrame(rows)


def compute_metrics(df, params, quality=None):
    all_rows = pd.DataFrame([metric_row(df, p) for p in params])
    all_rows.insert(0, "scope", "all_completed_reps")
    blocks = [all_rows]
    if quality is not None and "rep" in df:
        passed = quality.loc[quality["convergence_ok"], "rep"].astype(int).tolist()
        if passed:
            sub = df[df["rep"].isin(passed)]
            rows = pd.DataFrame([metric_row(sub, p) for p in params])
            rows.insert(0, "scope", "direct_convergence_passed_reps")
            blocks.append(rows)
    return pd.concat(blocks, ignore_index=True)


def limits_for(x, y):
    vals = np.concatenate([np.asarray(x, float), np.asarray(y, float)])
    vals = vals[np.isfinite(vals)]
    lo, hi = float(vals.min()), float(vals.max())
    pad = (abs(lo) * .1 if lo == hi and lo != 0 else .1) if lo == hi else .08 * (hi - lo)
    return lo - pad, hi + pad


def draw_one(ax, z, parameter):
    x = z["true"].to_numpy(float)
    y = z["recovered_mean"].to_numpy(float)
    ax.scatter(x, y, s=24, alpha=.75)
    lo, hi = limits_for(x, y)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("true value"); ax.set_ylabel("recovered posterior mean")
    ax.set_title(LABELS.get(parameter, parameter))
    m = metric_row(z, parameter)
    txt = f"r={m['correlation']:.2f}\nbias={m['bias_mean']:.3f}\nrmse={m['rmse']:.3f}\ncoverage={m['coverage_95']:.2f}"
    ax.text(.03, .97, txt, transform=ax.transAxes, ha="left", va="top", fontsize=8)


def save_individual(df, params, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in params:
        z = df[df["parameter"].eq(p)].replace([np.inf, -np.inf], np.nan).dropna(subset=["true", "recovered_mean"])
        if z.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 5.5))
        draw_one(ax, z, p)
        fig.tight_layout()
        fig.savefig(out_dir / f"recovery_{p}.png", dpi=200)
        plt.close(fig)


def save_combined(df, params, out_file, title):
    params = [p for p in params if df["parameter"].eq(p).any()]
    if not params:
        return
    ncols = 3
    nrows = math.ceil(len(params) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, p in zip(axes, params):
        z = df[df["parameter"].eq(p)].replace([np.inf, -np.inf], np.nan).dropna(subset=["true", "recovered_mean"])
        draw_one(ax, z, p)
    for ax in axes[len(params):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, choices=["study1", "study2"])
    ap.add_argument("--recovery-dir", type=Path, required=True)
    args = ap.parse_args()

    summary = args.recovery_dir / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    group, indiv = collect_reps(args.recovery_dir)
    group.to_csv(summary / "all_group_recovery.csv", index=False)
    if not indiv.empty:
        indiv.to_csv(summary / "all_individual_recovery.csv", index=False)

    quality = rep_quality(group)
    quality.to_csv(summary / "rep_convergence_summary.csv", index=False)
    gm = compute_metrics(group, DIRECT + DERIVED, quality)
    gm.to_csv(summary / "group_recovery_metrics.csv", index=False, float_format="%.8f")
    if not indiv.empty:
        im = compute_metrics(indiv, DIRECT + DERIVED, quality)
        im.to_csv(summary / "individual_recovery_metrics.csv", index=False, float_format="%.8f")

    plots = summary / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    save_combined(group, DIRECT, plots / "group_recovery_direct_combined.png", f"{args.study}: group recovery of directly fitted parameters")
    save_combined(group, DERIVED, plots / "group_recovery_derived_combined.png", f"{args.study}: group recovery of derived quantities")
    save_individual(group, DIRECT + DERIVED, plots / "group_individual")
    if not indiv.empty:
        save_combined(indiv, DIRECT, plots / "participant_recovery_direct_combined.png", f"{args.study}: participant recovery of directly fitted parameters")
        save_combined(indiv, DERIVED, plots / "participant_recovery_derived_combined.png", f"{args.study}: participant recovery of derived quantities")
        save_individual(indiv, DIRECT + DERIVED, plots / "participant_individual")

    report = {
        "study": args.study,
        "n_completed_reps": int(group["rep"].nunique()),
        "completed_reps": sorted(group["rep"].astype(int).unique().tolist()),
        "n_convergence_passed_reps": int(quality["convergence_ok"].sum()),
        "convergence_rule": "all directly fitted group parameters: R-hat <= 1.01, bulk ESS >= 400, and tail ESS >= 400",
        "direct_parameters": DIRECT,
        "derived_parameters": DERIVED,
    }
    (summary / "summary_metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\nGROUP RECOVERY METRICS\n" + gm.to_string(index=False), flush=True)
    print(f"\nsummary saved to {summary}", flush=True)


if __name__ == "__main__":
    main()
