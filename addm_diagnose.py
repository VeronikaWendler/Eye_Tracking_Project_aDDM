from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hddm
import kabuki



def save_core_posterior_summary(combined, out_file):
    """
    Save a concise group-level posterior summary for the basic aDDM.

    Short labels:
        a     = boundary separation
        t     = non-decision time
        z     = starting point
        b0    = v_Intercept
        b1    = v_AttentionW
        b2    = v_InattentionW
        theta = b2 / b1

    Mean/median/HDI are calculated from the combined MCMC posterior,
    """

    def get_trace(name):
        try:
            trace = combined.nodes_db.loc[name, "node"].trace()
            return np.asarray(trace, dtype=float).reshape(-1)
        except Exception:
            return None

    def summarize(short_name, hddm_name, trace):
        hdi = np.asarray(az.hdi(trace, hdi_prob=0.95)).reshape(-1)
        return {
            "Parameter": short_name,
            "HDDM_name": hddm_name,
            "Mean": float(np.mean(trace)),
            "Median": float(np.median(trace)),
            "HDI_lower": float(hdi[0]),
            "HDI_upper": float(hdi[-1]),
        }

    parameter_map = [
        ("a", "a"),
        ("t", "t"),
        ("z", "z"),
        ("b0", "v_Intercept"),
        ("b1", "v_AttentionW"),
        ("b2", "v_InattentionW"),
    ]

    rows = []
    traces = {}

    for short_name, hddm_name in parameter_map:
        trace = get_trace(hddm_name)

        if trace is None:
            print(f"Posterior summary: skipping missing parameter {hddm_name}")
            continue

        traces[short_name] = trace
        rows.append(summarize(short_name, hddm_name, trace))

    # Derived attentional discounting ratio, calculated draw-by-draw.
    if "b1" in traces and "b2" in traces:
        theta = traces["b2"] / traces["b1"]
        theta = theta[np.isfinite(theta)]

        if len(theta):
            rows.append(
                summarize(
                    "theta",
                    "v_InattentionW / v_AttentionW",
                    theta,
                )
            )

            # Important warning: theta becomes unstable if b1 is near/crosses zero.
            b1_hdi = np.asarray(
                az.hdi(traces["b1"], hdi_prob=0.95)
            ).reshape(-1)

            if b1_hdi[0] <= 0 <= b1_hdi[-1]:
                print(
                    "WARNING: 95% HDI for b1 includes zero. "
                    "Interpret theta = b2/b1 cautiously."
                )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_file, index=False)

    print("\nCore posterior summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved concise posterior summary to: {out_file}")

    return summary

def main():
    parser = argparse.ArgumentParser(
        description="Load and diagnose an already-fitted basic aDDM."
    )
    parser.add_argument("--phase", required=True, choices=["ES", "EE", "LE"])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--fig-dir", required=True)
    parser.add_argument("--chains", type=int, default=3)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--ppc-samples", type=int, default=50)
    args = parser.parse_args()

    model_name = args.model_name or f"basic_aDDM_{args.phase}"
    model_dir = Path(args.model_dir).resolve()
    fig_dir = Path(args.fig_dir).resolve()
    diag_dir = fig_dir / model_name / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Load fitted chains
    # ----------------------------------------------------------
    models = [
        hddm.load(str(model_dir / f"{model_name}_{i}.hddm"))
        for i in range(args.chains)
    ]

    print(f"Loaded {len(models)} chain(s) for {model_name}")

    # ----------------------------------------------------------
    # Convergence across chains
    # Gelman-Rubin requires >= 2 chains
    # ----------------------------------------------------------
    if args.chains >= 2:
        gr = hddm.analyze.gelman_rubin(models)
        pd.Series(gr, name="R_hat").to_csv(
            diag_dir / "gelman_rubin.csv"
        )
        print("Saved Gelman-Rubin diagnostics.")
    else:
        print(
            "Only one chain supplied: "
            "skipping Gelman-Rubin."
        )

    # ----------------------------------------------------------
    # Combine chains
    # ----------------------------------------------------------
    combined = kabuki.utils.concat_models(models)

    # DIC + full HDDM summary
    (diag_dir / "DIC.txt").write_text(
        f"DIC: {combined.dic}\n"
    )

    combined.gen_stats().to_csv(
        diag_dir / "posterior_summary.csv"
    )

    # ----------------------------------------------------------
    # Concise core summary
    # ----------------------------------------------------------
    save_core_posterior_summary(
        combined,
        diag_dir / "core_posterior_summary.csv",
    )

    # ----------------------------------------------------------
    # Posterior plots
    # ----------------------------------------------------------
    combined.plot_posteriors(
        save=True,
        path=str(diag_dir),
        format="pdf",
    )

    # ----------------------------------------------------------
    # Posterior predictive check
    # ----------------------------------------------------------
    size_plot = max(
        6,
        len(combined.data.subj_idx.unique()) / 3.0 * 1.5,
    )

    combined.plot_posterior_predictive(
        samples=args.ppc_samples,
        bins=100,
        figsize=(6, size_plot),
        save=True,
        path=str(diag_dir),
        format="pdf",
    )

    print(f"\nDiagnostics saved to: {diag_dir}")
    print(
        "Core model: "
        "v ~ 1 + AttentionW + InattentionW; "
        "free a, t, z"
    )


if __name__ == "__main__":
    main()
