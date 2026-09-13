from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hddm
import kabuki


def main():
    parser = argparse.ArgumentParser(description="Load and diagnose a basic aDDM fit.")
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

    models = [
        hddm.load(str(model_dir / f"{model_name}_{i}.hddm"))
        for i in range(args.chains)
    ]

    # Convergence across chains
    gr = hddm.analyze.gelman_rubin(models)
    pd.Series(gr, name="R_hat").to_csv(diag_dir / "gelman_rubin.csv")

    # Combine chains for summaries and standard HDDM plots
    combined = kabuki.utils.concat_models(models)

    (diag_dir / "DIC.txt").write_text(f"DIC: {combined.dic}\n")
    combined.gen_stats().to_csv(diag_dir / "posterior_summary.csv")

    # Basic posterior plots
    combined.plot_posteriors(save=True, path=str(diag_dir), format="pdf")

    # Posterior predictive check
    size_plot = max(6, len(combined.data.subj_idx.unique()) / 3.0 * 1.5)
    combined.plot_posterior_predictive(
        samples=args.ppc_samples,
        bins=100,
        figsize=(6, size_plot),
        save=True,
        path=str(diag_dir),
        format="pdf",
    )

    # Small human-readable summary for the parameters central to the basic aDDM.
    stats = combined.gen_stats()
    wanted = [
        idx for idx in stats.index
        if idx in {"a", "t", "v_AttentionW", "v_InattentionW"}
        or str(idx).startswith("v_AttentionW")
        or str(idx).startswith("v_InattentionW")
    ]
    if wanted:
        stats.loc[wanted].to_csv(diag_dir / "core_parameters.csv")

    print(f"Diagnostics saved to: {diag_dir}")
    print("Core model: v ~ 1 + AttentionW + InattentionW")


if __name__ == "__main__":
    main()
