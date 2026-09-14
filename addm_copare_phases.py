from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import arviz as az
import hddm
import kabuki


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_phase_models(model_dir: Path, phase: str, chains: int):
    """Load all HDDM chains for one phase."""
    models = []

    for chain in range(chains):
        path = model_dir / f"basic_aDDM_{phase}_{chain}.hddm"

        if not path.exists():
            raise FileNotFoundError(
                f"Could not find expected model file:\n{path}"
            )

        print(f"Loading {phase} chain {chain}: {path}")
        models.append(hddm.load(str(path)))

    print(f"Combining {chains} {phase} chains...")
    return kabuki.utils.concat_models(models)


def get_trace(model, node_name: str) -> np.ndarray:
    """Extract a posterior trace from a combined HDDM model."""
    if node_name not in model.nodes_db.index:
        available = list(model.nodes_db.index)

        raise KeyError(
            f"Node '{node_name}' was not found.\n"
            f"Some available nodes are:\n{available[:40]}"
        )

    trace = np.asarray(
        model.nodes_db.loc[node_name, "node"].trace(),
        dtype=float
    )

    trace = trace[np.isfinite(trace)]

    if trace.size == 0:
        raise ValueError(f"No finite samples found for {node_name}")

    return trace


def hdi(x: np.ndarray, prob: float = 0.95):
    """Return lower and upper HDI."""
    interval = np.asarray(az.hdi(x, hdi_prob=prob))
    return float(interval[0]), float(interval[1])


def make_theta(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Compute theta draw-by-draw:
        theta = b2 / b1
    """
    n = min(len(b1), len(b2))
    b1 = b1[:n]
    b2 = b2[:n]

    valid = (
        np.isfinite(b1)
        & np.isfinite(b2)
        & (np.abs(b1) > 1e-12)
    )

    theta = b2[valid] / b1[valid]
    theta = theta[np.isfinite(theta)]

    if theta.size == 0:
        raise ValueError("No valid theta samples could be calculated.")

    return theta


def compare_parameter(
    name: str,
    es: np.ndarray,
    ee: np.ndarray,
    rng: np.random.Generator,
    n_contrast_draws: int,
):
    """
    Construct posterior ES - EE by independently sampling
    from the separately fitted ES and EE posterior marginals.
    """

    # Independent sampling is deliberate here:
    # the ES and EE models were fitted separately, so their raw
    # MCMC draw indices have no meaningful pairing.
    es_draws = rng.choice(es, size=n_contrast_draws, replace=True)
    ee_draws = rng.choice(ee, size=n_contrast_draws, replace=True)

    diff = es_draws - ee_draws

    es_low, es_high = hdi(es)
    ee_low, ee_high = hdi(ee)
    diff_low, diff_high = hdi(diff)

    p_gt = float(np.mean(diff > 0))
    p_lt = float(np.mean(diff < 0))

    return {
        "Parameter": name,

        "Mean_ES": float(np.mean(es)),
        "Median_ES": float(np.median(es)),
        "HDI_ES_lower": es_low,
        "HDI_ES_upper": es_high,

        "Mean_EE": float(np.mean(ee)),
        "Median_EE": float(np.median(ee)),
        "HDI_EE_lower": ee_low,
        "HDI_EE_upper": ee_high,

        "Mean_Difference_ES_minus_EE": float(np.mean(diff)),
        "Median_Difference_ES_minus_EE": float(np.median(diff)),
        "HDI_Difference_lower": diff_low,
        "HDI_Difference_upper": diff_high,

        "P_ES_gt_EE": p_gt,
        "P_ES_lt_EE": p_lt,
        "P_direction": max(p_gt, p_lt),

        # Keep the draws temporarily so we can save them if desired.
        "_difference_draws": diff,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc posterior comparison of separately fitted "
            "ES and EE basic aDDM models."
        )
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing basic_aDDM_ES_*.hddm and EE files.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory in which comparison results will be saved.",
    )

    parser.add_argument(
        "--chains",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--contrast-draws",
        type=int,
        default=100000,
        help="Monte Carlo samples used for each ES-EE contrast.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260914,
    )

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("POSTERIOR PHASE COMPARISON")
    print("ES - EE")
    print("=" * 70)
    print(f"Model directory: {args.model_dir}")
    print(f"Output directory: {args.out_dir}")
    print(f"Chains per phase: {args.chains}")
    print(f"Contrast draws: {args.contrast_draws}")
    print()

    # -------------------------------------------------------------
    # Load and combine models
    # -------------------------------------------------------------

    es_model = load_phase_models(
        args.model_dir,
        "ES",
        args.chains
    )

    ee_model = load_phase_models(
        args.model_dir,
        "EE",
        args.chains
    )

    # -------------------------------------------------------------
    # Core posterior traces
    # -------------------------------------------------------------

    node_names = {
        "a": "a",
        "t": "t",
        "z": "z",
        "b0": "v_Intercept",
        "b1": "v_AttentionW",
        "b2": "v_InattentionW",
    }

    es_traces = {}
    ee_traces = {}

    for short_name, hddm_name in node_names.items():
        es_traces[short_name] = get_trace(es_model, hddm_name)
        ee_traces[short_name] = get_trace(ee_model, hddm_name)

        print(
            f"{short_name}: "
            f"ES n={len(es_traces[short_name])}, "
            f"EE n={len(ee_traces[short_name])}"
        )

    # -------------------------------------------------------------
    # theta = b2 / b1 calculated within each posterior draw
    # -------------------------------------------------------------

    es_traces["theta"] = make_theta(
        es_traces["b1"],
        es_traces["b2"]
    )

    ee_traces["theta"] = make_theta(
        ee_traces["b1"],
        ee_traces["b2"]
    )

    print(
        f"theta: ES n={len(es_traces['theta'])}, "
        f"EE n={len(ee_traces['theta'])}"
    )

    # -------------------------------------------------------------
    # Posterior contrasts
    # -------------------------------------------------------------

    rng = np.random.default_rng(args.seed)

    parameter_order = [
        "a",
        "t",
        "z",
        "b0",
        "b1",
        "b2",
        "theta",
    ]

    results = []
    difference_draws = {}

    for parameter in parameter_order:

        result = compare_parameter(
            name=parameter,
            es=es_traces[parameter],
            ee=ee_traces[parameter],
            rng=rng,
            n_contrast_draws=args.contrast_draws,
        )

        difference_draws[parameter] = result.pop("_difference_draws")
        results.append(result)

    results_df = pd.DataFrame(results)

    # -------------------------------------------------------------
    # Save concise summary
    # -------------------------------------------------------------

    summary_path = (
        args.out_dir /
        "ES_minus_EE_posterior_comparison.csv"
    )

    results_df.to_csv(
        summary_path,
        index=False,
        float_format="%.6f"
    )

    # Also save posterior difference samples.
    # These can later be used for plots or additional summaries.
    draws_df = pd.DataFrame(difference_draws)

    draws_path = (
        args.out_dir /
        "ES_minus_EE_difference_draws.csv"
    )

    draws_df.to_csv(
        draws_path,
        index=False,
        float_format="%.8f"
    )

    # -------------------------------------------------------------
    # Print compact result to log
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTS: ES - EE")
    print("=" * 70)

    display_cols = [
        "Parameter",
        "Mean_ES",
        "Mean_EE",
        "Mean_Difference_ES_minus_EE",
        "HDI_Difference_lower",
        "HDI_Difference_upper",
        "P_ES_gt_EE",
    ]

    print(
        results_df[display_cols]
        .to_string(index=False)
    )

    print()
    print(f"Saved summary to:\n{summary_path}")
    print()
    print(f"Saved difference draws to:\n{draws_path}")
    print()

    print(
        "These are post-hoc contrasts between the "
        "marginal posteriors of separately fitted ES and EE models. "
        "A joint phase model is still preferable for the formal "
        "within-participant phase test."
    )


if __name__ == "__main__":
    main()