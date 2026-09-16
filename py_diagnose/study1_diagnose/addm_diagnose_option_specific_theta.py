from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hddm
import kabuki


MODEL_NAME = "aDDM_ES_OPTION_SPECIFIC_THETA"


def hdi(x, prob=0.95):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    interval = np.asarray(az.hdi(x, hdi_prob=prob))
    return float(interval[0]), float(interval[1])


def get_trace(model, node_name):
    x = np.asarray(model.nodes_db.loc[node_name, "node"].trace(), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite posterior samples in node: %s" % node_name)
    return x


def resolve_group_node(model, exact_candidates, required_tokens):
    names = [str(x) for x in model.nodes_db.index]

    for candidate in exact_candidates:
        if candidate in names:
            return candidate

    matches = []
    for name in names:
        low = name.lower()
        if "subj" in low:
            continue
        if all(token.lower() in low for token in required_tokens):
            matches.append(name)

    if len(matches) == 1:
        return matches[0]

    raise KeyError(
        "Could not uniquely resolve group node.\n"
        "Exact candidates: %s\n"
        "Required tokens: %s\n"
        "Matches: %s\n\nAvailable nodes:\n%s"
        % (exact_candidates, required_tokens, matches, "\n".join(names))
    )


def resolve_nodes(model):
    return {
        "a": resolve_group_node(model, ["a"], ["a"]),
        "t": resolve_group_node(model, ["t"], ["t"]),
        "z": resolve_group_node(model, ["z"], ["z"]),
        "b0": resolve_group_node(model, ["v_Intercept"], ["v", "intercept"]),
        "bA": resolve_group_node(model, ["v_AttentionW"], ["v", "attentionw"]),
        "bE": resolve_group_node(model, ["v_InattentionW_E"], ["v", "inattentionw_e"]),
        "bS": resolve_group_node(model, ["v_InattentionW_S"], ["v", "inattentionw_s"]),
    }


def convergence_stats(chain_arrays):
    n = min(len(x) for x in chain_arrays)
    arr = np.vstack([np.asarray(x[:n], dtype=float) for x in chain_arrays])

    try:
        rhat = float(np.asarray(az.rhat(arr))) if arr.shape[0] >= 2 else np.nan
    except Exception:
        rhat = np.nan

    try:
        ess_bulk = float(np.asarray(az.ess(arr, method="bulk")))
    except Exception:
        ess_bulk = np.nan

    try:
        ess_tail = float(np.asarray(az.ess(arr, method="tail")))
    except Exception:
        ess_tail = np.nan

    return {
        "R_hat": rhat,
        "ESS_bulk": ess_bulk,
        "ESS_tail": ess_tail,
        "draws_per_chain_used": int(n),
    }


def summarize_quantity(name, draws, chains):
    low, high = hdi(draws)
    p_gt = float(np.mean(draws > 0))
    p_lt = float(np.mean(draws < 0))

    out = {
        "Parameter": name,
        "Mean": float(np.mean(draws)),
        "Median": float(np.median(draws)),
        "SD": float(np.std(draws, ddof=1)),
        "HDI_lower": low,
        "HDI_upper": high,
        "P_gt_0": p_gt,
        "P_lt_0": p_lt,
        "P_direction": max(p_gt, p_lt),
    }
    out.update(convergence_stats(chains))
    return out


def plot_posterior(name, draws, out_file):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(draws, bins=60, density=True, alpha=0.75)
    ax.axvline(0.0, linestyle="--", linewidth=1.5)

    mean = float(np.mean(draws))
    low, high = hdi(draws)
    ax.axvline(mean, linewidth=1.5)

    ax.set_xlabel(name)
    ax.set_ylabel("Posterior density")
    ax.set_title(
        "%s\nMean=%.4f, 95%% HDI=[%.4f, %.4f]"
        % (name, mean, low, high)
    )
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def load_models(model_dir, chains):
    models = []

    for chain in range(chains):
        path = model_dir / ("%s_%d.hddm" % (MODEL_NAME, chain))

        if not path.exists():
            raise FileNotFoundError("Missing expected model file: %s" % path)

        print("Loading chain %d: %s" % (chain, path), flush=True)
        models.append(hddm.load(str(path)))

    return models


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the ES option-specific-theta aDDM and derive "
            "theta_E, theta_S, and theta_E - theta_S draw-by-draw."
        )
    )

    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chains", type=int, default=3)

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.out_dir / "posterior_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76, flush=True)
    print("ES OPTION-SPECIFIC THETA aDDM DIAGNOSTICS", flush=True)
    print("=" * 76, flush=True)

    models = load_models(args.model_dir, args.chains)

    if args.chains >= 2:
        gr = hddm.analyze.gelman_rubin(models)
        pd.Series(gr, name="R_hat").to_csv(
            args.out_dir / "gelman_rubin_all_nodes.csv"
        )

    combined = kabuki.utils.concat_models(models)
    combined.gen_stats().to_csv(
        args.out_dir / "posterior_summary_all_nodes.csv"
    )

    try:
        dic_value = float(combined.dic)
        (args.out_dir / "DIC.txt").write_text(
            "%.12f\n" % dic_value,
            encoding="utf-8",
        )
    except Exception as exc:
        (args.out_dir / "DIC.txt").write_text(
            "DIC unavailable: %s: %s\n" % (type(exc).__name__, exc),
            encoding="utf-8",
        )

    nodes = resolve_nodes(models[0])

    print("\nResolved core nodes:", flush=True)
    print(json.dumps(nodes, indent=2), flush=True)

    (args.out_dir / "resolved_core_node_names.json").write_text(
        json.dumps(nodes, indent=2),
        encoding="utf-8",
    )

    traces = {
        key: []
        for key in ["a", "t", "z", "b0", "bA", "bE", "bS"]
    }

    for model in models:
        chain_nodes = resolve_nodes(model)
        for key in traces:
            traces[key].append(
                get_trace(model, chain_nodes[key])
            )

    theta_E_chains = []
    theta_S_chains = []
    delta_theta_chains = []
    delta_b_chains = []

    for chain in range(args.chains):
        bA = traces["bA"][chain]
        bE = traces["bE"][chain]
        bS = traces["bS"][chain]

        n = min(len(bA), len(bE), len(bS))
        bA = bA[:n]
        bE = bE[:n]
        bS = bS[:n]

        valid = (
            np.isfinite(bA)
            & np.isfinite(bE)
            & np.isfinite(bS)
            & (np.abs(bA) > 1e-12)
        )

        theta_E = bE[valid] / bA[valid]
        theta_S = bS[valid] / bA[valid]

        theta_E_chains.append(theta_E)
        theta_S_chains.append(theta_S)
        delta_theta_chains.append(theta_E - theta_S)
        delta_b_chains.append(bE[valid] - bS[valid])

    derived = {
        "theta_E": theta_E_chains,
        "theta_S": theta_S_chains,
        "delta_theta_E_minus_S": delta_theta_chains,
        "delta_bE_minus_bS": delta_b_chains,
    }

    rows = []

    for key in ["a", "t", "z", "b0", "bA", "bE", "bS"]:
        draws = np.concatenate(traces[key])
        rows.append(
            summarize_quantity(
                key,
                draws,
                traces[key],
            )
        )

    for key in [
        "theta_E",
        "theta_S",
        "delta_theta_E_minus_S",
        "delta_bE_minus_bS",
    ]:
        draws = np.concatenate(derived[key])
        rows.append(
            summarize_quantity(
                key,
                draws,
                derived[key],
            )
        )

    pd.DataFrame(rows).to_csv(
        args.out_dir / "option_specific_theta_summary.csv",
        index=False,
        float_format="%.8f",
    )

    theta_E_all = np.concatenate(theta_E_chains)
    theta_S_all = np.concatenate(theta_S_chains)
    delta_all = np.concatenate(delta_theta_chains)

    delta_low, delta_high = hdi(delta_all)
    delta_conv = convergence_stats(delta_theta_chains)

    theta_comparison = pd.DataFrame(
        [
            {
                "Mean_theta_E": float(np.mean(theta_E_all)),
                "Median_theta_E": float(np.median(theta_E_all)),
                "HDI_theta_E_lower": hdi(theta_E_all)[0],
                "HDI_theta_E_upper": hdi(theta_E_all)[1],
                "Mean_theta_S": float(np.mean(theta_S_all)),
                "Median_theta_S": float(np.median(theta_S_all)),
                "HDI_theta_S_lower": hdi(theta_S_all)[0],
                "HDI_theta_S_upper": hdi(theta_S_all)[1],
                "Mean_delta_theta_E_minus_S": float(np.mean(delta_all)),
                "Median_delta_theta_E_minus_S": float(np.median(delta_all)),
                "HDI_delta_lower": delta_low,
                "HDI_delta_upper": delta_high,
                "P_theta_E_gt_theta_S": float(np.mean(delta_all > 0)),
                "P_theta_E_lt_theta_S": float(np.mean(delta_all < 0)),
                "R_hat_delta_theta": delta_conv["R_hat"],
                "ESS_bulk_delta_theta": delta_conv["ESS_bulk"],
                "ESS_tail_delta_theta": delta_conv["ESS_tail"],
            }
        ]
    )

    theta_comparison.to_csv(
        args.out_dir / "theta_E_vs_theta_S.csv",
        index=False,
        float_format="%.8f",
    )

    min_total = min(
        len(theta_E_all),
        len(theta_S_all),
        len(delta_all),
    )

    pd.DataFrame(
        {
            "theta_E": theta_E_all[:min_total],
            "theta_S": theta_S_all[:min_total],
            "delta_theta_E_minus_S": delta_all[:min_total],
        }
    ).to_csv(
        args.out_dir / "theta_derived_posterior_draws.csv",
        index=False,
        float_format="%.10f",
    )

    coeff_all = {
        key: np.concatenate(traces[key])
        for key in ["bA", "bE", "bS"]
    }

    n_corr = min(len(x) for x in coeff_all.values())

    pd.DataFrame(
        {
            key: value[:n_corr]
            for key, value in coeff_all.items()
        }
    ).corr().to_csv(
        args.out_dir / "posterior_drift_coefficient_correlations.csv",
        float_format="%.8f",
    )

    bA_all = coeff_all["bA"]
    bA_low, bA_high = hdi(bA_all)

    ratio_warning = {
        "bA_mean": float(np.mean(bA_all)),
        "bA_HDI_lower": bA_low,
        "bA_HDI_upper": bA_high,
        "bA_HDI_contains_zero": bool(bA_low <= 0.0 <= bA_high),
        "note": (
            "theta ratios are unstable if bA has substantial posterior "
            "mass close to or across zero."
        ),
    }

    (args.out_dir / "theta_ratio_stability.json").write_text(
        json.dumps(ratio_warning, indent=2),
        encoding="utf-8",
    )

    plot_posterior(
        "theta_E",
        theta_E_all,
        plot_dir / "theta_E.png",
    )

    plot_posterior(
        "theta_S",
        theta_S_all,
        plot_dir / "theta_S.png",
    )

    plot_posterior(
        "theta_E - theta_S",
        delta_all,
        plot_dir / "delta_theta_E_minus_S.png",
    )

    print("\n" + "=" * 76, flush=True)
    print("KEY OPTION-SPECIFIC THETA RESULT", flush=True)
    print("=" * 76, flush=True)
    print(theta_comparison.to_string(index=False), flush=True)

    print(
        "\nPositive delta_theta means theta_E > theta_S. "
        "Because smaller theta means stronger attentional discounting, "
        "a positive delta means S is more strongly discounted when unattended.",
        flush=True,
    )

    print("\nSaved diagnostics to: %s" % args.out_dir, flush=True)


if __name__ == "__main__":
    main()
