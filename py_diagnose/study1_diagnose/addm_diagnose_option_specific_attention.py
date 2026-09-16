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


MODEL_NAME = "aDDM_ES_OPTION_SPECIFIC_ATTENTION"


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
        % (
            exact_candidates,
            required_tokens,
            matches,
            "\n".join(names),
        )
    )


def resolve_nodes(model):
    return {
        "a": resolve_group_node(model, ["a"], ["a"]),
        "t": resolve_group_node(model, ["t"], ["t"]),
        "z": resolve_group_node(model, ["z"], ["z"]),
        "b0": resolve_group_node(
            model,
            ["v_Intercept"],
            ["v", "intercept"],
        ),
        "bAE": resolve_group_node(
            model,
            ["v_AttentionW_E"],
            ["v", "attentionw_e"],
        ),
        "bAS": resolve_group_node(
            model,
            ["v_AttentionW_S"],
            ["v", "attentionw_s"],
        ),
        "bI": resolve_group_node(
            model,
            ["v_InattentionW"],
            ["v", "inattentionw"],
        ),
    }


def convergence_stats(chain_arrays):
    n = min(len(x) for x in chain_arrays)
    arr = np.vstack(
        [
            np.asarray(x[:n], dtype=float)
            for x in chain_arrays
        ]
    )

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
            "Diagnose the ES option-specific-attention aDDM and test "
            "whether attended E and attended S have different drift weights."
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
    print("ES OPTION-SPECIFIC ATTENTION aDDM DIAGNOSTICS", flush=True)
    print("=" * 76, flush=True)

    models = load_models(args.model_dir, args.chains)

    if args.chains >= 2:
        print("\nComputing Gelman-Rubin for all HDDM nodes...", flush=True)
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
        for key in ["a", "t", "z", "b0", "bAE", "bAS", "bI"]
    }

    for model in models:
        chain_nodes = resolve_nodes(model)
        for key in traces:
            traces[key].append(
                get_trace(model, chain_nodes[key])
            )

    delta_bA_chains = []
    theta_S_when_E_attended_chains = []
    theta_E_when_S_attended_chains = []
    delta_relative_theta_chains = []

    for chain in range(args.chains):
        bAE = traces["bAE"][chain]
        bAS = traces["bAS"][chain]
        bI = traces["bI"][chain]

        n = min(len(bAE), len(bAS), len(bI))
        bAE = bAE[:n]
        bAS = bAS[:n]
        bI = bI[:n]

        valid = (
            np.isfinite(bAE)
            & np.isfinite(bAS)
            & np.isfinite(bI)
        )

        bAE_v = bAE[valid]
        bAS_v = bAS[valid]
        bI_v = bI[valid]

        delta_bA_chains.append(
            bAE_v - bAS_v
        )

        ratio_valid = (
            (np.abs(bAE_v) > 1e-12)
            & (np.abs(bAS_v) > 1e-12)
        )

        theta_S_when_E_attended = (
            bI_v[ratio_valid] / bAE_v[ratio_valid]
        )

        theta_E_when_S_attended = (
            bI_v[ratio_valid] / bAS_v[ratio_valid]
        )

        theta_S_when_E_attended_chains.append(
            theta_S_when_E_attended
        )

        theta_E_when_S_attended_chains.append(
            theta_E_when_S_attended
        )

        delta_relative_theta_chains.append(
            theta_E_when_S_attended
            - theta_S_when_E_attended
        )

    derived = {
        "delta_bA_E_minus_S": delta_bA_chains,
        "theta_unattended_S_when_E_attended": theta_S_when_E_attended_chains,
        "theta_unattended_E_when_S_attended": theta_E_when_S_attended_chains,
        "delta_relative_theta_E_minus_S": delta_relative_theta_chains,
    }

    rows = []

    for key in ["a", "t", "z", "b0", "bAE", "bAS", "bI"]:
        draws = np.concatenate(traces[key])
        rows.append(
            summarize_quantity(
                key,
                draws,
                traces[key],
            )
        )

    for key in [
        "delta_bA_E_minus_S",
        "theta_unattended_S_when_E_attended",
        "theta_unattended_E_when_S_attended",
        "delta_relative_theta_E_minus_S",
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
        args.out_dir / "option_specific_attention_summary.csv",
        index=False,
        float_format="%.8f",
    )

    bAE_all = np.concatenate(traces["bAE"])
    bAS_all = np.concatenate(traces["bAS"])
    bI_all = np.concatenate(traces["bI"])
    delta_all = np.concatenate(delta_bA_chains)

    delta_low, delta_high = hdi(delta_all)
    delta_conv = convergence_stats(delta_bA_chains)

    primary = pd.DataFrame(
        [
            {
                "Mean_bAE": float(np.mean(bAE_all)),
                "Median_bAE": float(np.median(bAE_all)),
                "HDI_bAE_lower": hdi(bAE_all)[0],
                "HDI_bAE_upper": hdi(bAE_all)[1],

                "Mean_bAS": float(np.mean(bAS_all)),
                "Median_bAS": float(np.median(bAS_all)),
                "HDI_bAS_lower": hdi(bAS_all)[0],
                "HDI_bAS_upper": hdi(bAS_all)[1],

                "Mean_shared_bI": float(np.mean(bI_all)),
                "HDI_shared_bI_lower": hdi(bI_all)[0],
                "HDI_shared_bI_upper": hdi(bI_all)[1],

                "Mean_delta_bAE_minus_bAS": float(np.mean(delta_all)),
                "Median_delta_bAE_minus_bAS": float(np.median(delta_all)),
                "HDI_delta_lower": delta_low,
                "HDI_delta_upper": delta_high,

                "P_bAE_gt_bAS": float(np.mean(delta_all > 0)),
                "P_bAE_lt_bAS": float(np.mean(delta_all < 0)),

                "R_hat_delta_bA": delta_conv["R_hat"],
                "ESS_bulk_delta_bA": delta_conv["ESS_bulk"],
                "ESS_tail_delta_bA": delta_conv["ESS_tail"],
            }
        ]
    )

    primary.to_csv(
        args.out_dir / "attention_E_vs_S.csv",
        index=False,
        float_format="%.8f",
    )

    secondary_1 = np.concatenate(
        theta_S_when_E_attended_chains
    )

    secondary_2 = np.concatenate(
        theta_E_when_S_attended_chains
    )

    secondary_delta = np.concatenate(
        delta_relative_theta_chains
    )

    min_len = min(
        len(bAE_all),
        len(bAS_all),
        len(bI_all),
        len(delta_all),
        len(secondary_1),
        len(secondary_2),
        len(secondary_delta),
    )

    pd.DataFrame(
        {
            "bAE": bAE_all[:min_len],
            "bAS": bAS_all[:min_len],
            "shared_bI": bI_all[:min_len],
            "delta_bAE_minus_bAS": delta_all[:min_len],
            "theta_unattended_S_when_E_attended": secondary_1[:min_len],
            "theta_unattended_E_when_S_attended": secondary_2[:min_len],
            "delta_relative_theta_E_minus_S": secondary_delta[:min_len],
        }
    ).to_csv(
        args.out_dir / "attention_specific_posterior_draws.csv",
        index=False,
        float_format="%.10f",
    )

    coeff_all = {
        "bAE": bAE_all,
        "bAS": bAS_all,
        "bI": bI_all,
    }

    n_corr = min(
        len(x)
        for x in coeff_all.values()
    )

    pd.DataFrame(
        {
            key: value[:n_corr]
            for key, value in coeff_all.items()
        }
    ).corr().to_csv(
        args.out_dir / "posterior_drift_coefficient_correlations.csv",
        float_format="%.8f",
    )

    bAE_low, bAE_high = hdi(bAE_all)
    bAS_low, bAS_high = hdi(bAS_all)

    ratio_stability = {
        "bAE_mean": float(np.mean(bAE_all)),
        "bAE_HDI_lower": bAE_low,
        "bAE_HDI_upper": bAE_high,
        "bAE_HDI_contains_zero": bool(
            bAE_low <= 0.0 <= bAE_high
        ),
        "bAS_mean": float(np.mean(bAS_all)),
        "bAS_HDI_lower": bAS_low,
        "bAS_HDI_upper": bAS_high,
        "bAS_HDI_contains_zero": bool(
            bAS_low <= 0.0 <= bAS_high
        ),
        "note": (
            "The direct bAE-bAS contrast is the primary test. "
            "The theta-like ratios are secondary and become unstable "
            "if either attended coefficient approaches zero."
        ),
    }

    (args.out_dir / "attention_ratio_stability.json").write_text(
        json.dumps(ratio_stability, indent=2),
        encoding="utf-8",
    )

    plot_posterior(
        "bAE: attended E weight",
        bAE_all,
        plot_dir / "bAE_attended_E.png",
    )

    plot_posterior(
        "bAS: attended S weight",
        bAS_all,
        plot_dir / "bAS_attended_S.png",
    )

    plot_posterior(
        "bAE - bAS",
        delta_all,
        plot_dir / "delta_bAE_minus_bAS.png",
    )

    plot_posterior(
        "shared bI",
        bI_all,
        plot_dir / "shared_inattention_weight.png",
    )

    print("\n" + "=" * 76, flush=True)
    print("PRIMARY RESULT: ATTENDED E VS ATTENDED S", flush=True)
    print("=" * 76, flush=True)
    print(primary.to_string(index=False), flush=True)

    print(
        "\nPrimary interpretation: delta_bA = bAE - bAS. "
        "Positive values mean attended E value has a stronger effect "
        "on drift than attended S value. Negative values mean attended "
        "S value has a stronger effect.",
        flush=True,
    )

    print(
        "\nSaved diagnostics to: %s"
        % args.out_dir,
        flush=True,
    )


if __name__ == "__main__":
    main()
