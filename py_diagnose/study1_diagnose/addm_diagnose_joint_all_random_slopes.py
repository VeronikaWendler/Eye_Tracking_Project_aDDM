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


MODEL_NAME = "aDDM_JOINT_ALL_RANDOM_SLOPES"


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
        "a0": resolve_group_node(
            model, ["a_Intercept"], ["a", "intercept"]
        ),
        "da": resolve_group_node(
            model, ["a_phase_ES"], ["a", "phase_es"]
        ),

        "t0": resolve_group_node(
            model, ["t_Intercept"], ["t", "intercept"]
        ),
        "dt": resolve_group_node(
            model, ["t_phase_ES"], ["t", "phase_es"]
        ),

        "z0": resolve_group_node(
            model, ["z_Intercept"], ["z", "intercept"]
        ),
        "dz": resolve_group_node(
            model, ["z_phase_ES"], ["z", "phase_es"]
        ),

        "b0": resolve_group_node(
            model, ["v_Intercept"], ["v", "intercept"]
        ),
        "db0": resolve_group_node(
            model, ["v_phase_ES"], ["v", "phase_es"]
        ),

        "b1": resolve_group_node(
            model, ["v_AttentionW"], ["v", "attentionw"]
        ),
        "db1": resolve_group_node(
            model,
            ["v_phase_ES:AttentionW", "v_AttentionW:phase_ES"],
            ["v", "phase_es", "attentionw"],
        ),

        "b2": resolve_group_node(
            model, ["v_InattentionW"], ["v", "inattentionw"]
        ),
        "db2": resolve_group_node(
            model,
            ["v_phase_ES:InattentionW", "v_InattentionW:phase_ES"],
            ["v", "phase_es", "inattentionw"],
        ),
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


def summarize_phase_difference(
    parameter,
    ee_draws,
    es_draws,
    diff_draws,
    diff_chains,
):
    ee_low, ee_high = hdi(ee_draws)
    es_low, es_high = hdi(es_draws)
    diff_low, diff_high = hdi(diff_draws)

    p_gt = float(np.mean(diff_draws > 0))
    p_lt = float(np.mean(diff_draws < 0))

    conv = convergence_stats(diff_chains)

    return {
        "Parameter": parameter,

        "Mean_EE": float(np.mean(ee_draws)),
        "Median_EE": float(np.median(ee_draws)),
        "HDI_EE_lower": ee_low,
        "HDI_EE_upper": ee_high,

        "Mean_ES": float(np.mean(es_draws)),
        "Median_ES": float(np.median(es_draws)),
        "HDI_ES_lower": es_low,
        "HDI_ES_upper": es_high,

        "Mean_Difference_ES_minus_EE": float(np.mean(diff_draws)),
        "Median_Difference_ES_minus_EE": float(np.median(diff_draws)),
        "HDI_Difference_lower": diff_low,
        "HDI_Difference_upper": diff_high,

        "P_ES_gt_EE": p_gt,
        "P_ES_lt_EE": p_lt,
        "P_direction": max(p_gt, p_lt),

        "R_hat_Difference": conv["R_hat"],
        "ESS_bulk_Difference": conv["ESS_bulk"],
        "ESS_tail_Difference": conv["ESS_tail"],
        "draws_per_chain_used": conv["draws_per_chain_used"],
    }


def plot_difference(parameter, diff_draws, out_file):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(diff_draws, bins=60, density=True, alpha=0.75)
    ax.axvline(0.0, linestyle="--", linewidth=1.5)

    mean = float(np.mean(diff_draws))
    low, high = hdi(diff_draws)
    ax.axvline(mean, linewidth=1.5)

    ax.set_xlabel("%s: ES - EE" % parameter)
    ax.set_ylabel("Posterior density")
    ax.set_title(
        "%s phase effect\nMean=%.4f, 95%% HDI=[%.4f, %.4f]"
        % (parameter, mean, low, high)
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
            "Diagnose the all-random-slopes joint aDDM and report "
            "direct hierarchical ES-EE regression effects."
        )
    )

    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chains", type=int, default=3)

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.out_dir / "posterior_difference_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76, flush=True)
    print("ALL-RANDOM-SLOPES JOINT aDDM DIAGNOSTICS", flush=True)
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

    print("\nResolved regression nodes:", flush=True)
    print(json.dumps(nodes, indent=2), flush=True)

    (args.out_dir / "resolved_regression_node_names.json").write_text(
        json.dumps(nodes, indent=2),
        encoding="utf-8",
    )

    base = {
        key: []
        for key in [
            "a0", "da",
            "t0", "dt",
            "z0", "dz",
            "b0", "db0",
            "b1", "db1",
            "b2", "db2",
        ]
    }

    for model in models:
        chain_nodes = resolve_nodes(model)

        for key in base:
            base[key].append(
                get_trace(model, chain_nodes[key])
            )

    phase = {
        p: {"EE": [], "ES": [], "D": []}
        for p in ["a", "t", "z", "b0", "b1", "b2", "theta"]
    }

    pair_map = {
        "a": ("a0", "da"),
        "t": ("t0", "dt"),
        "z": ("z0", "dz"),
        "b0": ("b0", "db0"),
        "b1": ("b1", "db1"),
        "b2": ("b2", "db2"),
    }

    for chain in range(args.chains):

        for parameter, keys in pair_map.items():
            intercept = base[keys[0]][chain]
            slope = base[keys[1]][chain]

            n = min(len(intercept), len(slope))
            intercept = intercept[:n]
            slope = slope[:n]

            valid = np.isfinite(intercept) & np.isfinite(slope)

            ee = intercept[valid]
            diff = slope[valid]
            es = ee + diff

            phase[parameter]["EE"].append(ee)
            phase[parameter]["ES"].append(es)
            phase[parameter]["D"].append(diff)

        b1_ee = phase["b1"]["EE"][chain]
        b1_es = phase["b1"]["ES"][chain]
        b2_ee = phase["b2"]["EE"][chain]
        b2_es = phase["b2"]["ES"][chain]

        n = min(
            len(b1_ee),
            len(b1_es),
            len(b2_ee),
            len(b2_es),
        )

        b1_ee = b1_ee[:n]
        b1_es = b1_es[:n]
        b2_ee = b2_ee[:n]
        b2_es = b2_es[:n]

        valid = (
            np.isfinite(b1_ee)
            & np.isfinite(b1_es)
            & np.isfinite(b2_ee)
            & np.isfinite(b2_es)
            & (np.abs(b1_ee) > 1e-12)
            & (np.abs(b1_es) > 1e-12)
        )

        theta_ee = b2_ee[valid] / b1_ee[valid]
        theta_es = b2_es[valid] / b1_es[valid]

        phase["theta"]["EE"].append(theta_ee)
        phase["theta"]["ES"].append(theta_es)
        phase["theta"]["D"].append(theta_es - theta_ee)

    rows = []
    saved_diffs = {}

    for parameter in ["a", "t", "z", "b0", "b1", "b2", "theta"]:
        ee_all = np.concatenate(phase[parameter]["EE"])
        es_all = np.concatenate(phase[parameter]["ES"])
        diff_all = np.concatenate(phase[parameter]["D"])

        saved_diffs[parameter] = diff_all

        rows.append(
            summarize_phase_difference(
                parameter,
                ee_all,
                es_all,
                diff_all,
                phase[parameter]["D"],
            )
        )

        plot_difference(
            parameter,
            diff_all,
            plot_dir / ("delta_%s_ES_minus_EE.png" % parameter),
        )

    comparison = pd.DataFrame(rows)

    comparison.to_csv(
        args.out_dir / "random_slopes_phase_comparison.csv",
        index=False,
        float_format="%.8f",
    )

    min_len = min(len(x) for x in saved_diffs.values())

    pd.DataFrame(
        {
            key: value[:min_len]
            for key, value in saved_diffs.items()
        }
    ).to_csv(
        args.out_dir / "random_slopes_phase_difference_draws.csv",
        index=False,
        float_format="%.10f",
    )

    coef_rows = []

    for key in base:
        draws = np.concatenate(base[key])
        low, high = hdi(draws)
        conv = convergence_stats(base[key])

        coef_rows.append(
            {
                "Coefficient": key,
                "Mean": float(np.mean(draws)),
                "Median": float(np.median(draws)),
                "SD": float(np.std(draws, ddof=1)),
                "HDI_lower": low,
                "HDI_upper": high,
                "P_gt_0": float(np.mean(draws > 0)),
                "P_lt_0": float(np.mean(draws < 0)),
                "R_hat": conv["R_hat"],
                "ESS_bulk": conv["ESS_bulk"],
                "ESS_tail": conv["ESS_tail"],
            }
        )

    pd.DataFrame(coef_rows).to_csv(
        args.out_dir / "group_regression_coefficients.csv",
        index=False,
        float_format="%.8f",
    )

    b1_ee_all = np.concatenate(phase["b1"]["EE"])
    b1_es_all = np.concatenate(phase["b1"]["ES"])

    ee_low, ee_high = hdi(b1_ee_all)
    es_low, es_high = hdi(b1_es_all)

    ratio_stability = {
        "b1_EE_mean": float(np.mean(b1_ee_all)),
        "b1_EE_HDI_lower": ee_low,
        "b1_EE_HDI_upper": ee_high,
        "b1_EE_HDI_contains_zero": bool(ee_low <= 0.0 <= ee_high),

        "b1_ES_mean": float(np.mean(b1_es_all)),
        "b1_ES_HDI_lower": es_low,
        "b1_ES_HDI_upper": es_high,
        "b1_ES_HDI_contains_zero": bool(es_low <= 0.0 <= es_high),

        "note": (
            "theta=b2/b1 can be unstable if the attended-value coefficient "
            "has substantial posterior mass close to or across zero."
        ),
    }

    (args.out_dir / "theta_ratio_stability.json").write_text(
        json.dumps(ratio_stability, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 76, flush=True)
    print("DIRECT HIERARCHICAL PHASE EFFECTS: ES - EE", flush=True)
    print("=" * 76, flush=True)

    display_cols = [
        "Parameter",
        "Mean_EE",
        "Mean_ES",
        "Mean_Difference_ES_minus_EE",
        "HDI_Difference_lower",
        "HDI_Difference_upper",
        "P_ES_gt_EE",
        "R_hat_Difference",
        "ESS_bulk_Difference",
    ]

    print(
        comparison[display_cols].to_string(index=False),
        flush=True,
    )

    print(
        "\nFor a, t, z, b0, b1 and b2, the ES-EE difference is the "
        "fitted group-level phase regression coefficient itself. "
        "theta is derived draw-by-draw from the two phase-specific ratios.",
        flush=True,
    )

    print("\nSaved diagnostics to: %s" % args.out_dir, flush=True)


if __name__ == "__main__":
    main()
