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


MODEL_NAME = "aDDM_ES_IDENTITY_S_UPPER_OPTION_SPECIFIC_THETA"


def hdi(x, prob=0.95):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    interval = np.asarray(
        az.hdi(
            x,
            hdi_prob=prob,
        )
    )
    return float(interval[0]), float(interval[1])


def get_trace(model, node_name):
    x = np.asarray(
        model.nodes_db.loc[
            node_name,
            "node",
        ].trace(),
        dtype=float,
    )
    x = x[np.isfinite(x)]

    if x.size == 0:
        raise ValueError(
            f"No finite posterior samples in node {node_name}"
        )

    return x


def resolve_group_node(
    model,
    exact_candidates,
    required_tokens,
):
    names = [
        str(x)
        for x in model.nodes_db.index
    ]

    for candidate in exact_candidates:
        if candidate in names:
            return candidate

    matches = []

    for name in names:
        low = name.lower()

        if "subj" in low:
            continue

        if all(
            token.lower() in low
            for token in required_tokens
        ):
            matches.append(name)

    if len(matches) == 1:
        return matches[0]

    raise KeyError(
        "Could not uniquely resolve group node.\n"
        f"Exact candidates: {exact_candidates}\n"
        f"Required tokens: {required_tokens}\n"
        f"Matches: {matches}\n\n"
        "Available nodes:\n"
        + "\n".join(names)
    )


def resolve_nodes(model):
    return {
        "a": resolve_group_node(
            model,
            ["a"],
            ["a"],
        ),
        "t": resolve_group_node(
            model,
            ["t"],
            ["t"],
        ),
        "z": resolve_group_node(
            model,
            ["z"],
            ["z"],
        ),
        "b0": resolve_group_node(
            model,
            ["v_Intercept"],
            ["v", "intercept"],
        ),
        "bA": resolve_group_node(
            model,
            ["v_AttentionW_SE"],
            ["v", "attentionw_se"],
        ),
        "bIS": resolve_group_node(
            model,
            ["v_Unattended_S"],
            ["v", "unattended_s"],
        ),
        "bIE": resolve_group_node(
            model,
            ["v_Unattended_E"],
            ["v", "unattended_e"],
        ),
    }


def convergence_stats(chain_arrays):
    n = min(
        len(x)
        for x in chain_arrays
    )

    arr = np.vstack(
        [
            np.asarray(
                x[:n],
                dtype=float,
            )
            for x in chain_arrays
        ]
    )

    try:
        rhat = float(
            np.asarray(
                az.rhat(arr)
            )
        ) if arr.shape[0] >= 2 else np.nan
    except Exception:
        rhat = np.nan

    try:
        ess_bulk = float(
            np.asarray(
                az.ess(
                    arr,
                    method="bulk",
                )
            )
        )
    except Exception:
        ess_bulk = np.nan

    try:
        ess_tail = float(
            np.asarray(
                az.ess(
                    arr,
                    method="tail",
                )
            )
        )
    except Exception:
        ess_tail = np.nan

    return {
        "R_hat": rhat,
        "ESS_bulk": ess_bulk,
        "ESS_tail": ess_tail,
        "draws_per_chain_used": int(n),
    }


def summarize(
    name,
    chain_arrays,
):
    draws = np.concatenate(
        chain_arrays
    )

    lo, hi = hdi(draws)

    p_gt = float(
        np.mean(
            draws > 0
        )
    )

    p_lt = float(
        np.mean(
            draws < 0
        )
    )

    out = {
        "Parameter": name,
        "Mean": float(np.mean(draws)),
        "Median": float(np.median(draws)),
        "SD": float(
            np.std(
                draws,
                ddof=1,
            )
        ),
        "HDI_lower": lo,
        "HDI_upper": hi,
        "P_gt_0": p_gt,
        "P_lt_0": p_lt,
        "P_direction": max(
            p_gt,
            p_lt,
        ),
    }

    out.update(
        convergence_stats(
            chain_arrays
        )
    )

    return out


def plot_posterior(
    label,
    draws,
    out_file,
    zero_line=True,
):
    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    ax.hist(
        draws,
        bins=60,
        density=True,
        alpha=0.75,
    )

    if zero_line:
        ax.axvline(
            0.0,
            linestyle="--",
            linewidth=1.5,
        )

    mean = float(
        np.mean(draws)
    )

    lo, hi = hdi(draws)

    ax.axvline(
        mean,
        linewidth=1.5,
    )

    ax.set_xlabel(label)
    ax.set_ylabel("Posterior density")
    ax.set_title(
        "%s\nMean=%.4f, 95%% HDI=[%.4f, %.4f]"
        % (
            label,
            mean,
            lo,
            hi,
        )
    )

    fig.tight_layout()

    fig.savefig(
        out_file,
        dpi=180,
    )

    plt.close(fig)


def load_models(
    model_dir,
    chains,
):
    models = []

    for chain in range(chains):
        path = (
            model_dir
            / f"{MODEL_NAME}_{chain}.hddm"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing expected model: {path}"
            )

        print(
            f"Loading chain {chain}: {path}",
            flush=True,
        )

        models.append(
            hddm.load(
                str(path)
            )
        )

    return models


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the S-upper/E-lower option-specific-theta aDDM."
        )
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--chains",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_dir = (
        args.out_dir
        / "posterior_plots"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    models = load_models(
        args.model_dir,
        args.chains,
    )

    if args.chains >= 2:
        gr = hddm.analyze.gelman_rubin(
            models
        )

        pd.Series(
            gr,
            name="R_hat",
        ).to_csv(
            args.out_dir
            / "gelman_rubin_all_nodes.csv"
        )

    combined = kabuki.utils.concat_models(
        models
    )

    combined.gen_stats().to_csv(
        args.out_dir
        / "posterior_summary_all_nodes.csv"
    )

    try:
        dic = float(
            combined.dic
        )

        (
            args.out_dir
            / "DIC.txt"
        ).write_text(
            f"{dic:.12f}\n",
            encoding="utf-8",
        )
    except Exception as exc:
        (
            args.out_dir
            / "DIC.txt"
        ).write_text(
            f"DIC unavailable: {type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )

    nodes = resolve_nodes(
        models[0]
    )

    (
        args.out_dir
        / "resolved_core_node_names.json"
    ).write_text(
        json.dumps(
            nodes,
            indent=2,
        ),
        encoding="utf-8",
    )

    traces = {
        key: []
        for key in [
            "a",
            "t",
            "z",
            "b0",
            "bA",
            "bIS",
            "bIE",
        ]
    }

    for model in models:
        chain_nodes = resolve_nodes(
            model
        )

        for key in traces:
            traces[key].append(
                get_trace(
                    model,
                    chain_nodes[key],
                )
            )

    theta_S_chains = []
    theta_E_chains = []
    delta_theta_chains = []
    delta_bI_chains = []

    for chain in range(args.chains):
        bA = traces["bA"][chain]
        bIS = traces["bIS"][chain]
        bIE = traces["bIE"][chain]

        n = min(
            len(bA),
            len(bIS),
            len(bIE),
        )

        bA = bA[:n]
        bIS = bIS[:n]
        bIE = bIE[:n]

        valid = (
            np.isfinite(bA)
            & np.isfinite(bIS)
            & np.isfinite(bIE)
            & (np.abs(bA) > 1e-12)
        )

        theta_S = (
            bIS[valid]
            / bA[valid]
        )

        theta_E = (
            bIE[valid]
            / bA[valid]
        )

        theta_S_chains.append(
            theta_S
        )

        theta_E_chains.append(
            theta_E
        )

        delta_theta_chains.append(
            theta_S
            - theta_E
        )

        delta_bI_chains.append(
            bIS[valid]
            - bIE[valid]
        )

    derived = {
        "theta_S": theta_S_chains,
        "theta_E": theta_E_chains,
        "delta_theta_S_minus_E": delta_theta_chains,
        "delta_bI_S_minus_E": delta_bI_chains,
    }

    rows = []

    for key in traces:
        rows.append(
            summarize(
                key,
                traces[key],
            )
        )

    for key in derived:
        rows.append(
            summarize(
                key,
                derived[key],
            )
        )

    pd.DataFrame(
        rows
    ).to_csv(
        args.out_dir
        / "option_specific_theta_S_upper_summary.csv",
        index=False,
        float_format="%.8f",
    )

    theta_S_all = np.concatenate(
        theta_S_chains
    )

    theta_E_all = np.concatenate(
        theta_E_chains
    )

    delta_theta_all = np.concatenate(
        delta_theta_chains
    )

    bA_all = np.concatenate(
        traces["bA"]
    )

    bIS_all = np.concatenate(
        traces["bIS"]
    )

    bIE_all = np.concatenate(
        traces["bIE"]
    )

    primary = pd.DataFrame(
        [
            {
                "Mean_bA_common_attended": float(
                    np.mean(bA_all)
                ),
                "HDI_bA_lower": hdi(
                    bA_all
                )[0],
                "HDI_bA_upper": hdi(
                    bA_all
                )[1],

                "Mean_bIS_unattended_S": float(
                    np.mean(bIS_all)
                ),
                "HDI_bIS_lower": hdi(
                    bIS_all
                )[0],
                "HDI_bIS_upper": hdi(
                    bIS_all
                )[1],

                "Mean_bIE_unattended_E": float(
                    np.mean(bIE_all)
                ),
                "HDI_bIE_lower": hdi(
                    bIE_all
                )[0],
                "HDI_bIE_upper": hdi(
                    bIE_all
                )[1],

                "Mean_theta_S": float(
                    np.mean(theta_S_all)
                ),
                "HDI_theta_S_lower": hdi(
                    theta_S_all
                )[0],
                "HDI_theta_S_upper": hdi(
                    theta_S_all
                )[1],

                "Mean_theta_E": float(
                    np.mean(theta_E_all)
                ),
                "HDI_theta_E_lower": hdi(
                    theta_E_all
                )[0],
                "HDI_theta_E_upper": hdi(
                    theta_E_all
                )[1],

                "Mean_delta_theta_S_minus_E": float(
                    np.mean(delta_theta_all)
                ),
                "HDI_delta_theta_lower": hdi(
                    delta_theta_all
                )[0],
                "HDI_delta_theta_upper": hdi(
                    delta_theta_all
                )[1],

                "P_theta_S_gt_theta_E": float(
                    np.mean(
                        delta_theta_all > 0
                    )
                ),

                "P_theta_E_gt_theta_S": float(
                    np.mean(
                        delta_theta_all < 0
                    )
                ),

                "R_hat_delta_theta": convergence_stats(
                    delta_theta_chains
                )["R_hat"],

                "ESS_bulk_delta_theta": convergence_stats(
                    delta_theta_chains
                )["ESS_bulk"],

                "ESS_tail_delta_theta": convergence_stats(
                    delta_theta_chains
                )["ESS_tail"],
            }
        ]
    )

    primary.to_csv(
        args.out_dir
        / "theta_S_vs_E_primary_results.csv",
        index=False,
        float_format="%.8f",
    )

    # Ratio stability
    bA_lo, bA_hi = hdi(
        bA_all
    )

    stability = {
        "bA_mean": float(
            np.mean(bA_all)
        ),
        "bA_HDI_lower": bA_lo,
        "bA_HDI_upper": bA_hi,
        "bA_HDI_contains_zero": bool(
            bA_lo <= 0 <= bA_hi
        ),
        "theta_ratio_interpretation_safe": bool(
            not (
                bA_lo <= 0 <= bA_hi
            )
        ),
        "note": (
            "theta_S=bIS/bA and theta_E=bIE/bA. "
            "Ratios are unstable if bA has substantial posterior mass near zero."
        ),
    }

    (
        args.out_dir
        / "theta_ratio_stability.json"
    ).write_text(
        json.dumps(
            stability,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Posterior correlations among drift coefficients
    n_corr = min(
        len(bA_all),
        len(bIS_all),
        len(bIE_all),
    )

    pd.DataFrame(
        {
            "bA": bA_all[:n_corr],
            "bIS": bIS_all[:n_corr],
            "bIE": bIE_all[:n_corr],
        }
    ).corr().to_csv(
        args.out_dir
        / "posterior_drift_coefficient_correlations.csv",
        float_format="%.8f",
    )

    # Save derived posterior draws
    n_draw = min(
        len(theta_S_all),
        len(theta_E_all),
        len(delta_theta_all),
    )

    pd.DataFrame(
        {
            "theta_S": theta_S_all[:n_draw],
            "theta_E": theta_E_all[:n_draw],
            "delta_theta_S_minus_E": delta_theta_all[:n_draw],
        }
    ).to_csv(
        args.out_dir
        / "option_specific_theta_posterior_draws.csv",
        index=False,
        float_format="%.10f",
    )

    plot_posterior(
        "theta_S",
        theta_S_all,
        plot_dir / "theta_S.png",
        zero_line=True,
    )

    plot_posterior(
        "theta_E",
        theta_E_all,
        plot_dir / "theta_E.png",
        zero_line=True,
    )

    plot_posterior(
        "theta_S - theta_E",
        delta_theta_all,
        plot_dir / "delta_theta_S_minus_E.png",
        zero_line=True,
    )

    print(
        "\n" + "=" * 76,
        flush=True,
    )

    print(
        "PRIMARY OPTION-SPECIFIC THETA RESULT",
        flush=True,
    )

    print(
        "=" * 76,
        flush=True,
    )

    print(
        primary.to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "\nDirection reminder:",
        flush=True,
    )

    print(
        "smaller theta_S -> S is more strongly discounted while unattended",
        flush=True,
    )

    print(
        "smaller theta_E -> E is more strongly discounted while unattended",
        flush=True,
    )

    print(
        "positive theta_S - theta_E -> E is more strongly discounted than S",
        flush=True,
    )

    print(
        "negative theta_S - theta_E -> S is more strongly discounted than E",
        flush=True,
    )


if __name__ == "__main__":
    main()
