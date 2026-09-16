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


MODEL_NAME = "aDDM_ES_IDENTITY_S_UPPER_JOINT_CONTRASTS"


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
        "bI": resolve_group_node(
            model,
            ["v_InattentionW_SE"],
            ["v", "inattentionw_se"],
        ),
        "deltaA": resolve_group_node(
            model,
            ["v_AttentionContrast_SE"],
            ["v", "attentioncontrast_se"],
        ),
        "deltaI": resolve_group_node(
            model,
            ["v_InattentionContrast_SE"],
            ["v", "inattentioncontrast_se"],
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
        np.mean(draws > 0)
    )

    p_lt = float(
        np.mean(draws < 0)
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
            "Diagnose ES S-upper/E-lower identity joint E/S contrast aDDM."
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

    dic_value = np.nan

    try:
        dic_value = float(
            combined.dic
        )

        (
            args.out_dir
            / "DIC.txt"
        ).write_text(
            "%.12f\n"
            % dic_value,
            encoding="utf-8",
        )

    except Exception as exc:
        (
            args.out_dir
            / "DIC.txt"
        ).write_text(
            "DIC unavailable: %s: %s\n"
            % (
                type(exc).__name__,
                exc,
            ),
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
            "bI",
            "deltaA",
            "deltaI",
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

    derived = {
        "bAS": [],
        "bAE": [],
        "bIS": [],
        "bIE": [],
        "theta_S": [],
        "theta_E": [],
        "delta_theta_S_minus_E": [],
    }

    ratio_safety_rows = []

    for chain in range(
        args.chains
    ):
        bA = traces["bA"][chain]
        bI = traces["bI"][chain]
        deltaA = traces["deltaA"][chain]
        deltaI = traces["deltaI"][chain]

        n = min(
            len(bA),
            len(bI),
            len(deltaA),
            len(deltaI),
        )

        bA = bA[:n]
        bI = bI[:n]
        deltaA = deltaA[:n]
        deltaI = deltaI[:n]

        # In the S-upper identity parameterization:
        # deltaA = bAS - bAE
        # deltaI = bIS - bIE
        bAS = (
            bA
            + deltaA / 2.0
        )

        bAE = (
            bA
            - deltaA / 2.0
        )

        bIS = (
            bI
            + deltaI / 2.0
        )

        bIE = (
            bI
            - deltaI / 2.0
        )

        valid = (
            np.isfinite(bAS)
            & np.isfinite(bAE)
            & np.isfinite(bIS)
            & np.isfinite(bIE)
            & (np.abs(bAS) > 1e-12)
            & (np.abs(bAE) > 1e-12)
        )

        theta_S = (
            bIS[valid]
            / bAS[valid]
        )

        theta_E = (
            bIE[valid]
            / bAE[valid]
        )

        derived["bAS"].append(
            bAS
        )

        derived["bAE"].append(
            bAE
        )

        derived["bIS"].append(
            bIS
        )

        derived["bIE"].append(
            bIE
        )

        derived["theta_S"].append(
            theta_S
        )

        derived["theta_E"].append(
            theta_E
        )

        derived[
            "delta_theta_S_minus_E"
        ].append(
            theta_S
            - theta_E
        )

        ratio_safety_rows.append(
            {
                "chain": chain,
                "P_abs_bAS_lt_0p1": float(
                    np.mean(np.abs(bAS) < 0.1)
                ),
                "P_abs_bAE_lt_0p1": float(
                    np.mean(np.abs(bAE) < 0.1)
                ),
            }
        )

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

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        args.out_dir
        / "identity_joint_es_contrast_summary.csv",
        index=False,
        float_format="%.8f",
    )

    deltaA_all = np.concatenate(
        traces["deltaA"]
    )

    deltaI_all = np.concatenate(
        traces["deltaI"]
    )

    theta_S_all = np.concatenate(
        derived["theta_S"]
    )

    theta_E_all = np.concatenate(
        derived["theta_E"]
    )

    delta_theta_all = np.concatenate(
        derived["delta_theta_S_minus_E"]
    )

    primary = pd.DataFrame(
        [
            {
                "Mean_deltaA_attended_S_minus_E": float(
                    np.mean(deltaA_all)
                ),
                "HDI_deltaA_lower": hdi(
                    deltaA_all
                )[0],
                "HDI_deltaA_upper": hdi(
                    deltaA_all
                )[1],
                "P_deltaA_gt_0": float(
                    np.mean(deltaA_all > 0)
                ),
                "R_hat_deltaA": convergence_stats(
                    traces["deltaA"]
                )["R_hat"],
                "ESS_bulk_deltaA": convergence_stats(
                    traces["deltaA"]
                )["ESS_bulk"],

                "Mean_deltaI_unattended_S_minus_E": float(
                    np.mean(deltaI_all)
                ),
                "HDI_deltaI_lower": hdi(
                    deltaI_all
                )[0],
                "HDI_deltaI_upper": hdi(
                    deltaI_all
                )[1],
                "P_deltaI_gt_0": float(
                    np.mean(deltaI_all > 0)
                ),
                "R_hat_deltaI": convergence_stats(
                    traces["deltaI"]
                )["R_hat"],
                "ESS_bulk_deltaI": convergence_stats(
                    traces["deltaI"]
                )["ESS_bulk"],

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

                "Mean_theta_S_minus_E": float(
                    np.mean(delta_theta_all)
                ),
                "HDI_theta_S_minus_E_lower": hdi(
                    delta_theta_all
                )[0],
                "HDI_theta_S_minus_E_upper": hdi(
                    delta_theta_all
                )[1],
                "P_theta_S_gt_theta_E": float(
                    np.mean(delta_theta_all > 0)
                ),
                "R_hat_theta_S_minus_E": convergence_stats(
                    derived["delta_theta_S_minus_E"]
                )["R_hat"],
                "ESS_bulk_theta_S_minus_E": convergence_stats(
                    derived["delta_theta_S_minus_E"]
                )["ESS_bulk"],

                "DIC": dic_value,
            }
        ]
    )

    primary.to_csv(
        args.out_dir
        / "identity_joint_primary_results.csv",
        index=False,
        float_format="%.8f",
    )

    drift_all = {
        "bA": np.concatenate(
            traces["bA"]
        ),
        "bI": np.concatenate(
            traces["bI"]
        ),
        "deltaA": deltaA_all,
        "deltaI": deltaI_all,
    }

    n_corr = min(
        len(x)
        for x in drift_all.values()
    )

    pd.DataFrame(
        {
            key: value[:n_corr]
            for key, value in drift_all.items()
        }
    ).corr().to_csv(
        args.out_dir
        / "posterior_drift_coefficient_correlations.csv",
        float_format="%.8f",
    )

    bAS_all = np.concatenate(
        derived["bAS"]
    )

    bAE_all = np.concatenate(
        derived["bAE"]
    )

    bAS_hdi = hdi(bAS_all)
    bAE_hdi = hdi(bAE_all)

    ratio_stability = {
        "bAS_mean": float(np.mean(bAS_all)),
        "bAS_HDI_lower": bAS_hdi[0],
        "bAS_HDI_upper": bAS_hdi[1],
        "bAS_HDI_contains_zero": bool(
            bAS_hdi[0] <= 0 <= bAS_hdi[1]
        ),
        "bAE_mean": float(np.mean(bAE_all)),
        "bAE_HDI_lower": bAE_hdi[0],
        "bAE_HDI_upper": bAE_hdi[1],
        "bAE_HDI_contains_zero": bool(
            bAE_hdi[0] <= 0 <= bAE_hdi[1]
        ),
        "theta_ratio_interpretation_safe": bool(
            not (bAS_hdi[0] <= 0 <= bAS_hdi[1])
            and not (bAE_hdi[0] <= 0 <= bAE_hdi[1])
        ),
        "note": (
            "theta_S=bIS/bAS and theta_E=bIE/bAE. "
            "Ratios become unstable if either attended denominator has "
            "substantial posterior mass near zero."
        ),
        "per_chain_near_zero_probability": ratio_safety_rows,
    }

    (
        args.out_dir
        / "theta_ratio_stability.json"
    ).write_text(
        json.dumps(
            ratio_stability,
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_posterior(
        "deltaA: attended S - E",
        deltaA_all,
        plot_dir
        / "deltaA_attended_S_minus_E.png",
    )

    plot_posterior(
        "deltaI: unattended S - E",
        deltaI_all,
        plot_dir
        / "deltaI_unattended_S_minus_E.png",
    )

    plot_posterior(
        "theta_S",
        theta_S_all,
        plot_dir
        / "theta_S.png",
        zero_line=True,
    )

    plot_posterior(
        "theta_E",
        theta_E_all,
        plot_dir
        / "theta_E.png",
        zero_line=True,
    )

    plot_posterior(
        "theta_S - theta_E",
        delta_theta_all,
        plot_dir
        / "delta_theta_S_minus_E.png",
    )

    print(
        "\nPRIMARY S-UPPER IDENTITY JOINT-CONTRAST RESULTS",
        flush=True,
    )

    print(
        primary.to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "\nInterpretation: positive deltaA means stronger attended S than E "
        "weighting; positive deltaI means stronger unattended S than E "
        "weighting. theta_S and theta_E each use their own attended "
        "denominator.",
        flush=True,
    )

    print(
        "\nTHETA RATIO STABILITY",
        flush=True,
    )

    print(
        json.dumps(
            ratio_stability,
            indent=2,
        ),
        flush=True,
    )

    print(
        "\nSaved diagnostics to: %s"
        % args.out_dir,
        flush=True,
    )


if __name__ == "__main__":
    main()
