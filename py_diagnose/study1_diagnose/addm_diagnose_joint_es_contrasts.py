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


MODEL_NAME = "aDDM_ES_JOINT_ES_CONTRASTS"


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
    x = x[
        np.isfinite(x)
    ]
    if x.size == 0:
        raise ValueError(
            "No finite posterior draws for %s"
            % node_name
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
            matches.append(
                name
            )

    if len(matches) == 1:
        return matches[0]

    raise KeyError(
        "Could not uniquely resolve node.\n"
        "Candidates=%s\nTokens=%s\nMatches=%s\n\nAvailable:\n%s"
        % (
            exact_candidates,
            required_tokens,
            matches,
            "\n".join(names),
        )
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
            ["v_AttentionW"],
            ["v", "attentionw"],
        ),
        "bI": resolve_group_node(
            model,
            ["v_InattentionW"],
            ["v", "inattentionw"],
        ),
        "deltaA": resolve_group_node(
            model,
            ["v_AttentionContrast"],
            ["v", "attentioncontrast"],
        ),
        "deltaI": resolve_group_node(
            model,
            ["v_InattentionContrast"],
            ["v", "inattentioncontrast"],
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

    low, high = hdi(
        draws
    )

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
        "Mean": float(
            np.mean(draws)
        ),
        "Median": float(
            np.median(draws)
        ),
        "SD": float(
            np.std(
                draws,
                ddof=1,
            )
        ),
        "HDI_lower": low,
        "HDI_upper": high,
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
    name,
    draws,
    out_file,
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

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.5,
    )

    mean = float(
        np.mean(draws)
    )

    low, high = hdi(
        draws
    )

    ax.axvline(
        mean,
        linewidth=1.5,
    )

    ax.set_xlabel(
        name
    )

    ax.set_ylabel(
        "Posterior density"
    )

    ax.set_title(
        "%s\nMean=%.4f, 95%% HDI=[%.4f, %.4f]"
        % (
            name,
            mean,
            low,
            high,
        )
    )

    fig.tight_layout()

    fig.savefig(
        out_file,
        dpi=180,
    )

    plt.close(
        fig
    )


def load_models(
    model_dir,
    chains,
):
    models = []

    for chain in range(
        chains
    ):
        path = (
            model_dir
            / (
                "%s_%d.hddm"
                % (
                    MODEL_NAME,
                    chain,
                )
            )
        )

        if not path.exists():
            raise FileNotFoundError(
                "Missing model: %s"
                % path
            )

        print(
            "Loading chain %d: %s"
            % (
                chain,
                path,
            ),
            flush=True,
        )

        models.append(
            hddm.load(
                str(path)
            )
        )

    return models


def main():
    parser = argparse.ArgumentParser()

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
        "bAE": [],
        "bAS": [],
        "bIE": [],
        "bIS": [],
        "theta_E": [],
        "theta_S": [],
        "delta_theta_E_minus_S": [],
    }

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

        bAE = (
            bA
            + deltaA / 2.0
        )

        bAS = (
            bA
            - deltaA / 2.0
        )

        bIE = (
            bI
            + deltaI / 2.0
        )

        bIS = (
            bI
            - deltaI / 2.0
        )

        valid = (
            np.isfinite(bAE)
            & np.isfinite(bAS)
            & np.isfinite(bIE)
            & np.isfinite(bIS)
            & (np.abs(bAE) > 1e-12)
            & (np.abs(bAS) > 1e-12)
        )

        theta_E = (
            bIE[valid]
            / bAE[valid]
        )

        theta_S = (
            bIS[valid]
            / bAS[valid]
        )

        derived["bAE"].append(
            bAE
        )

        derived["bAS"].append(
            bAS
        )

        derived["bIE"].append(
            bIE
        )

        derived["bIS"].append(
            bIS
        )

        derived["theta_E"].append(
            theta_E
        )

        derived["theta_S"].append(
            theta_S
        )

        derived[
            "delta_theta_E_minus_S"
        ].append(
            theta_E
            - theta_S
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
        / "joint_es_contrast_summary.csv",
        index=False,
        float_format="%.8f",
    )

    deltaA_all = np.concatenate(
        traces["deltaA"]
    )

    deltaI_all = np.concatenate(
        traces["deltaI"]
    )

    primary = pd.DataFrame(
        [
            {
                "Mean_deltaA_attended_E_minus_S": float(
                    np.mean(
                        deltaA_all
                    )
                ),
                "HDI_deltaA_lower": hdi(
                    deltaA_all
                )[0],
                "HDI_deltaA_upper": hdi(
                    deltaA_all
                )[1],
                "P_deltaA_gt_0": float(
                    np.mean(
                        deltaA_all > 0
                    )
                ),
                "R_hat_deltaA": convergence_stats(
                    traces["deltaA"]
                )["R_hat"],
                "ESS_bulk_deltaA": convergence_stats(
                    traces["deltaA"]
                )["ESS_bulk"],

                "Mean_deltaI_unattended_E_minus_S": float(
                    np.mean(
                        deltaI_all
                    )
                ),
                "HDI_deltaI_lower": hdi(
                    deltaI_all
                )[0],
                "HDI_deltaI_upper": hdi(
                    deltaI_all
                )[1],
                "P_deltaI_gt_0": float(
                    np.mean(
                        deltaI_all > 0
                    )
                ),
                "R_hat_deltaI": convergence_stats(
                    traces["deltaI"]
                )["R_hat"],
                "ESS_bulk_deltaI": convergence_stats(
                    traces["deltaI"]
                )["ESS_bulk"],
            }
        ]
    )

    primary.to_csv(
        args.out_dir
        / "joint_ES_attended_unattended_contrasts.csv",
        index=False,
        float_format="%.8f",
    )

    # Posterior correlation among the four drift regression coefficients.
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

    plot_posterior(
        "deltaA: attended E - S",
        deltaA_all,
        plot_dir
        / "deltaA_attended_E_minus_S.png",
    )

    plot_posterior(
        "deltaI: unattended E - S",
        deltaI_all,
        plot_dir
        / "deltaI_unattended_E_minus_S.png",
    )

    plot_posterior(
        "theta_E - theta_S",
        np.concatenate(
            derived[
                "delta_theta_E_minus_S"
            ]
        ),
        plot_dir
        / "delta_theta_E_minus_S.png",
    )

    print(
        "\nPRIMARY JOINT CONTRAST RESULTS",
        flush=True,
    )

    print(
        primary.to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "\nPositive deltaA means stronger attended E than attended S weighting. "
        "Positive deltaI means stronger unattended E than unattended S weighting.",
        flush=True,
    )

    print(
        "\nSaved diagnostics to: %s"
        % args.out_dir,
        flush=True,
    )


if __name__ == "__main__":
    main()