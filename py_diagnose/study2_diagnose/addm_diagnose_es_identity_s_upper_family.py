from __future__ import annotations

from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hddm
import kabuki


STUDY_LABEL = "study 2"


def model_spec(model_kind: str, include_z: bool):
    if model_kind not in {"basic", "contrast"}:
        raise ValueError(f"unknown model kind: {model_kind}")

    tag = f"{model_kind}_{'z' if include_z else 'noz'}"
    model_name = f"aDDM_ES_IDENTITY_S_UPPER_{tag.upper()}_NO_INTERCEPT"

    if model_kind == "basic":
        formula = "v ~ 0 + AttentionW_SE + InattentionW_SE"
    else:
        formula = (
            "v ~ 0 + AttentionW_SE + InattentionW_SE "
            "+ AttentionContrast_SE + InattentionContrast_SE"
        )

    return tag, model_name, formula


def hdi(draws, prob=0.95):
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return np.nan, np.nan

    interval = np.asarray(
        az.hdi(draws, hdi_prob=prob)
    ).reshape(-1)

    return float(interval[0]), float(interval[-1])


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
        "could not uniquely resolve group-level node.\n"
        f"exact candidates: {exact_candidates}\n"
        f"required tokens: {required_tokens}\n"
        f"matches: {matches}\n\n"
        "available nodes:\n"
        + "\n".join(names)
    )


def resolve_nodes(model, model_kind: str, include_z: bool):
    nodes = {
        "a": resolve_group_node(model, ["a"], ["a"]),
        "t": resolve_group_node(model, ["t"], ["t"]),
        "b_a": resolve_group_node(
            model,
            ["v_AttentionW_SE"],
            ["v", "attentionw_se"],
        ),
        "b_i": resolve_group_node(
            model,
            ["v_InattentionW_SE"],
            ["v", "inattentionw_se"],
        ),
    }

    if include_z:
        nodes["z"] = resolve_group_node(model, ["z"], ["z"])

    if model_kind == "contrast":
        nodes["delta_a"] = resolve_group_node(
            model,
            ["v_AttentionContrast_SE"],
            ["v", "attentioncontrast_se"],
        )
        nodes["delta_i"] = resolve_group_node(
            model,
            ["v_InattentionContrast_SE"],
            ["v", "inattentioncontrast_se"],
        )

    return nodes


def get_trace(model, node_name):
    trace = np.asarray(
        model.nodes_db.loc[node_name, "node"].trace(),
        dtype=float,
    ).reshape(-1)

    trace = trace[np.isfinite(trace)]

    if trace.size == 0:
        raise ValueError(
            f"no finite posterior samples in node {node_name}"
        )

    return trace


def convergence_stats(chain_arrays):
    n = min(len(x) for x in chain_arrays)

    arr = np.vstack(
        [
            np.asarray(x[:n], dtype=float)
            for x in chain_arrays
        ]
    )

    try:
        r_hat = (
            float(np.asarray(az.rhat(arr)))
            if arr.shape[0] >= 2
            else np.nan
        )
    except Exception:
        r_hat = np.nan

    try:
        ess_bulk = float(
            np.asarray(
                az.ess(arr, method="bulk")
            )
        )
    except Exception:
        ess_bulk = np.nan

    try:
        ess_tail = float(
            np.asarray(
                az.ess(arr, method="tail")
            )
        )
    except Exception:
        ess_tail = np.nan

    return {
        "r_hat": r_hat,
        "ess_bulk": ess_bulk,
        "ess_tail": ess_tail,
        "draws_per_chain_used": int(n),
    }


def summarize(name, chain_arrays):
    draws = np.concatenate(chain_arrays)
    lo, hi = hdi(draws)

    p_gt_0 = float(np.mean(draws > 0))
    p_lt_0 = float(np.mean(draws < 0))

    row = {
        "parameter": name,
        "mean": float(np.mean(draws)),
        "median": float(np.median(draws)),
        "sd": float(np.std(draws, ddof=1)),
        "hdi_lower": lo,
        "hdi_upper": hi,
        "p_gt_0": p_gt_0,
        "p_lt_0": p_lt_0,
        "p_direction": max(p_gt_0, p_lt_0),
    }

    row.update(convergence_stats(chain_arrays))
    return row


def load_models(model_dir: Path, model_name: str, chains: int):
    models = []

    for chain in range(chains):
        path = model_dir / f"{model_name}_{chain}.hddm"

        if not path.exists():
            raise FileNotFoundError(
                f"missing expected chain: {path}"
            )

        print(
            f"loading chain {chain}: {path}",
            flush=True,
        )

        models.append(
            hddm.load(str(path))
        )

    return models


def plot_posterior(label, draws, out_file, zero_line=False):
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]

    mean = float(np.mean(draws))
    lo, hi = hdi(draws)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(draws, bins=60, density=True, alpha=0.75)

    if zero_line:
        ax.axvline(0.0, linestyle="--", linewidth=1.4)

    ax.axvline(mean, linewidth=1.5)

    ax.set_xlabel(label)
    ax.set_ylabel("posterior density")
    ax.set_title(
        f"{label}\nmean={mean:.4f}, 95% hdi=[{lo:.4f}, {hi:.4f}]"
    )

    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_trace(label, chain_arrays, out_file):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for chain, draws in enumerate(chain_arrays):
        ax.plot(
            np.arange(len(draws)),
            draws,
            linewidth=0.75,
            alpha=0.85,
            label=f"chain {chain}",
        )

    ax.set_xlabel("posterior draw")
    ax.set_ylabel(label)
    ax.set_title(f"trace: {label}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_combined_posteriors(plot_items, out_file):
    n = len(plot_items)
    ncols = 3
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15, 4.1 * nrows),
    )

    axes = np.atleast_1d(axes).reshape(-1)

    for ax, (label, draws, zero_line) in zip(
        axes,
        plot_items,
    ):
        draws = np.asarray(draws, dtype=float)
        draws = draws[np.isfinite(draws)]

        mean = float(np.mean(draws))
        lo, hi = hdi(draws)

        ax.hist(
            draws,
            bins=50,
            density=True,
            alpha=0.75,
        )

        if zero_line:
            ax.axvline(
                0.0,
                linestyle="--",
                linewidth=1.2,
            )

        ax.axvline(mean, linewidth=1.3)
        ax.set_xlabel(label)
        ax.set_ylabel("posterior density")
        ax.set_title(
            f"{label}\n"
            f"mean={mean:.3f}, 95% hdi=[{lo:.3f}, {hi:.3f}]"
        )

    for ax in axes[len(plot_items):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def plot_combined_traces(trace_items, out_file):
    n = len(trace_items)
    ncols = 2
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14, 3.5 * nrows),
    )

    axes = np.atleast_1d(axes).reshape(-1)

    for ax, (label, chain_arrays) in zip(
        axes,
        trace_items,
    ):
        for chain, draws in enumerate(chain_arrays):
            ax.plot(
                np.arange(len(draws)),
                draws,
                linewidth=0.6,
                alpha=0.8,
                label=f"chain {chain}",
            )

        ax.set_xlabel("posterior draw")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)

    for ax in axes[len(trace_items):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            f"{STUDY_LABEL} diagnostics for the no-intercept "
            "es s-upper / e-lower addm family."
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
        "--model",
        choices=["basic", "contrast"],
        required=True,
    )
    parser.add_argument(
        "--include-z",
        action="store_true",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--ppc-samples",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--skip-ppc",
        action="store_true",
    )

    args = parser.parse_args()

    tag, model_name, formula = model_spec(
        args.model,
        args.include_z,
    )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    posterior_dir = args.out_dir / "posterior_plots"
    trace_dir = args.out_dir / "trace_plots"

    posterior_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    trace_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78, flush=True)
    print(
        f"{STUDY_LABEL.lower()} diagnostics: {tag}",
        flush=True,
    )
    print(f"model: {model_name}", flush=True)
    print(f"drift: {formula}", flush=True)
    print(
        "z: estimated"
        if args.include_z
        else "z: not included in the fitted model",
        flush=True,
    )
    print(f"chains: {args.chains}", flush=True)
    print("=" * 78, flush=True)

    models = load_models(
        args.model_dir,
        model_name,
        args.chains,
    )

    # ----------------------------------------------------------
    # all-node gelman-rubin, matching the old diagnostics
    # ----------------------------------------------------------
    if args.chains >= 2:
        gr = hddm.analyze.gelman_rubin(models)
        pd.Series(
            gr,
            name="r_hat",
        ).to_csv(
            args.out_dir
            / "gelman_rubin_all_nodes.csv"
        )

    combined = kabuki.utils.concat_models(models)

    # full hddm summary
    combined.gen_stats().to_csv(
        args.out_dir
        / "posterior_summary_all_nodes.csv"
    )

    # dic
    try:
        dic_value = float(combined.dic)
        (
            args.out_dir
            / "dic.txt"
        ).write_text(
            f"{dic_value:.12f}\n",
            encoding="utf-8",
        )
    except Exception as exc:
        dic_value = np.nan
        (
            args.out_dir
            / "dic.txt"
        ).write_text(
            f"dic unavailable: {type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )

    # ----------------------------------------------------------
    # resolve group-level nodes and extract per-chain traces
    # ----------------------------------------------------------
    resolved = resolve_nodes(
        models[0],
        args.model,
        args.include_z,
    )

    (
        args.out_dir
        / "resolved_core_node_names.json"
    ).write_text(
        json.dumps(
            resolved,
            indent=2,
        ),
        encoding="utf-8",
    )

    traces = {
        key: []
        for key in resolved
    }

    for model in models:
        chain_nodes = resolve_nodes(
            model,
            args.model,
            args.include_z,
        )

        for key in traces:
            traces[key].append(
                get_trace(
                    model,
                    chain_nodes[key],
                )
            )

    derived = {}
    ratio_stability = {}

    # ----------------------------------------------------------
    # basic model:
    # theta = b_i / b_a
    # ----------------------------------------------------------
    if args.model == "basic":
        derived["theta"] = []

        near_zero = []

        for chain in range(args.chains):
            b_a = traces["b_a"][chain]
            b_i = traces["b_i"][chain]

            n = min(
                len(b_a),
                len(b_i),
            )

            b_a = b_a[:n]
            b_i = b_i[:n]

            valid = (
                np.isfinite(b_a)
                & np.isfinite(b_i)
                & (np.abs(b_a) > 1e-12)
            )

            derived["theta"].append(
                b_i[valid] / b_a[valid]
            )

            near_zero.append(
                {
                    "chain": chain,
                    "p_abs_b_a_lt_0p1": float(
                        np.mean(
                            np.abs(b_a) < 0.1
                        )
                    ),
                }
            )

        b_a_all = np.concatenate(
            traces["b_a"]
        )
        b_a_hdi = hdi(b_a_all)

        ratio_stability = {
            "b_a_mean": float(
                np.mean(b_a_all)
            ),
            "b_a_hdi_lower": b_a_hdi[0],
            "b_a_hdi_upper": b_a_hdi[1],
            "b_a_hdi_contains_zero": bool(
                b_a_hdi[0]
                <= 0
                <= b_a_hdi[1]
            ),
            "theta_ratio_interpretation_safe": bool(
                not (
                    b_a_hdi[0]
                    <= 0
                    <= b_a_hdi[1]
                )
            ),
            "note": (
                "theta = b_i / b_a. "
                "the ratio becomes unstable if the attended "
                "coefficient b_a has substantial posterior mass near zero."
            ),
            "per_chain_near_zero_probability": near_zero,
        }

    # ----------------------------------------------------------
    # contrast model:
    #
    # b_as = b_a + delta_a/2
    # b_ae = b_a - delta_a/2
    # b_is = b_i + delta_i/2
    # b_ie = b_i - delta_i/2
    #
    # theta_s = b_is / b_as
    # theta_e = b_ie / b_ae
    # ----------------------------------------------------------
    else:
        for key in [
            "b_as",
            "b_ae",
            "b_is",
            "b_ie",
            "theta_s",
            "theta_e",
            "delta_theta_s_minus_e",
        ]:
            derived[key] = []

        near_zero = []

        for chain in range(args.chains):
            b_a = traces["b_a"][chain]
            b_i = traces["b_i"][chain]
            delta_a = traces["delta_a"][chain]
            delta_i = traces["delta_i"][chain]

            n = min(
                len(b_a),
                len(b_i),
                len(delta_a),
                len(delta_i),
            )

            b_a = b_a[:n]
            b_i = b_i[:n]
            delta_a = delta_a[:n]
            delta_i = delta_i[:n]

            b_as = b_a + delta_a / 2.0
            b_ae = b_a - delta_a / 2.0
            b_is = b_i + delta_i / 2.0
            b_ie = b_i - delta_i / 2.0

            valid = (
                np.isfinite(b_as)
                & np.isfinite(b_ae)
                & np.isfinite(b_is)
                & np.isfinite(b_ie)
                & (np.abs(b_as) > 1e-12)
                & (np.abs(b_ae) > 1e-12)
            )

            theta_s = (
                b_is[valid]
                / b_as[valid]
            )

            theta_e = (
                b_ie[valid]
                / b_ae[valid]
            )

            derived["b_as"].append(b_as)
            derived["b_ae"].append(b_ae)
            derived["b_is"].append(b_is)
            derived["b_ie"].append(b_ie)
            derived["theta_s"].append(theta_s)
            derived["theta_e"].append(theta_e)
            derived[
                "delta_theta_s_minus_e"
            ].append(
                theta_s - theta_e
            )

            near_zero.append(
                {
                    "chain": chain,
                    "p_abs_b_as_lt_0p1": float(
                        np.mean(
                            np.abs(b_as) < 0.1
                        )
                    ),
                    "p_abs_b_ae_lt_0p1": float(
                        np.mean(
                            np.abs(b_ae) < 0.1
                        )
                    ),
                }
            )

        b_as_all = np.concatenate(
            derived["b_as"]
        )
        b_ae_all = np.concatenate(
            derived["b_ae"]
        )

        b_as_hdi = hdi(b_as_all)
        b_ae_hdi = hdi(b_ae_all)

        ratio_stability = {
            "b_as_mean": float(
                np.mean(b_as_all)
            ),
            "b_as_hdi_lower": b_as_hdi[0],
            "b_as_hdi_upper": b_as_hdi[1],
            "b_as_hdi_contains_zero": bool(
                b_as_hdi[0]
                <= 0
                <= b_as_hdi[1]
            ),
            "b_ae_mean": float(
                np.mean(b_ae_all)
            ),
            "b_ae_hdi_lower": b_ae_hdi[0],
            "b_ae_hdi_upper": b_ae_hdi[1],
            "b_ae_hdi_contains_zero": bool(
                b_ae_hdi[0]
                <= 0
                <= b_ae_hdi[1]
            ),
            "theta_ratio_interpretation_safe": bool(
                not (
                    b_as_hdi[0]
                    <= 0
                    <= b_as_hdi[1]
                )
                and not (
                    b_ae_hdi[0]
                    <= 0
                    <= b_ae_hdi[1]
                )
            ),
            "note": (
                "theta_s = b_is / b_as and theta_e = b_ie / b_ae. "
                "the ratios become unstable if either attended denominator "
                "has substantial posterior mass near zero."
            ),
            "per_chain_near_zero_probability": near_zero,
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

    # ----------------------------------------------------------
    # concise posterior summary for fitted + derived quantities
    # ----------------------------------------------------------
    summary_rows = []

    fitted_order = [
        "a",
        "t",
    ]

    if args.include_z:
        fitted_order.append("z")

    fitted_order += [
        "b_a",
        "b_i",
    ]

    if args.model == "contrast":
        fitted_order += [
            "delta_a",
            "delta_i",
        ]

    for key in fitted_order:
        summary_rows.append(
            summarize(
                key,
                traces[key],
            )
        )

    for key in derived:
        summary_rows.append(
            summarize(
                key,
                derived[key],
            )
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary["dic"] = dic_value

    summary.to_csv(
        args.out_dir
        / "core_posterior_summary.csv",
        index=False,
        float_format="%.8f",
    )

    print(
        "\ncore posterior summary",
        flush=True,
    )
    print(
        summary.to_string(index=False),
        flush=True,
    )

    # ----------------------------------------------------------
    # posterior coefficient correlation matrix
    # ----------------------------------------------------------
    coefficient_keys = [
        "b_a",
        "b_i",
    ]

    if args.model == "contrast":
        coefficient_keys += [
            "delta_a",
            "delta_i",
        ]

    coefficient_draws = {
        key: np.concatenate(traces[key])
        for key in coefficient_keys
    }

    n_corr = min(
        len(x)
        for x in coefficient_draws.values()
    )

    pd.DataFrame(
        {
            key: value[:n_corr]
            for key, value
            in coefficient_draws.items()
        }
    ).corr().to_csv(
        args.out_dir
        / "posterior_drift_coefficient_correlations.csv",
        float_format="%.8f",
    )

    # ----------------------------------------------------------
    # dedicated posterior plots + trace plots
    # ----------------------------------------------------------
    labels = {
        "a": "a: boundary separation",
        "t": "t: non-decision time",
        "z": "z: starting point",
        "b_a": "b_a: attended-value coefficient",
        "b_i": "b_i: unattended-value coefficient",
        "theta": "theta = b_i / b_a",
        "delta_a": "delta_a: attended s - e slope difference",
        "delta_i": "delta_i: unattended s - e slope difference",
        "b_as": "b_as: attended s slope",
        "b_ae": "b_ae: attended e slope",
        "b_is": "b_is: unattended s slope",
        "b_ie": "b_ie: unattended e slope",
        "theta_s": "theta_s = b_is / b_as",
        "theta_e": "theta_e = b_ie / b_ae",
        "delta_theta_s_minus_e": "theta_s - theta_e",
    }

    plot_items = []
    trace_items = []

    for key in fitted_order:
        all_draws = np.concatenate(
            traces[key]
        )

        zero_line = key not in {
            "a",
            "t",
            "z",
        }

        plot_posterior(
            labels[key],
            all_draws,
            posterior_dir / f"{key}.png",
            zero_line=zero_line,
        )

        plot_trace(
            labels[key],
            traces[key],
            trace_dir / f"{key}.png",
        )

        plot_items.append(
            (
                labels[key],
                all_draws,
                zero_line,
            )
        )

        trace_items.append(
            (
                labels[key],
                traces[key],
            )
        )

    for key in derived:
        all_draws = np.concatenate(
            derived[key]
        )

        zero_line = True

        plot_posterior(
            labels[key],
            all_draws,
            posterior_dir / f"{key}.png",
            zero_line=zero_line,
        )

        plot_trace(
            labels[key],
            derived[key],
            trace_dir / f"{key}.png",
        )

        plot_items.append(
            (
                labels[key],
                all_draws,
                zero_line,
            )
        )

        trace_items.append(
            (
                labels[key],
                derived[key],
            )
        )

    plot_combined_posteriors(
        plot_items,
        args.out_dir
        / "key_parameter_posteriors.png",
    )

    plot_combined_traces(
        trace_items,
        args.out_dir
        / "key_parameter_traces.png",
    )

    # ----------------------------------------------------------
    # keep the old hddm-style posterior plots as well
    # ----------------------------------------------------------
    try:
        hddm_plot_dir = (
            args.out_dir
            / "hddm_posterior_plots"
        )

        hddm_plot_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        combined.plot_posteriors(
            save=True,
            path=str(hddm_plot_dir),
            format="pdf",
        )

        (
            args.out_dir
            / "hddm_posterior_plot_status.txt"
        ).write_text(
            "completed\n",
            encoding="utf-8",
        )

    except Exception as exc:
        (
            args.out_dir
            / "hddm_posterior_plot_status.txt"
        ).write_text(
            (
                "failed: "
                f"{type(exc).__name__}: {exc}\n"
            ),
            encoding="utf-8",
        )

    # ----------------------------------------------------------
    # posterior predictive check, matching old diagnostics
    # ----------------------------------------------------------
    if not args.skip_ppc:
        try:
            ppc_dir = (
                args.out_dir
                / "posterior_predictive"
            )

            ppc_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            size_plot = max(
                6,
                len(
                    combined.data
                    .subj_idx
                    .unique()
                )
                / 3.0
                * 1.5,
            )

            combined.plot_posterior_predictive(
                samples=args.ppc_samples,
                bins=100,
                figsize=(6, size_plot),
                save=True,
                path=str(ppc_dir),
                format="pdf",
            )

            (
                args.out_dir
                / "posterior_predictive_status.txt"
            ).write_text(
                (
                    "completed with "
                    f"{args.ppc_samples} posterior predictive samples\n"
                ),
                encoding="utf-8",
            )

        except Exception as exc:
            (
                args.out_dir
                / "posterior_predictive_status.txt"
            ).write_text(
                (
                    "failed: "
                    f"{type(exc).__name__}: {exc}\n"
                ),
                encoding="utf-8",
            )

    else:
        (
            args.out_dir
            / "posterior_predictive_status.txt"
        ).write_text(
            "skipped by --skip-ppc\n",
            encoding="utf-8",
        )

    # ----------------------------------------------------------
    # compact primary result output
    # ----------------------------------------------------------
    primary_parameters = [
        "a",
        "t",
    ]

    if args.include_z:
        primary_parameters.append("z")

    primary_parameters += [
        "b_a",
        "b_i",
    ]

    if args.model == "basic":
        primary_parameters += [
            "theta",
        ]
    else:
        primary_parameters += [
            "delta_a",
            "delta_i",
            "b_as",
            "b_ae",
            "b_is",
            "b_ie",
            "theta_s",
            "theta_e",
            "delta_theta_s_minus_e",
        ]

    primary = (
        summary.loc[
            summary["parameter"].isin(
                primary_parameters
            )
        ]
        .copy()
    )

    primary.to_csv(
        args.out_dir
        / "primary_results.csv",
        index=False,
        float_format="%.8f",
    )

    metadata = {
        "study": STUDY_LABEL.lower(),
        "model_tag": tag,
        "model_name": model_name,
        "drift_formula": formula,
        "drift_intercept": "absent",
        "z_estimated": bool(args.include_z),
        "chains": int(args.chains),
        "ppc_samples": (
            0
            if args.skip_ppc
            else int(args.ppc_samples)
        ),
        "model_dir": str(args.model_dir),
        "out_dir": str(args.out_dir),
    }

    (
        args.out_dir
        / "diagnostic_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nprimary results",
        flush=True,
    )
    print(
        primary.to_string(index=False),
        flush=True,
    )

    print(
        "\ntheta ratio stability",
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
        f"\ndiagnostics saved to: {args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
