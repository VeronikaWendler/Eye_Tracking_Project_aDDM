from __future__ import annotations
from pathlib import Path
import argparse, json, re
import numpy as np
import pandas as pd
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hddm
import kabuki

MODEL_NAME = "phase_specific_aDDM_JOINT"

def hdi(x, prob=0.95):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    z = np.asarray(az.hdi(x, hdi_prob=prob))
    return float(z[0]), float(z[1])

def get_trace(model, node_name):
    x = np.asarray(model.nodes_db.loc[node_name, "node"].trace(), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError(f"No finite samples in {node_name}")
    return x

def resolve_node(model, parameter, phase=None, exact=None):
    names = [str(x) for x in model.nodes_db.index]
    for cand in (exact or []):
        if cand in names:
            return cand
    if phase is not None:
        for cand in [f"{parameter}({phase})", f"{parameter}_{phase}", f"{parameter}[{phase}]"]:
            if cand in names:
                return cand
    matches = []
    for name in names:
        low = name.lower()
        if "subj" in low:
            continue
        if parameter.lower() not in low:
            continue
        if phase is not None and phase.lower() not in low:
            continue
        matches.append(name)
    if parameter in {"a","t","z"}:
        strict = [x for x in matches if re.match(rf"^{re.escape(parameter)}(?:\(|\[|_|$)", x)]
        if len(strict) == 1:
            return strict[0]
        if strict:
            matches = strict
    if len(matches) == 1:
        return matches[0]
    raise KeyError(
        f"Could not uniquely resolve parameter={parameter}, phase={phase}. "
        f"Matches={matches}\nAvailable nodes:\n" + "\n".join(names)
    )

def conv_stats(chains):
    n = min(len(x) for x in chains)
    arr = np.vstack([np.asarray(x[:n], dtype=float) for x in chains])
    out = {"draws_per_chain_used": n}
    try:
        out["R_hat"] = float(np.asarray(az.rhat(arr))) if arr.shape[0] >= 2 else np.nan
    except Exception:
        out["R_hat"] = np.nan
    try:
        out["ESS_bulk"] = float(np.asarray(az.ess(arr, method="bulk")))
    except Exception:
        out["ESS_bulk"] = np.nan
    try:
        out["ESS_tail"] = float(np.asarray(az.ess(arr, method="tail")))
    except Exception:
        out["ESS_tail"] = np.nan
    return out

def discover(model):
    return {
        "a_EE": resolve_node(model, "a", "EE"),
        "a_ES": resolve_node(model, "a", "ES"),
        "t_EE": resolve_node(model, "t", "EE"),
        "t_ES": resolve_node(model, "t", "ES"),
        "z_EE": resolve_node(model, "z", "EE"),
        "z_ES": resolve_node(model, "z", "ES"),
        "b0_EE": resolve_node(model, "drift0_EE", exact=["v_drift0_EE"]),
        "b0_ES": resolve_node(model, "drift0_ES", exact=["v_drift0_ES"]),
        "b1_EE": resolve_node(model, "AttentionW_EE", exact=["v_AttentionW_EE"]),
        "b1_ES": resolve_node(model, "AttentionW_ES", exact=["v_AttentionW_ES"]),
        "b2_EE": resolve_node(model, "InattentionW_EE", exact=["v_InattentionW_EE"]),
        "b2_ES": resolve_node(model, "InattentionW_ES", exact=["v_InattentionW_ES"]),
    }

def load_models(model_dir, chains):
    out = []
    for c in range(chains):
        p = model_dir / f"{MODEL_NAME}_{c}.hddm"
        if not p.exists():
            raise FileNotFoundError(p)
        print(f"Loading chain {c}: {p}", flush=True)
        out.append(hddm.load(str(p)))
    return out

def phase_summary(param, phase, draws, chains):
    lo, hi = hdi(draws)
    d = {
        "Parameter": param, "Phase": phase,
        "Mean": float(np.mean(draws)),
        "Median": float(np.median(draws)),
        "SD": float(np.std(draws, ddof=1)),
        "HDI_lower": lo, "HDI_upper": hi,
    }
    d.update(conv_stats(chains))
    return d

def diff_summary(param, ee, es, diff, diff_chains):
    ee_lo, ee_hi = hdi(ee); es_lo, es_hi = hdi(es); d_lo, d_hi = hdi(diff)
    pgt = float(np.mean(diff > 0)); plt_ = float(np.mean(diff < 0))
    d = {
        "Parameter": param,
        "Mean_EE": float(np.mean(ee)), "Median_EE": float(np.median(ee)),
        "HDI_EE_lower": ee_lo, "HDI_EE_upper": ee_hi,
        "Mean_ES": float(np.mean(es)), "Median_ES": float(np.median(es)),
        "HDI_ES_lower": es_lo, "HDI_ES_upper": es_hi,
        "Mean_Difference_ES_minus_EE": float(np.mean(diff)),
        "Median_Difference_ES_minus_EE": float(np.median(diff)),
        "HDI_Difference_lower": d_lo, "HDI_Difference_upper": d_hi,
        "P_ES_gt_EE": pgt, "P_ES_lt_EE": plt_, "P_direction": max(pgt, plt_),
    }
    cs = conv_stats(diff_chains)
    d.update({
        "R_hat_Difference": cs["R_hat"],
        "ESS_bulk_Difference": cs["ESS_bulk"],
        "ESS_tail_Difference": cs["ESS_tail"],
        "draws_per_chain_used": cs["draws_per_chain_used"],
    })
    return d

def plot_diff(param, diff, out):
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.hist(diff, bins=60, density=True, alpha=.75)
    ax.axvline(0, linestyle="--", linewidth=1.5)
    lo, hi = hdi(diff); mean = float(np.mean(diff))
    ax.axvline(mean, linewidth=1.5)
    ax.set_xlabel(f"{param}: ES - EE")
    ax.set_ylabel("Posterior density")
    ax.set_title(f"{param} phase difference\nMean={mean:.4f}, 95% HDI=[{lo:.4f}, {hi:.4f}]")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--chains", type=int, default=3)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.out_dir / "posterior_difference_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    models = load_models(args.model_dir, args.chains)

    if args.chains >= 2:
        gr = hddm.analyze.gelman_rubin(models)
        pd.Series(gr, name="R_hat").to_csv(args.out_dir/"gelman_rubin_all_nodes.csv")

    combined = kabuki.utils.concat_models(models)
    combined.gen_stats().to_csv(args.out_dir/"posterior_summary_all_nodes.csv")
    try:
        (args.out_dir/"DIC.txt").write_text(f"{float(combined.dic)}\n", encoding="utf-8")
    except Exception as e:
        (args.out_dir/"DIC.txt").write_text(f"DIC unavailable: {type(e).__name__}: {e}\n", encoding="utf-8")

    nodes = discover(models[0])
    print("Resolved core nodes:", json.dumps(nodes, indent=2), flush=True)
    (args.out_dir/"resolved_core_node_names.json").write_text(json.dumps(nodes, indent=2), encoding="utf-8")

    core = {p: {"EE": [], "ES": []} for p in ["a","t","z","b0","b1","b2"]}

    for m in models:
        nd = discover(m)
        for p in core:
            core[p]["EE"].append(get_trace(m, nd[f"{p}_EE"]))
            core[p]["ES"].append(get_trace(m, nd[f"{p}_ES"]))

    theta = {"EE": [], "ES": []}
    for c in range(args.chains):
        for ph in ["EE","ES"]:
            b1, b2 = core["b1"][ph][c], core["b2"][ph][c]
            n = min(len(b1), len(b2))
            b1, b2 = b1[:n], b2[:n]
            ok = np.isfinite(b1) & np.isfinite(b2) & (np.abs(b1) > 1e-12)
            theta[ph].append(b2[ok] / b1[ok])

    all_draws = {}
    for p in core:
        all_draws[p] = {ph: np.concatenate(core[p][ph]) for ph in ["EE","ES"]}
    all_draws["theta"] = {ph: np.concatenate(theta[ph]) for ph in ["EE","ES"]}

    phase_rows = []
    for p in ["a","t","z","b0","b1","b2","theta"]:
        src = theta if p == "theta" else core[p]
        for ph in ["EE","ES"]:
            phase_rows.append(phase_summary(p, ph, all_draws[p][ph], src[ph]))
    pd.DataFrame(phase_rows).to_csv(
        args.out_dir/"core_phase_posterior_summary.csv", index=False, float_format="%.8f"
    )

    comp_rows, saved_diffs = [], {}
    for p in ["a","t","z","b0","b1","b2","theta"]:
        src = theta if p == "theta" else core[p]
        diff_chains = []
        for c in range(args.chains):
            ee, es = src["EE"][c], src["ES"][c]
            n = min(len(ee), len(es))
            ok = np.isfinite(ee[:n]) & np.isfinite(es[:n])
            diff_chains.append(es[:n][ok] - ee[:n][ok])
        diff = np.concatenate(diff_chains)
        saved_diffs[p] = diff
        comp_rows.append(diff_summary(p, all_draws[p]["EE"], all_draws[p]["ES"], diff, diff_chains))
        plot_diff(p, diff, plot_dir/f"delta_{p}_ES_minus_EE.png")

    comp = pd.DataFrame(comp_rows)
    comp.to_csv(args.out_dir/"joint_phase_comparison.csv", index=False, float_format="%.8f")
    pd.DataFrame(saved_diffs).to_csv(
        args.out_dir/"joint_phase_difference_draws.csv", index=False, float_format="%.10f"
    )

    corrs = []
    for ph in ["EE","ES"]:
        b1, b2 = all_draws["b1"][ph], all_draws["b2"][ph]
        n = min(len(b1), len(b2))
        corrs.append({"Phase": ph, "Posterior_correlation_b1_b2": float(np.corrcoef(b1[:n], b2[:n])[0,1])})
    pd.DataFrame(corrs).to_csv(
        args.out_dir/"posterior_b1_b2_correlations.csv", index=False, float_format="%.8f"
    )

    cols = [
        "Parameter","Mean_EE","Mean_ES","Mean_Difference_ES_minus_EE",
        "HDI_Difference_lower","HDI_Difference_upper","P_ES_gt_EE",
        "R_hat_Difference","ESS_bulk_Difference"
    ]
    print("\nDIRECT JOINT POSTERIOR COMPARISONS: ES - EE", flush=True)
    print(comp[cols].to_string(index=False), flush=True)
    print(f"\nSaved diagnostics to: {args.out_dir}", flush=True)

if __name__ == "__main__":
    main()
