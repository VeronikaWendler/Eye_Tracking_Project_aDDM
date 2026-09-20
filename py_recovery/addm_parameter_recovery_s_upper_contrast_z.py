from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import arviz as az
import hddm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

DIRECT = ["a", "t", "z", "b_a", "b_i", "delta_a", "delta_i"]
DERIVED = ["b_as", "b_ae", "b_is", "b_ie", "theta_s", "theta_e", "delta_theta_s_minus_e"]
REGRESSOR_COLUMNS = ["AttentionW_SE", "InattentionW_SE", "AttentionContrast_SE", "InattentionContrast_SE"]
GROUP_NODE_EXACT = {
    "a": "a",
    "t": "t",
    "z": "z",
    "b_a": "v_AttentionW_SE",
    "b_i": "v_InattentionW_SE",
    "delta_a": "v_AttentionContrast_SE",
    "delta_i": "v_InattentionContrast_SE",
}


def hdi(draws, prob=0.95):
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    out = np.asarray(az.hdi(draws, hdi_prob=prob)).reshape(-1)
    return float(out[0]), float(out[-1])


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
    return {"r_hat": rhat, "ess_bulk": ess_bulk, "ess_tail": ess_tail, "draws_per_chain_used": int(n)}


def node_names(model):
    return [str(x) for x in model.nodes_db.index]


def resolve_group_nodes(model):
    names = node_names(model)
    out = {}
    for key, candidate in GROUP_NODE_EXACT.items():
        if candidate not in names:
            raise KeyError(f"missing group node {candidate!r}; available nodes begin: {names[:40]}")
        out[key] = candidate
    return out


def subject_id_from_node(name):
    for pat in [
        r"subj(?:_idx)?[.\-_(]*([0-9]+)\)?$",
        r"subj(?:_idx)?[.\-_(]*([0-9]+)\)?(?:\.|$)",
    ]:
        m = re.search(pat, str(name), flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def resolve_subject_node(model, group_node, subj):
    names = node_names(model)
    subj = int(subj)
    for candidate in [
        f"{group_node}_subj.{subj}",
        f"{group_node}_subj({subj})",
        f"{group_node}_subj_{subj}",
        f"{group_node}_subj-{subj}",
    ]:
        if candidate in names:
            return candidate
    prefix = group_node.lower() + "_subj"
    matches = [n for n in names if n.lower().startswith(prefix) and subject_id_from_node(n) == subj]
    if len(matches) != 1:
        raise KeyError(f"could not resolve subject node for {group_node!r}, subject={subj}; matches={matches}")
    return matches[0]


def get_trace(model, node):
    x = np.asarray(model.nodes_db.loc[node, "node"].trace(), dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if not len(x):
        raise ValueError(f"empty trace for {node}")
    return x


def discover_chain_files(model_dir, chains):
    model_dir = Path(model_dir)
    out = []
    for c in range(chains):
        matches = sorted(model_dir.glob(f"*_{c}.hddm"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one '*_{c}.hddm' in {model_dir}, found {matches}")
        out.append(matches[0])
    return out


def load_models(model_dir, chains):
    files = discover_chain_files(model_dir, chains)
    models = []
    for c, path in enumerate(files):
        print(f"loading empirical chain {c}: {path}", flush=True)
        models.append(hddm.load(str(path)))
    return models, files


def derive_from_direct(p):
    b_as = float(p["b_a"]) + float(p["delta_a"]) / 2.0
    b_ae = float(p["b_a"]) - float(p["delta_a"]) / 2.0
    b_is = float(p["b_i"]) + float(p["delta_i"]) / 2.0
    b_ie = float(p["b_i"]) - float(p["delta_i"]) / 2.0
    theta_s = b_is / b_as if abs(b_as) > 1e-12 else np.nan
    theta_e = b_ie / b_ae if abs(b_ae) > 1e-12 else np.nan
    return {
        "b_as": b_as, "b_ae": b_ae, "b_is": b_is, "b_ie": b_ie,
        "theta_s": theta_s, "theta_e": theta_e,
        "delta_theta_s_minus_e": theta_s - theta_e,
    }


def derive_chain_traces(direct):
    out = {k: [] for k in DERIVED}
    for c in range(len(direct["b_a"])):
        n = min(len(direct[k][c]) for k in ["b_a", "b_i", "delta_a", "delta_i"])
        b_a = direct["b_a"][c][:n]
        b_i = direct["b_i"][c][:n]
        da = direct["delta_a"][c][:n]
        di = direct["delta_i"][c][:n]
        b_as = b_a + da / 2.0
        b_ae = b_a - da / 2.0
        b_is = b_i + di / 2.0
        b_ie = b_i - di / 2.0
        theta_s = np.divide(b_is, b_as, out=np.full(n, np.nan), where=np.abs(b_as) > 1e-12)
        theta_e = np.divide(b_ie, b_ae, out=np.full(n, np.nan), where=np.abs(b_ae) > 1e-12)
        out["b_as"].append(b_as)
        out["b_ae"].append(b_ae)
        out["b_is"].append(b_is)
        out["b_ie"].append(b_ie)
        out["theta_s"].append(theta_s[np.isfinite(theta_s)])
        out["theta_e"].append(theta_e[np.isfinite(theta_e)])
        ok = np.isfinite(theta_s) & np.isfinite(theta_e)
        out["delta_theta_s_minus_e"].append(theta_s[ok] - theta_e[ok])
    return out


def summarize_parameter(parameter, kind, true_value, chains):
    chains = [np.asarray(x, dtype=float)[np.isfinite(x)] for x in chains]
    draws = np.concatenate(chains)
    low, high = hdi(draws)
    row = {
        "parameter": parameter,
        "kind": kind,
        "true": float(true_value),
        "recovered_mean": float(np.mean(draws)),
        "recovered_median": float(np.median(draws)),
        "hdi_lower": low,
        "hdi_upper": high,
        "covered_95": bool(low <= true_value <= high),
        "hdi_width": float(high - low),
    }
    row.update(convergence_stats(chains))
    return row


def extract_truth(source_model, design_df, rep, seed):
    rng = np.random.default_rng(seed + rep)
    group_nodes = resolve_group_nodes(source_model)
    subjects = sorted(design_df["subj_idx"].astype(int).unique().tolist())
    subject_nodes = {
        s: {k: resolve_subject_node(source_model, group_nodes[k], s) for k in DIRECT}
        for s in subjects
    }
    cache = {}
    for node in group_nodes.values():
        cache[node] = get_trace(source_model, node)
    for s in subjects:
        for node in subject_nodes[s].values():
            if node not in cache:
                cache[node] = get_trace(source_model, node)
    n = min(len(x) for x in cache.values())
    if n < 50:
        raise RuntimeError(f"only {n} common source posterior draws")
    draw = int(rng.integers(0, n))
    group = {k: float(cache[node][draw]) for k, node in group_nodes.items()}
    individual = {
        s: {k: float(cache[subject_nodes[s][k]][draw]) for k in DIRECT}
        for s in subjects
    }
    return group, individual, group_nodes, draw, n


def simulate_dataset(design_df, true_individual, seed):
    rng = np.random.default_rng(seed)
    rows = []
    design = design_df[["subj_idx", *REGRESSOR_COLUMNS]].copy().reset_index(drop=True)
    for _, tr in design.iterrows():
        subj = int(tr["subj_idx"])
        p = true_individual[subj]
        v = (
            p["b_a"] * float(tr["AttentionW_SE"])
            + p["b_i"] * float(tr["InattentionW_SE"])
            + p["delta_a"] * float(tr["AttentionContrast_SE"])
            + p["delta_i"] * float(tr["InattentionContrast_SE"])
        )
        np.random.seed(int(rng.integers(1, 2_000_000_000)))
        one, _ = hddm.generate.gen_rand_data(
            {"a": float(p["a"]), "t": float(p["t"]), "z": float(p["z"]), "v": float(v)},
            size=1, subjs=1,
        )
        r0 = one.iloc[0]
        rt = float(r0["rt"])
        response = int(r0["response"]) if "response" in one.columns else int(rt > 0)
        if response not in {0, 1}:
            raise ValueError(f"invalid simulated response {response}")
        rows.append({
            "subj_idx": subj,
            "rt": abs(rt),
            "response": response,
            **{col: float(tr[col]) for col in REGRESSOR_COLUMNS},
        })
    return pd.DataFrame(rows)


def build_model(sim_df):
    v_reg = {
        "model": "v ~ 0 + AttentionW_SE + InattentionW_SE + AttentionContrast_SE + InattentionContrast_SE",
        "link_func": lambda x: x,
    }
    return hddm.HDDMRegressor(
        sim_df,
        v_reg,
        include=["a", "t", "v", "z"],
        p_outlier=0.05,
        group_only_regressors=False,
        keep_regressor_trace=True,
    )


def fit_one_chain(chain, sim_df, out_dir, samples, burn, seed):
    np.random.seed(seed + chain)
    model = build_model(sim_df)
    print(f"recovery chain {chain}: starting values", flush=True)
    model.find_starting_values()
    print(f"recovery chain {chain}: sampling {samples}, burn {burn}", flush=True)
    db_path = Path(out_dir) / f"recovery_chain_{chain}_db"
    model.sample(
        samples,
        burn=burn,
        dbname=str(db_path),
        db="pickle",
    )
    out = Path(out_dir) / f"recovery_chain_{chain}.hddm"
    model.save(str(out))
    print(f"recovery chain {chain}: saved {out}", flush=True)
    return str(out)


def collect_recovered(models, true_group, true_individual):
    group_rows, indiv_rows = [], []
    group_nodes = [resolve_group_nodes(m) for m in models]
    direct_group = {
        k: [get_trace(m, group_nodes[c][k]) for c, m in enumerate(models)]
        for k in DIRECT
    }
    for k in DIRECT:
        group_rows.append(summarize_parameter(k, "direct", true_group[k], direct_group[k]))
    true_group_derived = derive_from_direct(true_group)
    derived_group = derive_chain_traces(direct_group)
    for k in DERIVED:
        group_rows.append(summarize_parameter(k, "derived", true_group_derived[k], derived_group[k]))

    for subj in sorted(true_individual):
        subj_direct = {}
        for k in DIRECT:
            arrays = []
            for c, model in enumerate(models):
                node = resolve_subject_node(model, group_nodes[c][k], subj)
                arrays.append(get_trace(model, node))
            subj_direct[k] = arrays
            row = summarize_parameter(k, "direct", true_individual[subj][k], arrays)
            row["subj_idx"] = subj
            indiv_rows.append(row)
        true_derived = derive_from_direct(true_individual[subj])
        subj_derived = derive_chain_traces(subj_direct)
        for k in DERIVED:
            row = summarize_parameter(k, "derived", true_derived[k], subj_derived[k])
            row["subj_idx"] = subj
            indiv_rows.append(row)
    return pd.DataFrame(group_rows), pd.DataFrame(indiv_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, choices=["study1", "study2"])
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--recovery-dir", type=Path, required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--empirical-chains", type=int, default=3)
    ap.add_argument("--recovery-chains", type=int, default=3)
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--burn", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.burn >= args.samples:
        raise ValueError("burn must be smaller than samples")

    rep_dir = args.recovery_dir / f"rep_{args.rep:03d}"
    flag = rep_dir / "complete.flag"
    if flag.exists() and not args.force:
        print(f"{rep_dir} already complete; skipping", flush=True)
        return
    rep_dir.mkdir(parents=True, exist_ok=True)

    empirical_models, empirical_files = load_models(args.model_dir, args.empirical_chains)
    design = empirical_models[0].data.copy().reset_index(drop=True)
    required = {"subj_idx", "rt", "response", *REGRESSOR_COLUMNS}
    missing = sorted(required - set(design.columns))
    if missing:
        raise ValueError(f"empirical model data missing {missing}")

    source_chain = args.rep % args.empirical_chains
    true_group, true_individual, source_nodes, source_draw, source_n = extract_truth(
        empirical_models[source_chain], design, args.rep, args.seed
    )

    true_group_derived = derive_from_direct(true_group)
    pd.DataFrame(
        [{"parameter": k, "kind": "direct", "true": true_group[k]} for k in DIRECT]
        + [{"parameter": k, "kind": "derived", "true": true_group_derived[k]} for k in DERIVED]
    ).to_csv(rep_dir / "true_parameters_group.csv", index=False)

    true_indiv_rows = []
    for subj, p in true_individual.items():
        d = derive_from_direct(p)
        true_indiv_rows += [
            {"subj_idx": subj, "parameter": k, "kind": "direct", "true": p[k]} for k in DIRECT
        ]
        true_indiv_rows += [
            {"subj_idx": subj, "parameter": k, "kind": "derived", "true": d[k]} for k in DERIVED
        ]
    pd.DataFrame(true_indiv_rows).to_csv(rep_dir / "true_parameters_individual.csv", index=False)

    sim_df = simulate_dataset(design, true_individual, args.seed + 100000 + args.rep)
    sim_df.to_csv(rep_dir / "synthetic_data.csv", index=False)

    chain_dir = rep_dir / "chains"
    chain_dir.mkdir(parents=True, exist_ok=True)
    paths = Parallel(n_jobs=args.recovery_chains)(
        delayed(fit_one_chain)(
            c, sim_df, chain_dir, args.samples, args.burn,
            args.seed + args.rep * 1000 + 500000,
        )
        for c in range(args.recovery_chains)
    )
    recovered_models = [hddm.load(p) for p in paths]
    group, individual = collect_recovered(recovered_models, true_group, true_individual)
    group.insert(0, "rep", args.rep)
    individual.insert(0, "rep", args.rep)
    group.to_csv(rep_dir / "recovery_group.csv", index=False, float_format="%.10f")
    individual.to_csv(rep_dir / "recovery_individual.csv", index=False, float_format="%.10f")

    metadata = {
        "study": args.study,
        "rep": args.rep,
        "source_empirical_chain": source_chain,
        "source_empirical_file": str(empirical_files[source_chain]),
        "source_posterior_draw_index": source_draw,
        "source_draws_available": source_n,
        "recovery_chains": args.recovery_chains,
        "samples_per_chain": args.samples,
        "burn_per_chain": args.burn,
        "n_rows": int(len(sim_df)),
        "n_participants": int(sim_df["subj_idx"].nunique()),
        "response_0_n": int((sim_df["response"] == 0).sum()),
        "response_1_n": int((sim_df["response"] == 1).sum()),
        "drift_formula": "v ~ 0 + AttentionW_SE + InattentionW_SE + AttentionContrast_SE + InattentionContrast_SE",
        "response_coordinate": "response=1 S upper; response=0 E lower",
        "direct_recovery_targets": DIRECT,
        "derived_recovery_targets": DERIVED,
        "source_group_nodes": source_nodes,
        "method": "posterior-informed conditional parameter recovery using one coherent empirical posterior iteration and the exact empirical design matrix",
    }
    (rep_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    flag.write_text("complete\n", encoding="utf-8")
    print("\nGROUP RECOVERY\n" + group.to_string(index=False), flush=True)
    print(f"\ncompleted rep {args.rep}: {rep_dir}", flush=True)


if __name__ == "__main__":
    main()
