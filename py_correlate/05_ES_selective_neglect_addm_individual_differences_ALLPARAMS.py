#!/usr/bin/env python3
"""
05 — COMPACT INDIVIDUAL-DIFFERENCE ANALYSIS
Garcia-style value sensitivity + behavioural performance x final ESaDDM+z

Final fitted model:
    A_DDM_ES_IDENTITY_S_UPPER_CONTRAST_Z_NO_INTERCEPT
    response=1 -> S (upper), response=0 -> E (lower)

    v ~ 0 + AttentionW_SE + InattentionW_SE
          + AttentionContrast_SE + InattentionContrast_SE

This version intentionally produces FEW figures:
    1) one compact Pearson-r heatmap covering the behavioural outcomes
       and all substantive participant-level aDDM parameters.

It saves detailed CSVs for inspection instead of dozens of scatterplots.

Primary behavioural score:
    selective_vs_EE = EE slope - ES slope

Primary mechanistic tests:
    selective_vs_EE ~ theta_S - theta_E
    selective_vs_EE ~ delta_I
    selective_vs_EE ~ delta_A
    selective_vs_EE ~ z

Performance metrics:
    accuracy_ES
    accuracy_EE
    accuracy_SP_exact
        IMPORTANT: SP "corr" in the conversion code is 1 only when the slider
        estimate equals the true probability exactly. This is NOT equivalent to
        binary choice accuracy.
    accuracy_choice_ES_EE
        all ES + EE choice trials pooled
    accuracy_all_raw_ES_EE_SP
        raw "corr" pooled across ES + EE + SP; descriptive only because SP uses
        a different criterion
    SP_E_MAE
        mean absolute error of stated probability for E symbols
    SP_E_estimation_accuracy
        1 - SP_E_MAE; continuous calibration-like accuracy score

z extraction:
    The group-level diagnostic code finds the group z node. This script needs a
    participant-level z. It therefore resolves participant-specific z directly
    from each fitted model's nodes_db, accepting raw z or transformed z nodes.
    Missing z now raises an informative error instead of silently returning NaN.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re
import json
import dill

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


ROOTS = {
    1: Path("/rds/projects/z/zhanglp-vwendler-core/Study1_aDDM"),
    2: Path("/rds/projects/z/zhanglp-vwendler-core/Study2_aDDM"),
}

FIT_REL = Path("derivatives/models/es_identity_S_upper_contrast_z_final")
OUT_REL = Path(
    "derivatives/figures/es_identity_S_upper_contrast_z_final/"
    "individual_differences_compact"
)

MODEL_NAME_TOKEN = "CONTRAST_Z_NO_INTERCEPT"

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
})


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def expit(x):
    x = np.asarray(x, dtype=float)
    # numerically stable enough for HDDM traces here
    return 1.0 / (1.0 + np.exp(-x))


def summarize_draws(x, prefix):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q025": np.nan,
            f"{prefix}_q975": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_q025": float(np.quantile(x, 0.025)),
        f"{prefix}_q975": float(np.quantile(x, 0.975)),
    }


def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(len(pvals), np.nan, dtype=float)
    good = np.isfinite(pvals)
    if not np.any(good):
        return out

    p = pvals[good]
    order = np.argsort(p)
    adj_sorted = np.empty(len(p), dtype=float)
    running = 0.0

    for rank, idx in enumerate(order):
        val = (len(p) - rank) * p[idx]
        running = max(running, val)
        adj_sorted[rank] = min(running, 1.0)

    tmp = np.empty(len(p), dtype=float)
    for rank, idx in enumerate(order):
        tmp[idx] = adj_sorted[rank]

    out[good] = tmp
    return out


def fdr_bh_adjust(pvals):
    """Benjamini-Hochberg FDR adjustment; NaNs preserved."""
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(len(pvals), np.nan, dtype=float)
    good_idx = np.where(np.isfinite(pvals))[0]
    if good_idx.size == 0:
        return out

    p = pvals[good_idx]
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)

    adj = ranked * m / np.arange(1, m + 1)
    # enforce monotonicity from the end
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)

    tmp = np.empty(m, dtype=float)
    tmp[order] = adj
    out[good_idx] = tmp
    return out


def p_text(p):
    if not np.isfinite(p):
        return "NA"
    return "<.001" if p < .001 else f"{p:.3f}"


# ---------------------------------------------------------------------
# Load fitted chains
# ---------------------------------------------------------------------

def load_chain_models_and_traces(model_dir: Path):
    pkls = sorted(model_dir.glob(f"*{MODEL_NAME_TOKEN}_*.pkl"))
    if not pkls:
        pkls = sorted(model_dir.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(
            f"No HDDM chain .pkl files found in:\n{model_dir}"
        )

    models = []
    traces = []

    for chain_i, p in enumerate(pkls):
        print(f"Loading chain: {p.name}", flush=True)
        with open(p, "rb") as f:
            model = dill.load(f)

        if not hasattr(model, "get_traces"):
            raise AttributeError(f"{p.name}: loaded model has no get_traces()")

        tr = model.get_traces().copy()
        tr["__chain__"] = chain_i
        tr["__chain_file__"] = p.name

        models.append(model)
        traces.append(tr)

    combined = pd.concat(traces, ignore_index=True)

    print(f"Loaded {len(models)} chains", flush=True)
    print(f"Combined posterior draws: {len(combined):,}", flush=True)

    return models, combined


# ---------------------------------------------------------------------
# Participant-level regression coefficients from get_traces()
# ---------------------------------------------------------------------

SUBJ_PATTERN = re.compile(
    r"^(?P<base>.+)_subj(?:\((?P<mod>.+?)\))?\.(?P<sid>\d+)$"
)


def index_subject_trace_columns(traces: pd.DataFrame):
    out = {}
    for col in traces.columns:
        m = SUBJ_PATTERN.match(str(col))
        if not m:
            continue
        sid = int(m.group("sid"))
        base = m.group("base")
        mod = m.group("mod")
        full_base = f"{base}({mod})" if mod else base
        out.setdefault(sid, {})[norm(full_base)] = str(col)
    return out


def find_subject_col(index, sid, aliases, label, required=True):
    available = index.get(int(sid), {})
    aliases_n = [norm(a) for a in aliases]

    for a in aliases_n:
        if a in available:
            return available[a]

    hits = []
    for base_n, col in available.items():
        if any(a in base_n for a in aliases_n):
            hits.append(col)
    hits = list(dict.fromkeys(hits))

    if len(hits) == 1:
        return hits[0]

    if not required:
        return None

    raise KeyError(
        f"\nParticipant {sid}: could not identify {label}.\n"
        f"Aliases tried: {aliases}\n"
        f"Candidate hits: {hits}\n"
        "Available participant-level bases:\n  "
        + "\n  ".join(sorted(available.keys()))
    )


# ---------------------------------------------------------------------
# Robust participant-level z extraction DIRECTLY from model.nodes_db
# ---------------------------------------------------------------------

def _subject_from_name(name: str):
    s = str(name)

    patterns = [
        r"subj\.(\d+)(?:$|[^0-9])",
        r"subj\((\d+)\)",
        r"subj\[(\d+)\]",
        r"subject\.(\d+)(?:$|[^0-9])",
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _row_subject(row):
    for col in ["subj", "subj_idx", "subject", "subject_id"]:
        if col in row.index:
            v = row[col]
            try:
                if pd.notna(v):
                    return int(float(v))
            except Exception:
                pass
    return None


def _z_candidate_score(name, draws, sid):
    low = str(name).lower()
    n = norm(name)

    score = 0

    # Participant specificity
    parsed = _subject_from_name(name)
    if parsed == sid:
        score += 100

    # Prefer actual z over hyperparameters
    if "std" in low or "tau" in low or "var" in low:
        score -= 500
    if "mean" in low or "mu" in low:
        score -= 150

    # Prefer raw z subject node if present
    if n.startswith("zsubj"):
        score += 80
    elif n.startswith("ztranssubj"):
        score += 70
    elif "ztrans" in n:
        score += 50
    elif n.startswith("z"):
        score += 30

    finite = draws[np.isfinite(draws)]
    if finite.size:
        if finite.min() >= 0.0 and finite.max() <= 1.0:
            score += 25

    return score


def resolve_subject_z_chain(model, sid):
    """
    Find participant-specific z inside one fitted chain.

    Returns
    -------
    z_draws : ndarray on natural [0,1] scale
    source_name : str
    audit_candidates : list[str]
    """
    if not hasattr(model, "nodes_db"):
        raise AttributeError("Loaded HDDM model has no nodes_db")

    candidates = []

    for name, row in model.nodes_db.iterrows():
        name_s = str(name)
        low = name_s.lower()
        n = norm(name_s)

        # Must genuinely be a z-related node and not obvious hyperparameter noise.
        if not n.startswith("z"):
            continue
        if any(tok in low for tok in ["std", "tau", "var"]):
            continue

        sid_from_name = _subject_from_name(name_s)
        sid_from_row = _row_subject(row)

        is_subject_match = (
            sid_from_name == int(sid)
            or sid_from_row == int(sid)
        )
        if not is_subject_match:
            continue

        try:
            node = row["node"]
            draws = np.asarray(node.trace(), dtype=float).reshape(-1)
        except Exception:
            continue

        draws = draws[np.isfinite(draws)]
        if draws.size == 0:
            continue

        score = _z_candidate_score(name_s, draws, int(sid))
        candidates.append((score, name_s, draws))

    # Fallback: broader scan through get_traces(), without the old restrictive regex.
    if not candidates:
        try:
            tr = model.get_traces()
            for col in tr.columns:
                name_s = str(col)
                n = norm(name_s)
                if not n.startswith("z"):
                    continue
                if any(tok in name_s.lower() for tok in ["std", "tau", "var"]):
                    continue

                parsed = _subject_from_name(name_s)

                # Generic trailing ID fallback, but only for z + subj-like names.
                if parsed is None and "subj" in name_s.lower():
                    m = re.search(r"[.(\[](\d+)[)\]]?$", name_s)
                    if m:
                        parsed = int(m.group(1))

                if parsed != int(sid):
                    continue

                draws = pd.to_numeric(
                    tr[col], errors="coerce"
                ).to_numpy(float)
                draws = draws[np.isfinite(draws)]
                if draws.size == 0:
                    continue

                score = _z_candidate_score(name_s, draws, int(sid))
                candidates.append((score, name_s, draws))
        except Exception:
            pass

    if not candidates:
        z_names = [
            str(x)
            for x in model.nodes_db.index
            if norm(x).startswith("z")
        ]
        raise KeyError(
            f"Could not resolve participant-level z for participant {sid}.\n"
            "All z-related model node names were:\n  "
            + "\n  ".join(z_names)
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, source, raw = candidates[0]

    finite = raw[np.isfinite(raw)]

    # Natural z is bounded [0,1]. If this is a transformed latent trace, map it.
    if (
        "trans" in source.lower()
        or finite.min() < 0.0
        or finite.max() > 1.0
    ):
        z = expit(raw)
        transform = "expit"
    else:
        z = raw
        transform = "already_[0,1]"

    candidate_names = [x[1] for x in candidates]

    return z, source, transform, candidate_names


def extract_subject_z_all_chains(models, subject_ids):
    rows = []
    draw_map = {}

    for sid in sorted(map(int, subject_ids)):
        chain_draws = []
        sources = []
        transforms = []
        candidate_union = []

        for chain_i, model in enumerate(models):
            z, source, transform, candidates = resolve_subject_z_chain(model, sid)
            chain_draws.append(z)
            sources.append(f"chain{chain_i}:{source}")
            transforms.append(transform)
            candidate_union.extend(candidates)

        z_all = np.concatenate(chain_draws)
        draw_map[sid] = z_all

        row = {"sub_id": sid}
        row.update(summarize_draws(z_all, "z"))
        row["z_S_bias_mean"] = row["z_mean"] - 0.5
        row["z_E_bias_mean"] = 0.5 - row["z_mean"]
        row["trace_z"] = " | ".join(sources)
        row["trace_z_transform"] = " | ".join(transforms)
        row["z_candidate_nodes"] = " | ".join(sorted(set(candidate_union)))
        rows.append(row)

    return pd.DataFrame(rows), draw_map


# ---------------------------------------------------------------------
# Full participant aDDM extraction
# ---------------------------------------------------------------------

def extract_individual_addm_parameters(traces, models, subject_ids):
    idx = index_subject_trace_columns(traces)

    missing = sorted(set(map(int, subject_ids)) - set(idx))
    if missing:
        raise KeyError(
            "Behavioural IDs with no participant-level regression trace columns: "
            f"{missing}"
        )

    # z is resolved separately/directly from nodes_db.
    z_df, _ = extract_subject_z_all_chains(models, subject_ids)
    z_df = z_df.set_index("sub_id")

    rows = []

    for sid in sorted(map(int, subject_ids)):
        b_a_col = find_subject_col(
            idx, sid,
            ["v_AttentionW_SE", "AttentionW_SE"],
            "b_A",
        )
        b_i_col = find_subject_col(
            idx, sid,
            ["v_InattentionW_SE", "InattentionW_SE"],
            "b_I",
        )
        da_col = find_subject_col(
            idx, sid,
            ["v_AttentionContrast_SE", "AttentionContrast_SE"],
            "delta_A",
        )
        di_col = find_subject_col(
            idx, sid,
            ["v_InattentionContrast_SE", "InattentionContrast_SE"],
            "delta_I",
        )

        a_col = find_subject_col(idx, sid, ["a"], "a", required=False)
        t_col = find_subject_col(idx, sid, ["t"], "t", required=False)

        b_A = pd.to_numeric(traces[b_a_col], errors="coerce").to_numpy(float)
        b_I = pd.to_numeric(traces[b_i_col], errors="coerce").to_numpy(float)
        delta_A = pd.to_numeric(traces[da_col], errors="coerce").to_numpy(float)
        delta_I = pd.to_numeric(traces[di_col], errors="coerce").to_numpy(float)

        b_AS = b_A + delta_A / 2.0
        b_AE = b_A - delta_A / 2.0
        b_IS = b_I + delta_I / 2.0
        b_IE = b_I - delta_I / 2.0

        with np.errstate(divide="ignore", invalid="ignore"):
            theta_S = b_IS / b_AS
            theta_E = b_IE / b_AE

        a = (
            pd.to_numeric(traces[a_col], errors="coerce").to_numpy(float)
            if a_col is not None
            else np.repeat(np.nan, len(b_A))
        )
        t = (
            pd.to_numeric(traces[t_col], errors="coerce").to_numpy(float)
            if t_col is not None
            else np.repeat(np.nan, len(b_A))
        )

        row = {
            "sub_id": sid,
            "trace_b_A": b_a_col,
            "trace_b_I": b_i_col,
            "trace_delta_A": da_col,
            "trace_delta_I": di_col,
            "trace_a": a_col if a_col is not None else "NOT FOUND",
            "trace_t": t_col if t_col is not None else "NOT FOUND",
            "n_posterior_draws": int(len(b_A)),
            "p_abs_b_AS_lt_0p05": float(np.nanmean(np.abs(b_AS) < 0.05)),
            "p_abs_b_AE_lt_0p05": float(np.nanmean(np.abs(b_AE) < 0.05)),
        }

        quantities = {
            "b_A": b_A,
            "b_I": b_I,
            "delta_A": delta_A,
            "delta_I": delta_I,
            "b_AS": b_AS,
            "b_AE": b_AE,
            "b_IS": b_IS,
            "b_IE": b_IE,
            "theta_S": theta_S,
            "theta_E": theta_E,
            "theta_S_minus_E": theta_S - theta_E,
            "a": a,
            "t": t,
        }

        for name, values in quantities.items():
            row.update(summarize_draws(values, name))

        # Attach robustly extracted participant z.
        zrow = z_df.loc[sid]
        for c in [
            "z_mean", "z_median", "z_q025", "z_q975",
            "z_S_bias_mean", "z_E_bias_mean",
            "trace_z", "trace_z_transform", "z_candidate_nodes",
        ]:
            row[c] = zrow[c]

        row["theta_ratio_warning"] = bool(
            row["p_abs_b_AS_lt_0p05"] > 0.05
            or row["p_abs_b_AE_lt_0p05"] > 0.05
        )

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("sub_id").reset_index(drop=True)

    # z must be present now; fail loudly otherwise.
    if out["z_mean"].isna().any():
        bad = out.loc[out["z_mean"].isna(), "sub_id"].tolist()
        raise ValueError(f"Participant z extraction produced NaN for IDs: {bad}")

    return out


# ---------------------------------------------------------------------
# Garcia slopes
# ---------------------------------------------------------------------

def read_behavioural_slopes(path: Path):
    df = pd.read_csv(path).copy()

    if "sub_id" not in df.columns:
        raise KeyError(f"{path}: no sub_id column")

    df["sub_id"] = pd.to_numeric(
        df["sub_id"], errors="raise"
    ).astype(int)

    if {"ES", "EE", "SP"}.issubset(df.columns):
        df["ES"] = pd.to_numeric(df["ES"], errors="coerce")
        df["EE"] = pd.to_numeric(df["EE"], errors="coerce")
        df["SP"] = pd.to_numeric(df["SP"], errors="coerce")
    else:
        needed = {
            "ES_slope_all_trials",
            "EE_slope_all_trials",
            "SP_slope_E_only",
        }
        missing = needed - set(df.columns)
        if missing:
            raise KeyError(
                f"{path}: expected ES/EE/SP or canonical full-table columns. "
                f"Missing {sorted(missing)}"
            )

        df["ES"] = pd.to_numeric(
            df["ES_slope_all_trials"], errors="coerce"
        )
        df["EE"] = pd.to_numeric(
            df["EE_slope_all_trials"], errors="coerce"
        )
        df["SP"] = pd.to_numeric(
            df["SP_slope_E_only"], errors="coerce"
        )

    df["ES_raw_compression"] = 1.0 - df["ES"]
    df["selective_vs_EE"] = df["EE"] - df["ES"]
    df["selective_vs_SP"] = df["SP"] - df["ES"]
    df["nonES_mean"] = df[["EE", "SP"]].mean(axis=1)
    df["selective_composite"] = df["nonES_mean"] - df["ES"]

    return df


# ---------------------------------------------------------------------
# Accuracy/performance from analysis-ready behavioural data
# ---------------------------------------------------------------------

def participant_performance(path: Path, subject_ids):
    df = pd.read_csv(path, low_memory=False).copy()

    required = {"sub_id", "phase", "corr", "p1", "cho", "op1"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"{path}: missing columns required for performance analysis: "
            f"{sorted(missing)}"
        )

    df["sub_id"] = pd.to_numeric(
        df["sub_id"], errors="raise"
    ).astype(int)
    df["phase"] = df["phase"].astype(str).str.strip()

    for c in ["corr", "p1", "cho"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    keep_ids = set(map(int, subject_ids))
    df = df.loc[df["sub_id"].isin(keep_ids)].copy()

    rows = []

    for sid in sorted(keep_ids):
        g = df.loc[df["sub_id"].eq(sid)].copy()

        es = g.loc[g["phase"].eq("ES")]
        ee = g.loc[g["phase"].eq("EE")]
        sp = g.loc[g["phase"].eq("SP")]

        if es.empty or ee.empty or sp.empty:
            raise ValueError(
                f"Participant {sid}: expected ES, EE and SP rows; "
                f"counts={{'ES':{len(es)}, 'EE':{len(ee)}, 'SP':{len(sp)}}}"
            )

        acc_es = float(es["corr"].mean())
        acc_ee = float(ee["corr"].mean())

        # This is the raw exact-match variable from the SP conversion.
        acc_sp_exact = float(sp["corr"].mean())

        # Pooled choice accuracy across the two actual choice phases.
        choice = pd.concat(
            [es[["corr"]], ee[["corr"]]],
            ignore_index=True,
        )
        acc_choice_es_ee = float(choice["corr"].mean())

        # Requested raw combined score. Kept, but interpretation is limited
        # because SP corr uses exact probability matching rather than a choice.
        raw_all = pd.concat(
            [es[["corr"]], ee[["corr"]], sp[["corr"]]],
            ignore_index=True,
        )
        acc_all_raw = float(raw_all["corr"].mean())

        # Better continuous SP calibration metric, matched to E-only SP slope.
        sp_e = sp.loc[
            sp["op1"].astype(str).str.strip().eq("E"),
            ["p1", "cho"],
        ].dropna()

        if sp_e.empty:
            raise ValueError(
                f"Participant {sid}: no E-only SP ratings for MAE."
            )

        abs_err = np.abs(
            sp_e["cho"].to_numpy(float)
            - sp_e["p1"].to_numpy(float)
        )
        sq_err = (
            sp_e["cho"].to_numpy(float)
            - sp_e["p1"].to_numpy(float)
        ) ** 2

        sp_mae = float(np.mean(abs_err))
        sp_rmse = float(np.sqrt(np.mean(sq_err)))
        sp_accuracy_cont = float(1.0 - sp_mae)

        rows.append({
            "sub_id": sid,
            "accuracy_ES": acc_es,
            "accuracy_EE": acc_ee,
            "accuracy_SP_exact": acc_sp_exact,
            "accuracy_choice_ES_EE": acc_choice_es_ee,
            "accuracy_all_raw_ES_EE_SP": acc_all_raw,
            "SP_E_MAE": sp_mae,
            "SP_E_RMSE": sp_rmse,
            "SP_E_estimation_accuracy": sp_accuracy_cont,
            "n_ES": int(len(es)),
            "n_EE": int(len(ee)),
            "n_SP": int(len(sp)),
            "n_SP_E": int(len(sp_e)),
        })

    out = pd.DataFrame(rows)

    missing_ids = sorted(keep_ids - set(out["sub_id"]))
    if missing_ids:
        raise ValueError(
            f"Behavioural performance missing participant IDs: {missing_ids}"
        )

    return out



# ---------------------------------------------------------------------
# SP confidence summaries
# ---------------------------------------------------------------------

def read_sp_confidence_summary(path: Path, subject_ids):
    df = pd.read_csv(path).copy()

    required = {
        "sub_id",
        "SP_confidence_all",
        "SP_confidence_E",
        "SP_confidence_S",
        "SP_confidence_S_minus_E",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"{path}: missing SP confidence summary columns: {sorted(missing)}"
        )

    df["sub_id"] = pd.to_numeric(
        df["sub_id"], errors="raise"
    ).astype(int)

    for c in sorted(required - {"sub_id"}):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    keep_ids = set(map(int, subject_ids))
    df = df.loc[df["sub_id"].isin(keep_ids)].copy()

    if df["sub_id"].duplicated().any():
        bad = df.loc[df["sub_id"].duplicated(keep=False), "sub_id"].tolist()
        raise ValueError(
            f"{path}: duplicate participant IDs in confidence summary: {bad}"
        )

    missing_ids = sorted(keep_ids - set(df["sub_id"]))
    if missing_ids:
        raise ValueError(
            f"{path}: confidence summary missing participant IDs: {missing_ids}"
        )

    return df.sort_values("sub_id").reset_index(drop=True)


# ---------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------

def leave_one_out_pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    vals = []

    for i in range(len(x)):
        xx = np.delete(x, i)
        yy = np.delete(y, i)
        if (
            len(xx) >= 3
            and np.std(xx) > 0
            and np.std(yy) > 0
        ):
            r_loo, _ = stats.pearsonr(xx, yy)
            vals.append(float(r_loo))

    if not vals:
        return np.nan, np.nan

    return float(np.min(vals)), float(np.max(vals))


def association_stats(df, x_col, y_col):
    d = df[[x_col, y_col]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    x = d[x_col].to_numpy(float)
    y = d[y_col].to_numpy(float)
    n = len(d)

    if n < 5 or np.std(x) == 0 or np.std(y) == 0:
        return None

    pear_r, pear_p = stats.pearsonr(x, y)
    spear_rho, spear_p = stats.spearmanr(x, y)

    reg_slope, reg_intercept, reg_r, reg_p, reg_stderr = stats.linregress(
        x, y
    )

    tcrit = stats.t.ppf(0.975, df=n - 2)
    slope_lo = reg_slope - tcrit * reg_stderr
    slope_hi = reg_slope + tcrit * reg_stderr

    loo_lo, loo_hi = leave_one_out_pearson(x, y)

    return {
        "parameter": x_col,
        "behaviour": y_col,
        "n": n,
        "pearson_r": float(pear_r),
        "pearson_p": float(pear_p),
        "spearman_rho": float(spear_rho),
        "spearman_p": float(spear_p),
        "regression_slope": float(reg_slope),
        "regression_slope_ci_low": float(slope_lo),
        "regression_slope_ci_high": float(slope_hi),
        "regression_intercept": float(reg_intercept),
        "r2": float(reg_r ** 2),
        "loo_pearson_r_min": loo_lo,
        "loo_pearson_r_max": loo_hi,
    }


PARAMETERS = [
    "b_A_mean",
    "b_I_mean",
    "delta_A_mean",
    "delta_I_mean",
    "b_AS_mean",
    "b_AE_mean",
    "b_IS_mean",
    "b_IE_mean",
    "theta_S_mean",
    "theta_E_mean",
    "theta_S_minus_E_mean",
    "z_mean",
    "a_mean",
    "t_mean",
]

BEHAVIOURS = [
    # primary/selective
    "selective_vs_EE",
    # raw value-sensitivity
    "ES",
    "EE",
    "SP",
    # actual choice accuracy
    "accuracy_ES",
    "accuracy_EE",
    "accuracy_choice_ES_EE",
    # requested SP/raw-all scores
    "accuracy_SP_exact",
    "accuracy_all_raw_ES_EE_SP",
    # better continuous SP calibration
    "SP_E_estimation_accuracy",
    "SP_E_MAE",
    # SP confidence ratings
    "SP_confidence_all",
    "SP_confidence_E",
    "SP_confidence_S",
    "SP_confidence_S_minus_E",
]

PARAM_LABELS = {
    "b_A_mean": "b_A",
    "b_I_mean": "b_I",
    "delta_A_mean": "δ_A",
    "delta_I_mean": "δ_I",
    "b_AS_mean": "b_AS",
    "b_AE_mean": "b_AE",
    "b_IS_mean": "b_IS",
    "b_IE_mean": "b_IE",
    "theta_S_mean": "θ_S",
    "theta_E_mean": "θ_E",
    "theta_S_minus_E_mean": "θ_S−θ_E",
    "z_mean": "z",
    "a_mean": "a",
    "t_mean": "t",
}

BEHAVIOUR_LABELS = {
    "selective_vs_EE": "EE−ES slope",
    "ES": "ES slope",
    "EE": "EE slope",
    "SP": "SP slope",
    "accuracy_ES": "ES accuracy",
    "accuracy_EE": "EE accuracy",
    "accuracy_choice_ES_EE": "ES+EE choice accuracy",
    "accuracy_SP_exact": "SP exact-match accuracy",
    "accuracy_all_raw_ES_EE_SP": "Raw ES+EE+SP accuracy",
    "SP_E_estimation_accuracy": "SP E estimation accuracy (1−MAE)",
    "SP_E_MAE": "SP E MAE",
    "SP_confidence_all": "SP confidence (all)",
    "SP_confidence_E": "SP confidence (E)",
    "SP_confidence_S": "SP confidence (S)",
    "SP_confidence_S_minus_E": "SP confidence S−E",
}


def build_all_associations(merged):
    rows = []

    for behaviour in BEHAVIOURS:
        if behaviour not in merged.columns:
            continue

        for param in PARAMETERS:
            if param not in merged.columns:
                continue

            s = association_stats(merged, param, behaviour)
            if s is not None:
                rows.append(s)

    out = pd.DataFrame(rows)

    if len(out):
        out["pearson_p_fdr_all"] = fdr_bh_adjust(
            out["pearson_p"].to_numpy(float)
        )
        out["spearman_p_fdr_all"] = fdr_bh_adjust(
            out["spearman_p"].to_numpy(float)
        )

    return out


# ---------------------------------------------------------------------
# ONE compact heatmap
# ---------------------------------------------------------------------

def plot_correlation_heatmap(assoc, out_file: Path, study: int):
    behaviours = [
        x for x in BEHAVIOURS
        if x in set(assoc["behaviour"])
    ]
    params = [
        x for x in PARAMETERS
        if x in set(assoc["parameter"])
    ]

    mat = np.full((len(behaviours), len(params)), np.nan)
    p_raw = np.full_like(mat, np.nan)
    p_fdr = np.full_like(mat, np.nan)

    for i, beh in enumerate(behaviours):
        for j, par in enumerate(params):
            row = assoc.loc[
                assoc["behaviour"].eq(beh)
                & assoc["parameter"].eq(par)
            ]
            if len(row) == 1:
                mat[i, j] = float(row.iloc[0]["pearson_r"])
                p_raw[i, j] = float(row.iloc[0]["pearson_p"])
                p_fdr[i, j] = float(row.iloc[0]["pearson_p_fdr_all"])

    fig_w = max(12, 0.85 * len(params))
    fig_h = max(8, 0.70 * len(behaviours))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        mat,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(params)))
    ax.set_xticklabels(
        [PARAM_LABELS.get(x, x) for x in params],
        rotation=45,
        ha="right",
    )

    ax.set_yticks(np.arange(len(behaviours)))
    ax.set_yticklabels(
        [BEHAVIOUR_LABELS.get(x, x) for x in behaviours]
    )

    for i in range(len(behaviours)):
        for j in range(len(params)):
            if not np.isfinite(mat[i, j]):
                continue

            mark = ""
            if np.isfinite(p_fdr[i, j]) and p_fdr[i, j] < .05:
                mark = "**"
            elif np.isfinite(p_raw[i, j]) and p_raw[i, j] < .05:
                mark = "*"

            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}{mark}",
                ha="center",
                va="center",
                fontsize=8,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Pearson r")

    ax.set_title(
        f"Study {study}: behaviour × ESaDDM individual differences\n"
        "* raw p < .05; ** survives FDR across the full exploratory matrix",
        pad=14,
    )
    ax.set_xlabel("Participant-level ESaDDM parameter")
    ax.set_ylabel("Behavioural individual-difference measure")

    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(path, study, merged, planned, assoc):
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"STUDY {study} — COMPACT BEHAVIOUR x ESaDDM INDIVIDUAL DIFFERENCES\n"
        )
        f.write("=" * 92 + "\n\n")

        f.write("Z EXTRACTION\n")
        f.write(
            "Participant-specific z was resolved directly from HDDM model.nodes_db.\n"
        )
        zcols = [
            "sub_id", "z_mean", "trace_z", "trace_z_transform"
        ]
        f.write(
            merged[zcols].to_string(index=False)
            + "\n\n"
        )

        f.write("PERFORMANCE DESCRIPTIVES\n")
        perf_cols = [
            "accuracy_ES",
            "accuracy_EE",
            "accuracy_choice_ES_EE",
            "accuracy_SP_exact",
            "accuracy_all_raw_ES_EE_SP",
            "SP_E_estimation_accuracy",
            "SP_E_MAE",
        ]
        for col in perf_cols:
            vals = pd.to_numeric(
                merged[col], errors="coerce"
            ).dropna().to_numpy(float)
            if vals.size:
                f.write(
                    f"{col}: N={len(vals)}, M={np.mean(vals):.4f}, "
                    f"SD={np.std(vals, ddof=1):.4f}, "
                    f"range=[{np.min(vals):.4f}, {np.max(vals):.4f}]\n"
                )

        f.write(
            "\nNOTE: SP exact-match accuracy uses corr=1 only when the slider "
            "estimate equals the objective probability exactly. It is not "
            "equivalent to ES/EE binary choice accuracy. SP_E_MAE and "
            "SP_E_estimation_accuracy are the more graded SP performance measures.\n\n"
        )

        f.write("PRIMARY PLANNED ASSOCIATIONS — HOLM CORRECTED\n")
        if len(planned):
            cols = [
                "parameter", "behaviour", "n",
                "pearson_r", "pearson_p", "pearson_p_holm",
                "spearman_rho", "spearman_p",
                "regression_slope",
                "regression_slope_ci_low",
                "regression_slope_ci_high",
                "loo_pearson_r_min",
                "loo_pearson_r_max",
            ]
            f.write(planned[cols].to_string(index=False) + "\n")

        f.write("\nSP CONFIDENCE DESCRIPTIVES\n")
        conf_cols = [
            "SP_confidence_all",
            "SP_confidence_E",
            "SP_confidence_S",
            "SP_confidence_S_minus_E",
        ]
        for col in conf_cols:
            if col in merged.columns:
                vals = pd.to_numeric(
                    merged[col], errors="coerce"
                ).dropna().to_numpy(float)
                if vals.size:
                    f.write(
                        f"{col}: N={len(vals)}, M={np.mean(vals):.4f}, "
                        f"SD={np.std(vals, ddof=1):.4f}, "
                        f"range=[{np.min(vals):.4f}, {np.max(vals):.4f}]\n"
                    )

        f.write("\nTOP EXPLORATORY PEARSON ASSOCIATIONS\n")
        if len(assoc):
            top = assoc.sort_values(
                ["pearson_p", "parameter", "behaviour"]
            ).head(20)
            cols = [
                "behaviour", "parameter", "n",
                "pearson_r", "pearson_p", "pearson_p_fdr_all",
                "spearman_rho", "spearman_p",
                "loo_pearson_r_min", "loo_pearson_r_max",
            ]
            f.write(top[cols].to_string(index=False) + "\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        type=int,
        choices=[1, 2],
        required=True,
    )
    parser.add_argument(
        "--slopes",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--behaviour-data",
        type=Path,
        required=True,
        help=(
            "StudyX_Behaviour_with_Gaze_AnalysisReady.csv containing "
            "sub_id, phase, corr, p1, cho, op1."
        ),
    )
    parser.add_argument(
        "--confidence-summary",
        type=Path,
        required=True,
        help=(
            "Participant-level SP confidence summary CSV produced by "
            "06_build_SP_confidence_summary.py."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    root = ROOTS[args.study]
    model_dir = args.model_dir or (root / FIT_REL)
    out_dir = args.out_dir or (root / OUT_REL)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 92, flush=True)
    print(f"STUDY {args.study}", flush=True)
    print("Slope file:", args.slopes, flush=True)
    print("Behaviour file:", args.behaviour_data, flush=True)
    print("Confidence summary:", args.confidence_summary, flush=True)
    print("Model dir:", model_dir, flush=True)
    print("Output dir:", out_dir, flush=True)
    print("=" * 92, flush=True)

    slopes = read_behavioural_slopes(args.slopes)
    subject_ids = slopes["sub_id"].tolist()

    performance = participant_performance(
        args.behaviour_data,
        subject_ids,
    )

    confidence = read_sp_confidence_summary(
        args.confidence_summary,
        subject_ids,
    )

    models, traces = load_chain_models_and_traces(model_dir)

    addm = extract_individual_addm_parameters(
        traces,
        models,
        subject_ids,
    )

    addm_file = out_dir / f"Study{args.study}_addm_individual_params.csv"
    perf_file = out_dir / f"Study{args.study}_participant_performance.csv"
    conf_file = out_dir / f"Study{args.study}_SP_confidence_summary_used.csv"
    addm.to_csv(addm_file, index=False)
    performance.to_csv(perf_file, index=False)
    confidence.to_csv(conf_file, index=False)

    merged = (
        slopes
        .merge(performance, on="sub_id", how="inner", validate="one_to_one")
        .merge(confidence, on="sub_id", how="inner", validate="one_to_one")
        .merge(addm, on="sub_id", how="inner", validate="one_to_one")
        .sort_values("sub_id")
        .reset_index(drop=True)
    )

    if len(merged) != len(slopes):
        raise ValueError(
            f"Slope N={len(slopes)} but merged N={len(merged)}; ID mismatch."
        )

    merged_file = out_dir / f"Study{args.study}_behaviour_addm_merged.csv"
    merged.to_csv(merged_file, index=False)

    # Primary planned tests
    planned_pairs = [
        ("theta_S_minus_E_mean", "selective_vs_EE"),
        ("delta_I_mean", "selective_vs_EE"),
        ("delta_A_mean", "selective_vs_EE"),
        ("z_mean", "selective_vs_EE"),
    ]

    planned_rows = []
    for param, behaviour in planned_pairs:
        s = association_stats(merged, param, behaviour)
        if s is not None:
            planned_rows.append(s)

    planned = pd.DataFrame(planned_rows)
    if len(planned):
        planned["pearson_p_holm"] = holm_adjust(
            planned["pearson_p"].to_numpy(float)
        )
        planned.to_csv(
            out_dir / f"Study{args.study}_planned_associations.csv",
            index=False,
        )


    # Focused exploratory SP-confidence × z tests.
    # Because z>0.5 is an S starting-point bias and z<0.5 is an E bias,
    # SP_confidence_S_minus_E is the most directly interpretable contrast.
    confidence_z_behaviours = [
        "SP_confidence_all",
        "SP_confidence_E",
        "SP_confidence_S",
        "SP_confidence_S_minus_E",
    ]

    confidence_z_rows = []
    for behaviour in confidence_z_behaviours:
        s = association_stats(merged, "z_mean", behaviour)
        if s is not None:
            confidence_z_rows.append(s)

    confidence_z = pd.DataFrame(confidence_z_rows)
    if len(confidence_z):
        confidence_z["pearson_p_holm"] = holm_adjust(
            confidence_z["pearson_p"].to_numpy(float)
        )
        confidence_z.to_csv(
            out_dir / f"Study{args.study}_confidence_z_associations.csv",
            index=False,
        )

    assoc = build_all_associations(merged)
    assoc.to_csv(
        out_dir / f"Study{args.study}_all_associations.csv",
        index=False,
    )

    plot_correlation_heatmap(
        assoc,
        out_dir / f"Study{args.study}_CORRELATION_OVERVIEW.png",
        args.study,
    )

    # z audit specifically, so it can be checked without opening the huge merged table.
    z_audit_cols = [
        "sub_id", "z_mean", "z_median", "z_q025", "z_q975",
        "z_S_bias_mean", "trace_z", "trace_z_transform",
        "z_candidate_nodes",
    ]
    addm[z_audit_cols].to_csv(
        out_dir / f"Study{args.study}_z_extraction_audit.csv",
        index=False,
    )

    write_summary(
        out_dir / f"Study{args.study}_SUMMARY.txt",
        args.study,
        merged,
        planned,
        assoc,
    )

    print("\nDONE", flush=True)
    print("Main figure:", out_dir / f"Study{args.study}_CORRELATION_OVERVIEW.png")
    print("Associations:", out_dir / f"Study{args.study}_all_associations.csv")
    print("Confidence × z:", out_dir / f"Study{args.study}_confidence_z_associations.csv")
    print("z audit:", out_dir / f"Study{args.study}_z_extraction_audit.csv")
    print("Summary:", out_dir / f"Study{args.study}_SUMMARY.txt")

    print("\nPRIMARY PLANNED ASSOCIATIONS", flush=True)
    if len(planned):
        print(
            planned[
                [
                    "parameter", "behaviour", "n",
                    "pearson_r", "pearson_p",
                    "pearson_p_holm",
                    "spearman_rho", "spearman_p",
                ]
            ].to_string(index=False),
            flush=True,
        )


if __name__ == "__main__":
    main()
