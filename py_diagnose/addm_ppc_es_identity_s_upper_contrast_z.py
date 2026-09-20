from __future__ import annotations

import argparse
import gc
import json
import math
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hddm

# ============================================================
# constants
# ============================================================

RT_QUANTILES = [0.10, 0.30, 0.50, 0.70, 0.90]
RT_QUANTILE_LABELS = ["q10", "q30", "q50", "q70", "q90"]
DWELL_LABELS = ["E>>S", "E>S", "S~E", "S>E", "S>>E"]
RT_QUINTILE_LABELS = ["1", "2", "3", "4", "5"]

# Consistent presentation colours:
# empirical E/S = bright blue / dark orchid
# posterior-predictive E/S = darker colours from the same families
OBS_E_COLOR = "deepskyblue"
PPC_E_COLOR = "steelblue"
OBS_S_COLOR = "darkorchid"
PPC_S_COLOR = "indigo"


# ============================================================
# generic helpers
# ============================================================

def safe_reset_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    names = []
    for i, name in enumerate(df.index.names):
        base = name if name is not None else f"index_level_{i}"
        new = str(base)
        k = 1
        while new in df.columns or new in names:
            new = f"idx_{base}_{k}"
            k += 1
        names.append(new)
    df.index = df.index.set_names(names)
    return df.reset_index()


def detect_sample_column(df: pd.DataFrame) -> str:
    for col in ["sample", "draw"]:
        if col in df.columns:
            return col
    for col in df.columns:
        if str(col).startswith("level_") or str(col).startswith("idx_sample"):
            return col
    raise ValueError(
        "Could not detect posterior-predictive sample column. "
        f"Available columns: {list(df.columns)}"
    )


def discover_chain_files(model_dir: Path, chains: int) -> list[Path]:
    out = []
    for c in range(chains):
        matches = sorted(model_dir.glob(f"*_{c}.hddm"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one '*_{c}.hddm' in {model_dir}; found: {matches}"
            )
        out.append(matches[0])
    return out


def split_counts(total: int, n: int) -> list[int]:
    if total < n:
        raise ValueError(f"ppc total ({total}) must be at least the number of chains ({n})")
    q, r = divmod(total, n)
    return [q + (1 if i < r else 0) for i in range(n)]


def attach_identity_gaze_from_analysisready(
    obs: pd.DataFrame,
    analysisready_path: Path,
) -> pd.DataFrame:
    """
    Attach the corrected E/S identity-coordinate gaze variables to the exact
    trial rows used in the fitted HDDM model.

    MAIN old-style PPC variable:
        DwellTimeAdvantage_ES = Dwell_S - Dwell_E
        positive -> more dwell time on S
        negative -> more dwell time on E

    Additional identity-coordinate proportional check:
        DwellPropAdvantage_ES = PropDwell_S - PropDwell_E

    The merge is strict and one-to-one on:
        sub_id + phase + trial

    This deliberately never uses the spatial left/right DwellTimeAdvantage.
    """
    required_model = {
        "sub_id", "phase", "trial",
        "PropDwell_E", "PropDwell_S",
    }
    missing_model = sorted(required_model - set(obs.columns))
    if missing_model:
        raise ValueError(
            "Fitted model data are missing columns required for the strict "
            f"identity-gaze merge: {missing_model}"
        )

    ar = pd.read_csv(analysisready_path, low_memory=False)

    required_ar = {
        "sub_id", "phase", "trial",
        "Dwell_E", "Dwell_S",
        "PropDwell_E", "PropDwell_S",
        "DwellTimeAdvantage_ES",
        "DwellPropAdvantage_ES",
    }
    missing_ar = sorted(required_ar - set(ar.columns))
    if missing_ar:
        raise ValueError(
            "AnalysisReady data are missing required corrected E/S identity "
            f"gaze columns: {missing_ar}"
        )

    ar = ar.loc[
        ar["phase"].astype(str).str.strip().eq("ES"),
        sorted(required_ar),
    ].copy()

    for col in [
        "sub_id", "trial",
        "Dwell_E", "Dwell_S",
        "PropDwell_E", "PropDwell_S",
        "DwellTimeAdvantage_ES",
        "DwellPropAdvantage_ES",
    ]:
        ar[col] = pd.to_numeric(ar[col], errors="coerce")

    ar["phase"] = ar["phase"].astype(str).str.strip()

    if ar.duplicated(["sub_id", "phase", "trial"]).any():
        bad = ar.loc[
            ar.duplicated(["sub_id", "phase", "trial"], keep=False),
            ["sub_id", "phase", "trial"],
        ]
        raise ValueError(
            "AnalysisReady ES data have duplicate identity merge keys:\n"
            + bad.head(30).to_string(index=False)
        )

    # Validate the explicit corrected identity-coordinate definitions.
    time_diff = (
        ar["DwellTimeAdvantage_ES"]
        - (ar["Dwell_S"] - ar["Dwell_E"])
    ).abs()
    prop_diff = (
        ar["DwellPropAdvantage_ES"]
        - (ar["PropDwell_S"] - ar["PropDwell_E"])
    ).abs()

    if time_diff.dropna().empty or float(time_diff.dropna().max()) > 1e-8:
        raise ValueError(
            "DwellTimeAdvantage_ES does not equal Dwell_S - Dwell_E "
            "in the AnalysisReady data."
        )
    if prop_diff.dropna().empty or float(prop_diff.dropna().max()) > 1e-8:
        raise ValueError(
            "DwellPropAdvantage_ES does not equal PropDwell_S - PropDwell_E "
            "in the AnalysisReady data."
        )

    z = obs.copy().reset_index(drop=True)
    z["__model_row"] = np.arange(len(z))
    z["phase"] = z["phase"].astype(str).str.strip()

    merge_cols = [
        "sub_id", "phase", "trial",
        "Dwell_E", "Dwell_S",
        "PropDwell_E", "PropDwell_S",
        "DwellTimeAdvantage_ES",
        "DwellPropAdvantage_ES",
    ]

    # Avoid name collisions so we can explicitly compare the model's rounded
    # PropDwell_E/S with the AnalysisReady identity values.
    ar_merge = ar[merge_cols].rename(columns={
        "PropDwell_E": "PropDwell_E_analysisready",
        "PropDwell_S": "PropDwell_S_analysisready",
    })

    z = z.merge(
        ar_merge,
        on=["sub_id", "phase", "trial"],
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("__model_row").reset_index(drop=True)

    needed = [
        "DwellTimeAdvantage_ES",
        "DwellPropAdvantage_ES",
        "Dwell_E", "Dwell_S",
        "PropDwell_E_analysisready",
        "PropDwell_S_analysisready",
    ]
    if z[needed].isna().any().any():
        bad = z.loc[
            z[needed].isna().any(axis=1),
            ["sub_id", "phase", "trial", *needed],
        ]
        raise ValueError(
            "Not every fitted HDDM trial matched exactly one AnalysisReady ES "
            "identity-gaze row. First unmatched/problem rows:\n"
            + bad.head(30).to_string(index=False)
        )

    # The identity model preparation rounds model PropDwell_E/S to 3 decimals.
    # Check that the attached raw AnalysisReady identity proportions agree.
    if not np.allclose(
        pd.to_numeric(z["PropDwell_E"], errors="coerce"),
        pd.to_numeric(z["PropDwell_E_analysisready"], errors="coerce"),
        atol=0.002,
        rtol=0,
    ):
        raise ValueError(
            "Model PropDwell_E does not agree with the matched AnalysisReady "
            "identity-coordinate PropDwell_E."
        )

    if not np.allclose(
        pd.to_numeric(z["PropDwell_S"], errors="coerce"),
        pd.to_numeric(z["PropDwell_S_analysisready"], errors="coerce"),
        atol=0.002,
        rtol=0,
    ):
        raise ValueError(
            "Model PropDwell_S does not agree with the matched AnalysisReady "
            "identity-coordinate PropDwell_S."
        )

    z = z.drop(columns=["__model_row"])
    return z


def compute_quantile_edges(series, n_quantiles=5):
    s = pd.Series(series).dropna().astype(float).copy()
    if s.empty:
        raise ValueError("Cannot compute quintiles from an empty series.")
    # tiny deterministic jitter prevents qcut failures from tied values
    s_jittered = s + np.linspace(0, 1e-10, len(s))
    _, edges = pd.qcut(
        s_jittered,
        q=n_quantiles,
        retbins=True,
        duplicates="drop",
    )
    edges = np.unique(edges)
    if len(edges) - 1 != n_quantiles:
        raise ValueError(
            f"Expected {n_quantiles} bins but obtained {len(edges)-1}. "
            "Inspect the predictor distribution before changing the analysis."
        )
    return edges


def assign_bins_from_edges(series, edges, labels):
    return pd.cut(
        pd.to_numeric(series, errors="coerce"),
        bins=edges,
        labels=labels,
        include_lowest=True,
    )


def assign_rank_quintiles(series, labels=RT_QUINTILE_LABELS):
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=x.index, dtype="object")
    ok = x.notna()
    if ok.sum():
        ranks = x.loc[ok].rank(method="first")
        out.loc[ok] = pd.qcut(
            ranks,
            q=len(labels),
            labels=labels,
            duplicates="drop",
        ).astype(str)
    return pd.Categorical(out, categories=labels, ordered=True)


def bootstrap_mean_ci(values, n_boot=5000, ci=0.95, seed=123):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        return values[0], values[0], values[0]
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = np.mean(rng.choice(values, size=n, replace=True))
    alpha = 1.0 - ci
    return (
        float(np.mean(values)),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def predictive_summary(values):
    x = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {
            "predictive_mean": np.nan,
            "predictive_median": np.nan,
            "predictive_pi95_low": np.nan,
            "predictive_pi95_high": np.nan,
            "n_predictive_draws": 0,
        }
    return {
        "predictive_mean": float(x.mean()),
        "predictive_median": float(x.median()),
        "predictive_pi95_low": float(x.quantile(0.025)),
        "predictive_pi95_high": float(x.quantile(0.975)),
        "n_predictive_draws": int(len(x)),
    }


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_close(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# load models and generate posterior predictive datasets
# ============================================================

def load_models(model_dir: Path, chains: int):
    files = discover_chain_files(model_dir, chains)
    models = []
    for c, path in enumerate(files):
        print(f"loading chain {c}: {path}", flush=True)
        models.append(hddm.load(str(path)))
    return models, files


def verify_observed_data_match(models):
    first = models[0].data.reset_index(drop=True).copy()
    key_cols = [c for c in ["subj_idx", "rt", "response"] if c in first.columns]
    for c, m in enumerate(models[1:], start=1):
        other = m.data.reset_index(drop=True)
        if len(other) != len(first):
            raise ValueError(f"chain {c} has {len(other)} observed rows; chain 0 has {len(first)}")
        for col in key_cols:
            a = pd.to_numeric(first[col], errors="coerce").to_numpy()
            b = pd.to_numeric(other[col], errors="coerce").to_numpy()
            if not np.allclose(a, b, equal_nan=True):
                raise ValueError(f"observed column {col!r} differs between chain 0 and chain {c}")
    return first


def attach_observed_columns_by_row_order(
    obs_df: pd.DataFrame,
    ppc_df: pd.DataFrame,
    draw_col: str,
    cols_to_attach: list[str],
) -> pd.DataFrame:
    """
    Reattach the ORIGINAL observed trial mapping to every posterior-predictive
    draw by row order.

    Important: append_data=True may already return columns such as subj_idx,
    but those columns are not trusted for the trial-to-participant mapping.
    Requested observed columns are therefore deliberately overwritten from the
    fitted observed data, matching the safe behaviour of the older PPC script.
    """
    missing_obs = [c for c in cols_to_attach if c not in obs_df.columns]
    if missing_obs:
        raise ValueError(
            "Observed model data are missing columns requested for PPC remapping: "
            f"{missing_obs}"
        )

    obs_map = obs_df[cols_to_attach].copy().reset_index(drop=True)
    obs_map["orig_row"] = np.arange(len(obs_map))

    out = ppc_df.copy()
    out["orig_row"] = out.groupby(draw_col, sort=False).cumcount()

    rows_per_draw = out.groupby(draw_col, sort=False)["orig_row"].max() + 1
    if not (rows_per_draw == len(obs_map)).all():
        raise ValueError(
            "Could not safely reattach observed columns because posterior-"
            "predictive draws do not contain exactly the observed number of rows. "
            f"Expected {len(obs_map)}; got summary:\n{rows_per_draw.describe()}"
        )

    # Deliberately remove stale append_data copies and restore the original
    # trial-level mapping from model.data.
    out = out.drop(columns=cols_to_attach, errors="ignore")
    out = out.merge(
        obs_map,
        on="orig_row",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    return out


def validate_ppc_participant_mapping(
    obs_df: pd.DataFrame,
    ppc_df: pd.DataFrame,
    draw_col: str,
    subject_col: str,
) -> dict:
    """
    Hard fail unless every PPC draw contains exactly the same participants and
    exactly the same per-participant row counts as the observed fitted data.
    """
    expected_counts = (
        obs_df.groupby(subject_col, dropna=False)
        .size()
        .sort_index()
    )
    expected_subjects = list(expected_counts.index)
    expected_n = len(expected_subjects)

    draw_subject_counts = (
        ppc_df.groupby(draw_col, sort=False)[subject_col]
        .nunique(dropna=True)
    )
    bad_n = draw_subject_counts.loc[draw_subject_counts != expected_n]
    if len(bad_n):
        raise ValueError(
            "PPC participant mapping failed: some draws do not contain the "
            f"expected {expected_n} participants. First failures:\n"
            + bad_n.head(20).to_string()
        )

    problems = []
    for draw, g in ppc_df.groupby(draw_col, sort=False):
        got = g.groupby(subject_col, dropna=False).size().sort_index()
        if not got.index.equals(expected_counts.index):
            problems.append(
                f"{draw}: participant IDs differ; got={list(got.index)}"
            )
            continue
        if not np.array_equal(got.to_numpy(), expected_counts.to_numpy()):
            bad = pd.DataFrame({
                "expected": expected_counts,
                "got": got,
            })
            bad = bad.loc[bad["expected"] != bad["got"]]
            problems.append(
                f"{draw}: per-participant trial counts differ:\n"
                + bad.head(20).to_string()
            )
        if len(problems) >= 10:
            break

    if problems:
        raise ValueError(
            "PPC participant/trial mapping failed.\n"
            + "\n\n".join(problems)
        )

    return {
        "expected_participants": int(expected_n),
        "participant_ids": [
            int(x) if isinstance(x, (int, np.integer)) or (
                isinstance(x, float) and float(x).is_integer()
            ) else str(x)
            for x in expected_subjects
        ],
        "expected_rows_per_draw": int(len(obs_df)),
        "draws_checked": int(ppc_df[draw_col].nunique()),
        "min_participants_per_draw": int(draw_subject_counts.min()),
        "max_participants_per_draw": int(draw_subject_counts.max()),
        "status": "passed",
    }


def generate_chain_ppc(model, chain: int, samples: int):
    print(f"chain {chain}: generating {samples} posterior-predictive datasets", flush=True)
    raw = hddm.utils.post_pred_gen(
        model,
        samples=samples,
        append_data=True,
    )
    flat = safe_reset_index(raw)
    sample_col = detect_sample_column(flat)

    if "response_sampled" not in flat.columns:
        raise ValueError(
            f"chain {chain}: expected response_sampled in PPC output; columns={list(flat.columns)}"
        )
    if "rt_sampled" not in flat.columns:
        raise ValueError(
            f"chain {chain}: expected rt_sampled in PPC output; columns={list(flat.columns)}"
        )

    flat["chain"] = int(chain)
    flat["chain_draw"] = pd.to_numeric(flat[sample_col], errors="coerce").astype("Int64")
    if flat["chain_draw"].isna().any():
        raise ValueError(f"chain {chain}: sample/draw column contains non-numeric values")
    flat["ppc_draw"] = flat["chain"].astype(str) + "_" + flat["chain_draw"].astype(str)
    return raw, flat, sample_col


# ============================================================
# standard HDDM outputs retained from the user's previous PPC
# ============================================================

def save_parameter_summary(model, path: Path):
    try:
        model.gen_stats().to_csv(path)
    except Exception as exc:
        path.with_suffix(".error.txt").write_text(str(exc), encoding="utf-8")


def save_hddm_post_pred_stats(model, raw_ppc, path: Path):
    try:
        stats = hddm.utils.post_pred_stats(model.data, raw_ppc)
        stats.to_csv(path)
    except Exception as exc:
        path.with_suffix(".error.txt").write_text(str(exc), encoding="utf-8")


def save_builtin_hddm_plots(model, out_dir: Path, chain: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        model.plot_posterior_predictive(figsize=(12, 10))
        plt.suptitle(f"HDDM posterior predictive plot: chain {chain}")
        plt.tight_layout()
        plt.savefig(
            out_dir / f"builtin_posterior_predictive_chain_{chain}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    except Exception as exc:
        (out_dir / f"builtin_posterior_predictive_chain_{chain}.error.txt").write_text(
            str(exc), encoding="utf-8"
        )
        plt.close("all")

    try:
        model.plot_posterior_quantiles(columns=3, hexbin=True)
        plt.suptitle(f"HDDM posterior quantiles: chain {chain}")
        plt.tight_layout()
        plt.savefig(
            out_dir / f"builtin_posterior_quantiles_chain_{chain}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    except Exception as exc:
        (out_dir / f"builtin_posterior_quantiles_chain_{chain}.error.txt").write_text(
            str(exc), encoding="utf-8"
        )
        plt.close("all")


# ============================================================
# observed and simulated summaries
# ============================================================

def subjectwise_observed_bin_summary(
    df,
    subject_col,
    bin_col,
    response_col,
    order,
    n_boot,
    ci,
    seed,
):
    tmp = df[[subject_col, bin_col, response_col]].dropna().copy()
    tmp[bin_col] = pd.Categorical(tmp[bin_col], categories=order, ordered=True)

    subj_bin = (
        tmp.groupby([subject_col, bin_col], observed=False)[response_col]
        .mean()
        .reset_index()
        .rename(columns={response_col: "subject_mean"})
    )

    rows = []
    for i, b in enumerate(order):
        vals = subj_bin.loc[subj_bin[bin_col] == b, "subject_mean"].dropna().to_numpy(float)
        mean, low, high = bootstrap_mean_ci(
            vals,
            n_boot=n_boot,
            ci=ci,
            seed=seed + i,
        )
        rows.append({
            "bin": b,
            "mean": mean,
            "boot_ci_low": low,
            "boot_ci_high": high,
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "sem": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan,
            "n_subjects": int(len(vals)),
        })
    return pd.DataFrame(rows), subj_bin


def simulated_bin_summary(
    df,
    draw_col,
    subject_col,
    bin_col,
    response_col,
    order,
):
    tmp = df[[draw_col, subject_col, bin_col, response_col]].dropna().copy()
    tmp[bin_col] = pd.Categorical(tmp[bin_col], categories=order, ordered=True)

    subj_bin = (
        tmp.groupby([draw_col, subject_col, bin_col], observed=False)[response_col]
        .mean()
        .reset_index()
        .rename(columns={response_col: "subject_mean"})
    )

    draw_bin = (
        subj_bin.groupby([draw_col, bin_col], observed=False)["subject_mean"]
        .agg(p_choose_s="mean", draw_subject_sd="std", n_subjects="count")
        .reset_index()
    )
    draw_bin["draw_subject_sem"] = (
        draw_bin["draw_subject_sd"] / np.sqrt(draw_bin["n_subjects"].replace(0, np.nan))
    )
    draw_bin[bin_col] = pd.Categorical(draw_bin[bin_col], categories=order, ordered=True)
    return draw_bin, subj_bin


def build_bin_comparison(observed_summary, simulated_long, bin_col, order):
    obs = observed_summary.copy().set_index("bin").reindex(order)
    sim = (
        simulated_long.groupby(bin_col, observed=False)["p_choose_s"]
        .agg(
            predictive_mean="mean",
            predictive_sd="std",
            predictive_median="median",
            predictive_pi95_low=lambda s: s.quantile(0.025),
            predictive_pi95_high=lambda s: s.quantile(0.975),
            n_predictive_draws="count",
        )
        .reindex(order)
    )
    out = obs.join(sim)
    out["observed_inside_predictive_95"] = (
        (out["mean"] >= out["predictive_pi95_low"])
        & (out["mean"] <= out["predictive_pi95_high"])
    )
    out["predictive_mean_minus_observed"] = out["predictive_mean"] - out["mean"]
    return out.reset_index().rename(columns={"index": "bin"})


# ============================================================
# plots
# ============================================================

def plot_choice_bin_ppc(observed_summary, simulated_long, bin_col, order, title, xlabel, path):
    obs = observed_summary.set_index("bin").reindex(order)
    model = (
        simulated_long.groupby(bin_col, observed=False)["p_choose_s"]
        .agg(
            mean="mean",
            low=lambda s: s.quantile(0.025),
            high=lambda s: s.quantile(0.975),
        )
        .reindex(order)
    )

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10.5, 7.2))

    obs_mean = 100 * obs["mean"].to_numpy(float)
    obs_low = 100 * obs["boot_ci_low"].to_numpy(float)
    obs_high = 100 * obs["boot_ci_high"].to_numpy(float)

    ax.bar(
        x,
        obs_mean,
        width=0.72,
        facecolor=OBS_S_COLOR,
        edgecolor="black",
        linewidth=1.8,
        alpha=0.82,
        label="Empirical P(S)",
        zorder=1,
    )
    ax.errorbar(
        x,
        obs_mean,
        yerr=np.vstack([obs_mean - obs_low, obs_high - obs_mean]),
        fmt="none",
        ecolor="black",
        elinewidth=1.8,
        capsize=6,
        zorder=4,
        label="Empirical 95% bootstrap CI",
    )

    m = 100 * model["mean"].to_numpy(float)
    lo = 100 * model["low"].to_numpy(float)
    hi = 100 * model["high"].to_numpy(float)
    ax.fill_between(
        x, lo, hi,
        color=PPC_S_COLOR,
        alpha=0.16,
        label="Posterior-predictive 95% interval",
        zorder=2,
    )
    ax.plot(
        x, m,
        color=PPC_S_COLOR,
        linewidth=3,
        marker="o",
        label="Posterior-predictive P(S)",
        zorder=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0, 100)
    ax.set_ylabel("P(choose S) in %")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    style_ax(ax)
    save_close(fig, path)


def choice_es_bin_table(observed_summary, simulated_long, bin_col, order):
    """Long comparison table for E and S probabilities in each bin."""
    obs = observed_summary.set_index("bin").reindex(order)
    sim_s = (
        simulated_long.groupby(bin_col, observed=False)["p_choose_s"]
        .agg(
            ppc_mean="mean",
            ppc_pi95_low=lambda s: s.quantile(0.025),
            ppc_pi95_high=lambda s: s.quantile(0.975),
        )
        .reindex(order)
    )

    rows = []
    for b in order:
        s_obs = float(obs.loc[b, "mean"])
        s_lo = float(obs.loc[b, "boot_ci_low"])
        s_hi = float(obs.loc[b, "boot_ci_high"])
        s_ppc = float(sim_s.loc[b, "ppc_mean"])
        s_ppc_lo = float(sim_s.loc[b, "ppc_pi95_low"])
        s_ppc_hi = float(sim_s.loc[b, "ppc_pi95_high"])

        rows.append({
            "bin": b,
            "choice": "E",
            "observed_mean": 1.0 - s_obs,
            "observed_boot_ci_low": 1.0 - s_hi,
            "observed_boot_ci_high": 1.0 - s_lo,
            "ppc_mean": 1.0 - s_ppc,
            "ppc_pi95_low": 1.0 - s_ppc_hi,
            "ppc_pi95_high": 1.0 - s_ppc_lo,
        })
        rows.append({
            "bin": b,
            "choice": "S",
            "observed_mean": s_obs,
            "observed_boot_ci_low": s_lo,
            "observed_boot_ci_high": s_hi,
            "ppc_mean": s_ppc,
            "ppc_pi95_low": s_ppc_lo,
            "ppc_pi95_high": s_ppc_hi,
        })

    out = pd.DataFrame(rows)
    out["observed_inside_predictive_95"] = (
        (out["observed_mean"] >= out["ppc_pi95_low"])
        & (out["observed_mean"] <= out["ppc_pi95_high"])
    )
    return out


def plot_choice_es_bin_ppc(observed_summary, simulated_long, bin_col, order, title, xlabel, path):
    """
    Empirical E/S grouped bars plus posterior-predictive E/S means and 95%
    predictive intervals. E is always blue-family; S is always purple-family.
    """
    tab = choice_es_bin_table(
        observed_summary,
        simulated_long,
        bin_col,
        order,
    )

    x = np.arange(len(order))
    width = 0.34
    fig, ax = plt.subplots(figsize=(11.2, 7.2))

    for choice, offset, obs_color, ppc_color in [
        ("E", -width / 2, OBS_E_COLOR, PPC_E_COLOR),
        ("S", +width / 2, OBS_S_COLOR, PPC_S_COLOR),
    ]:
        z = tab.loc[tab["choice"] == choice].set_index("bin").reindex(order)
        xpos = x + offset

        obs_mean = 100 * z["observed_mean"].to_numpy(float)
        obs_low = 100 * z["observed_boot_ci_low"].to_numpy(float)
        obs_high = 100 * z["observed_boot_ci_high"].to_numpy(float)

        ax.bar(
            xpos,
            obs_mean,
            width=width * 0.92,
            color=obs_color,
            edgecolor="black",
            linewidth=1.4,
            alpha=0.82,
            label=f"Empirical {choice}",
            zorder=1,
        )
        ax.errorbar(
            xpos,
            obs_mean,
            yerr=np.vstack([obs_mean - obs_low, obs_high - obs_mean]),
            fmt="none",
            ecolor="black",
            elinewidth=1.5,
            capsize=5,
            zorder=4,
        )

        ppc_mean = 100 * z["ppc_mean"].to_numpy(float)
        ppc_low = 100 * z["ppc_pi95_low"].to_numpy(float)
        ppc_high = 100 * z["ppc_pi95_high"].to_numpy(float)

        ax.plot(
            xpos,
            ppc_mean,
            color=ppc_color,
            marker="o",
            markersize=7,
            linewidth=2.8,
            label=f"Posterior predictive {choice}",
            zorder=6,
        )
        ax.errorbar(
            xpos,
            ppc_mean,
            yerr=np.vstack([ppc_mean - ppc_low, ppc_high - ppc_mean]),
            fmt="none",
            ecolor=ppc_color,
            elinewidth=2,
            capsize=5,
            zorder=5,
        )

    ax.axhline(50, color="0.55", linestyle=":", linewidth=1.2, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Choice probability (%)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    style_ax(ax)
    save_close(fig, path)

def plot_rt_distribution(obs, ppc, path):
    obs_rt = pd.to_numeric(obs["rt"], errors="coerce").abs().dropna()
    sim_rt = pd.to_numeric(ppc["rt_sampled"], errors="coerce").abs().dropna()

    pooled = pd.concat([obs_rt, sim_rt], ignore_index=True)
    bins = np.histogram_bin_edges(pooled, bins=50)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(obs_rt, bins=bins, density=True, histtype="step", linewidth=2.5, label="Observed")
    ax.hist(sim_rt, bins=bins, density=True, histtype="step", linewidth=2.5, label="Posterior predictive")
    ax.set_xlabel("Reaction time (s)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior predictive check: RT distribution")
    ax.legend(frameon=False)
    style_ax(ax)
    save_close(fig, path)


def overall_choice_group_plot_table(
    obs,
    ppc,
    subject_col,
    n_boot,
    ci,
    seed,
):
    obs_sub = (
        obs.groupby(subject_col)["response"]
        .mean()
        .astype(float)
    )
    obs_s_mean, obs_s_low, obs_s_high = bootstrap_mean_ci(
        obs_sub.to_numpy(float),
        n_boot=n_boot,
        ci=ci,
        seed=seed,
    )

    sim_sub = (
        ppc.groupby(["ppc_draw", subject_col])["response_sampled"]
        .mean()
        .rename("p_s")
        .reset_index()
    )
    sim_draw = (
        sim_sub.groupby("ppc_draw")["p_s"]
        .mean()
    )
    pred_s = predictive_summary(sim_draw)

    rows = [
        {
            "source": "Observed",
            "choice": "E",
            "mean": 1.0 - obs_s_mean,
            "low": 1.0 - obs_s_high,
            "high": 1.0 - obs_s_low,
        },
        {
            "source": "Observed",
            "choice": "S",
            "mean": obs_s_mean,
            "low": obs_s_low,
            "high": obs_s_high,
        },
        {
            "source": "Posterior predictive",
            "choice": "E",
            "mean": 1.0 - pred_s["predictive_mean"],
            "low": 1.0 - pred_s["predictive_pi95_high"],
            "high": 1.0 - pred_s["predictive_pi95_low"],
        },
        {
            "source": "Posterior predictive",
            "choice": "S",
            "mean": pred_s["predictive_mean"],
            "low": pred_s["predictive_pi95_low"],
            "high": pred_s["predictive_pi95_high"],
        },
    ]
    return pd.DataFrame(rows)


def plot_overall_choice(group_table, path):
    source_order = ["Observed", "Posterior predictive"]
    choices = ["E", "S"]
    x = np.arange(len(source_order))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    for choice, offset, colors in [
        ("E", -width / 2, [OBS_E_COLOR, PPC_E_COLOR]),
        ("S", +width / 2, [OBS_S_COLOR, PPC_S_COLOR]),
    ]:
        z = (
            group_table.loc[group_table["choice"] == choice]
            .set_index("source")
            .reindex(source_order)
        )
        means = 100 * z["mean"].to_numpy(float)
        lows = 100 * z["low"].to_numpy(float)
        highs = 100 * z["high"].to_numpy(float)

        for i in range(len(source_order)):
            ax.bar(
                x[i] + offset,
                means[i],
                width=width * 0.92,
                color=colors[i],
                edgecolor="black",
                linewidth=1.4,
                label=choice if i == 0 else None,
                zorder=1,
            )
            ax.errorbar(
                [x[i] + offset],
                [means[i]],
                yerr=[[means[i] - lows[i]], [highs[i] - means[i]]],
                fmt="none",
                ecolor="black",
                elinewidth=1.5,
                capsize=5,
                zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(source_order)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Choice probability (%)")
    ax.set_title("Overall choice PPC")
    ax.legend(title="Choice", frameon=False)
    style_ax(ax)
    save_close(fig, path)


def rt_by_response_group_plot_table(
    obs,
    ppc,
    subject_col,
    n_boot,
    ci,
    seed,
):
    obs2 = obs.copy()
    obs2["rt_abs"] = pd.to_numeric(obs2["rt"], errors="coerce").abs()
    ppc2 = ppc.copy()
    ppc2["rt_abs_sampled"] = pd.to_numeric(ppc2["rt_sampled"], errors="coerce").abs()

    rows = []
    for response, choice in [(0, "E"), (1, "S")]:
        obs_sub = (
            obs2.loc[pd.to_numeric(obs2["response"], errors="coerce") == response]
            .groupby(subject_col)["rt_abs"]
            .median()
            .dropna()
        )
        obs_mean, obs_low, obs_high = bootstrap_mean_ci(
            obs_sub.to_numpy(float),
            n_boot=n_boot,
            ci=ci,
            seed=seed + response,
        )
        rows.append({
            "source": "Observed",
            "choice": choice,
            "mean": obs_mean,
            "low": obs_low,
            "high": obs_high,
            "n_subjects": int(len(obs_sub)),
        })

        sim_sub = (
            ppc2.loc[pd.to_numeric(ppc2["response_sampled"], errors="coerce") == response]
            .groupby(["ppc_draw", subject_col])["rt_abs_sampled"]
            .median()
            .rename("median_rt")
            .reset_index()
        )
        sim_draw = sim_sub.groupby("ppc_draw")["median_rt"].mean()
        pred = predictive_summary(sim_draw)
        rows.append({
            "source": "Posterior predictive",
            "choice": choice,
            "mean": pred["predictive_mean"],
            "low": pred["predictive_pi95_low"],
            "high": pred["predictive_pi95_high"],
            "n_subjects": int(obs[subject_col].nunique()),
        })

    return pd.DataFrame(rows)


def plot_rt_by_response(group_table, path):
    choices = ["E", "S"]
    x = np.arange(len(choices))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    specs = [
        ("Observed", -width / 2, [OBS_E_COLOR, OBS_S_COLOR]),
        ("Posterior predictive", +width / 2, [PPC_E_COLOR, PPC_S_COLOR]),
    ]
    for source, offset, colors in specs:
        z = (
            group_table.loc[group_table["source"] == source]
            .set_index("choice")
            .reindex(choices)
        )
        means = z["mean"].to_numpy(float)
        lows = z["low"].to_numpy(float)
        highs = z["high"].to_numpy(float)

        ax.bar(
            x + offset,
            means,
            width=width * 0.92,
            color=colors,
            edgecolor="black",
            linewidth=1.4,
            label=source,
            zorder=1,
        )
        ax.errorbar(
            x + offset,
            means,
            yerr=np.vstack([means - lows, highs - means]),
            fmt="none",
            ecolor="black",
            elinewidth=1.5,
            capsize=5,
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(choices)
    ax.set_ylabel("Participant median RT (s)")
    ax.set_title("RT conditional on choice")
    ax.legend(frameon=False)
    style_ax(ax)
    save_close(fig, path)



def plot_rt_quantiles_by_response(summary_df, path):
    fig, ax = plt.subplots(figsize=(10, 7))
    qx = np.arange(len(RT_QUANTILES))
    for response, label in [(0, "E"), (1, "S")]:
        z = summary_df.loc[summary_df["response"] == response].set_index("quantile").reindex(RT_QUANTILE_LABELS)
        ax.plot(qx, z["observed"].to_numpy(float), marker="o", linewidth=2.5, label=f"Observed {label}")
        ax.plot(qx, z["predictive_mean"].to_numpy(float), marker="o", linestyle="--", linewidth=2.5, label=f"Predicted {label}")
        ax.fill_between(
            qx,
            z["predictive_pi95_low"].to_numpy(float),
            z["predictive_pi95_high"].to_numpy(float),
            alpha=0.14,
        )
    ax.set_xticks(qx)
    ax.set_xticklabels(["10", "30", "50", "70", "90"])
    ax.set_xlabel("RT percentile")
    ax.set_ylabel("RT (s)")
    ax.set_title("RT quantiles by response")
    ax.legend(frameon=False)
    style_ax(ax)
    save_close(fig, path)


def plot_participant_scatter(table, observed_col, predicted_col, low_col, high_col, title, xlabel, ylabel, path):
    z = table.dropna(subset=[observed_col, predicted_col]).copy()
    if z.empty:
        return

    x = z[observed_col].to_numpy(float)
    y = z[predicted_col].to_numpy(float)
    low = z[low_col].to_numpy(float)
    high = z[high_col].to_numpy(float)
    lo = np.nanmin(np.concatenate([x, low]))
    hi = np.nanmax(np.concatenate([x, high]))
    pad = 0.05 * (hi - lo) if hi > lo else 0.1

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.errorbar(
        x,
        y,
        yerr=np.vstack([y - low, high - y]),
        fmt="o",
        alpha=0.75,
        capsize=2,
    )
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1.5)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_ax(ax)
    save_close(fig, path)


# ============================================================
# core PPC calculations
# ============================================================

def overall_choice_table(obs, ppc):
    observed = float(pd.to_numeric(obs["response"], errors="coerce").mean())
    by_draw = (
        ppc.groupby("ppc_draw")["response_sampled"]
        .mean()
        .rename("p_choose_s")
        .reset_index()
    )
    pred = predictive_summary(by_draw["p_choose_s"])
    summary = pd.DataFrame([{
        "statistic": "p_choose_s",
        "observed": observed,
        **pred,
        "observed_inside_predictive_95": (
            pred["predictive_pi95_low"] <= observed <= pred["predictive_pi95_high"]
        ),
    }])
    return summary, by_draw


def overall_rt_table(obs, ppc):
    obs_rt = pd.to_numeric(obs["rt"], errors="coerce").abs()
    sim = ppc[["ppc_draw", "rt_sampled"]].copy()
    sim["rt_abs"] = pd.to_numeric(sim["rt_sampled"], errors="coerce").abs()

    rows = []
    funcs = {
        "mean_rt": lambda x: x.mean(),
        "median_rt": lambda x: x.median(),
        "sd_rt": lambda x: x.std(ddof=1),
        "q10_rt": lambda x: x.quantile(0.10),
        "q30_rt": lambda x: x.quantile(0.30),
        "q50_rt": lambda x: x.quantile(0.50),
        "q70_rt": lambda x: x.quantile(0.70),
        "q90_rt": lambda x: x.quantile(0.90),
    }

    for name, fn in funcs.items():
        observed = float(fn(obs_rt.dropna()))
        draw_stats = sim.groupby("ppc_draw")["rt_abs"].apply(fn)
        pred = predictive_summary(draw_stats)
        rows.append({
            "statistic": name,
            "observed": observed,
            **pred,
            "observed_inside_predictive_95": (
                pred["predictive_pi95_low"] <= observed <= pred["predictive_pi95_high"]
            ),
        })
    return pd.DataFrame(rows)


def rt_by_response_table(obs, ppc):
    obs2 = obs.copy()
    obs2["rt_abs"] = pd.to_numeric(obs2["rt"], errors="coerce").abs()
    ppc2 = ppc.copy()
    ppc2["rt_abs_sampled"] = pd.to_numeric(ppc2["rt_sampled"], errors="coerce").abs()

    rows = []
    draw_rows = []
    for response in [0, 1]:
        observed = float(
            obs2.loc[pd.to_numeric(obs2["response"], errors="coerce") == response, "rt_abs"].median()
        )
        draw_stats = (
            ppc2.loc[pd.to_numeric(ppc2["response_sampled"], errors="coerce") == response]
            .groupby("ppc_draw")["rt_abs_sampled"]
            .median()
        )
        pred = predictive_summary(draw_stats)
        rows.append({
            "response": response,
            "label": "E" if response == 0 else "S",
            "observed_median_rt": observed,
            **pred,
            "observed_inside_predictive_95": (
                pred["predictive_pi95_low"] <= observed <= pred["predictive_pi95_high"]
            ),
        })
        for draw, val in draw_stats.items():
            draw_rows.append({
                "ppc_draw": draw,
                "response": response,
                "median_rt": float(val),
            })
    return pd.DataFrame(rows), pd.DataFrame(draw_rows)


def rt_quantiles_by_response_table(obs, ppc):
    obs2 = obs.copy()
    obs2["rt_abs"] = pd.to_numeric(obs2["rt"], errors="coerce").abs()
    ppc2 = ppc.copy()
    ppc2["rt_abs_sampled"] = pd.to_numeric(ppc2["rt_sampled"], errors="coerce").abs()

    summary_rows = []
    draw_rows = []
    for response in [0, 1]:
        obs_r = obs2.loc[pd.to_numeric(obs2["response"], errors="coerce") == response, "rt_abs"].dropna()
        sim_r = ppc2.loc[pd.to_numeric(ppc2["response_sampled"], errors="coerce") == response]
        for q, qlab in zip(RT_QUANTILES, RT_QUANTILE_LABELS):
            observed = float(obs_r.quantile(q))
            draw_stats = sim_r.groupby("ppc_draw")["rt_abs_sampled"].quantile(q)
            pred = predictive_summary(draw_stats)
            summary_rows.append({
                "response": response,
                "label": "E" if response == 0 else "S",
                "quantile": qlab,
                "quantile_probability": q,
                "observed": observed,
                **pred,
                "observed_inside_predictive_95": (
                    pred["predictive_pi95_low"] <= observed <= pred["predictive_pi95_high"]
                ),
            })
            for draw, val in draw_stats.items():
                draw_rows.append({
                    "ppc_draw": draw,
                    "response": response,
                    "quantile": qlab,
                    "quantile_probability": q,
                    "rt_quantile": float(val),
                })
    return pd.DataFrame(summary_rows), pd.DataFrame(draw_rows)


def participant_tables(obs, ppc, subject_col):
    obs2 = obs.copy()
    obs2["rt_abs"] = pd.to_numeric(obs2["rt"], errors="coerce").abs()
    ppc2 = ppc.copy()
    ppc2["rt_abs_sampled"] = pd.to_numeric(ppc2["rt_sampled"], errors="coerce").abs()

    obs_choice = (
        obs2.groupby(subject_col)["response"].mean().rename("observed_p_choose_s")
    )
    sim_choice = (
        ppc2.groupby(["ppc_draw", subject_col])["response_sampled"].mean().rename("p_choose_s").reset_index()
    )

    choice_rows = []
    for subj, observed in obs_choice.items():
        vals = sim_choice.loc[sim_choice[subject_col] == subj, "p_choose_s"]
        pred = predictive_summary(vals)
        choice_rows.append({
            subject_col: subj,
            "observed_p_choose_s": float(observed),
            **pred,
            "observed_inside_predictive_95": (
                pred["predictive_pi95_low"] <= observed <= pred["predictive_pi95_high"]
            ),
        })

    obs_rt = obs2.groupby(subject_col)["rt_abs"].median().rename("observed_median_rt")
    sim_rt = (
        ppc2.groupby(["ppc_draw", subject_col])["rt_abs_sampled"].median().rename("median_rt").reset_index()
    )

    rt_rows = []
    for subj, observed in obs_rt.items():
        vals = sim_rt.loc[sim_rt[subject_col] == subj, "median_rt"]
        pred = predictive_summary(vals)
        rt_rows.append({
            subject_col: subj,
            "observed_median_rt": float(observed),
            **pred,
            "observed_inside_predictive_95": (
                pred["predictive_pi95_low"] <= observed <= pred["predictive_pi95_high"]
            ),
        })

    return (
        pd.DataFrame(choice_rows),
        pd.DataFrame(rt_rows),
        sim_choice,
        sim_rt,
    )


# ============================================================
# main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, choices=["study1", "study2"])
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--chains", type=int, default=3)
    ap.add_argument("--ppc-total", type=int, default=1000)
    ap.add_argument("--bootstrap-samples", type=int, default=5000)
    ap.add_argument("--bootstrap-ci", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--subject-col", default="subj_idx")
    ap.add_argument("--analysisready", type=Path, required=True)
    ap.add_argument("--skip-builtin-plots", action="store_true")
    ap.add_argument(
        "--save-replot-data",
        action="store_true",
        help=(
            "Save compact observed and posterior-predictive trial-level data "
            "after all PPC bin assignments so new figures can be made later "
            "without rerunning HDDM."
        ),
    )
    args = ap.parse_args()

    np.random.seed(args.seed)

    out = args.output_dir
    figures = out / "figures"
    tables = out / "tables"
    hddm_standard = out / "hddm_standard"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    hddm_standard.mkdir(parents=True, exist_ok=True)

    models, model_files = load_models(args.model_dir, args.chains)
    obs = verify_observed_data_match(models)

    required = {args.subject_col, "rt", "response"}
    missing = sorted(required - set(obs.columns))
    if missing:
        raise ValueError(f"Observed model data are missing required columns: {missing}")

    obs = attach_identity_gaze_from_analysisready(obs, args.analysisready)
    dwell_col = "DwellTimeAdvantage_ES"
    dwell_prop_col = "DwellPropAdvantage_ES"
    counts = split_counts(args.ppc_total, args.chains)
    print(f"PPC allocation across chains: {counts}", flush=True)
    print("Main dwell PPC variable: DwellTimeAdvantage_ES = Dwell_S - Dwell_E", flush=True)
    print("Additional proportional gaze PPC: DwellPropAdvantage_ES = PropDwell_S - PropDwell_E", flush=True)

    # Save chain DICs as descriptive chain metadata only.
    dic_rows = []
    for c, model in enumerate(models):
        try:
            dic = float(model.dic)
        except Exception:
            dic = np.nan
        dic_rows.append({
            "chain": c,
            "model_file": str(model_files[c]),
            "dic": dic,
            "ppc_datasets_requested": counts[c],
        })
    pd.DataFrame(dic_rows).to_csv(tables / "chain_metadata.csv", index=False)

    raw_ppcs = []
    flat_ppcs = []

    for c, (model, n_samples) in enumerate(zip(models, counts)):
        raw, flat, _ = generate_chain_ppc(model, c, n_samples)
        cols_to_restore = [
            args.subject_col,
            "sub_id",
            "phase",
            "trial",
            "response",
            "rt",
            dwell_col,
            dwell_prop_col,
        ]
        # De-duplicate in case --subject-col is itself one of the listed names.
        cols_to_restore = list(dict.fromkeys(cols_to_restore))
        flat = attach_observed_columns_by_row_order(
            obs,
            flat,
            "ppc_draw",
            cols_to_restore,
        )
        raw_ppcs.append(raw)
        flat_ppcs.append(flat)

        save_parameter_summary(
            model,
            hddm_standard / f"parameter_summary_chain_{c}.csv",
        )
        save_hddm_post_pred_stats(
            model,
            raw,
            hddm_standard / f"posterior_predictive_summary_chain_{c}.csv",
        )

        if not args.skip_builtin_plots:
            save_builtin_hddm_plots(model, hddm_standard, c)

    ppc = pd.concat(flat_ppcs, ignore_index=True)

    participant_mapping_audit = validate_ppc_participant_mapping(
        obs,
        ppc,
        "ppc_draw",
        args.subject_col,
    )
    print(
        "PPC participant mapping audit PASSED: "
        f"{participant_mapping_audit['expected_participants']} participants, "
        f"{participant_mapping_audit['expected_rows_per_draw']} rows/draw, "
        f"{participant_mapping_audit['draws_checked']} draws checked.",
        flush=True,
    )

    # Basic checks on draw counts.
    actual_draws = int(ppc["ppc_draw"].nunique())
    if actual_draws != args.ppc_total:
        warnings.warn(
            f"Requested {args.ppc_total} posterior-predictive datasets but detected "
            f"{actual_draws} unique draw identifiers."
        )

    # --------------------------------------------------------
    # overall choice
    # --------------------------------------------------------
    choice_summary, choice_by_draw = overall_choice_table(obs, ppc)
    choice_summary.to_csv(tables / "overall_choice_ppc.csv", index=False)
    choice_by_draw.to_csv(tables / "overall_choice_by_draw.csv", index=False)

    overall_choice_group = overall_choice_group_plot_table(
        obs,
        ppc,
        args.subject_col,
        args.bootstrap_samples,
        args.bootstrap_ci,
        args.seed + 10,
    )
    overall_choice_group.to_csv(
        tables / "overall_choice_es_group_plot.csv",
        index=False,
    )
    plot_overall_choice(
        overall_choice_group,
        figures / "overall_choice_ppc.png",
    )

    # --------------------------------------------------------
    # overall RT
    # --------------------------------------------------------
    rt_summary = overall_rt_table(obs, ppc)
    rt_summary.to_csv(tables / "overall_rt_ppc.csv", index=False)
    plot_rt_distribution(obs, ppc, figures / "rt_distribution_ppc.png")

    # --------------------------------------------------------
    # RT by response
    # --------------------------------------------------------
    rt_resp, rt_resp_draw = rt_by_response_table(obs, ppc)
    rt_resp.to_csv(tables / "rt_by_response_ppc.csv", index=False)
    rt_resp_draw.to_csv(tables / "rt_by_response_by_draw.csv", index=False)

    rt_resp_group = rt_by_response_group_plot_table(
        obs,
        ppc,
        args.subject_col,
        args.bootstrap_samples,
        args.bootstrap_ci,
        args.seed + 20,
    )
    rt_resp_group.to_csv(
        tables / "rt_by_response_group_plot.csv",
        index=False,
    )
    plot_rt_by_response(
        rt_resp_group,
        figures / "rt_by_response_ppc.png",
    )

    # --------------------------------------------------------
    # standard RT quantiles by response
    # --------------------------------------------------------
    rtq, rtq_draw = rt_quantiles_by_response_table(obs, ppc)
    rtq.to_csv(tables / "rt_quantiles_by_response_ppc.csv", index=False)
    rtq_draw.to_csv(tables / "rt_quantiles_by_response_by_draw.csv", index=False)
    plot_rt_quantiles_by_response(rtq, figures / "rt_quantiles_by_response_ppc.png")

    # --------------------------------------------------------
    # dwell-time quintiles: observed bins carried into PPC draws
    # --------------------------------------------------------
    dwell_edges = compute_quantile_edges(obs[dwell_col], n_quantiles=5)
    obs = obs.copy()
    obs["dwell_quintile"] = assign_bins_from_edges(
        obs[dwell_col],
        dwell_edges,
        DWELL_LABELS,
    )

    # Prefer the actual dwell column retained by append_data. If it was attached by
    # fallback, it is still the observed trial-level value, which is what this
    # conditional PPC requires.
    ppc["dwell_quintile"] = assign_bins_from_edges(
        ppc[dwell_col],
        dwell_edges,
        DWELL_LABELS,
    )

    obs_dwell, obs_dwell_subj = subjectwise_observed_bin_summary(
        obs,
        args.subject_col,
        "dwell_quintile",
        "response",
        DWELL_LABELS,
        args.bootstrap_samples,
        args.bootstrap_ci,
        args.seed,
    )
    sim_dwell, sim_dwell_subj = simulated_bin_summary(
        ppc,
        "ppc_draw",
        args.subject_col,
        "dwell_quintile",
        "response_sampled",
        DWELL_LABELS,
    )
    dwell_cmp = build_bin_comparison(
        obs_dwell,
        sim_dwell,
        "dwell_quintile",
        DWELL_LABELS,
    )

    # filenames intentionally parallel the user's old PPC outputs
    obs_dwell.to_csv(tables / "observed_dwell_summary_combined.csv", index=False)
    obs_dwell_subj.to_csv(tables / "observed_dwell_subjectwise_combined.csv", index=False)
    sim_dwell.to_csv(tables / "simulated_dwell_by_draw_combined.csv", index=False)
    sim_dwell_subj.to_csv(tables / "simulated_dwell_subjectwise_combined.csv", index=False)
    dwell_cmp.to_csv(tables / "dwell_model_vs_observed_comparison_combined.csv", index=False)
    pd.DataFrame({
        "edge_index": np.arange(len(dwell_edges)),
        "edge_value": dwell_edges,
    }).to_csv(tables / "dwell_quintile_edges.csv", index=False)

    plot_choice_bin_ppc(
        obs_dwell,
        sim_dwell,
        "dwell_quintile",
        DWELL_LABELS,
        "P(choose S) by dwell-time quintile",
        "Dwell-time advantage quintile (E/S identity; S - E)",
        figures / "p_choose_s_by_dwell_quintile_combined.png",
    )

    # --------------------------------------------------------
    # Additional identity-coordinate proportional dwell PPC
    # Does not replace the old-style millisecond dwell-time PPC.
    # --------------------------------------------------------
    dwell_prop_edges = compute_quantile_edges(obs[dwell_prop_col], n_quantiles=5)
    obs["dwell_prop_quintile"] = assign_bins_from_edges(
        obs[dwell_prop_col],
        dwell_prop_edges,
        DWELL_LABELS,
    )
    ppc["dwell_prop_quintile"] = assign_bins_from_edges(
        ppc[dwell_prop_col],
        dwell_prop_edges,
        DWELL_LABELS,
    )

    obs_dwell_prop, obs_dwell_prop_subj = subjectwise_observed_bin_summary(
        obs,
        args.subject_col,
        "dwell_prop_quintile",
        "response",
        DWELL_LABELS,
        args.bootstrap_samples,
        args.bootstrap_ci,
        args.seed + 50,
    )
    sim_dwell_prop, sim_dwell_prop_subj = simulated_bin_summary(
        ppc,
        "ppc_draw",
        args.subject_col,
        "dwell_prop_quintile",
        "response_sampled",
        DWELL_LABELS,
    )
    dwell_prop_cmp = build_bin_comparison(
        obs_dwell_prop,
        sim_dwell_prop,
        "dwell_prop_quintile",
        DWELL_LABELS,
    )

    obs_dwell_prop.to_csv(tables / "observed_dwell_prop_summary_identity.csv", index=False)
    obs_dwell_prop_subj.to_csv(tables / "observed_dwell_prop_subjectwise_identity.csv", index=False)
    sim_dwell_prop.to_csv(tables / "simulated_dwell_prop_by_draw_identity.csv", index=False)
    sim_dwell_prop_subj.to_csv(tables / "simulated_dwell_prop_subjectwise_identity.csv", index=False)
    dwell_prop_cmp.to_csv(tables / "dwell_prop_model_vs_observed_comparison_identity.csv", index=False)
    pd.DataFrame({
        "edge_index": np.arange(len(dwell_prop_edges)),
        "edge_value": dwell_prop_edges,
    }).to_csv(tables / "dwell_prop_quintile_edges_identity.csv", index=False)

    plot_choice_bin_ppc(
        obs_dwell_prop,
        sim_dwell_prop,
        "dwell_prop_quintile",
        DWELL_LABELS,
        "P(choose S) by proportional dwell quintile",
        "Proportional dwell advantage quintile (S - E)",
        figures / "p_choose_s_by_dwell_prop_quintile_identity.png",
    )

    # --------------------------------------------------------
    # RT quintiles: user's original rank-within-each-PPC-draw approach
    # --------------------------------------------------------
    obs["rt_abs"] = pd.to_numeric(obs["rt"], errors="coerce").abs()
    obs["rt_quintile"] = assign_rank_quintiles(obs["rt_abs"])

    ppc["rt_abs_sampled"] = pd.to_numeric(ppc["rt_sampled"], errors="coerce").abs()
    rt_quintile_out = pd.Series(pd.NA, index=ppc.index, dtype="object")
    for draw, idx in ppc.groupby("ppc_draw", sort=False).groups.items():
        idx = list(idx)
        rt_quintile_out.loc[idx] = pd.Series(
            assign_rank_quintiles(ppc.loc[idx, "rt_abs_sampled"]),
            index=idx,
        ).astype("object")
    ppc["rt_quintile"] = rt_quintile_out
    ppc["rt_quintile"] = pd.Categorical(
        ppc["rt_quintile"],
        categories=RT_QUINTILE_LABELS,
        ordered=True,
    )

    obs_rtq, obs_rtq_subj = subjectwise_observed_bin_summary(
        obs,
        args.subject_col,
        "rt_quintile",
        "response",
        RT_QUINTILE_LABELS,
        args.bootstrap_samples,
        args.bootstrap_ci,
        args.seed + 100,
    )
    sim_rtq, sim_rtq_subj = simulated_bin_summary(
        ppc,
        "ppc_draw",
        args.subject_col,
        "rt_quintile",
        "response_sampled",
        RT_QUINTILE_LABELS,
    )
    rtq_cmp = build_bin_comparison(
        obs_rtq,
        sim_rtq,
        "rt_quintile",
        RT_QUINTILE_LABELS,
    )

    # filenames intentionally parallel the user's old PPC outputs
    obs_rtq.to_csv(tables / "observed_rt_summary_combined.csv", index=False)
    obs_rtq_subj.to_csv(tables / "observed_rt_subjectwise_combined.csv", index=False)
    sim_rtq.to_csv(tables / "simulated_rt_by_draw_combined.csv", index=False)
    sim_rtq_subj.to_csv(tables / "simulated_rt_subjectwise_combined.csv", index=False)
    rtq_cmp.to_csv(tables / "rt_model_vs_observed_comparison_combined.csv", index=False)

    plot_choice_bin_ppc(
        obs_rtq,
        sim_rtq,
        "rt_quintile",
        RT_QUINTILE_LABELS,
        "P(choose S) by RT quintile",
        "RT quintile (1 = fastest, 5 = slowest)",
        figures / "p_choose_s_by_rt_quintile_combined.png",
    )

    rt_choice_es = choice_es_bin_table(
        obs_rtq,
        sim_rtq,
        "rt_quintile",
        RT_QUINTILE_LABELS,
    )
    rt_choice_es.to_csv(
        tables / "choice_e_s_by_rt_quintile_ppc.csv",
        index=False,
    )
    plot_choice_es_bin_ppc(
        obs_rtq,
        sim_rtq,
        "rt_quintile",
        RT_QUINTILE_LABELS,
        "Probability of choosing E vs S by RT quintile",
        "RT quintile (1 = fastest, 5 = slowest)",
        figures / "p_choose_e_s_by_rt_quintile_ppc.png",
    )

    # --------------------------------------------------------
    # Additional RT-quintile check using empirical cutpoints
    # This does NOT replace the user's rank-based quintile PPC above.
    # --------------------------------------------------------
    rt_edges = compute_quantile_edges(obs["rt_abs"], n_quantiles=5)
    obs["rt_quintile_empirical_edges"] = assign_bins_from_edges(
        obs["rt_abs"],
        rt_edges,
        RT_QUINTILE_LABELS,
    )
    ppc["rt_quintile_empirical_edges"] = assign_bins_from_edges(
        ppc["rt_abs_sampled"],
        rt_edges,
        RT_QUINTILE_LABELS,
    )

    obs_rte, _ = subjectwise_observed_bin_summary(
        obs,
        args.subject_col,
        "rt_quintile_empirical_edges",
        "response",
        RT_QUINTILE_LABELS,
        args.bootstrap_samples,
        args.bootstrap_ci,
        args.seed + 200,
    )
    sim_rte, _ = simulated_bin_summary(
        ppc,
        "ppc_draw",
        args.subject_col,
        "rt_quintile_empirical_edges",
        "response_sampled",
        RT_QUINTILE_LABELS,
    )
    rte_cmp = build_bin_comparison(
        obs_rte,
        sim_rte,
        "rt_quintile_empirical_edges",
        RT_QUINTILE_LABELS,
    )
    rte_cmp.to_csv(tables / "rt_empirical_cutpoint_model_vs_observed_comparison.csv", index=False)
    pd.DataFrame({
        "edge_index": np.arange(len(rt_edges)),
        "edge_value_seconds": rt_edges,
    }).to_csv(tables / "rt_empirical_quintile_edges.csv", index=False)
    plot_choice_bin_ppc(
        obs_rte,
        sim_rte,
        "rt_quintile_empirical_edges",
        RT_QUINTILE_LABELS,
        "P(choose S) by empirical RT cutpoints",
        "Empirical RT quintile cutpoint bin",
        figures / "p_choose_s_by_rt_empirical_cutpoints.png",
    )

    rte_choice_es = choice_es_bin_table(
        obs_rte,
        sim_rte,
        "rt_quintile_empirical_edges",
        RT_QUINTILE_LABELS,
    )
    rte_choice_es.to_csv(
        tables / "choice_e_s_by_rt_empirical_cutpoints_ppc.csv",
        index=False,
    )
    plot_choice_es_bin_ppc(
        obs_rte,
        sim_rte,
        "rt_quintile_empirical_edges",
        RT_QUINTILE_LABELS,
        "Probability of choosing E vs S by empirical RT cutpoints",
        "Empirical RT quintile cutpoint bin",
        figures / "p_choose_e_s_by_rt_empirical_cutpoints_ppc.png",
    )

    # --------------------------------------------------------
    # participant-level PPCs
    # --------------------------------------------------------
    p_choice, p_rt, p_choice_draw, p_rt_draw = participant_tables(
        obs,
        ppc,
        args.subject_col,
    )
    p_choice.to_csv(tables / "participant_choice_ppc.csv", index=False)
    p_rt.to_csv(tables / "participant_rt_ppc.csv", index=False)
    p_choice_draw.to_csv(tables / "participant_choice_by_draw.csv", index=False)
    p_rt_draw.to_csv(tables / "participant_rt_by_draw.csv", index=False)

    plot_participant_scatter(
        p_choice,
        "observed_p_choose_s",
        "predictive_mean",
        "predictive_pi95_low",
        "predictive_pi95_high",
        "Participant-level choice PPC",
        "Observed P(choose S)",
        "Posterior-predictive P(choose S)",
        figures / "participant_choice_ppc.png",
    )
    plot_participant_scatter(
        p_rt,
        "observed_median_rt",
        "predictive_mean",
        "predictive_pi95_low",
        "predictive_pi95_high",
        "Participant-level median RT PPC",
        "Observed median RT (s)",
        "Posterior-predictive median RT (s)",
        figures / "participant_rt_ppc.png",
    )

    # --------------------------------------------------------
    # Optional compact trial-level data for later re-plotting.
    # Full launchers enable this; smoke launchers do not.
    # --------------------------------------------------------
    if args.save_replot_data:
        observed_replot_cols = [
            args.subject_col,
            "sub_id",
            "phase",
            "trial",
            "response",
            "rt",
            dwell_col,
            dwell_prop_col,
            "dwell_quintile",
            "dwell_prop_quintile",
            "rt_abs",
            "rt_quintile",
            "rt_quintile_empirical_edges",
        ]
        observed_replot_cols = [
            c for c in dict.fromkeys(observed_replot_cols)
            if c in obs.columns
        ]
        obs[observed_replot_cols].to_csv(
            tables / "observed_replot_trials.csv.gz",
            index=False,
            compression="gzip",
        )

        ppc_replot_cols = [
            "ppc_draw",
            "chain",
            "chain_draw",
            args.subject_col,
            "sub_id",
            "phase",
            "trial",
            "response",
            "rt",
            "response_sampled",
            "rt_sampled",
            dwell_col,
            dwell_prop_col,
            "dwell_quintile",
            "dwell_prop_quintile",
            "rt_abs_sampled",
            "rt_quintile",
            "rt_quintile_empirical_edges",
        ]
        ppc_replot_cols = [
            c for c in dict.fromkeys(ppc_replot_cols)
            if c in ppc.columns
        ]
        ppc[ppc_replot_cols].to_csv(
            tables / "ppc_replot_trials.csv.gz",
            index=False,
            compression="gzip",
        )

    # --------------------------------------------------------
    # metadata / manifest
    # --------------------------------------------------------
    manifest = {
        "study": args.study,
        "model_dir": str(args.model_dir),
        "model_files": [str(x) for x in model_files],
        "chains": args.chains,
        "requested_total_ppc_datasets": args.ppc_total,
        "detected_total_ppc_draws": actual_draws,
        "allocation_across_chains": counts,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_ci": args.bootstrap_ci,
        "seed": args.seed,
        "response_coordinate": "response=1 S upper; response=0 E lower",
        "analysisready_file": str(args.analysisready),
        "main_dwell_ppc_column": dwell_col,
        "main_dwell_ppc_definition": "DwellTimeAdvantage_ES = Dwell_S - Dwell_E; positive means more ms on S",
        "additional_dwell_prop_column": dwell_prop_col,
        "additional_dwell_prop_definition": "DwellPropAdvantage_ES = PropDwell_S - PropDwell_E; positive means greater proportional dwell on S",
        "dwell_bins": DWELL_LABELS,
        "rt_quintile_method_primary": "rank-based quintiles recomputed independently within each posterior-predictive dataset",
        "rt_quintile_method_additional": "empirical observed RT quintile cutpoints applied to posterior-predictive RTs",
        "standard_rt_quantiles": RT_QUANTILES,
        "participant_mapping_audit": participant_mapping_audit,
        "replot_trial_data_saved": bool(args.save_replot_data),
        "plot_colour_rule": (
            "E empirical=deepskyblue, E PPC=steelblue; "
            "S empirical=darkorchid, S PPC=indigo"
        ),
        "important_note": (
            "All substantive combined PPC summaries use all converged chains. "
            "Chain DIC values are saved only as descriptive metadata and are not "
            "used to select a 'best chain'."
        ),
    }
    (out / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    output_index = [
        "figures/overall_choice_ppc.png",
        "figures/rt_distribution_ppc.png",
        "figures/rt_by_response_ppc.png",
        "figures/rt_quantiles_by_response_ppc.png",
        "figures/p_choose_s_by_dwell_quintile_combined.png",
        "figures/p_choose_s_by_dwell_prop_quintile_identity.png",
        "figures/p_choose_s_by_rt_quintile_combined.png",
        "figures/p_choose_e_s_by_rt_quintile_ppc.png",
        "figures/p_choose_s_by_rt_empirical_cutpoints.png",
        "figures/p_choose_e_s_by_rt_empirical_cutpoints_ppc.png",
        "figures/participant_choice_ppc.png",
        "figures/participant_rt_ppc.png",
        "tables/overall_choice_ppc.csv",
        "tables/overall_choice_es_group_plot.csv",
        "tables/rt_by_response_group_plot.csv",
        "tables/choice_e_s_by_rt_quintile_ppc.csv",
        "tables/choice_e_s_by_rt_empirical_cutpoints_ppc.csv",
        "tables/observed_replot_trials.csv.gz (full run only)",
        "tables/ppc_replot_trials.csv.gz (full run only)",
        "tables/overall_rt_ppc.csv",
        "tables/rt_by_response_ppc.csv",
        "tables/rt_quantiles_by_response_ppc.csv",
        "tables/observed_dwell_summary_combined.csv",
        "tables/simulated_dwell_by_draw_combined.csv",
        "tables/dwell_model_vs_observed_comparison_combined.csv",
        "tables/dwell_prop_model_vs_observed_comparison_identity.csv",
        "tables/observed_rt_summary_combined.csv",
        "tables/simulated_rt_by_draw_combined.csv",
        "tables/rt_model_vs_observed_comparison_combined.csv",
        "tables/participant_choice_ppc.csv",
        "tables/participant_rt_ppc.csv",
        "hddm_standard/parameter_summary_chain_*.csv",
        "hddm_standard/posterior_predictive_summary_chain_*.csv",
        "hddm_standard/builtin_posterior_predictive_chain_*.png",
        "hddm_standard/builtin_posterior_quantiles_chain_*.png",
    ]
    (out / "output_manifest.txt").write_text("\n".join(output_index) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2), flush=True)
    print(f"\nPPC outputs saved to: {out}", flush=True)

    del ppc, raw_ppcs, flat_ppcs, models
    gc.collect()


if __name__ == "__main__":
    main()
