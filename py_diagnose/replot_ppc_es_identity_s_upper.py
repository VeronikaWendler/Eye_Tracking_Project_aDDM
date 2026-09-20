from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OBS_E_COLOR = "deepskyblue"
PPC_E_COLOR = "steelblue"
OBS_S_COLOR = "darkorchid"
PPC_S_COLOR = "indigo"

RT_ORDER = ["1", "2", "3", "4", "5"]


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_close(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def overall_choice_plot(tab, path):
    source_order = ["Observed", "Posterior predictive"]
    x = np.arange(2)
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for choice, offset, colors in [
        ("E", -width / 2, [OBS_E_COLOR, PPC_E_COLOR]),
        ("S", +width / 2, [OBS_S_COLOR, PPC_S_COLOR]),
    ]:
        z = tab.loc[tab["choice"] == choice].set_index("source").reindex(source_order)
        means = 100 * z["mean"].to_numpy(float)
        lows = 100 * z["low"].to_numpy(float)
        highs = 100 * z["high"].to_numpy(float)
        for i in range(2):
            ax.bar(
                x[i] + offset, means[i], width=width * 0.92,
                color=colors[i], edgecolor="black", linewidth=1.4,
                label=choice if i == 0 else None,
            )
            ax.errorbar(
                [x[i] + offset], [means[i]],
                yerr=[[means[i] - lows[i]], [highs[i] - means[i]]],
                fmt="none", ecolor="black", capsize=5, linewidth=1.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(source_order)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Choice probability (%)")
    ax.set_title("Overall choice PPC")
    ax.legend(title="Choice", frameon=False)
    style_ax(ax)
    save_close(fig, path)


def rt_choice_plot(tab, path):
    choices = ["E", "S"]
    x = np.arange(2)
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    for source, offset, colors in [
        ("Observed", -width / 2, [OBS_E_COLOR, OBS_S_COLOR]),
        ("Posterior predictive", +width / 2, [PPC_E_COLOR, PPC_S_COLOR]),
    ]:
        z = tab.loc[tab["source"] == source].set_index("choice").reindex(choices)
        means = z["mean"].to_numpy(float)
        lows = z["low"].to_numpy(float)
        highs = z["high"].to_numpy(float)
        ax.bar(
            x + offset, means, width=width * 0.92,
            color=colors, edgecolor="black", linewidth=1.4, label=source,
        )
        ax.errorbar(
            x + offset, means,
            yerr=np.vstack([means - lows, highs - means]),
            fmt="none", ecolor="black", capsize=5, linewidth=1.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(choices)
    ax.set_ylabel("Participant median RT (s)")
    ax.set_title("RT conditional on choice")
    ax.legend(frameon=False)
    style_ax(ax)
    save_close(fig, path)


def choice_es_bin_plot(tab, title, xlabel, path):
    order = RT_ORDER
    x = np.arange(len(order))
    width = 0.34
    fig, ax = plt.subplots(figsize=(11.2, 7.2))

    for choice, offset, obs_color, ppc_color in [
        ("E", -width / 2, OBS_E_COLOR, PPC_E_COLOR),
        ("S", +width / 2, OBS_S_COLOR, PPC_S_COLOR),
    ]:
        z = tab.loc[tab["choice"] == choice].copy()
        z["bin"] = z["bin"].astype(str)
        z = z.set_index("bin").reindex(order)
        xpos = x + offset

        obs_mean = 100 * z["observed_mean"].to_numpy(float)
        obs_low = 100 * z["observed_boot_ci_low"].to_numpy(float)
        obs_high = 100 * z["observed_boot_ci_high"].to_numpy(float)
        ax.bar(
            xpos, obs_mean, width=width * 0.92,
            color=obs_color, edgecolor="black", linewidth=1.4, alpha=0.82,
            label=f"Empirical {choice}",
        )
        ax.errorbar(
            xpos, obs_mean,
            yerr=np.vstack([obs_mean - obs_low, obs_high - obs_mean]),
            fmt="none", ecolor="black", capsize=5, linewidth=1.5,
        )

        ppc_mean = 100 * z["ppc_mean"].to_numpy(float)
        ppc_low = 100 * z["ppc_pi95_low"].to_numpy(float)
        ppc_high = 100 * z["ppc_pi95_high"].to_numpy(float)
        ax.plot(
            xpos, ppc_mean, color=ppc_color, marker="o",
            markersize=7, linewidth=2.8,
            label=f"Posterior predictive {choice}",
        )
        ax.errorbar(
            xpos, ppc_mean,
            yerr=np.vstack([ppc_mean - ppc_low, ppc_high - ppc_mean]),
            fmt="none", ecolor=ppc_color, capsize=5, linewidth=2,
        )

    ax.axhline(50, color="0.55", linestyle=":", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Choice probability (%)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    style_ax(ax)
    save_close(fig, path)


def p_s_bin_plot(obs_path, sim_path, bin_col, title, xlabel, path):
    obs = pd.read_csv(obs_path)
    sim = pd.read_csv(sim_path)

    order = [str(x) for x in obs["bin"].tolist()]
    obs["bin"] = obs["bin"].astype(str)
    sim[bin_col] = sim[bin_col].astype(str)

    obs = obs.set_index("bin").reindex(order)
    model = (
        sim.groupby(bin_col)["p_choose_s"]
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
        x, obs_mean, width=0.72, color=OBS_S_COLOR,
        edgecolor="black", linewidth=1.8, alpha=0.82, label="Empirical P(S)",
    )
    ax.errorbar(
        x, obs_mean,
        yerr=np.vstack([obs_mean - obs_low, obs_high - obs_mean]),
        fmt="none", ecolor="black", capsize=6, linewidth=1.8,
    )

    m = 100 * model["mean"].to_numpy(float)
    lo = 100 * model["low"].to_numpy(float)
    hi = 100 * model["high"].to_numpy(float)
    ax.fill_between(x, lo, hi, color=PPC_S_COLOR, alpha=0.16)
    ax.plot(x, m, color=PPC_S_COLOR, linewidth=3, marker="o",
            label="Posterior-predictive P(S)")

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0, 100)
    ax.set_ylabel("P(choose S) in %")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    style_ax(ax)
    save_close(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    out = args.output_dir
    tables = out / "tables"
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    overall_choice_plot(
        pd.read_csv(tables / "overall_choice_es_group_plot.csv"),
        figures / "overall_choice_ppc.png",
    )
    rt_choice_plot(
        pd.read_csv(tables / "rt_by_response_group_plot.csv"),
        figures / "rt_by_response_ppc.png",
    )
    choice_es_bin_plot(
        pd.read_csv(tables / "choice_e_s_by_rt_quintile_ppc.csv"),
        "Probability of choosing E vs S by RT quintile",
        "RT quintile (1 = fastest, 5 = slowest)",
        figures / "p_choose_e_s_by_rt_quintile_ppc.png",
    )
    choice_es_bin_plot(
        pd.read_csv(tables / "choice_e_s_by_rt_empirical_cutpoints_ppc.csv"),
        "Probability of choosing E vs S by empirical RT cutpoints",
        "Empirical RT quintile cutpoint bin",
        figures / "p_choose_e_s_by_rt_empirical_cutpoints_ppc.png",
    )

    # Existing P(S)-only conditional PPCs can also be regenerated from tables.
    p_s_bin_plot(
        tables / "observed_dwell_summary_combined.csv",
        tables / "simulated_dwell_by_draw_combined.csv",
        "dwell_quintile",
        "P(choose S) by dwell-time quintile",
        "Dwell-time advantage quintile (E/S identity; S - E)",
        figures / "p_choose_s_by_dwell_quintile_combined.png",
    )
    p_s_bin_plot(
        tables / "observed_dwell_prop_summary_identity.csv",
        tables / "simulated_dwell_prop_by_draw_identity.csv",
        "dwell_prop_quintile",
        "P(choose S) by proportional dwell quintile",
        "Proportional dwell advantage quintile (S - E)",
        figures / "p_choose_s_by_dwell_prop_quintile_identity.png",
    )
    p_s_bin_plot(
        tables / "observed_rt_summary_combined.csv",
        tables / "simulated_rt_by_draw_combined.csv",
        "rt_quintile",
        "P(choose S) by RT quintile",
        "RT quintile (1 = fastest, 5 = slowest)",
        figures / "p_choose_s_by_rt_quintile_combined.png",
    )

    print(f"Replotted PPC figures from saved outputs in: {out}")


if __name__ == "__main__":
    main()
