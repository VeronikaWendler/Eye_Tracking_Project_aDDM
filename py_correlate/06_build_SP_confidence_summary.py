#!/usr/bin/env python3
"""
06 — BUILD PARTICIPANT-LEVEL SP CONFIDENCE SUMMARIES

Reads the original per-participant SP CSV files produced by the MAT->CSV
conversion scripts and summarizes confidence separately for E and S items.

Outputs:
  data_sets/study1_data_sets/Study1_SP_confidence_summary.csv
  data_sets/study2_data_sets/Study2_SP_confidence_summary.csv

Primary contrast for z:
  SP_confidence_S_minus_E = mean confidence on S items - mean confidence on E items

Because the ESaDDM uses S as the upper boundary:
  z > .5 = S starting-point bias
  z < .5 = E starting-point bias

Therefore a positive correlation between SP_confidence_S_minus_E and z would
mean that participants who are relatively more confident about S than E also
start relatively closer to the S boundary in ES choice.

Important: confidence is measured after ES, so this is an association, not a
causal test.
"""

from pathlib import Path
import pandas as pd
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

STUDIES = {
    1: {
        "raw_root": Path(
            "D:/Birmingham_Uni_Oct25/Eye_Tracking_Project/Study1/Study1_Data"
        ),
        "exclude": {1, 4, 5, 6, 14, 99},
        "filename": "Study1_SP_Phase_Participant_{sid}.csv",
        "out": REPO_ROOT
        / "data_sets/study1_data_sets/Study1_SP_confidence_summary.csv",
    },
    2: {
        "raw_root": Path(
            "D:/Birmingham_Uni_Oct25/Eye_Tracking_Project/Study2/Study2_Data"
        ),
        "exclude": {2, 6, 9, 14, 18, 20, 26},
        "filename": "Study2_SP_Phase_Participant_{sid}.csv",
        "out": REPO_ROOT
        / "data_sets/study2_data_sets/Study2_SP_confidence_summary.csv",
    },
}


CONF_COL = "confidenceLevelsArrayEXP"


def mean_sd(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    if len(x) == 0:
        return np.nan, np.nan, 0
    sd = float(np.std(x, ddof=1)) if len(x) >= 2 else np.nan
    return float(np.mean(x)), sd, int(len(x))


def run_study(study, cfg):
    root = cfg["raw_root"]
    exclude = set(cfg["exclude"])
    out = cfg["out"]

    if not root.exists():
        raise FileNotFoundError(
            f"Study {study}: raw data root does not exist:\n{root}"
        )

    participant_dirs = sorted(
        [
            p for p in root.glob("Participant_*")
            if p.is_dir()
        ],
        key=lambda p: int(p.name.split("_")[-1]),
    )

    rows = []

    for folder in participant_dirs:
        sid = int(folder.name.split("_")[-1])

        if sid in exclude:
            continue

        f = folder / cfg["filename"].format(sid=sid)
        if not f.exists():
            raise FileNotFoundError(
                f"Study {study}, participant {sid}: missing SP file:\n{f}"
            )

        df = pd.read_csv(f)

        required = {CONF_COL, "op1"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(
                f"Study {study}, participant {sid}: missing columns "
                f"{sorted(missing)} in {f}"
            )

        d = df[[CONF_COL, "op1"]].copy()
        d[CONF_COL] = pd.to_numeric(d[CONF_COL], errors="coerce")
        d["op1"] = d["op1"].astype(str).str.strip()

        all_mean, all_sd, n_all = mean_sd(d[CONF_COL])
        e_mean, e_sd, n_e = mean_sd(
            d.loc[d["op1"].eq("E"), CONF_COL]
        )
        s_mean, s_sd, n_s = mean_sd(
            d.loc[d["op1"].eq("S"), CONF_COL]
        )

        if n_e == 0 or n_s == 0:
            raise ValueError(
                f"Study {study}, participant {sid}: expected both E and S "
                f"confidence ratings; n_E={n_e}, n_S={n_s}"
            )

        rows.append({
            "study": f"Study{study}",
            "sub_id": sid,
            "SP_confidence_all": all_mean,
            "SP_confidence_E": e_mean,
            "SP_confidence_S": s_mean,
            "SP_confidence_S_minus_E": s_mean - e_mean,
            "SP_confidence_E_minus_S": e_mean - s_mean,
            "SP_confidence_all_sd": all_sd,
            "SP_confidence_E_sd": e_sd,
            "SP_confidence_S_sd": s_sd,
            "SP_confidence_n_all": n_all,
            "SP_confidence_n_E": n_e,
            "SP_confidence_n_S": n_s,
        })

    result = pd.DataFrame(rows).sort_values("sub_id").reset_index(drop=True)

    if result.empty:
        raise ValueError(f"Study {study}: no included participants found.")

    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    print("=" * 78)
    print(f"Study {study}")
    print(f"Included N: {len(result)}")
    print(f"Participant IDs: {result['sub_id'].tolist()}")
    print(
        "Mean confidence all / E / S / S-E:",
        result[
            [
                "SP_confidence_all",
                "SP_confidence_E",
                "SP_confidence_S",
                "SP_confidence_S_minus_E",
            ]
        ].mean().round(4).to_dict(),
    )
    print("Saved:", out)

    return result


def main():
    for study, cfg in STUDIES.items():
        run_study(study, cfg)


if __name__ == "__main__":
    main()
