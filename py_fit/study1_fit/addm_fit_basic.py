from __future__ import annotations

from pathlib import Path
import argparse
import os
import time
import numpy as np
import pandas as pd
import arviz as az
import dill as pickle
from joblib import Parallel, delayed

# Keep cluster plotting headless.
import matplotlib
matplotlib.use("Agg")

import hddm

from addm_prepare_data import prepare_addm_data


def sanitize_infdata(infdata):
    """Replace pandas missing objects with np.nan before NetCDF export."""
    for group in infdata._groups_all:
        if hasattr(infdata, group):
            dataset = getattr(infdata, group)
            for var in dataset.data_vars:
                values = dataset[var].values
                if isinstance(values, np.ndarray) and values.dtype == "object":
                    mask = pd.isna(values)
                    if mask.any():
                        values = values.copy()
                        values[mask] = np.nan
                        dataset[var].values = values
    return infdata


def build_basic_addm(data: pd.DataFrame):
    """
    Basic value-ranked aDDM.

    response = 1 means the participant chose the higher-valued option.
    Therefore positive drift points toward the higher-valued boundary.

    AttentionW / InattentionW are already value-ranked in the AnalysisReady file,
    so the SAME model specification is used for ES and EE.
    """
    v_reg = {
        "model": "v ~ 1 + AttentionW + InattentionW",
        "link_func": lambda x: x,
    }

    model = hddm.models.HDDMRegressor(
        data,
        [v_reg],
        p_outlier=0.05,
        include=["a", "t", "v", "z"],   
        group_only_regressors=False,
        keep_regressor_trace=True,
    )
    model.find_starting_values()
    return model


def run_chain(chain_id, data, model_dir, model_name, samples, burn):
    model = build_basic_addm(data)

    dbname = model_dir / f"{model_name}_db{chain_id}"
    infdata = model.sample(
        samples,
        burn=burn,
        dbname=str(dbname),
        db="pickle",
        return_infdata=True,
        loglike=True,
        ppc=True,
    )
    return model, sanitize_infdata(infdata)


def save_chain(model, infdata, model_dir: Path, model_name: str, chain_id: int):
    stem = model_dir / f"{model_name}_{chain_id}"

    model.save(str(stem) + ".hddm")

    with open(str(stem) + ".pkl", "wb") as f:
        pickle.dump(model, f)

    az.to_netcdf(infdata, str(stem) + ".nc")


def main():
    parser = argparse.ArgumentParser(description="Fit the basic corrected aDDM separately to ES or EE.")
    parser.add_argument("--data", required=True, help="Correct Study1_Behaviour_with_Gaze_AnalysisReady.csv")
    parser.add_argument("--phase", required=True, choices=["ES", "EE", "LE"])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--burn", type=int, default=500)
    parser.add_argument("--chains", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--min-rt", type=float, default=0.250)
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    if args.phase == "LE":
        print(
            "WARNING: LE is allowed as a scaffold only. "
            "The current AttentionW/InattentionW use objective trial values. "
            "If LE should use learned/Q values, rebuild those regressors before interpreting the fit."
        )

    model_dir = Path(args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    data, audit = prepare_addm_data(args.data, phase=args.phase, min_rt=args.min_rt)

    model_name = args.model_name or f"basic_aDDM_{args.phase}"
    exact_input = model_dir / f"{model_name}_MODEL_INPUT.csv"
    data.to_csv(exact_input, index=False)

    print("\n=== EXACT MODEL INPUT ===")
    print(f"phase:        {args.phase}")
    print(f"rows:         {len(data):,}")
    print(f"participants: {data['subj_idx'].nunique()}")
    print(f"response:     {data['response'].value_counts().sort_index().to_dict()}")
    print(f"saved input:  {exact_input}")

    n_jobs = args.jobs if args.jobs is not None else args.chains
    start = time.time()

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_chain)(
            chain_id=i,
            data=data,
            model_dir=model_dir,
            model_name=model_name,
            samples=args.samples,
            burn=args.burn,
        )
        for i in range(args.chains)
    )

    for i, (model, infdata) in enumerate(results):
        save_chain(model, infdata, model_dir, model_name, i)

    print(f"\nCompleted {args.chains} chains in {(time.time() - start)/60:.1f} minutes.")
    print(f"Models saved in: {model_dir}")


if __name__ == "__main__":
    main()
