"""
Запуск воспроизводимого эксперимента агрегация результатов (вычисление mean +/- std).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from run_exp1 import run_single

DEFAULT_START_SEED = 1
DEFAULT_NUM_SEEDS = 50
METRICS = ["relerr_Q", "relerr_xs", "relerr_ys", "param_relerr_mean",
           "test_RMSE", "test_MAE", "test_relL2", "test_interior_RMSE",
           "test_interior_MAE", "test_interior_relL2", "observation_RMSE",
           "PDE_residual_RMS_dimensionless", "training_seconds", "trainable_parameters",
           "lbfgsb_success", "lbfgsb_nit", "lbfgsb_nfev"]


def main():
    ap = argparse.ArgumentParser(description="Run a reproducible seed series for PINN-Raissi vs M1 PI-DeepONet")
    ap.add_argument("--start-seed", type=int, default=DEFAULT_START_SEED,
                    help="First random seed (default: 1)")
    ap.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS,
                    help="Number of consecutive seeds (default: 50)")
    ap.add_argument("--output", default="results_50seeds")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--adam-epochs", type=int, default=2000)
    ap.add_argument("--lbfgs-max-iter", type=int, default=5000)
    ap.add_argument("--lf-pretrain-epochs", type=int, default=1000)
    ap.add_argument("--collocation", type=int, default=256)
    ap.add_argument("--n-boundary", type=int, default=128)
    ap.add_argument("--lf-parameters", type=int, default=40)
    ap.add_argument("--points-per-parameter", type=int, default=384)
    ap.add_argument("--n-test-axis", type=int, default=33)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--lf-scheme", choices=["central", "upwind"], default="central")
    a = ap.parse_args()
    if a.num_seeds < 1:
        ap.error("--num-seeds must be at least 1")
    if a.start_seed < 0:
        ap.error("--start-seed must be non-negative")
    seeds = list(range(a.start_seed, a.start_seed + a.num_seeds))
    torch.set_num_threads(1)
    root = Path(a.output); root.mkdir(parents=True, exist_ok=True)
    all_rows, failures = [], []
    cfg = dict(device=a.device, adam_epochs=a.adam_epochs,
               lbfgs_max_iter=a.lbfgs_max_iter,
               lf_pretrain_epochs=a.lf_pretrain_epochs,
               collocation=a.collocation, n_boundary=a.n_boundary,
               n_lf_parameters=a.lf_parameters,
               points_per_parameter=a.points_per_parameter,
               n_test_axis=a.n_test_axis, learning_rate=a.learning_rate,
               lf_scheme=a.lf_scheme)
    for seed in seeds:
        seed_dir = root / f"seed_{seed}"
        try:
            df = run_single(seed=seed, output=seed_dir, model="both", **cfg)
            df.insert(0, "seed", seed)
            all_rows.append(df)
            print(f"[seed {seed}] completed")
        except Exception as exc:
            failures.append({"seed": seed, "error": repr(exc)})
            print(f"[seed {seed}] FAILED: {exc}")
    if all_rows:
        per_seed = pd.concat(all_rows, ignore_index=True)
        per_seed.to_csv(root/"per_seed_results.csv", index=False)
        summary_rows = []
        for method, grp in per_seed.groupby("method"):
            row = {"method": method, "n_completed_runs": int(len(grp))}
            for metric in METRICS:
                if metric in grp:
                    vals = pd.to_numeric(grp[metric], errors="coerce").dropna()
                    row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
                    row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals)>1 else (0.0 if len(vals)==1 else np.nan)
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(root/"mean_std_summary.csv", index=False)
    (root/"failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    (root/"run_config.json").write_text(json.dumps({"seeds": seeds, **cfg,
           "n_completed_seeds": len({r["seed"] for d in all_rows for r in d.to_dict("records")}),
           "n_failed_seeds": len(failures), "HF_scheme": "central",
           "LF_scheme": a.lf_scheme}, indent=2), encoding="utf-8")
    print(f"Completed seeds: {len(all_rows)} / {len(seeds)}; failures: {len(failures)}")


if __name__ == "__main__":
    main()
