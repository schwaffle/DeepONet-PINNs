"""Direct bounded HF least-squares inverse reference (not a neural method)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import least_squares
from datasets import PARAM_BOUNDS
from solver import ExperimentConfig, solve, interpolate_trajectory


def run(data_file: str | Path, output: str | Path, seed: int = 7, max_nfev: int = 80):
    data = np.load(data_file)
    cfg = ExperimentConfig(seed=seed)
    coords = data["obs_coords"].astype(np.float64)
    values = data["obs_values"].astype(np.float64)
    calls = {"n": 0}

    def residual(theta):
        calls["n"] += 1
        tr = solve(tuple(map(float, theta)), cfg.n_hf, "central", cfg)
        pred = interpolate_trajectory(tr, coords)
        # Scale residuals by the training-observation noise std (not test data).
        sigma = max(float(data["noise_std"]), 1e-10)
        return (pred-values)/sigma

    t0 = time.perf_counter()
    result = least_squares(residual, x0=np.asarray([0.9, 0.5, 0.5]),
                           bounds=(PARAM_BOUNDS[:,0], PARAM_BOUNDS[:,1]),
                           max_nfev=max_nfev, diff_step=1e-4, verbose=0)
    elapsed = time.perf_counter()-t0
    theta = result.x
    fitted_residual = residual(theta)
    true = data["theta_true"]
    rel = np.abs(theta-true)/np.maximum(np.abs(true), 1e-12)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "Direct_HF_LS_control", "seed": seed,
        "theta_estimated": theta.tolist(), "theta_true": true.tolist(),
        "relerr_Q": float(rel[0]), "relerr_xs": float(rel[1]),
        "relerr_ys": float(rel[2]), "param_relerr_mean": float(rel.mean()),
        "observation_RMSE": float(np.sqrt(np.mean((fitted_residual*max(float(data["noise_std"]),1e-10))**2))),
        "observation_RMSE_in_noise_units": float(np.sqrt(np.mean(fitted_residual**2))),
        "cost": float(result.cost), "optimality": float(result.optimality),
        "success": bool(result.success), "status": int(result.status),
        "message": str(result.message), "nfev": int(result.nfev),
        "solver_calls": int(calls["n"]), "seconds": float(elapsed),
        "note": "Numerical optimizer over the same discrete HF solver; not a neural baseline."
    }
    (out/"direct_hf_ls_reference.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", default="results_exp1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-nfev", type=int, default=80)
    a=ap.parse_args()
    print(json.dumps(run(a.data,a.output,a.seed,a.max_nfev),indent=2))

if __name__ == "__main__":
    main()
