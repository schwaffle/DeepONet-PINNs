"""LF operator-training and HF inverse-identification datasets for Exp. 1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np

from solver import (ExperimentConfig, solve, sample_hf_observations,
                    sample_operator_query_points, interpolate_trajectory)

PARAM_BOUNDS = np.asarray([[0.4, 1.6], [0.1, 0.9], [0.1, 0.9]], dtype=np.float64)


def source_profile_numpy(theta: np.ndarray, sensor_xy: np.ndarray, sigma_s: float) -> np.ndarray:
    q, xs, ys = np.asarray(theta, dtype=np.float64)
    r2 = (sensor_xy[:, 0]-xs)**2 + (sensor_xy[:, 1]-ys)**2
    return q*np.exp(-r2/(2.0*sigma_s**2))


def make_sensor_grid(n: int = 12) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, n)
    sx, sy = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack([sx.ravel(), sy.ravel()]).astype(np.float32)


def make_lf_feature_coords(nxy: int = 11, T: float = 0.5, times_frac=(0.25, 0.50, 0.75, 1.0)):
    """Fixed LF solution features used by the linear and nonlinear HF blocks.

    The 11x11 spatial mesh follows the downsampled LF-output feature idea in
    Howard et al. (2023). Time slices are added because this application is
    time-dependent; this 3-D space-time extension is an explicit adaptation.
    """
    axis = np.linspace(0.02, 0.98, nxy)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    rows = []
    for tf in times_frac:
        rows.append(np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, tf*T)]))
    return np.concatenate(rows, axis=0).astype(np.float32)


def make_lf_sub_indices(nxy: int = 11, ntime: int = 4,
                        spatial_indices=(1, 3, 5, 7, 9), time_indices=(0, 2, 3)) -> np.ndarray:
    idx = []
    for ti in time_indices:
        for ix in spatial_indices:
            for iy in spatial_indices:
                idx.append(ti*nxy*nxy + ix*nxy + iy)
    return np.asarray(idx, dtype=np.int64)


def _sample_parameter_functions(n: int, rng: np.random.Generator, theta_true):
    rows = []
    while len(rows) < n:
        cand = rng.uniform(PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])
        # Target inverse problem is deliberately held out from LF operator training.
        if np.linalg.norm((cand-np.asarray(theta_true))/(PARAM_BOUNDS[:, 1]-PARAM_BOUNDS[:, 0])) < 1e-5:
            continue
        rows.append(cand)
    return np.asarray(rows, dtype=np.float64)


def generate_datasets(out_dir: str | Path, seed: int = 7,
                      n_lf_parameters: int = 40, points_per_parameter: int = 384,
                      n_test_axis: int = 33, sensor_axis_n: int = 12,
                      lf_scheme: str = "central") -> Path:
    """Create one reproducible Exp. 1 dataset.

    LF train/validation is split by whole source functions, not by points.
    HF test coordinates are clean and have exact coordinate-time overlaps with
    training observations removed. The full coarse plotting grid is also saved
    separately for field visualisation only; it is never used for metrics.
    """
    generation_t0 = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(seed=seed)
    if lf_scheme not in {"central", "upwind"}:
        raise ValueError("lf_scheme must be central or upwind")

    sensor_xy = make_sensor_grid(sensor_axis_n)
    rng_lf = np.random.default_rng(seed + 101)
    theta_all = _sample_parameter_functions(n_lf_parameters, rng_lf, cfg.theta_true)
    perm = rng_lf.permutation(n_lf_parameters)
    n_val = max(1, int(round(0.2*n_lf_parameters))) if n_lf_parameters >= 5 else 1
    val_set = set(perm[:n_val].tolist())

    chunks = {f"lf_{split}_{name}": [] for split in ("train", "val")
              for name in ("source", "coords", "values", "theta")}
    lf_solver_seconds = 0.0
    for i, theta in enumerate(theta_all):
        solve_t0 = time.perf_counter()
        lf_traj = solve(tuple(theta.tolist()), cfg.n_lf, lf_scheme, cfg)
        lf_solver_seconds += time.perf_counter() - solve_t0
        qcoords = sample_operator_query_points(theta, cfg, points_per_parameter, rng_lf)
        qvalues = interpolate_trajectory(lf_traj, qcoords)
        src = source_profile_numpy(theta, sensor_xy, cfg.sigma_s).astype(np.float32)
        split = "val" if i in val_set else "train"
        chunks[f"lf_{split}_source"].append(np.repeat(src[None, :], points_per_parameter, axis=0))
        chunks[f"lf_{split}_coords"].append(qcoords)
        chunks[f"lf_{split}_values"].append(qvalues[:, None].astype(np.float32))
        chunks[f"lf_{split}_theta"].append(np.repeat(theta[None, :], points_per_parameter, axis=0).astype(np.float32))

    arrays = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
    arrays["lf_parameters_train"] = np.asarray([theta_all[i] for i in range(n_lf_parameters) if i not in val_set], dtype=np.float32)
    arrays["lf_parameters_val"] = np.asarray([theta_all[i] for i in range(n_lf_parameters) if i in val_set], dtype=np.float32)

    # HF target trajectory: central differences on the 129x129 grid.
    hf_t0 = time.perf_counter()
    hf = solve(cfg.theta_true, cfg.n_hf, "central", cfg)
    hf_solver_seconds = time.perf_counter() - hf_t0
    rng_obs = np.random.default_rng(seed)
    obs_coords, obs_values, noise_std = sample_hf_observations(hf, cfg, rng_obs)

    # Full clean HF test: every saved HF grid-time point except exact noisy
    # observation locations. This makes test metrics representative of the
    # resolved HF trajectory rather than only of a visualisation sub-grid.
    times, xg, yg, fields = hf
    all_coords = np.asarray([(x, y, t) for t in times for x in xg for y in yg], dtype=np.float32)
    all_values = fields.reshape(-1).astype(np.float32)
    obs_keys = {(round(float(x), 7), round(float(y), 7), round(float(t), 7)) for x, y, t in obs_coords}
    keep = np.asarray([(round(float(x), 7), round(float(y), 7), round(float(t), 7)) not in obs_keys
                       for x, y, t in all_coords], dtype=bool)
    test_coords = all_coords[keep]
    test_values = all_values[keep]

    # Separate structured subset for figures only. It is not used for metrics,
    # loss construction, scaling, validation, or early stopping.
    ids = np.unique(np.linspace(0, cfg.n_hf-1, n_test_axis).round().astype(int))
    plot_coords = np.asarray([(xg[ix], yg[iy], t)
                              for t in times for ix in ids for iy in ids], dtype=np.float32)
    plot_values = np.asarray([fields[it, ix, iy]
                              for it in range(len(times)) for ix in ids for iy in ids], dtype=np.float32)

    lf_feature_coords = make_lf_feature_coords(nxy=11, T=cfg.T)
    lf_sub_idx = make_lf_sub_indices(nxy=11, ntime=4)
    # Include LF scheme in the filename. Otherwise a previous Central/Central
    # dataset could silently be reused when the user requests LF-Upwind/HF-Central.
    npz_path = out / f"exp1_multifidelity_seed_{seed}_lf_{lf_scheme}_hf_central.npz"
    np.savez_compressed(
        npz_path, **arrays,
        obs_coords=obs_coords, obs_values=obs_values,
        noise_std=np.asarray(noise_std, dtype=np.float32),
        test_coords=test_coords, test_values=test_values,
        plot_coords=plot_coords, plot_values=plot_values,
        test_times=times.astype(np.float32),
        test_x_axis=xg[ids].astype(np.float32), test_y_axis=yg[ids].astype(np.float32),
        theta_true=np.asarray(cfg.theta_true, dtype=np.float32),
        sensor_xy=sensor_xy, lf_feature_coords=lf_feature_coords,
        lf_sub_idx=lf_sub_idx, param_bounds=PARAM_BOUNDS.astype(np.float32),
    )
    dx_lf = 1.0 / (cfg.n_lf - 1)
    dx_hf = 1.0 / (cfg.n_hf - 1)

    # Diagnostic-only target LF solves. They are NOT appended to LF operator
    # train/validation data and are never passed to train.py. They quantify the
    # fidelity gap and the same-grid scheme effect for the held-out inverse target.
    target_lf = solve(cfg.theta_true, cfg.n_lf, lf_scheme, cfg)
    target_lf_central = (target_lf if lf_scheme == "central"
                         else solve(cfg.theta_true, cfg.n_lf, "central", cfg))
    tx, ty = np.meshgrid(target_lf[1], target_lf[2], indexing="ij")
    lf_diag_coords = np.asarray(
        [(x, y, t) for t in target_lf[0] for x in target_lf[1] for y in target_lf[2]],
        dtype=np.float32)
    hf_at_lf = interpolate_trajectory(hf, lf_diag_coords).reshape(len(target_lf[0]), cfg.n_lf, cfg.n_lf)
    target_lf_err = target_lf[3] - hf_at_lf
    target_lf_hf_gap = {
        "MAE_on_LF_grid": float(np.mean(np.abs(target_lf_err))),
        "relative_L2_on_LF_grid": float(np.linalg.norm(target_lf_err)/(np.linalg.norm(hf_at_lf)+1e-15)),
        "note": "Diagnostic-only target solve; not included in LF operator training or validation."
    }
    central_lf_coords = np.asarray(
        [(x, y, t) for t in target_lf_central[0]
         for x in target_lf_central[1] for y in target_lf_central[2]], dtype=np.float32)
    central_lf_vs_hf = interpolate_trajectory(hf, central_lf_coords).reshape(
        len(target_lf_central[0]), cfg.n_lf, cfg.n_lf)
    central_lf_err = target_lf_central[3] - central_lf_vs_hf
    central_lf_hf_gap = {
        "MAE_on_LF_grid": float(np.mean(np.abs(central_lf_err))),
        "relative_L2_on_LF_grid": float(np.linalg.norm(central_lf_err)/(np.linalg.norm(central_lf_vs_hf)+1e-15)),
        "note": "Diagnostic-only Central 21x21 vs HF Central 129x129 resolution-control gap; not used in training."
    }

    same_grid_scheme_gap = None
    if lf_scheme == "upwind":
        central_at_upwind_times = interpolate_trajectory(
            target_lf_central,
            np.asarray([(x, y, t) for t in target_lf[0]
                        for x in target_lf[1] for y in target_lf[2]], dtype=np.float32)
        ).reshape(len(target_lf[0]), cfg.n_lf, cfg.n_lf)
        scheme_err = target_lf[3] - central_at_upwind_times
        same_grid_scheme_gap = {
            "MAE_same_21x21_grid": float(np.mean(np.abs(scheme_err))),
            "relative_L2_same_21x21_grid": float(np.linalg.norm(scheme_err)/(np.linalg.norm(central_at_upwind_times)+1e-15)),
            "note": "Diagnostic-only comparison of Upwind and Central advection on identical 21x21 nodes; not used in training."
        }

    metadata = {
        "experiment": "Exp. 1: multi-fidelity source identification, 2-D advection-diffusion",
        "seed": seed, "theta_true": list(cfg.theta_true), "param_bounds": PARAM_BOUNDS.tolist(),
        "LF": {"grid": [cfg.n_lf, cfg.n_lf], "scheme": lf_scheme,
               "n_realizations": int(n_lf_parameters),
               "train_realizations": int(len(arrays['lf_parameters_train'])),
               "validation_realizations": int(len(arrays['lf_parameters_val'])),
               "points_per_realization": points_per_parameter},
        "HF": {"grid": [cfg.n_hf, cfg.n_hf], "scheme": "central",
               "n_observations": int(len(obs_values)), "noise_fraction_of_peak": cfg.noise_fraction,
               "noise_std": noise_std},
        "test": {"n_clean_points": int(len(test_values)),
                 "n_full_HF_grid_time_points_before_exclusion": int(len(all_values)),
                 "n_plot_points": int(len(plot_values)),
                 "observation_overlaps_removed": int(len(all_values)-len(test_values)),
                 "test_coords_disjoint_from_observation_coords": True,
                 "plot_grid_is_visualisation_only": True},
        "operator_architecture": {
            "branch_input": "Gaussian spatial source profile at fixed 12x12 sensors",
            "trunk_input": "physical (x,y,t), normalized internally to [-1,1]",
            "LF_feature_mesh": "11x11 spatial mesh x 4 times; 5x5x3 subset used in nonlinear branch"
        },
        "design": f"LF {lf_scheme} on {cfg.n_lf}x{cfg.n_lf}; HF central on {cfg.n_hf}x{cfg.n_hf}.",
        "scheme_diagnostics": {
            "dx_lf": dx_lf, "dx_hf": dx_hf,
            "target_LF_vs_HF": target_lf_hf_gap,
            "resolution_only_Central21_vs_Central129": central_lf_hf_gap,
            "same_grid_LF_scheme_effect": same_grid_scheme_gap,
            "LF_cell_Peclet_x": abs(cfg.vx)*dx_lf/(2.0*cfg.kappa),
            "LF_cell_Peclet_y": abs(cfg.vy)*dx_lf/(2.0*cfg.kappa),
            "HF_cell_Peclet_x": abs(cfg.vx)*dx_hf/(2.0*cfg.kappa),
            "HF_cell_Peclet_y": abs(cfg.vy)*dx_hf/(2.0*cfg.kappa),
            "LF_upwind_modified_diffusion_x": abs(cfg.vx)*dx_lf/2.0 if lf_scheme == "upwind" else 0.0,
            "LF_upwind_modified_diffusion_y": abs(cfg.vy)*dx_lf/2.0 if lf_scheme == "upwind" else 0.0,
            "LF_upwind_effective_kappa_x": cfg.kappa + abs(cfg.vx)*dx_lf/2.0 if lf_scheme == "upwind" else cfg.kappa,
            "LF_upwind_effective_kappa_y": cfg.kappa + abs(cfg.vy)*dx_lf/2.0 if lf_scheme == "upwind" else cfg.kappa,
            "warning": "Central advection has no first-order upwind artificial diffusion, but it is not exact and may show dispersive/oscillatory error at high cell Peclet number. Upwind modified-equation diffusion is a leading-order estimate."
        },
        "timing": {"lf_solver_seconds": float(lf_solver_seconds),
                   "hf_solver_seconds": float(hf_solver_seconds),
                   "dataset_generation_seconds": float(time.perf_counter()-generation_t0)},
        "note": "The target source theta_true is excluded from the LF function ensemble."
    }
    (out / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return npz_path


def main():
    ap = argparse.ArgumentParser(description="Generate LF operator and HF inverse data")
    ap.add_argument("--output", default="results")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lf-parameters", type=int, default=40)
    ap.add_argument("--points-per-parameter", type=int, default=384)
    ap.add_argument("--n-test-axis", type=int, default=33)
    ap.add_argument("--sensor-axis-n", type=int, default=12)
    ap.add_argument("--lf-scheme", choices=["central", "upwind"], default="central")
    args = ap.parse_args()
    path = generate_datasets(args.output, args.seed, args.lf_parameters,
                             args.points_per_parameter, args.n_test_axis,
                             args.sensor_axis_n, args.lf_scheme)
    print(f"Dataset written: {path}")


if __name__ == "__main__":
    main()
