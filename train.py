"""Train PINN-Raissi and M1 composite PI-DeepONet on the same inverse problem.

The six loss terms (lambda_1..lambda_6) reproduce Howard et al. (2023), Eq. 15.
For source identification we additionally need sparse HF observations of the unknown
source realization. Therefore we add lambda_obs * L_HFobs as a clearly documented
seventh, task-specific term. The same observation term is used for the PINN baseline.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from datasets import PARAM_BOUNDS, generate_datasets
from models import PINN, ModifiedDeepONet, MultiFidelityPIDeepONet
from solver import ExperimentConfig


DEFAULT_LOSSES = {
    "lambda_obs": 10.0,
    "lambda1_physics": 10.0,
    "lambda2_lf": 1.0,
    "lambda3_nonlinear_branch_reg": 1e-6,
    "lambda4_lf_branch_reg": 1e-6,
    "lambda5_ic": 20.0,
    "lambda6_bc": 1.0,
}


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_torch(a, device="cpu"):
    return torch.as_tensor(a, dtype=torch.float32, device=device)


def source_torch(coords: torch.Tensor, theta: torch.Tensor, cfg: ExperimentConfig):
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)
    if theta.shape[0] == 1 and coords.shape[0] > 1:
        theta = theta.expand(coords.shape[0], -1)
    q, xs, ys = theta[:, 0:1], theta[:, 1:2], theta[:, 2:3]
    x, y, t = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
    r2 = (x-xs)**2 + (y-ys)**2
    return q*torch.exp(-r2/(2.0*cfg.sigma_s**2))*(-torch.expm1(-t/cfg.tau))


def pde_residual(model, coords: torch.Tensor, u_scale: float, cfg: ExperimentConfig):
    """Dimensionless PDE residual; AD is with respect to physical coordinates.

    Models normalize coordinates internally. Autograd therefore applies the chain
    rule for x, y and t automatically; no manual derivative rescaling is required.
    Multiplication by T/u_scale makes the residual dimensionless.
    """
    z = coords.detach().clone().requires_grad_(True)
    u = model(z)*u_scale
    grad = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
    ux, uy, ut = grad[:, 0:1], grad[:, 1:2], grad[:, 2:3]
    uxx = torch.autograd.grad(ux.sum(), z, create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(uy.sum(), z, create_graph=True)[0][:, 1:2]
    theta = model.theta_physical()
    residual = ut + cfg.vx*ux + cfg.vy*uy - cfg.kappa * \
        (uxx+uyy) - source_torch(z, theta, cfg)
    return residual*(cfg.T/max(u_scale, 1e-12))


def sample_collocation(n: int, cfg: ExperimentConfig, rng: np.random.Generator):
    x = rng.uniform(0.01, 0.99, n)
    y = rng.uniform(0.01, 0.99, n)
    t = rng.uniform(0.0, cfg.T, n)
    near = rng.random(n) < 0.55
    xs = rng.uniform(PARAM_BOUNDS[1, 0], PARAM_BOUNDS[1, 1], n)
    ys = rng.uniform(PARAM_BOUNDS[2, 0], PARAM_BOUNDS[2, 1], n)
    x[near] = np.clip(rng.normal(xs[near] + cfg.vx*t[near], 0.14), 0.01, 0.99)
    y[near] = np.clip(rng.normal(ys[near] + cfg.vy*t[near], 0.14), 0.01, 0.99)
    return np.column_stack([x, y, t]).astype(np.float32)


def sample_boundary_initial(n: int, cfg: ExperimentConfig, rng: np.random.Generator):
    side = rng.integers(0, 4, size=n)
    x = rng.uniform(0.0, 1.0, n)
    y = rng.uniform(0.0, 1.0, n)
    x[side == 0] = 0.0
    x[side == 1] = 1.0
    y[side == 2] = 0.0
    y[side == 3] = 1.0
    bc = np.column_stack([x, y, rng.uniform(0.0, cfg.T, n)]).astype(np.float32)
    ic = np.column_stack([rng.uniform(0.0, 1.0, n),
                          rng.uniform(0.0, 1.0, n), np.zeros(n)]).astype(np.float32)
    return bc, ic


def _sum_squared(params):
    if not params:
        return torch.zeros(())
    return torch.stack([p.square().sum() for p in params]).sum()


def compute_losses(model, D, col, bc, ic, weights, lf_batch=None):
    obs_pred = model(D["obs_coords"])
    l_obs = torch.mean((obs_pred - D["obs_y"])**2)
    res = pde_residual(model, col, D["u_scale"], D["cfg"])
    l_phys = torch.mean(res**2)
    l_bc = torch.mean(model(bc)**2)
    l_ic = torch.mean(model(ic)**2)
    zero = torch.zeros((), dtype=torch.float32, device=D["device"])
    l_lf, reg_nl, reg_lf = zero, zero, zero

    if isinstance(model, MultiFidelityPIDeepONet):
        if lf_batch is None:
            raise ValueError("M1 loss requires a low-fidelity training batch")
        lf_pred = model.lf(lf_batch["source"], lf_batch["coords"])
        l_lf = torch.mean((lf_pred - lf_batch["values"])**2)
        nl_params, lf_params = model.regularization_parameters()
        reg_nl = _sum_squared(nl_params)
        reg_lf = _sum_squared(lf_params)
        loss = (weights["lambda_obs"]*l_obs
                + weights["lambda1_physics"]*l_phys
                + weights["lambda2_lf"]*l_lf
                + weights["lambda3_nonlinear_branch_reg"]*reg_nl
                + weights["lambda4_lf_branch_reg"]*reg_lf
                + weights["lambda5_ic"]*l_ic
                + weights["lambda6_bc"]*l_bc)
    else:
        loss = (weights["lambda_obs"]*l_obs
                + weights["lambda1_physics"]*l_phys
                + weights["lambda5_ic"]*l_ic
                + weights["lambda6_bc"]*l_bc)
    comps = {"obs": l_obs, "pde": l_phys, "lf": l_lf,
             "reg_nl": reg_nl, "reg_lf": reg_lf, "ic": l_ic, "bc": l_bc}
    return loss, comps


def _record(history, stage, iteration, loss, comps, model):
    theta = model.theta_physical().detach().cpu().numpy()
    row = {"stage": stage, "iteration": int(iteration), "loss": float(loss),
           **{k: float(v.detach().cpu()) for k, v in comps.items()},
           "Q": float(theta[0]), "xs": float(theta[1]), "ys": float(theta[2])}
    history.append(row)


def train_lf_operator(lf_model, data, u_scale: float, device: str,
                      epochs: int, batch_size: int, seed: int):
    """Warm-start the LF operator; model selection uses source-disjoint LF val only."""
    seed_everything(seed)
    source = to_torch(data["lf_train_source"], device)
    coords = to_torch(data["lf_train_coords"], device)
    values = to_torch(data["lf_train_values"], device)/u_scale
    v_source = to_torch(data["lf_val_source"], device)
    v_coords = to_torch(data["lf_val_coords"], device)
    v_values = to_torch(data["lf_val_values"], device)/u_scale
    optimizer = torch.optim.Adam(lf_model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed+91)
    history, best_val, best_state = [], float("inf"), None
    for ep in range(1, epochs+1):
        ids = rng.integers(0, len(values), size=min(batch_size, len(values)))
        pred = lf_model(source[ids], coords[ids])
        loss = torch.mean((pred-values[ids])**2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if ep == 1 or ep % max(1, epochs//40) == 0 or ep == epochs:
            with torch.no_grad():
                val = torch.mean((lf_model(v_source, v_coords)-v_values)**2)
            history.append({"epoch": ep, "train_mse": float(loss.detach().cpu()),
                            "val_mse": float(val.detach().cpu())})
            if float(val) < best_val:
                best_val = float(val)
                best_state = {k: v.detach().cpu().clone()
                              for k, v in lf_model.state_dict().items()}
    if best_state is not None:
        lf_model.load_state_dict(best_state)
    return history, best_val


def _flat_parameters(parameters):
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in parameters]).astype(np.float64)


def _assign_flat(parameters, x):
    offset = 0
    with torch.no_grad():
        for p in parameters:
            count = p.numel()
            block = torch.as_tensor(
                x[offset:offset+count], dtype=p.dtype, device=p.device).reshape_as(p)
            p.copy_(block)
            offset += count


def _flat_gradients(parameters):
    grads = []
    for p in parameters:
        if p.grad is None:
            grads.append(np.zeros(p.numel(), dtype=np.float64))
        else:
            grads.append(p.grad.detach().cpu(
            ).numpy().ravel().astype(np.float64))
    return np.concatenate(grads)


def scipy_lbfgsb(model, loss_closure, max_iter: int, history: list, start_iter: int):
    """SciPy L-BFGS-B over flattened PyTorch parameters (bounded theta via sigmoid)."""
    params = [p for p in model.parameters() if p.requires_grad]
    x0 = _flat_parameters(params)
    eval_counter = [0]

    def fun(x):
        _assign_flat(params, x)
        model.zero_grad(set_to_none=True)
        loss, comps = loss_closure()
        loss.backward()
        grad = _flat_gradients(params)
        eval_counter[0] += 1
        return float(loss.detach().cpu()), grad

    callback_count = [0]

    def callback(xk):
        _assign_flat(params, xk)
        with torch.enable_grad():
            loss, comps = loss_closure()
        callback_count[0] += 1
        _record(history, "L-BFGS-B", start_iter +
                callback_count[0], loss.detach(), comps, model)

    result = minimize(fun, x0, method="L-BFGS-B", jac=True, callback=callback,
                      options={"maxiter": int(max_iter), "maxfun": max(20, int(max_iter)*5),
                               "ftol": 1e-12, "gtol": 1e-8, "maxls": 20, "maxcor": 50})
    _assign_flat(params, result.x)
    return {"success": bool(result.success), "status": int(result.status),
            "message": str(result.message), "nit": int(result.nit),
            "nfev": int(result.nfev), "fun": float(result.fun)}


def train_inverse_model(name, model, D, args, weights, seed: int):
    rng = np.random.default_rng(seed+401)
    rng_lf = np.random.default_rng(seed+801)
    history = []
    t0 = time.perf_counter()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lf_s = D.get("lf_train_source_t")
    lf_c = D.get("lf_train_coords_t")
    lf_y = D.get("lf_train_values_t")

    # Adam: collocation and BC/IC points are re-sampled each epoch.
    for ep in range(1, args.adam_epochs+1):
        col = to_torch(sample_collocation(
            args.collocation, D["cfg"], rng), args.device)
        b_np, i_np = sample_boundary_initial(args.n_boundary, D["cfg"], rng)
        bc, ic = to_torch(b_np, args.device), to_torch(i_np, args.device)
        lf_batch = None
        if isinstance(model, MultiFidelityPIDeepONet):
            ids = rng_lf.integers(0, len(lf_y), size=min(
                args.lf_joint_batch, len(lf_y)))
            lf_batch = {"source": lf_s[ids],
                        "coords": lf_c[ids], "values": lf_y[ids]}
        loss, comps = compute_losses(model, D, col, bc, ic, weights, lf_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if ep == 1 or ep % max(1, args.adam_epochs//50) == 0 or ep == args.adam_epochs:
            _record(history, "Adam", ep, loss.detach(), comps, model)

    # L-BFGS-B: fixed HF observations, collocation, boundary/initial, and LF subset.
    col = to_torch(sample_collocation(
        args.collocation, D["cfg"], rng), args.device)
    b_np, i_np = sample_boundary_initial(args.n_boundary, D["cfg"], rng)
    bc, ic = to_torch(b_np, args.device), to_torch(i_np, args.device)
    lf_batch = None
    if isinstance(model, MultiFidelityPIDeepONet):
        ids = rng_lf.choice(len(lf_y), size=min(
            args.lf_lbfgs_batch, len(lf_y)), replace=False)
        lf_batch = {"source": lf_s[ids],
                    "coords": lf_c[ids], "values": lf_y[ids]}

    def fixed_closure():
        return compute_losses(model, D, col, bc, ic, weights, lf_batch)

    lbfgs_stats = scipy_lbfgsb(model, fixed_closure, args.lbfgs_max_iter,
                               history, start_iter=args.adam_epochs)
    seconds = time.perf_counter()-t0
    return history, seconds, lbfgs_stats


def _save_checkpoint(out: Path, name: str, model, seed: int, cfg,
                     u_scale: float, hyper: dict):
    payload = {
        "method": name, "seed": int(seed), "model_state": model.state_dict(),
        "u_scale": float(u_scale), "theta_bounds": PARAM_BOUNDS.tolist(),
        "coord_min": [0.0, 0.0, 0.0], "coord_max": [1.0, 1.0, cfg.T],
        "solver_config": cfg.__dict__, "hyperparameters": hyper,
    }
    if isinstance(model, MultiFidelityPIDeepONet):
        payload.update({
            "sensor_xy": model.sensor_xy.detach().cpu().numpy(),
            "lf_feature_coords": model.lf_feature_coords.detach().cpu().numpy(),
            "lf_sub_idx": model.lf_sub_idx.detach().cpu().numpy(),
            "sigma_s": model.sigma_s,
        })
    torch.save(payload, out/f"{name}.pt")


def _predict(model, coords, device, batch_size=4096):
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(coords), batch_size):
            z = to_torch(coords[start:start+batch_size], device)
            out.append(model(z).squeeze(-1).detach().cpu().numpy())
    return np.concatenate(out)


def evaluate_model(name, model, D, seconds, lbfgs_stats):
    pred = _predict(model, D["test_coords"], D["device"])*D["u_scale"]
    err = pred-D["test_values"]
    obs_pred = _predict(model, D["obs_coords_np"], D["device"])*D["u_scale"]
    obs_err = obs_pred-D["obs_values"]
    theta = model.theta_physical().detach().cpu().numpy()
    rel = np.abs(theta-D["theta_true"]) / \
        np.maximum(np.abs(D["theta_true"]), 1e-12)
    test_coords = D["test_coords"]
    interior_mask = ((test_coords[:, 0] > 1e-7) & (test_coords[:, 0] < 1.0-1e-7)
                     & (test_coords[:, 1] > 1e-7) & (test_coords[:, 1] < 1.0-1e-7)
                     & (test_coords[:, 2] > 1e-7))
    interior_err = err[interior_mask]
    interior_truth = D["test_values"][interior_mask]
    with torch.enable_grad():
        resid = pde_residual(model, D["eval_col"], D["u_scale"], D["cfg"])
    return {
        "method": name, "theta_Q": float(theta[0]), "theta_xs": float(theta[1]), "theta_ys": float(theta[2]),
        "relerr_Q": float(rel[0]), "relerr_xs": float(rel[1]), "relerr_ys": float(rel[2]),
        "param_relerr_mean": float(np.mean(rel)),
        "test_RMSE": float(np.sqrt(np.mean(err**2))),
        "test_MAE": float(np.mean(np.abs(err))),
        "test_relL2": float(np.linalg.norm(err)/(np.linalg.norm(D["test_values"])+1e-12)),
        "test_interior_points": int(interior_mask.sum()),
        "test_interior_RMSE": float(np.sqrt(np.mean(interior_err**2))),
        "test_interior_MAE": float(np.mean(np.abs(interior_err))),
        "test_interior_relL2": float(np.linalg.norm(interior_err)/(np.linalg.norm(interior_truth)+1e-12)),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "observation_RMSE": float(np.sqrt(np.mean(obs_err**2))),
        "PDE_residual_RMS_dimensionless": float(torch.sqrt(torch.mean(resid.detach()**2)).cpu()),
        "training_seconds": float(seconds),
        "lbfgsb_success": bool(lbfgs_stats["success"]),
        "lbfgsb_status": int(lbfgs_stats["status"]),
        "lbfgsb_message": lbfgs_stats["message"],
        "lbfgsb_nit": int(lbfgs_stats["nit"]),
        "lbfgsb_nfev": int(lbfgs_stats["nfev"]),
        "lbfgsb_final_loss": float(lbfgs_stats["fun"]),
    }


def train_experiment(data_file: str | Path, out_dir: str | Path, seed: int = 7,
                     model_choice: str = "both", adam_epochs: int = 2000,
                     lbfgs_max_iter: int = 5000, collocation: int = 256,
                     n_boundary: int = 128, lf_pretrain_epochs: int = 1000,
                     lf_batch_size: int = 1024, lf_joint_batch: int = 1024,
                     lf_lbfgs_batch: int = 2048, learning_rate: float = 1e-3,
                     device: str = "cpu", fast_dev: bool = False):
    if model_choice not in {"both", "pinn", "m1"}:
        raise ValueError("model_choice must be both, pinn, or m1")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    raw = np.load(data_file)
    data = {k: raw[k] for k in raw.files}
    cfg = ExperimentConfig(seed=seed)
    # Scaling uses only the common, noisy HF training observations (never test targets).
    u_scale = float(max(np.max(np.abs(data["obs_values"])), 1e-8))
    D = {
        "device": device, "cfg": cfg, "u_scale": u_scale,
        "obs_coords": to_torch(data["obs_coords"], device),
        "obs_coords_np": data["obs_coords"].astype(np.float32),
        "obs_y": to_torch(data["obs_values"][:, None]/u_scale, device),
        "obs_values": data["obs_values"].astype(np.float64),
        "test_coords": data["test_coords"].astype(np.float32),
        "test_values": data["test_values"].astype(np.float64),
        "plot_coords": data["plot_coords"].astype(np.float32),
        "plot_values": data["plot_values"].astype(np.float64),
        "theta_true": data["theta_true"].astype(np.float64),
        "eval_col": to_torch(sample_collocation(512, cfg, np.random.default_rng(seed+900)), device),
    }
    if model_choice in {"both", "m1"}:
        D["lf_train_source_t"] = to_torch(data["lf_train_source"], device)
        D["lf_train_coords_t"] = to_torch(data["lf_train_coords"], device)
        D["lf_train_values_t"] = to_torch(
            data["lf_train_values"]/u_scale, device)

    weights = dict(DEFAULT_LOSSES)
    if fast_dev:
        adam_epochs = min(adam_epochs, 2)
        lbfgs_max_iter = min(lbfgs_max_iter, 2)
        collocation = min(collocation, 16)
        n_boundary = min(n_boundary, 16)
        lf_pretrain_epochs = min(lf_pretrain_epochs, 2)
        lf_batch_size = min(lf_batch_size, 32)
        lf_joint_batch = min(lf_joint_batch, 32)
        lf_lbfgs_batch = min(lf_lbfgs_batch, 32)

    args = SimpleNamespace(adam_epochs=adam_epochs, lbfgs_max_iter=lbfgs_max_iter,
                           collocation=collocation, n_boundary=n_boundary,
                           lf_joint_batch=lf_joint_batch, lf_lbfgs_batch=lf_lbfgs_batch,
                           learning_rate=learning_rate, device=device)
    coord_min, coord_max = [0.0, 0.0, 0.0], [1.0, 1.0, cfg.T]
    theta_init = [0.90, 0.50, 0.50]
    rows, histories, models = [], {}, {}

    if model_choice in {"both", "pinn"}:
        seed_everything(seed+300)
        pinn = PINN(theta_init, coord_min, coord_max,
                    width=20, depth=8).to(device)
        hist, seconds, opt_stats = train_inverse_model(
            "PINN_Raissi", pinn, D, args, weights, seed+701)
        hyper = {"width": 20, "depth": 8, "optimizer": "Adam + SciPy L-BFGS-B",
                 "loss_weights": weights}
        _save_checkpoint(out, "PINN_Raissi", pinn, seed, cfg, u_scale, hyper)
        (out/"history_PINN_Raissi.json").write_text(json.dumps(hist,
                                                               indent=2), encoding="utf-8")
        pinn_row = evaluate_model("PINN_Raissi", pinn, D, seconds, opt_stats)
        pinn_row["lf_pretraining_seconds"] = np.nan
        pinn_row["joint_inverse_training_seconds"] = float(seconds)
        rows.append(pinn_row)
        histories["PINN_Raissi"], models["PINN_Raissi"] = hist, pinn

    if model_choice in {"both", "m1"}:
        lf_pretrain_t0 = time.perf_counter()
        seed_everything(seed+500)
        lf = ModifiedDeepONet(
            branch_dim=data["sensor_xy"].shape[0], coord_min=coord_min, coord_max=coord_max,
            latent_dim=64, hidden=30, depth=3, output_activation=False).to(device)
        lf_hist, lf_best_val = train_lf_operator(
            lf, data, u_scale, device, lf_pretrain_epochs, lf_batch_size, seed+501)
        lf_pretrain_seconds = time.perf_counter() - lf_pretrain_t0
        (out/"history_LF_operator.json").write_text(json.dumps(lf_hist,
                                                               indent=2), encoding="utf-8")
        m1 = MultiFidelityPIDeepONet(
            lf_model=lf, sensor_xy=data["sensor_xy"],
            lf_feature_coords=data["lf_feature_coords"], lf_sub_idx=data["lf_sub_idx"],
            theta_init=theta_init, coord_min=coord_min, coord_max=coord_max,
            sigma_s=cfg.sigma_s, linear_latent=5, linear_hidden=30,
            nonlinear_latent=20, nonlinear_hidden=20, nonlinear_depth=2).to(device)
        hist, seconds_joint, opt_stats = train_inverse_model(
            "M1_PI_DeepONet", m1, D, args, weights, seed+701)
        seconds_total = seconds_joint + lf_pretrain_seconds
        hyper = {"lf_kind": "ModifiedDeepONet", "lf_latent": 64, "lf_hidden": 30,
                 "lf_depth": 3, "linear_latent": 5, "nonlinear_latent": 20,
                 "nonlinear_hidden": 20, "nonlinear_depth": 2,
                 "lf_pretrain_epochs": lf_pretrain_epochs,
                 "lf_validation_mse_best": lf_best_val, "lf_pretraining_seconds": lf_pretrain_seconds,
                 "optimizer": "Adam + SciPy L-BFGS-B", "loss_weights": weights,
                 "note": "Composite MF PI-DeepONet Eq. (15) + task-specific HF observation loss."}
        _save_checkpoint(out, "M1_PI_DeepONet", m1, seed, cfg, u_scale, hyper)
        (out/"history_M1_PI_DeepONet.json").write_text(json.dumps(hist,
                                                                  indent=2), encoding="utf-8")
        row = evaluate_model("M1_PI_DeepONet", m1, D, seconds_total, opt_stats)
        row["lf_pretraining_seconds"] = float(lf_pretrain_seconds)
        row["joint_inverse_training_seconds"] = float(seconds_joint)
        row["lf_pretrain_best_val_MSE"] = float(lf_best_val)
        rows.append(row)
        histories["M1_PI_DeepONet"], models["M1_PI_DeepONet"] = hist, m1

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out/"results_seed.csv", index=False)
    metrics.to_csv(out/"exp1_comparison_seed.csv", index=False)
    (out/"results_seed.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    dataset_meta_path = Path(data_file).parent/"dataset_metadata.json"
    dataset_meta = json.loads(dataset_meta_path.read_text(
        encoding="utf-8")) if dataset_meta_path.exists() else {}
    timing_meta = dataset_meta.get("timing", {})
    for row in rows:
        row["lf_solver_generation_seconds"] = (float(timing_meta.get("lf_solver_seconds", np.nan))
                                               if row["method"] == "M1_PI_DeepONet" else np.nan)
        row["shared_hf_solver_generation_seconds"] = float(
            timing_meta.get("hf_solver_seconds", np.nan))
        row["end_to_end_method_seconds_excluding_shared_HF_data"] = (
            float(row["training_seconds"]) +
            float(timing_meta.get("lf_solver_seconds", 0.0))
            if row["method"] == "M1_PI_DeepONet" else float(row["training_seconds"]))

    # Rebuild tables after attaching the separately measured LF solver cost.
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out/"results_seed.csv", index=False)
    metrics.to_csv(out/"exp1_comparison_seed.csv", index=False)
    (out/"results_seed.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    metadata = {
        "seed": seed, "model_choice": model_choice, "u_scale_from_HF_training_observations_only": u_scale,
        "optimizer": "Adam + SciPy L-BFGS-B", "adam_epochs": adam_epochs,
        "lbfgs_max_iter": lbfgs_max_iter, "collocation_points": collocation,
        "boundary_initial_points_per_epoch": n_boundary,
        "loss_weights": weights,
        "HF_grid": [cfg.n_hf, cfg.n_hf], "LF_grid": [cfg.n_lf, cfg.n_lf],
        "HF_scheme": "central", "LF_scheme": str(dataset_meta.get("LF", {}).get("scheme", "central")),
        "n_HF_observations": int(len(data["obs_values"])),
        "n_HF_test_points_disjoint": int(len(data["test_values"])),
        "interior_test_metric_definition": "x,y strictly inside (0,1), t>0; excludes analytically prescribed BC and IC points",
        "data_generation_timing": timing_meta,
        "test_targets_used_for_scaling_or_early_stopping": False,
        "M1_composite_formula": "u_HF = F_linear + F_nonlinear",
        "M1_loss": "lambda_obs*L_HFobs + Eq.(15): lambda1*L_physics + lambda2*L_LF + lambda3*R_nl + lambda4*R_LF + lambda5*L_IC + lambda6*L_BC",
        "PI_DeepONet_reference": "Howard et al. 2023, JCP 493, 112462, Eq. (15); modified DeepONet based on Wang et al.",
        "pinn_reference": "Raissi et al. 2019, JCP 378, 686-707; fully-connected tanh baseline adapted to 2D+time inverse source problem",
    }
    (out/"training_metadata.json").write_text(json.dumps(metadata,
                                                         indent=2, ensure_ascii=False), encoding="utf-8")

    # Save both metric test predictions and a separate structured visualization grid.
    pred_payload = {
        "test_coords": D["test_coords"], "test_truth": D["test_values"]}
    plot_payload = {
        "plot_coords": D["plot_coords"], "plot_truth": D["plot_values"]}
    for name, model_obj in models.items():
        pred_payload[name] = _predict(
            model_obj, D["test_coords"], device)*u_scale
        plot_payload[name] = _predict(
            model_obj, D["plot_coords"], device)*u_scale
    np.savez_compressed(out/"predictions_test.npz", **pred_payload)
    np.savez_compressed(out/"predictions_plot.npz", **plot_payload,
                        plot_times=data["test_times"], plot_x_axis=data["test_x_axis"],
                        plot_y_axis=data["test_y_axis"])
    from make_report import build_report
    build_report(out, dataset=str(data_file), device=device)
    return metrics


def main():
    ap = argparse.ArgumentParser(
        description="Compare PINN-Raissi and M1 PI-DeepONet")
    ap.add_argument("--model", choices=["both", "pinn", "m1"], default="both")
    ap.add_argument("--data", default=None)
    ap.add_argument("--output", default="results_exp1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--adam-epochs", type=int, default=2000)
    ap.add_argument("--lbfgs-max-iter", type=int, default=5000)
    ap.add_argument("--collocation", type=int, default=256)
    ap.add_argument("--n-boundary", type=int, default=128)
    ap.add_argument("--lf-pretrain-epochs", type=int, default=1000)
    ap.add_argument("--lf-batch-size", type=int, default=1024)
    ap.add_argument("--lf-joint-batch", type=int, default=1024)
    ap.add_argument("--lf-lbfgs-batch", type=int, default=2048)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data_file = Path(args.data) if args.data else out / \
        f"exp1_multifidelity_seed_{args.seed}.npz"
    if not data_file.exists():
        data_file = generate_datasets(out, seed=args.seed)
    df = train_experiment(
        data_file, out, seed=args.seed, model_choice=args.model,
        adam_epochs=args.adam_epochs, lbfgs_max_iter=args.lbfgs_max_iter,
        collocation=args.collocation, n_boundary=args.n_boundary,
        lf_pretrain_epochs=args.lf_pretrain_epochs, lf_batch_size=args.lf_batch_size,
        lf_joint_batch=args.lf_joint_batch, lf_lbfgs_batch=args.lf_lbfgs_batch,
        learning_rate=args.learning_rate, device=args.device)
    print(df.to_string(index=False))
    print(f"Results saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
