"""Load a trained PINN-Raissi or composite M1 PI-DeepONet checkpoint."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from models import PINN, ModifiedDeepONet, MultiFidelityPIDeepONet
from solver import ExperimentConfig
from datasets import PARAM_BOUNDS


def load_model(checkpoint: str | Path, device: str = "cpu"):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ExperimentConfig(**payload["solver_config"])
    cmin, cmax = payload["coord_min"], payload["coord_max"]
    method = payload["method"]
    hp = payload.get("hyperparameters", {})
    theta0 = [0.9, 0.5, 0.5]
    if method == "PINN_Raissi":
        model = PINN(theta0, cmin, cmax,
                     width=int(hp.get("width", 20)), depth=int(hp.get("depth", 8)))
    elif method == "M1_PI_DeepONet":
        sensor_xy = payload["sensor_xy"]
        feature_coords = payload["lf_feature_coords"]
        sub_idx = payload["lf_sub_idx"]
        lf = ModifiedDeepONet(
            branch_dim=len(sensor_xy), coord_min=cmin, coord_max=cmax,
            latent_dim=int(hp.get("lf_latent", 64)),
            hidden=int(hp.get("lf_hidden", 30)),
            depth=int(hp.get("lf_depth", 3)), output_activation=False)
        model = MultiFidelityPIDeepONet(
            lf_model=lf, sensor_xy=sensor_xy, lf_feature_coords=feature_coords,
            lf_sub_idx=sub_idx, theta_init=theta0,
            coord_min=cmin, coord_max=cmax,
            sigma_s=float(payload.get("sigma_s", 0.10)),
            linear_latent=int(hp.get("linear_latent", 5)),
            linear_hidden=int(hp.get("linear_hidden", 30)),
            nonlinear_latent=int(hp.get("nonlinear_latent", 20)),
            nonlinear_hidden=int(hp.get("nonlinear_hidden", 20)),
            nonlinear_depth=int(hp.get("nonlinear_depth", 2)))
    else:
        raise ValueError(f"Unknown checkpoint method: {method}")
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, float(payload["u_scale"]), cfg, np.asarray(payload.get("theta_bounds", PARAM_BOUNDS))


def set_source_parameters(model, theta, bounds):
    theta = np.asarray(theta, dtype=np.float64)
    lo, hi = bounds[:, 0], bounds[:, 1]
    p = np.clip((theta-lo)/(hi-lo), 1e-6, 1-1e-6)
    raw = np.log(p/(1-p))
    with torch.no_grad():
        model.theta_raw.copy_(torch.as_tensor(raw, dtype=model.theta_raw.dtype,
                                              device=model.theta_raw.device))


def predict_points(model, coords: np.ndarray, u_scale: float, device="cpu", batch_size=4096):
    coords = np.asarray(coords, dtype=np.float32)
    preds = []
    with torch.no_grad():
        for start in range(0, len(coords), batch_size):
            z = torch.as_tensor(coords[start:start+batch_size], dtype=torch.float32, device=device)
            preds.append(model(z).squeeze(-1).cpu().numpy())
    return np.concatenate(preds)*u_scale


def main():
    ap = argparse.ArgumentParser(description="Predict concentration with a trained model")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--coords", required=True, help="NPZ containing coords array [N,3] in physical units")
    ap.add_argument("--output", default="prediction.npy")
    ap.add_argument("--theta", nargs=3, type=float, default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    model, u_scale, cfg, bounds = load_model(args.checkpoint, args.device)
    if args.theta is not None:
        set_source_parameters(model, args.theta, bounds)
    with np.load(args.coords) as raw:
        coords = raw["coords"]
    pred = predict_points(model, coords, u_scale, args.device)
    np.save(args.output, pred)
    print(f"Saved predictions: {Path(args.output).resolve()}")
    print("Identified source parameters:", model.theta_physical().detach().cpu().numpy())


if __name__ == "__main__":
    main()
