"""Загрузка обученной модели PINN-Raissi или составной M1 PI-DeepONet из чекпоинта."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from datasets import PARAM_BOUNDS
from models import PINN, ModifiedDeepONet, MultiFidelityPIDeepONet
from solver import ExperimentConfig


def load_model(checkpoint: str | Path, device: str = "cpu"):
    """
    Загружает обученную модель из файла чекпоинта.

    Восстанавливает архитектуру сети на основе метаданных, сохранённых
    при обучении, загружает веса и переводит модель в режим eval.
    Возвращает модель, масштабный коэффициент u_scale, конфигурацию
    солвера и границы параметров источника.
    """
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ExperimentConfig(**payload["solver_config"])
    cmin, cmax = payload["coord_min"], payload["coord_max"]
    method = payload["method"]
    hp = payload.get("hyperparameters", {})
    theta0 = [0.9, 0.5, 0.5]

    if method == "PINN_Raissi":
        model = PINN(theta0, cmin, cmax,
                     width=int(hp.get("width", 20)),
                     depth=int(hp.get("depth", 8)))
    elif method == "M1_PI_DeepONet":
        sensor_xy = payload["sensor_xy"]
        feature_coords = payload["lf_feature_coords"]
        sub_idx = payload["lf_sub_idx"]
        lf = ModifiedDeepONet(
            branch_dim=len(sensor_xy), coord_min=cmin, coord_max=cmax,
            latent_dim=int(hp.get("lf_latent", 64)),
            hidden=int(hp.get("lf_hidden", 30)),
            depth=int(hp.get("lf_depth", 3)),
            output_activation=False,
        )
        model = MultiFidelityPIDeepONet(
            lf_model=lf, sensor_xy=sensor_xy,
            lf_feature_coords=feature_coords, lf_sub_idx=sub_idx,
            theta_init=theta0,
            coord_min=cmin, coord_max=cmax,
            sigma_s=float(payload.get("sigma_s", 0.10)),
            linear_latent=int(hp.get("linear_latent", 5)),
            linear_hidden=int(hp.get("linear_hidden", 30)),
            nonlinear_latent=int(hp.get("nonlinear_latent", 20)),
            nonlinear_hidden=int(hp.get("nonlinear_hidden", 20)),
            nonlinear_depth=int(hp.get("nonlinear_depth", 2)),
        )
    else:
        raise ValueError(f"Неизвестный метод в чекпоинте: {method}")

    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, float(payload["u_scale"]), cfg, np.asarray(
        payload.get("theta_bounds", PARAM_BOUNDS)
    )


def set_source_parameters(model, theta, bounds):
    """
    Устанавливает физические параметры источника (Q, xs, ys) в модели.

    Преобразует физические значения в unconstrained raw-параметры
    через обратную сигмоиду (logit) и записывает их напрямую в модель.
    Используется для ручной подстановки известных параметров при инференсе.
    """
    theta = np.asarray(theta, dtype=np.float64)
    lo, hi = bounds[:, 0], bounds[:, 1]
    p = np.clip((theta - lo) / (hi - lo), 1e-6, 1 - 1e-6)
    raw = np.log(p / (1 - p))
    with torch.no_grad():
        model.theta_raw.copy_(torch.as_tensor(
            raw, dtype=model.theta_raw.dtype,
            device=model.theta_raw.device,
        ))


def predict_points(model, coords: np.ndarray, u_scale: float,
                   device: str = "cpu", batch_size: int = 4096):
    """
    Пакетное предсказание концентрации в заданных точках.

    Разбивает входные координаты на батчи для экономии памяти,
    пропускает через модель в режиме no_grad и возвращает
    физические значения концентрации (умноженные на u_scale).
    """
    coords = np.asarray(coords, dtype=np.float32)
    preds = []
    with torch.no_grad():
        for start in range(0, len(coords), batch_size):
            z = torch.as_tensor(
                coords[start:start + batch_size],
                dtype=torch.float32, device=device,
            )
            preds.append(model(z).squeeze(-1).cpu().numpy())
    return np.concatenate(preds) * u_scale


def main():
    """Точка входа CLI для инференса обученной модели."""
    ap = argparse.ArgumentParser(
        description="Предсказание концентрации обученной моделью"
    )
    ap.add_argument(
        "--checkpoint", required=True,
        help="Путь к файлу чекпоинта (.pt) обученной модели",
    )
    ap.add_argument(
        "--coords", required=True,
        help="NPZ-файл с массивом coords формы [N,3] в физических единицах (x,y,t)",
    )
    ap.add_argument(
        "--output", default="prediction.npy",
        help="Путь для сохранения предсказаний (по умолчанию: prediction.npy)",
    )
    ap.add_argument(
        "--theta", nargs=3, type=float, default=None,
        help="Необязательная подмена параметров источника Q, xs, ys",
    )
    ap.add_argument(
        "--device", default="cpu",
        help="Устройство для вычислений ('cpu' или 'cuda')",
    )
    args = ap.parse_args()

    model, u_scale, cfg, bounds = load_model(args.checkpoint, args.device)
    if args.theta is not None:
        set_source_parameters(model, args.theta, bounds)

    with np.load(args.coords) as raw:
        coords = raw["coords"]

    pred = predict_points(model, coords, u_scale, args.device)
    np.save(args.output, pred)
    print(f"Предсказания сохранены: {Path(args.output).resolve()}")
    print("Идентифицированные параметры источника:",
          model.theta_physical().detach().cpu().numpy())


if __name__ == "__main__":
    main()
