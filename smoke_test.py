"""Небольшой сквозной (end-to-end) программный тест.
Полученные результаты не являются научными или исследовательскими."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from datasets import generate_datasets
from inference import load_model, predict_points
from train import train_experiment


def check_coordinate_chain_rule():
    """
    Аналитическая проверка вычисления первых и вторых производных
    для преобразования физических координат в диапазон [-1, 1].
    """
    from models import normalize_coords

    z = torch.tensor([[0.23, 0.61, 0.17]], dtype=torch.float64, requires_grad=True)
    lo = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    hi = torch.tensor([1.0, 1.0, 0.5], dtype=torch.float64)

    zn = normalize_coords(z, lo, hi)

    # Тестовая функция для проверки градиентов
    f = zn[:, 0:1]**2 + 0.5*zn[:, 1:2]**2 + 2.0*zn[:, 2:3]

    # Первые производные
    g = torch.autograd.grad(f.sum(), z, create_graph=True)[0]

    # Вторые производные
    d2x = torch.autograd.grad(g[:, 0].sum(), z, create_graph=True)[0][:, 0]
    d2y = torch.autograd.grad(g[:, 1].sum(), z, create_graph=True)[0][:, 1]

    # Ожидаемые аналитические значения
    expected_dx = 4.0 * zn[:, 0] / (hi[0] - lo[0])
    expected_dy = 2.0 * zn[:, 1] / (hi[1] - lo[1])
    expected_dt = torch.full_like(g[:, 2], 4.0 / (hi[2] - lo[2]))
    expected_d2x = torch.full_like(d2x, 8.0 / (hi[0] - lo[0])**2)
    expected_d2y = torch.full_like(d2y, 4.0 / (hi[1] - lo[1])**2)

    for got, expected in [
        (g[:, 0], expected_dx),
        (g[:, 1], expected_dy),
        (g[:, 2], expected_dt),
        (d2x, expected_d2x),
        (d2y, expected_d2y)
    ]:
        if not torch.allclose(got, expected, atol=1e-10, rtol=1e-10):
            raise AssertionError(
                f"Несоответствие цепного правила для координат: got={got}, expected={expected}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="smoke_test_outputs")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # Ограничиваем число потоков для детерминированности и стабильности тестов
    torch.set_num_threads(1)

    # Проверка математической корректности нормализации координат
    check_coordinate_chain_rule()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Генерация крошечного датасета для быстрой проверки пайплайна
    data_path = generate_datasets(
        out, seed=7, n_lf_parameters=6,
        points_per_parameter=32, n_test_axis=9
    )

    # Проверка целостности данных и отсутствия утечек
    with np.load(data_path) as data:
        assert len(data["obs_values"]) == 80, "Ожидается ровно 80 наблюдений"

        # Проверка размера чистого тестового набора (все точки HF сетки минус точки наблюдений)
        assert len(data["test_values"]) == 21 * 129 * 129 - 80

        obs_keys = {tuple(np.round(z, 7)) for z in data["obs_coords"]}
        test_keys = {tuple(np.round(z, 7)) for z in data["test_coords"]}

        assert obs_keys.isdisjoint(test_keys), \
            "Координаты наблюдений попали в чистый тестовый набор (утечка данных)."

        assert not np.any(
            np.all(np.isclose(data["lf_parameters_train"], data["theta_true"]), axis=1)
        ), "Истинный источник не должен присутствовать в обучающей выборке LF."

    # Проверка архитектуры и инициализации PINN
    from models import PINN
    init_model = PINN([0.9, 0.5, 0.5], [0, 0, 0], [1, 1, 0.5])

    n_tanh_init = sum(isinstance(m, torch.nn.Tanh) for m in init_model.net.modules())
    assert n_tanh_init == 8, f"Ожидается 8 скрытых слоев tanh, получено {n_tanh_init}"

    for layer in init_model.net.modules():
        if isinstance(layer, torch.nn.Linear):
            assert torch.count_nonzero(layer.bias).item() == 0, \
                "Смещения (biases) в слоях PINN должны инициализироваться нулями"

    # Запуск сверхкороткого обучения для проверки пайплайна
    metrics = train_experiment(
        data_file=data_path, out_dir=out, seed=7, model_choice="both",
        adam_epochs=2, lbfgs_max_iter=2, collocation=16, n_boundary=16,
        lf_pretrain_epochs=2, lf_batch_size=32, lf_joint_batch=32,
        lf_lbfgs_batch=32, learning_rate=1e-3, device=args.device, fast_dev=True
    )

    expected_methods = {"PINN_Raissi", "M1_PI_DeepONet"}
    if set(metrics["method"]) != expected_methods:
        raise AssertionError(
            f"Ожидались методы {expected_methods}; получены {set(metrics['method'])}"
        )

    # Проверка сохранения, загрузки чекпоинтов и инференса
    query = np.asarray([[0.4, 0.4, 0.1], [0.5, 0.5, 0.25]], dtype=np.float32)

    for method in expected_methods:
        ckpt = out / f"{method}.pt"
        if not ckpt.exists():
            raise AssertionError(f"Чекпоинт отсутствует: {ckpt}")

        model, scale, cfg, bounds = load_model(ckpt, args.device)
        pred = predict_points(model, query, scale, args.device)

        if pred.shape != (len(query),) or not np.all(np.isfinite(pred)):
            raise AssertionError(f"Некорректный вывод при инференсе для {method}: {pred}")

        # Проверяем путь автоматического дифференцирования PDE после перезагрузки модели
        from solver import ExperimentConfig
        from train import pde_residual, sample_collocation

        z = torch.as_tensor(
            sample_collocation(8, ExperimentConfig(), np.random.default_rng(123)),
            dtype=torch.float32, device=args.device
        )
        residual = pde_residual(model, z, scale, cfg)

        if not torch.isfinite(residual).all():
            raise AssertionError(f"Неконечное (NaN/Inf) значение невязки PDE для {method}")

    print("SMOKE TEST ПРОЙДЕН: разделение данных/отсутствие утечек, аналитическая проверка ")
    print("цепного правила для координат, обучение PINN/M1, автодифференцирование PDE, ")
    print("сохранение/загрузка чекпоинтов и инференс.")
    print("Этот крошечный запуск является только программной проверкой; это не научный бенчмарк.")


if __name__ == "__main__":
    main()
