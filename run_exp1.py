"""
Запуск одного воспроизводимого сида (seed) Эксперимента 1:
сравнение PINN-Raissi и составной M1 PI-DeepONet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from datasets import generate_datasets
from train import train_experiment


def run_single(
    seed: int,
    output: str | Path,
    model: str = "both",
    device: str = "cpu",
    adam_epochs: int = 2000,
    lbfgs_max_iter: int = 5000,
    lf_pretrain_epochs: int = 1000,
    collocation: int = 256,
    n_boundary: int = 128,
    n_lf_parameters: int = 40,
    points_per_parameter: int = 384,
    n_test_axis: int = 33,
    learning_rate: float = 1e-3,
    fast_dev: bool = False,
    lf_scheme: str = "central",
):
    """
    Запускает процесс генерации данных и обучения для одного конкретного сида.

    Если файл датасета не существует, он генерируется с заданными параметрами.
    """
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    # Явное именование файла датасета для удобства отслеживания конфигурации
    data_file = out / f"exp1_multifidelity_seed_{seed}_lf_{lf_scheme}_hf_central.npz"

    if not data_file.exists():
        data_file = generate_datasets(
            out_dir=out,
            seed=seed,
            n_lf_parameters=n_lf_parameters,
            points_per_parameter=points_per_parameter,
            n_test_axis=n_test_axis,
            lf_scheme=lf_scheme,
        )

    return train_experiment(
        data_file=data_file,
        out_dir=out,
        seed=seed,
        model_choice=model,
        adam_epochs=adam_epochs,
        lbfgs_max_iter=lbfgs_max_iter,
        collocation=collocation,
        n_boundary=n_boundary,
        lf_pretrain_epochs=lf_pretrain_epochs,
        learning_rate=learning_rate,
        device=device,
        fast_dev=fast_dev,
    )


def main():
    """Точка входа CLI для настройки и запуска эксперимента."""
    p = argparse.ArgumentParser(
        description="Запуск Эксперимента 1: сравнение PINN-Raissi и M1 PI-DeepONet"
    )
    p.add_argument(
        "--seed", type=int, default=7,
        help="Случайное зерно (seed) для воспроизводимости результатов."
    )
    p.add_argument(
        "--model", choices=["both", "pinn", "m1"], default="both",
        help="Выбор модели для обучения: 'both' (обе), 'pinn' или 'm1'."
    )
    p.add_argument(
        "--output", default="results_exp1",
        help="Директория для сохранения всех артефактов эксперимента."
    )
    p.add_argument(
        "--device", default="cpu",
        help="Устройство для вычислений ('cpu' или 'cuda')."
    )
    p.add_argument(
        "--adam-epochs", type=int, default=2000,
        help="Количество эпох оптимизатора Adam."
    )
    p.add_argument(
        "--lbfgs-max-iter", type=int, default=5000,
        help="Максимальное количество итераций оптимизатора L-BFGS-B."
    )
    p.add_argument(
        "--lf-pretrain-epochs", type=int, default=1000,
        help="Количество эпох предварительного обучения (pretraining) LF-оператора."
    )
    p.add_argument(
        "--collocation", type=int, default=256,
        help="Количество коллокационных точек для вычисления невязки PDE на каждой эпохе."
    )
    p.add_argument(
        "--n-boundary", type=int, default=128,
        help="Количество точек для граничных (BC) и начальных (IC) условий на каждой эпохе Adam."
    )
    p.add_argument(
        "--lf-parameters", type=int, default=40,
        help="Количество различных реализаций источника в LF-ансамбле."
    )
    p.add_argument(
        "--points-per-parameter", type=int, default=384,
        help="Количество точек запроса (query points) на одну реализ
