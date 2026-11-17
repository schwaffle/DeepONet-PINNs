"""Run one reproducible seed of Exp. 1: PINN-Raissi vs M1 PI-DeepONet."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from datasets import generate_datasets
from train import train_experiment


def run_single(seed: int, output: str | Path, model: str = "both", device: str = "cpu",
               adam_epochs: int = 2000, lbfgs_max_iter: int = 5000,
               lf_pretrain_epochs: int = 1000, collocation: int = 256,
               n_boundary: int = 128, n_lf_parameters: int = 40,
               points_per_parameter: int = 384, n_test_axis: int = 33,
               learning_rate: float = 1e-3, fast_dev: bool = False,
               lf_scheme: str = "central"):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    data_file = out / f"exp1_multifidelity_seed_{seed}.npz"
    if not data_file.exists():
        data_file = generate_datasets(out, seed=seed,
            n_lf_parameters=n_lf_parameters,
            points_per_parameter=points_per_parameter,
            n_test_axis=n_test_axis,
            lf_scheme=lf_scheme)
    return train_experiment(
        data_file=data_file, out_dir=out, seed=seed, model_choice=model,
        adam_epochs=adam_epochs, lbfgs_max_iter=lbfgs_max_iter,
        collocation=collocation, n_boundary=n_boundary,
        lf_pretrain_epochs=lf_pretrain_epochs,
        learning_rate=learning_rate, device=device, fast_dev=fast_dev)


def main():
    p = argparse.ArgumentParser(description="Run Exp. 1: PINN-Raissi vs M1 PI-DeepONet")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--model", choices=["both", "pinn", "m1"], default="both")
    p.add_argument("--output", default="results_exp1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--adam-epochs", type=int, default=2000)
    p.add_argument("--lbfgs-max-iter", type=int, default=5000)
    p.add_argument("--lf-pretrain-epochs", type=int, default=1000)
    p.add_argument("--collocation", type=int, default=256)
    p.add_argument("--n-boundary", type=int, default=128)
    p.add_argument("--lf-parameters", type=int, default=40)
    p.add_argument("--points-per-parameter", type=int, default=384)
    p.add_argument("--n-test-axis", type=int, default=33)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--fast-dev", action="store_true", help="tiny smoke run; not scientific")
    p.add_argument("--lf-scheme", choices=["central", "upwind"], default="central",
                   help="LF advection scheme; HF remains central (cross-scheme experiment)")
    a = p.parse_args()
    torch.set_num_threads(1)
    df = run_single(a.seed, a.output, a.model, a.device,
                    a.adam_epochs, a.lbfgs_max_iter, a.lf_pretrain_epochs,
                    a.collocation, a.n_boundary, a.lf_parameters,
                    a.points_per_parameter, a.n_test_axis,
                    a.learning_rate, a.fast_dev, a.lf_scheme)
    print(df.to_string(index=False))
    print(f"Results: {Path(a.output).resolve()}")


if __name__ == "__main__":
    main()
