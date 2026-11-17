"""Small end-to-end software test. Results are not research results."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from datasets import generate_datasets
from train import train_experiment
from inference import load_model, predict_points


def check_coordinate_chain_rule():
    """Analytic derivative test for the physical->[-1,1] coordinate map."""
    from models import normalize_coords
    z = torch.tensor([[0.23, 0.61, 0.17]],
                     dtype=torch.float64, requires_grad=True)
    lo = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    hi = torch.tensor([1.0, 1.0, 0.5], dtype=torch.float64)
    zn = normalize_coords(z, lo, hi)
    f = zn[:, 0:1]**2 + 0.5*zn[:, 1:2]**2 + 2.0*zn[:, 2:3]
    g = torch.autograd.grad(f.sum(), z, create_graph=True)[0]
    d2x = torch.autograd.grad(g[:, 0].sum(), z, create_graph=True)[0][:, 0]
    d2y = torch.autograd.grad(g[:, 1].sum(), z, create_graph=True)[0][:, 1]
    expected_dx = 4.0*zn[:, 0]/(hi[0]-lo[0])
    expected_dy = 2.0*zn[:, 1]/(hi[1]-lo[1])
    expected_dt = torch.full_like(g[:, 2], 4.0/(hi[2]-lo[2]))
    expected_d2x = torch.full_like(d2x, 8.0/(hi[0]-lo[0])**2)
    expected_d2y = torch.full_like(d2y, 4.0/(hi[1]-lo[1])**2)
    for got, expected in [(g[:, 0], expected_dx), (g[:, 1], expected_dy),
                          (g[:, 2], expected_dt), (d2x, expected_d2x),
                          (d2y, expected_d2y)]:
        if not torch.allclose(got, expected, atol=1e-10, rtol=1e-10):
            raise AssertionError(
                f"Coordinate chain-rule mismatch: got={got}, expected={expected}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="smoke_test_outputs")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(1)
    check_coordinate_chain_rule()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data_path = generate_datasets(out, seed=7, n_lf_parameters=6,
                                  points_per_parameter=32, n_test_axis=9)
    with np.load(data_path) as data:
        assert len(data["obs_values"]) == 80
        assert len(data["test_values"]) == 21*129*129 - 80
        obs_keys = {tuple(np.round(z, 7)) for z in data["obs_coords"]}
        test_keys = {tuple(np.round(z, 7)) for z in data["test_coords"]}
        assert obs_keys.isdisjoint(
            test_keys), "Observation coordinates leaked into clean test."
        assert not np.any(
            np.all(np.isclose(data["lf_parameters_train"], data["theta_true"]), axis=1))
    from models import PINN
    init_model = PINN([0.9, 0.5, 0.5], [0, 0, 0], [1, 1, 0.5])
    n_tanh_init = sum(isinstance(m, torch.nn.Tanh)
                      for m in init_model.net.modules())
    assert n_tanh_init == 8, f"Expected 8 tanh hidden layers, got {n_tanh_init}"
    for layer in init_model.net.modules():
        if isinstance(layer, torch.nn.Linear):
            assert torch.count_nonzero(layer.bias).item(
            ) == 0, "PINN biases must initialize to zero"
    metrics = train_experiment(
        data_file=data_path, out_dir=out, seed=7, model_choice="both",
        adam_epochs=2, lbfgs_max_iter=2, collocation=16, n_boundary=16,
        lf_pretrain_epochs=2, lf_batch_size=32, lf_joint_batch=32,
        lf_lbfgs_batch=32, learning_rate=1e-3, device=args.device, fast_dev=True)
    expected = {"PINN_Raissi", "M1_PI_DeepONet"}
    if set(metrics["method"]) != expected:
        raise AssertionError(
            f"Expected methods {expected}; got {set(metrics['method'])}")
    query = np.asarray([[0.4, 0.4, 0.1], [0.5, 0.5, 0.25]], dtype=np.float32)
    for method in expected:
        ckpt = out / f"{method}.pt"
        if not ckpt.exists():
            raise AssertionError(f"Checkpoint missing: {ckpt}")
        model, scale, cfg, bounds = load_model(ckpt, args.device)
        pred = predict_points(model, query, scale, args.device)
        if pred.shape != (len(query),) or not np.all(np.isfinite(pred)):
            raise AssertionError(
                f"Invalid inference output for {method}: {pred}")
        # Exercise the PDE automatic-differentiation path after checkpoint reload.
        from train import pde_residual, sample_collocation
        from solver import ExperimentConfig
        z = torch.as_tensor(sample_collocation(8, ExperimentConfig(), np.random.default_rng(123)),
                            dtype=torch.float32, device=args.device)
        residual = pde_residual(model, z, scale, cfg)
        if not torch.isfinite(residual).all():
            raise AssertionError(f"Non-finite PDE residual for {method}")
    print("SMOKE TEST PASSED: data split/leakage, analytic coordinate chain rule, PINN/M1 training, PDE autograd, checkpoints, reload and inference.")
    print("This tiny run is a software check only; it is not a scientific benchmark.")


if __name__ == "__main__":
    main()
