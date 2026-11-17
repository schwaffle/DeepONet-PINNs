# Implementation status — PINN-Raissi vs M1 PI-DeepONet

## What is included

- Main neural methods only: `PINN_Raissi` and `M1_PI_DeepONet`. MPINN-v2/M4 is absent from active models, training, CLI, inference, reports and 10-seed orchestration.
- M1: LF modified DeepONet (source-function operator) + linear HF DeepONet correction + nonlinear HF modified DeepONet correction; `u_HF = F_l + F_nl`.
- Modified DeepONet gates use `tanh`, matching the encoder/gating equations; do not interpret these as convex sigmoid gates.
- PINN baseline: 3 physical inputs `(x,y,t)`, 8 hidden layers × 20 `tanh` units, scalar concentration output, truncated-normal Xavier-style initialization and zero biases; source parameters are trainable bounded variables in the PDE forcing.
- Howard Eq. (15): six terms retained. A seventh explicit HF observation term is added for sparse inverse source identification and is also used by PINN.
- Same noisy HF observations and same per-seed collocation / BC / IC random samples for both methods. LF pretraining and LF solver cost are separately reported for M1.
- LF train/validation split by full source realization; true target source held out from LF parameter sweep.
- `u_scale` is derived from noisy HF training observations only. Clean test values are never used for scaling, tuning, validation or early stopping.
- Full clean test is all saved HF grid/time nodes excluding exact observation coordinate-time pairs. Additional interior-only metrics exclude the analytically imposed boundary and initial-condition nodes.

## Software checks actually executed

- `python -m py_compile *.py` — passed.
- `python smoke_test.py --output smoke_test_outputs` — passed. The test uses a tiny 2-epoch Adam + 2-iteration L-BFGS-B run and checks: data split/leakage, coordinate normalization chain rule, PINN/M1 training path, PDE automatic differentiation, checkpoint save/load and inference.
- `python run_exp1.py --seed 7 --model both --fast-dev --output cli_fast_dev_check_v3` — passed as a CLI smoke check. This also uses only 2 epochs / 2 L-BFGS-B iterations.
- `python run_exp1.py --help` and `python run_10seeds.py --help` — passed.

## What has NOT been executed

- No full-epoch scientific run for seed 7.
- No 10-seed series; no mean/std scientific table exists yet.
- No convergence conclusion for Adam/L-BFGS-B; the fast-development checks intentionally stop at two L-BFGS-B iterations.
- No WRF-Chem connection, WRF-Chem field mapping, or runtime validation. The HF source is still the local 129×129 central finite-difference solver.
- No full cross-scheme study. The CLI can generate an LF-upwind / HF-central variant, but it must be treated as a separate experiment from the central/central resolution comparison.

All numbers in `smoke_test_outputs/` and `cli_fast_dev_check_v3/` are software-test artifacts only and must not be cited as scientific results.
