# Audit of the uploaded project and implementation gaps

## Executive assessment

The uploaded source project was a mixed PINN/MPINN-v2 prototype with several M1 architecture components. It did **not** yet implement the requested PINN-vs-DeepONet scientific comparison end to end. The user's concern was correct: M1 did not enter the training run, result table, report or checkpoint pipeline.

## What was present

- `models.py`: vanilla `DeepONet`, `ModifiedDeepONet`, `MultiFidelityPIDeepONet` M1 class, `MPINNv2` and `PINN` classes were present.
- `datasets.py`: source-sensor functions, LF source ensemble, source-level train/validation split, HF noisy observations and clean test data were prepared.
- `train.py`: LF surrogate training was only for MPINN-v2; the actual selected-model branches were M4 and PINN. There was no M1 LF DeepONet pretraining and no composite M1 joint training branch.
- `make_report.py`: text/plot labels referenced M1, but plotting depends on prediction/checkpoint artifacts. Without M1 training artifacts, it could not show M1 results.
- `inference.py`: a loader branch for M1 was present in part, but a checkpoint was not being produced by training.
- `run_exp1.py` and `smoke_test.py`: choices did not include M1.

## Main corrections in this rewrite

- Implemented LF DeepONet pretraining on multiple independent Gaussian source functions, holding out full source functions for validation.
- Integrated M1 composite `F_l + F_nl` into the training path.
- Implemented Eq. (15) loss components and added an explicit HF-observation loss for inverse parameter identification.
- Kept PINN and M1 as the only main neural methods; MPINN-v2 is removed from the active experiment.
- Added checkpoint save/load and inference for both methods.
- Added clean HF test points across all saved HF grid-time nodes, removing exact observation coordinates; separate visualisation grid is not used for metrics. Reports include full-domain metrics and a second interior-only metric that excludes analytically prescribed boundary and initial-condition points.
- Added same per-seed HF observations and same random collocation/BC/IC sequences to PINN and M1.
- Added optional ten-seed runner, per-seed table, mean/std summary and failure log.
- Added Direct HF-LS only as a separate numerical control, not as a neural method.

## Important methodological qualifications

### 1. Which DeepONet source is used?

The `Physics-informed-DeepONets` repository by Wang, Wang and Perdikaris is a reference for physics-informed single-fidelity DeepONet and modified DeepONet building blocks. The **composite three-block M1** in this project is based on Howard et al. (2023), not merely a copy of the Wang repository.

### 2. Eq. (15) and inverse observations

Howard Eq. (15) contains exactly six terms: physics, LF data, nonlinear-branch regularization, LF-branch regularization, initial condition and boundary condition. The present source-identification task requires fitting 80 noisy HF observations to identify the unknown source. Therefore the implementation includes a separate seventh term `lambda_obs L_HFobs`. It is an explicit extension of the published loss, and the same observation term is used in PINN.

### 3. LF pretraining

The code performs LF operator warm-start pretraining and then unfreezes LF jointly with the two HF correction blocks. This follows the user's requested practical training sequence, while retaining the joint objective. A strict paper reproduction should document exact differences in training schedule and hyperparameters.

### 4. Fidelity and WRF-Chem

The current HF data come from a 129×129 central finite-difference solver. They are not WRF-Chem data. The current experiment isolates spatial discretization (21×21 vs 129×129, same central scheme). It does not yet compare central vs upwind as a multi-fidelity scheme pair. WRF-Chem integration, data-assimilation operators, units, domain/time alignment, runtime profile and cost ratio require a separate stage.

### 5. Statistical evidence

A smoke test verifies software flow only. A one-seed run is a demonstrator. `run_10seeds.py` provides the planned reproducible series, but mean/std only exist after the ten seed executions actually complete. A failure is retained in `failures.json`; it is not converted to a zero result.

## Readiness

The rewritten code is intended to make M1 a real runnable comparator against PINN. Before reporting scientific conclusions, run the smoke test, then one full seed, inspect convergence, and only then execute all ten seeds. Verify hyperparameter sensitivity without using clean test values for selection.
