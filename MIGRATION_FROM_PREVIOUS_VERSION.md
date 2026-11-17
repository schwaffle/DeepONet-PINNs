# Migration notes from the uploaded project

The uploaded DOCX project had a partially prepared M1 architecture in `models.py`, but no functional M1 training branch. `train.py` only executed PINN and MPINN-v2; CLI accepted `both`, `pinn`, `m4`; the checkpoint save code even contained a comment saying M1 checkpoint metadata should be added later. `make_report.py` advertised M1 in labels but depended on a result row and checkpoint that were never generated. Consequently there was no trained DeepONet curve/field in the delivered comparison.

This rewrite changes the experiment as follows:

1. Keep `PINN_Raissi` and add a fully integrated `M1_PI_DeepONet` training path.
2. Exclude `MPINN-v2` completely from active CLI, training, evaluation and figures.
3. Train LF ModifiedDeepONet on an ensemble of 40 source functions, split train/validation by entire functions; target source is excluded.
4. Build composite M1 as LF operator + linear HF correlation block + nonlinear HF correlation block; HF output is `F_l + F_nl`.
5. Implement all six Howard Eq. (15) terms, plus explicit `lambda_obs L_HFobs` for the inverse source-identification extension.
6. Share the same noisy HF observations, physics collocation points, boundary points and initial points between M1 and PINN for each seed.
7. Use HF clean metrics on every saved HF grid-time point excluding exact observation coordinate/time pairs; plot-grid points are a separate visualisation-only set.
8. Save M1-specific checkpoint metadata required by `inference.py`.
9. Use SciPy L-BFGS-B after Adam and freeze collocation/BC/IC/LF mini-batches throughout each quasi-Newton closure.
10. Keep WRF-Chem as future HF integration; it is not connected in this finite-difference Exp. 1.

The rewritten code is a starting point for a reproducible method comparison, not a claim that the current synthetic finite-difference HF experiment reproduces a real WRF-Chem workflow or that one seed establishes statistical superiority.
