# Exp. 1 — PINN-Raissi vs M1 PI-DeepONet

## Research question
Compare a single-fidelity inverse PINN with composite multi-fidelity PI-DeepONet for identifying a Gaussian source in 2-D advection–diffusion.

## Methods
- PINN-Raissi: fully connected tanh network; trainable bounded source parameters; Adam then SciPy L-BFGS-B.
- M1 PI-DeepONet: LF modified DeepONet + linear HF DeepONet + nonlinear HF modified DeepONet; HF output is the sum of the linear and nonlinear blocks.
- MPINN-v2 is excluded from this comparison.

## Data and fidelity protocol
LF simulations use a 21×21 grid; HF simulations use a 129×129 grid. Both use central differences. The target source is held out of the LF parameter-function ensemble. HF inverse training uses 80 noisy observations at noise_fraction=0.005. Clean HF test points are taken from all saved HF grid-time points, excluding exact observation coordinate-time locations. A separate downsampled grid is used only for figures.

## Objective
For M1, the six terms in Howard et al. (2023), Eq. (15), are retained: physics, LF data, nonlinear-branch regularization, LF-branch regularization, initial condition and boundary condition. Because this application is an inverse problem with sparse noisy HF measurements, we add λ_obs L_HF-observation as a seventh, task-specific term. The same HF-observation loss is used for PINN-Raissi. This is an explicit extension of Eq. (15), not a claim that Eq. (15) contains the observation term.

## Training and leakage prevention
The target clean test values are not used in training, scaling, hyperparameter selection or early stopping. Concentration scaling is computed from noisy training observations only. LF train/validation is split by complete source realizations. The same HF observations, collocation points and boundary/initial sampling seeds are used for both methods within each seed. L-BFGS-B uses fixed collocation, BC, IC and LF mini-batch points throughout its closure evaluations.

## Metrics (one seed)
| method         |   theta_Q |   theta_xs |   theta_ys |   relerr_Q |   relerr_xs |   relerr_ys |   param_relerr_mean |   test_RMSE |   test_MAE |   test_relL2 |   test_interior_points |   test_interior_RMSE |   test_interior_MAE |   test_interior_relL2 |   trainable_parameters |   observation_RMSE |   PDE_residual_RMS_dimensionless |   training_seconds | lbfgsb_success   |   lbfgsb_status | lbfgsb_message                              |   lbfgsb_nit |   lbfgsb_nfev |   lbfgsb_final_loss |   lf_pretraining_seconds |   joint_inverse_training_seconds |   lf_solver_generation_seconds |   shared_hf_solver_generation_seconds |   end_to_end_method_seconds_excluding_shared_HF_data |   lf_pretrain_best_val_MSE |
|:---------------|----------:|-----------:|-----------:|-----------:|------------:|------------:|--------------------:|------------:|-----------:|-------------:|-----------------------:|---------------------:|--------------------:|----------------------:|-----------------------:|-------------------:|---------------------------------:|-------------------:|:-----------------|----------------:|:--------------------------------------------|-------------:|--------------:|--------------------:|-------------------------:|---------------------------------:|-------------------------------:|--------------------------------------:|-----------------------------------------------------:|---------------------------:|
| PINN_Raissi    |   0.98204 |    0.35987 |    0.62929 |   0.017963 |  0.00035175 |   0.0011235 |           0.0064793 |   0.0016147 | 0.00096762 |     0.037551 |                 322500 |            0.0016604 |          0.0010004  |              0.037097 |                   3044 |          0.0010362 |                         0.035158 |             159.41 | False            |               1 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |         5000 |          5391 |          0.00034888 |                   nan    |                           159.41 |                      nan       |                               0.16685 |                                               159.41 |                 nan        |
| M1_PI_DeepONet |   0.97326 |    0.35902 |    0.63198 |   0.026744 |  0.0027284  |   0.0031369 |           0.01087   |   0.0013294 | 0.00080953 |     0.030914 |                 322500 |            0.0013642 |          0.00083002 |              0.030479 |                  29719 |          0.001506  |                         0.021513 |             674.51 | False            |               1 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |         5000 |          5625 |          0.0055788  |                    10.27 |                           664.24 |                        0.80236 |                               0.16685 |                                               675.31 |                   0.052549 |

`test_RMSE/MAE/relL2` use all clean, non-observed HF grid-time points, including analytically prescribed boundary and initial-condition points. `test_interior_*` excludes the four boundary sides and the t=0 plane, and is the more stringent field metric away from imposed constraints.

## Direct HF-LS control (not a neural method)
Not run.

## Reproduction
```bash
python -m pip install -r requirements.txt
python run_exp1.py --seed 7 --model both --device cpu
```

## Current limitations
The HF model is the project's finite-difference solver, not WRF-Chem. WRF-Chem coupling, runtime calibration (the intended expensive 1/5 real-time HF cost), multi-seed statistics, cross-scheme (central vs upwind) comparison and physical validation against atmospheric chemistry data remain future work.

## Figures
- figures/field_comparison.png
- figures/absolute_error_maps.png
- figures/parameter_errors.png
- figures/training_curves.png
- figures/parameter_convergence.png
