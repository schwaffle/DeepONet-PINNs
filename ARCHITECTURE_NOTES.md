# Architecture notes

## PINN-Raissi

`PINN` is an inverse continuous-time PINN with FCNN `[3, 20,20,20,20,20,20,20,20, 1]`: physical coordinates `(x,y,t)` are normalized internally to `[-1,1]`; eight hidden `tanh` layers; scalar concentration output. Parameters `Q,xs,ys` are trainable variables transformed from raw unconstrained variables through sigmoid to `PARAM_BOUNDS`. The source is included in the PDE forcing, not network input.

This follows the inverse Burgers PINN layout in Raissi's `continuous_time_identification (Burgers)/Burgers.py`: tanh FCNN, Xavier-style truncated-normal initialization, zero biases, coordinate scaling, AD residual, Adam then SciPy L-BFGS-B. The present code is a PyTorch adaptation and adds the zero initial and Dirichlet boundary losses required by this advection-diffusion setup.

## Composite M1 PI-DeepONet

The branch input is a discretized spatial Gaussian source profile at fixed sensor sites. The trunk input is physical query coordinates `(x,y,t)`, normalized inside the network. The LF modified DeepONet learns the LF solution operator from a source ensemble disjoint from the inverse target. HF corrections are:

- `F_l`: vanilla linear DeepONet; branch input is the LF output vector at fixed space-time feature coordinates.
- `F_nl`: modified nonlinear DeepONet; branch input concatenates the source sensor vector and a subset of LF outputs.

`u_HF = F_l + F_nl`, following the composite construction of Howard et al. (2023), JCP 493, 112462. LF features use 11×11 spatial sites at four nonzero time slices; the temporal stacking is an adaptation to the present transient 2D problem. The nonlinearity/encoder injection pattern follows the modified DeepONet family described by Wang et al.; gate activation is `tanh` as in the reference equations. The gate activation is `tanh`, following the modified DeepONet equations (the gate values are not constrained to `[0,1]`; this is not a convex mixture). This corrects an earlier local implementation choice that used sigmoid and therefore deviated from the reference activation.

## Objective

Howard Eq. (15) has six loss terms: PDE, LF data, nonlinear branch regularization, LF branch regularization, IC and BC. The source-identification experiment additionally has sparse noisy HF observations, so `lambda_obs * L_HFobs` is added as a task-specific seventh term. This is essential to identify an unknown source from its measurements and is **not** part of Howard Eq. (15) itself.

For PINN, the corresponding single-fidelity objective is `lambda_obs*L_HFobs + lambda1*L_PDE + lambda5*L_IC + lambda6*L_BC`.

## Derivatives and normalization

The network receives physical `(x,y,t)` and normalizes internally using fixed known domain bounds. PDE residual is differentiated with respect to the original physical coordinate tensor. PyTorch autograd traverses the normalization layer, thereby applying the chain rule for `x,y,t` (including the `2/T` derivative for time). No manual derivative scaling should be layered on top of this. Residual is nondimensionalized by `T/u_scale`.

## Fidelity design

Main Exp. 1 uses LF central `21×21` and HF central `129×129`: only spatial resolution changes. Upwind is retained in the solver but not used in the main experiment. A later cross-scheme experiment should keep grid resolution fixed while changing the advection discretization, and should separately measure the LF/HF discrepancies induced by the scheme.
