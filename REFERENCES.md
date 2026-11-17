# Method references used in the implementation

1. **Howard, A. A., Perego, M., Karniadakis, G. E., & Stinis, P. (2023).** Multifidelity deep operator networks for data-driven and physics-informed problems. *Journal of Computational Physics*, 493, 112462. DOI: https://doi.org/10.1016/j.jcp.2023.112462
   - Source for the composite three-block M1 architecture, the linear/nonlinear correction sum, and Eq. (15) six-term physics-informed multi-fidelity loss.

2. **Wang, S., Wang, H., & Perdikaris, P. (2021).** Learning the solution operator of parametric partial differential equations with physics-informed DeepONets. *Science Advances*, 7(40), eabi8605. DOI: https://doi.org/10.1126/sciadv.abi8605
   - Source for single-fidelity physics-informed DeepONet architecture and modified DeepONet building blocks. Reference repository: https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets

3. **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).** Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707. DOI: https://doi.org/10.1016/j.jcp.2018.10.045
   - Source for the inverse PINN baseline. Reference repository and Burgers inverse-identification example: https://github.com/maziarraissi/PINNs

## Implementation boundary

The code is a PyTorch translation/adaptation of these method ideas for the user's scalar 2-D advection–diffusion inverse source task. It is not a verbatim copy of the TensorFlow 1.x or JAX repository code. See `ARCHITECTURE_NOTES.md` and `AUDIT_REPORT.md` for exact adaptations and caveats.
