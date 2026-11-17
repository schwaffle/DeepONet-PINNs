"""PINN and composite physics-informed multi-fidelity DeepONet models.

M1 follows the composite architecture in Howard et al. (2023):
    u_HF = F_l(u_LF features, x,y,t) + F_nl(source sensors, LF subset, x,y,t).
The LF operator and the two HF correlation blocks are distinct networks.
The modified DeepONet update follows the encoder/gate equations of Wang et al.
(2021/2022) and Howard et al. (2023): the gate activation is tanh, matching the
reference formulation rather than replacing it with a sigmoid convex gate.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np
import torch
from torch import nn

from datasets import PARAM_BOUNDS


def normalize_coords(coords: torch.Tensor, coord_min: torch.Tensor,
                     coord_max: torch.Tensor) -> torch.Tensor:
    return 2.0 * (coords - coord_min) / (coord_max - coord_min) - 1.0


class InverseParameterMixin:
    def init_inverse_parameters(self, theta_init: Sequence[float],
                                 bounds: np.ndarray = PARAM_BOUNDS):
        bounds = np.asarray(bounds, dtype=np.float64)
        self.register_buffer("theta_bounds", torch.as_tensor(bounds, dtype=torch.float32))
        lo, hi = bounds[:, 0], bounds[:, 1]
        p = np.clip((np.asarray(theta_init, dtype=np.float64)-lo)/(hi-lo), 1e-6, 1-1e-6)
        raw = np.log(p/(1-p))
        self.theta_raw = nn.Parameter(torch.as_tensor(raw, dtype=torch.float32))

    def theta_physical(self) -> torch.Tensor:
        lo, hi = self.theta_bounds[:, 0], self.theta_bounds[:, 1]
        return lo + (hi-lo)*torch.sigmoid(self.theta_raw)


class MLP(nn.Module):
    def __init__(self, dims: Sequence[int], activation=nn.Tanh,
                 xavier: bool = True, linear: bool = False):
        super().__init__()
        layers = []
        for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(din, dout))
            if i < len(dims)-2 and not linear:
                layers.append(activation())
        self.net = nn.Sequential(*layers)
        if xavier:
            self.net.apply(self._init_xavier)

    @staticmethod
    def _init_xavier(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)


class DeepONet(nn.Module):
    """Vanilla DeepONet with dot-product merge and physical-coordinate scaling."""
    def __init__(self, branch_dim: int, coord_min, coord_max,
                 latent_dim: int = 10, hidden: int = 40, depth: int = 1,
                 linear: bool = False):
        super().__init__()
        self.branch = MLP([branch_dim] + [hidden]*depth + [latent_dim], linear=linear)
        self.trunk = MLP([3] + [hidden]*depth + [latent_dim], linear=linear)
        self.bias = nn.Parameter(torch.zeros(()))
        self.register_buffer("coord_min", torch.as_tensor(coord_min, dtype=torch.float32))
        self.register_buffer("coord_max", torch.as_tensor(coord_max, dtype=torch.float32))

    def forward(self, source_samples: torch.Tensor, coords: torch.Tensor):
        if source_samples.ndim == 1:
            source_samples = source_samples.unsqueeze(0)
        if coords.ndim == 1:
            coords = coords.unsqueeze(0)
        if coords.ndim == 3:
            bsz, npts, _ = coords.shape
            if source_samples.shape[0] == 1:
                source_samples = source_samples.expand(bsz, -1)
            if source_samples.shape[0] != bsz:
                raise ValueError("Branch and trunk batch dimensions do not match")
            b = self.branch(source_samples).unsqueeze(1)
            t = self.trunk(normalize_coords(coords, self.coord_min, self.coord_max))
            return torch.sum(b*t, dim=-1) + self.bias
        if source_samples.shape[0] == 1 and coords.shape[0] > 1:
            source_samples = source_samples.expand(coords.shape[0], -1)
        if source_samples.shape[0] != coords.shape[0]:
            raise ValueError("Branch and trunk batch dimensions do not match")
        b = self.branch(source_samples)
        t = self.trunk(normalize_coords(coords, self.coord_min, self.coord_max))
        return torch.sum(b*t, dim=-1, keepdim=True) + self.bias

    def branch_parameters(self):
        return list(self.branch.parameters())


class ModifiedDeepONet(nn.Module):
    """Encoder/gated DeepONet used for LF operator and nonlinear HF correction.

    Inputs are a source function sampled at fixed branch sensors and one or many
    physical query coordinates. Each layer blends branch/trunk states with a
    sigmoid gate; final branch/trunk latent vectors are merged by inner product.
    """
    def __init__(self, branch_dim: int, coord_min, coord_max,
                 latent_dim: int = 32, hidden: int = 48, depth: int = 2,
                 output_activation: bool = False):
        super().__init__()
        if depth < 1:
            raise ValueError("ModifiedDeepONet depth must be >= 1")
        self.enc_u = nn.Sequential(nn.Linear(branch_dim, hidden), nn.Tanh())
        self.enc_x = nn.Sequential(nn.Linear(3, hidden), nn.Tanh())
        self.gate_u = nn.ModuleList([nn.Linear(branch_dim if k == 0 else hidden, hidden)
                                     for k in range(depth)])
        self.gate_x = nn.ModuleList([nn.Linear(3 if k == 0 else hidden, hidden)
                                     for k in range(depth)])
        self.final_u = nn.Linear(hidden, latent_dim)
        self.final_x = nn.Linear(hidden, latent_dim)
        self.output_activation = nn.Tanh() if output_activation else nn.Identity()
        self.bias = nn.Parameter(torch.zeros(()))
        self.register_buffer("coord_min", torch.as_tensor(coord_min, dtype=torch.float32))
        self.register_buffer("coord_max", torch.as_tensor(coord_max, dtype=torch.float32))
        self.apply(MLP._init_xavier)

    def _core(self, source, coords):
        coords_n = normalize_coords(coords, self.coord_min, self.coord_max)
        U = self.enc_u(source)
        V = self.enc_x(coords_n)
        hu, hx = U, V
        for k in range(len(self.gate_u)):
            gu_in = source if k == 0 else hu
            gx_in = coords_n if k == 0 else hx
            zu = torch.tanh(self.gate_u[k](gu_in))
            zx = torch.tanh(self.gate_x[k](gx_in))
            hu = (1.0-zu)*U + zu*V
            hx = (1.0-zx)*U + zx*V
        hu = self.output_activation(self.final_u(hu))
        hx = self.output_activation(self.final_x(hx))
        return torch.sum(hu*hx, dim=-1, keepdim=True) + self.bias

    def forward(self, source_samples: torch.Tensor, coords: torch.Tensor):
        if source_samples.ndim == 1:
            source_samples = source_samples.unsqueeze(0)
        if coords.ndim == 1:
            coords = coords.unsqueeze(0)
        if coords.ndim == 3:
            bsz, npts, _ = coords.shape
            if source_samples.shape[0] == 1:
                source_samples = source_samples.expand(bsz, -1)
            if source_samples.shape[0] != bsz:
                raise ValueError("Branch and trunk batch dimensions do not match")
            uf = source_samples.repeat_interleave(npts, dim=0)
            xf = coords.reshape(bsz*npts, 3)
            return self._core(uf, xf).reshape(bsz, npts)
        if source_samples.shape[0] == 1 and coords.shape[0] > 1:
            source_samples = source_samples.expand(coords.shape[0], -1)
        if source_samples.shape[0] != coords.shape[0]:
            raise ValueError("Branch and trunk batch dimensions do not match")
        return self._core(source_samples, coords)

    def branch_parameters(self):
        params = list(self.enc_u.parameters()) + list(self.gate_u.parameters()) + list(self.final_u.parameters())
        return params


class PINN(nn.Module, InverseParameterMixin):
    """Raissi-style fully connected PINN for the inverse scalar PDE.

    Eight hidden layers of 20 tanh units are the reference architecture adopted
    for this baseline. The source parameters enter the PDE forcing, not the network
    input. Inputs are physical (x,y,t) and are mapped internally to [-1,1].
    """
    def __init__(self, theta_init, coord_min, coord_max,
                 width: int = 20, depth: int = 8, bounds=PARAM_BOUNDS):
        super().__init__()
        self.init_inverse_parameters(theta_init, bounds)
        self.register_buffer("coord_min", torch.as_tensor(coord_min, dtype=torch.float32))
        self.register_buffer("coord_max", torch.as_tensor(coord_max, dtype=torch.float32))
        dims = [3] + [width]*depth + [1]
        layers = []
        for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
            linear = nn.Linear(din, dout)
            # Raissi's TensorFlow reference draws truncated-normal weights with
            # std=sqrt(2/(fan_in+fan_out)); biases are exactly zero.
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = float(np.sqrt(2.0/(fan_in+fan_out)))
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=std,
                                  a=-2.0*std, b=2.0*std)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if i < len(dims)-2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        return self.net(normalize_coords(coords, self.coord_min, self.coord_max))


class MultiFidelityPIDeepONet(nn.Module, InverseParameterMixin):
    """M1: composite physics-informed multi-fidelity DeepONet (Howard et al. 2023).

    The LF operator is warm-started on low-fidelity simulations and then unfrozen.
    The HF prediction is F_l + F_nl (not u_LF + a residual):
      F_l: vanilla linear DeepONet, branch input = LF output feature vector;
      F_nl: modified nonlinear DeepONet, branch input = source sensors + LF subset.
    """
    def __init__(self, lf_model: ModifiedDeepONet, sensor_xy, lf_feature_coords,
                 lf_sub_idx, theta_init, coord_min, coord_max,
                 bounds=PARAM_BOUNDS, sigma_s: float = 0.10, linear_latent: int = 10,
                 linear_hidden: int = 40, nonlinear_latent: int = 32,
                 nonlinear_hidden: int = 32, nonlinear_depth: int = 2):
        super().__init__()
        self.init_inverse_parameters(theta_init, bounds)
        self.lf = lf_model
        self.sigma_s = float(sigma_s)
        self.register_buffer("sensor_xy", torch.as_tensor(sensor_xy, dtype=torch.float32))
        self.register_buffer("lf_feature_coords", torch.as_tensor(lf_feature_coords, dtype=torch.float32))
        self.register_buffer("lf_sub_idx", torch.as_tensor(lf_sub_idx, dtype=torch.long))
        self.register_buffer("coord_min", torch.as_tensor(coord_min, dtype=torch.float32))
        self.register_buffer("coord_max", torch.as_tensor(coord_max, dtype=torch.float32))
        n_features = int(self.lf_feature_coords.shape[0])
        self.linear_block = DeepONet(n_features, coord_min, coord_max,
                                     latent_dim=linear_latent, hidden=linear_hidden,
                                     depth=0, linear=True)
        self.nonlinear_block = ModifiedDeepONet(
            branch_dim=int(self.sensor_xy.shape[0] + self.lf_sub_idx.numel()),
            coord_min=coord_min, coord_max=coord_max,
            latent_dim=nonlinear_latent, hidden=nonlinear_hidden,
            depth=nonlinear_depth, output_activation=True)

    def source_samples(self):
        theta = self.theta_physical()
        sx, sy = self.sensor_xy[:, 0], self.sensor_xy[:, 1]
        q, xs, ys = theta[0], theta[1], theta[2]
        r2 = (sx-xs)**2 + (sy-ys)**2
        return q*torch.exp(-r2/(2.0*self.sigma_s**2))

    def forward_components(self, coords):
        if coords.ndim == 1:
            coords = coords.unsqueeze(0)
        source = self.source_samples().unsqueeze(0)  # [1, M]
        lf_features = self.lf(source, self.lf_feature_coords.unsqueeze(0))  # [1, P]
        bsz = coords.shape[0]
        lf_features_b = lf_features.expand(bsz, -1)
        source_b = source.expand(bsz, -1)
        nonlinear_branch = torch.cat([source_b, lf_features_b[:, self.lf_sub_idx]], dim=-1)
        f_linear = self.linear_block(lf_features_b, coords)
        f_nonlinear = self.nonlinear_block(nonlinear_branch, coords)
        return {"u_hf": f_linear + f_nonlinear,
                "f_linear": f_linear, "f_nonlinear": f_nonlinear,
                "lf_features": lf_features, "source_samples": source}

    def forward(self, coords):
        return self.forward_components(coords)["u_hf"]

    def regularization_parameters(self):
        # Eq. (15): regularize the nonlinear branch and LF branch weights/biases.
        return self.nonlinear_block.branch_parameters(), self.lf.branch_parameters()
