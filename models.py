"""
Модели PINN и составной физико-информированной мультифидельной DeepONet (PI-DeepONet).

Архитектура M1 следует составной структуре из Howard (2023):
    u_HF = F_l(признаки_LF, x, y, t) + F_nl(сенсоры_источника, подвыборка_LF, x, y, t).

LF-оператор и два HF-блока коррекции являются отдельными нейронными сетями.
Обновление ModifiedDeepONet следует уравнениям кодировщика/гейтов из Wang (2021/2022)
и Howard (2023): активация гейтов — это `tanh`, что соответствует оригинальной
формулировке (не выпуклый sigmoid-гейт).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from datasets import PARAM_BOUNDS


def normalize_coords(coords: torch.Tensor, coord_min: torch.Tensor,
                     coord_max: torch.Tensor) -> torch.Tensor:
    """Нормализует физические координаты (x, y, t) в диапазон [-1, 1]."""
    return 2.0 * (coords - coord_min) / (coord_max - coord_min) - 1.0


class InverseParameterMixin(nn.Module):
    """
    Миксин для обучаемых параметров обратной задачи (Q, xs, ys).

    Используется сигмоидальная параметризация: theta_raw оптимизируется без
    ограничений, а физические значения всегда лежат внутри PARAM_BOUNDS.

    ВАЖНО: миксин наследует nn.Module, чтобы методы register_buffer/parameters
    были видны анализатору типов. В классах-наследниках миксин должен стоять
    ПЕРВЫМ в списке базовых классов, иначе MRO не вычисляется.
    """

    # Атрибуты создаются в init_inverse_parameters через register_buffer/Parameter.
    theta_bounds: torch.Tensor
    theta_raw: nn.Parameter

    def init_inverse_parameters(self, theta_init: Sequence[float],
                                bounds: np.ndarray = PARAM_BOUNDS) -> None:
        bounds = np.asarray(bounds, dtype=np.float64)
        self.register_buffer("theta_bounds", torch.as_tensor(
            bounds, dtype=torch.float32))
        lo, hi = bounds[:, 0], bounds[:, 1]

        # Нормализуем старт в [0, 1] и обрезаем от точных 0 и 1 (иначе log(0)).
        p = np.clip((np.asarray(theta_init, dtype=np.float64) - lo)
                    / (hi - lo), 1e-6, 1 - 1e-6)

        # Обратная функция к sigmoid (logit): raw даёт ровно p при прямом проходе.
        raw = np.log(p / (1 - p))
        self.theta_raw = nn.Parameter(
            torch.as_tensor(raw, dtype=torch.float32))

    def theta_physical(self) -> torch.Tensor:
        """Физические значения параметров, гарантированно в границах."""
        lo, hi = self.theta_bounds[:, 0], self.theta_bounds[:, 1]
        return lo + (hi - lo) * torch.sigmoid(self.theta_raw)


class MLP(nn.Module):
    """
    Стандартная полносвязная сеть.

    Опция linear=True отключает активации — это необходимо для линейного
    блока F_l в M1 (линейная корреляция LF/HF по Howard, Sec. 2.2).
    """

    def __init__(self, dims: Sequence[int],
                 activation: Callable[[], nn.Module] = nn.Tanh,
                 xavier: bool = True, linear: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(din, dout))
            # Активация не добавляется после последнего слоя и в линейном режиме.
            if i < len(dims) - 2 and not linear:
                layers.append(activation())
        self.net = nn.Sequential(*layers)
        if xavier:
            self.net.apply(self._init_xavier)

    @staticmethod
    def _init_xavier(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepONet(nn.Module):
    """
    Vanilla DeepONet: branch кодирует входную функцию, trunk — координаты,
    выход — скалярное произведение латентных векторов плюс bias.
    """

    coord_min: torch.Tensor
    coord_max: torch.Tensor

    def __init__(self, branch_dim: int, coord_min, coord_max,
                 latent_dim: int = 10, hidden: int = 40, depth: int = 1,
                 linear: bool = False):
        super().__init__()
        self.branch = MLP([branch_dim] + [hidden] * depth +
                          [latent_dim], linear=linear)
        self.trunk = MLP([3] + [hidden] * depth + [latent_dim], linear=linear)
        self.bias = nn.Parameter(torch.zeros(()))
        self.register_buffer("coord_min", torch.as_tensor(
            coord_min, dtype=torch.float32))
        self.register_buffer("coord_max", torch.as_tensor(
            coord_max, dtype=torch.float32))

    def forward(self, source_samples: torch.Tensor,
                coords: torch.Tensor) -> torch.Tensor:
        if source_samples.ndim == 1:
            source_samples = source_samples.unsqueeze(0)
        if coords.ndim == 1:
            coords = coords.unsqueeze(0)
        # Случай "один источник на сетку точек": coords [B, N, 3].
        if coords.ndim == 3:
            bsz, npts, _ = coords.shape
            if source_samples.shape[0] == 1:
                source_samples = source_samples.expand(bsz, -1)
            if source_samples.shape[0] != bsz:
                raise ValueError(
                    "Branch and trunk batch dimensions do not match")
            b = self.branch(source_samples).unsqueeze(1)
            t = self.trunk(normalize_coords(
                coords, self.coord_min, self.coord_max))
            return torch.sum(b * t, dim=-1) + self.bias
        if source_samples.shape[0] == 1 and coords.shape[0] > 1:
            source_samples = source_samples.expand(coords.shape[0], -1)
        if source_samples.shape[0] != coords.shape[0]:
            raise ValueError(
                "Branch and trunk batch dimensions do not match")
        b = self.branch(source_samples)
        t = self.trunk(normalize_coords(
            coords, self.coord_min, self.coord_max))
        return torch.sum(b * t, dim=-1, keepdim=True) + self.bias

    def branch_parameters(self) -> List[nn.Parameter]:
        """Веса только branch-части (для регуляризации lambda_3/lambda_4)."""
        return list(self.branch.parameters())


class ModifiedDeepONet(nn.Module):
    """
    Modified DeepONet с энкодерами и гейтами (Wang 2022; Howard 2023, Eq. 2-6).

    Состояния branch и trunk смешиваются на каждом слое через tanh-гейт.
    Комбинация не выпуклая: гейт принимает значения в [-1, 1], а не [0, 1].
    """

    coord_min: torch.Tensor
    coord_max: torch.Tensor

    def __init__(self, branch_dim: int, coord_min, coord_max,
                 latent_dim: int = 32, hidden: int = 48, depth: int = 2,
                 output_activation: bool = False):
        super().__init__()
        if depth < 1:
            raise ValueError("ModifiedDeepONet depth must be >= 1")

        self.enc_u = nn.Sequential(nn.Linear(branch_dim, hidden), nn.Tanh())
        self.enc_x = nn.Sequential(nn.Linear(3, hidden), nn.Tanh())

        self.gate_u = nn.ModuleList(
            [nn.Linear(branch_dim if k == 0 else hidden, hidden)
             for k in range(depth)])
        self.gate_x = nn.ModuleList(
            [nn.Linear(3 if k == 0 else hidden, hidden)
             for k in range(depth)])
        self.final_u = nn.Linear(hidden, latent_dim)
        self.final_x = nn.Linear(hidden, latent_dim)
        self.output_activation: nn.Module = (
            nn.Tanh() if output_activation else nn.Identity())
        self.bias = nn.Parameter(torch.zeros(()))
        self.register_buffer("coord_min", torch.as_tensor(
            coord_min, dtype=torch.float32))
        self.register_buffer("coord_max", torch.as_tensor(
            coord_max, dtype=torch.float32))
        self.apply(MLP._init_xavier)

    def _core(self, source: torch.Tensor,
              coords: torch.Tensor) -> torch.Tensor:
        """Прямой проход для одной точки или плоского батча."""
        coords_n = normalize_coords(coords, self.coord_min, self.coord_max)
        U = self.enc_u(source)
        V = self.enc_x(coords_n)
        hu, hx = U, V

        for k in range(len(self.gate_u)):
            gu_in = source if k == 0 else hu
            gx_in = coords_n if k == 0 else hx
            zu = torch.tanh(self.gate_u[k](gu_in))
            zx = torch.tanh(self.gate_x[k](gx_in))

            hu = (1.0 - zu) * U + zu * V
            hx = (1.0 - zx) * U + zx * V

        hu = self.output_activation(self.final_u(hu))
        hx = self.output_activation(self.final_x(hx))
        return torch.sum(hu * hx, dim=-1, keepdim=True) + self.bias

    def forward(self, source_samples: torch.Tensor,
                coords: torch.Tensor) -> torch.Tensor:
        if source_samples.ndim == 1:
            source_samples = source_samples.unsqueeze(0)
        if coords.ndim == 1:
            coords = coords.unsqueeze(0)
        if coords.ndim == 3:
            bsz, npts, _ = coords.shape
            if source_samples.shape[0] == 1:
                source_samples = source_samples.expand(bsz, -1)
            if source_samples.shape[0] != bsz:
                raise ValueError(
                    "Branch and trunk batch dimensions do not match")
            # Повторяем источник для каждой точки сетки и выравниваем батч.
            uf = source_samples.repeat_interleave(npts, dim=0)
            xf = coords.reshape(bsz * npts, 3)
            return self._core(uf, xf).reshape(bsz, npts)
        if source_samples.shape[0] == 1 and coords.shape[0] > 1:
            source_samples = source_samples.expand(coords.shape[0], -1)
        if source_samples.shape[0] != coords.shape[0]:
            raise ValueError(
                "Branch and trunk batch dimensions do not match")
        return self._core(source_samples, coords)

    def branch_parameters(self) -> List[nn.Parameter]:
        """Веса branch-потока: энкодер + гейты + финальный слой."""
        params = (list(self.enc_u.parameters())
                  + list(self.gate_u.parameters())
                  + list(self.final_u.parameters()))
        return params


class PINN(InverseParameterMixin, nn.Module):
    """
    Полносвязный PINN в стиле Raissi для обратной скалярной PDE.

    Архитектура и инициализация взяты из референсного примера Burgers
    (continuous_time_identification): 3 входа (x, y, t), 8 скрытых слоёв
    по 20 tanh-единиц, 1 выход. Параметры источника входят в правую часть
    уравнения (PDE forcing), а не во вход сети. Координаты нормализуются
    внутри сети в [-1, 1].
    """

    coord_min: torch.Tensor
    coord_max: torch.Tensor

    def __init__(self, theta_init, coord_min, coord_max,
                 width: int = 20, depth: int = 8, bounds=PARAM_BOUNDS):
        super().__init__()

        # Обучаемые ограниченные параметры источника (Q, xs, ys).
        self.init_inverse_parameters(theta_init, bounds)

        # Границы физической области — константы задачи (buffer, не Parameter).
        self.register_buffer("coord_min", torch.as_tensor(
            coord_min, dtype=torch.float32))
        self.register_buffer("coord_max", torch.as_tensor(
            coord_max, dtype=torch.float32))

        # dims = [3, 20, ..., 20, 1]: вход, depth скрытых слоёв, скалярный выход.
        dims = [3] + [width] * depth + [1]
        layers: List[nn.Module] = []

        for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
            linear = nn.Linear(din, dout)

            # Инициализация Raissi: truncated normal со std = sqrt(2/(fan_in+fan_out)),
            # bias ровно нули. Для nn.Linear fan_in = din, fan_out = dout,
            # поэтому приватная функция torch не нужна — считаем напрямую.
            fan_in, fan_out = din, dout
            std = float(np.sqrt(2.0 / (fan_in + fan_out)))
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=std,
                                  a=-2.0 * std, b=2.0 * std)
            nn.init.zeros_(linear.bias)
            layers.append(linear)

            # Tanh обязательна: нужны вторые производные для PDE residual.
            if i < len(dims) - 2:
                layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(normalize_coords(
            coords, self.coord_min, self.coord_max))


class MultiFidelityPIDeepONet(InverseParameterMixin, nn.Module):
    """
    M1: составная мультифидельная PI-DeepONet (Howard 2023).

    u_HF = F_l + F_nl, где
      F_l  — vanilla DeepONet без активаций, branch = выходы LF-блока;
      F_nl — modified DeepONet с активацией на каждом слое,
             branch = сенсоры источника + подвыборка выходов LF.
    LF-блок — поставщик признаков и объект LF-loss; в сумму он не входит.
    """

    sensor_xy: torch.Tensor
    lf_feature_coords: torch.Tensor
    lf_sub_idx: torch.Tensor
    coord_min: torch.Tensor
    coord_max: torch.Tensor

    def __init__(self, lf_model: ModifiedDeepONet, sensor_xy, lf_feature_coords,
                 lf_sub_idx, theta_init, coord_min, coord_max,
                 bounds=PARAM_BOUNDS, sigma_s: float = 0.10,
                 linear_latent: int = 10, linear_hidden: int = 40,
                 nonlinear_latent: int = 32, nonlinear_hidden: int = 32,
                 nonlinear_depth: int = 2):
        super().__init__()
        self.init_inverse_parameters(theta_init, bounds)
        self.lf = lf_model
        self.sigma_s = float(sigma_s)
        self.register_buffer("sensor_xy", torch.as_tensor(
            sensor_xy, dtype=torch.float32))
        self.register_buffer("lf_feature_coords", torch.as_tensor(
            lf_feature_coords, dtype=torch.float32))
        self.register_buffer("lf_sub_idx", torch.as_tensor(
            lf_sub_idx, dtype=torch.long))
        self.register_buffer("coord_min", torch.as_tensor(
            coord_min, dtype=torch.float32))
        self.register_buffer("coord_max", torch.as_tensor(
            coord_max, dtype=torch.float32))
        n_features = int(self.lf_feature_coords.shape[0])

        # Линейный блок: одна линейная проекция branch/trunk, без активаций.
        self.linear_block = DeepONet(n_features, coord_min, coord_max,
                                     latent_dim=linear_latent,
                                     hidden=linear_hidden,
                                     depth=0, linear=True)
        # Нелинейный блок: branch = сенсоры (144) + подвыборка LF (75).
        self.nonlinear_block = ModifiedDeepONet(
            branch_dim=int(self.sensor_xy.shape[0] + self.lf_sub_idx.numel()),
            coord_min=coord_min, coord_max=coord_max,
            latent_dim=nonlinear_latent, hidden=nonlinear_hidden,
            depth=nonlinear_depth, output_activation=True)

    def source_samples(self) -> torch.Tensor:
        """
        Дифференцируемо строит sensor-вектор источника из theta_physical().

        Через эту операцию градиент от наблюдений и PDE течёт в параметры
        Q, xs, ys — именно так решается обратная задача в DeepONet.
        """
        theta = self.theta_physical()
        sx, sy = self.sensor_xy[:, 0], self.sensor_xy[:, 1]
        q, xs, ys = theta[0], theta[1], theta[2]
        r2 = (sx - xs) ** 2 + (sy - ys) ** 2
        return q * torch.exp(-r2 / (2.0 * self.sigma_s ** 2))

    def forward_components(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Возвращает итоговый u_HF и промежуточные величины для анализа/loss."""
        if coords.ndim == 1:
            coords = coords.unsqueeze(0)

        # [1, M] — один источник на весь батч (обратная задача, одна реализация).
        source = self.source_samples().unsqueeze(0)

        # [1, P] — признаки LF-оператора в фиксированных feature-точках.
        lf_features = self.lf(source, self.lf_feature_coords.unsqueeze(0))
        bsz = coords.shape[0]
        lf_features_b = lf_features.expand(bsz, -1)
        source_b = source.expand(bsz, -1)

        # Branch нелинейного блока: сенсоры + подвыборка LF-признаков.
        nonlinear_branch = torch.cat(
            [source_b, lf_features_b[:, self.lf_sub_idx]], dim=-1)
        f_linear = self.linear_block(lf_features_b, coords)
        f_nonlinear = self.nonlinear_block(nonlinear_branch, coords)
        return {"u_hf": f_linear + f_nonlinear,
                "f_linear": f_linear, "f_nonlinear": f_nonlinear,
                "lf_features": lf_features, "source_samples": source}

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.forward_components(coords)["u_hf"]

    def regularization_parameters(
            self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """Веса branch нелинейного и LF блоков — члены lambda_3 и lambda_4."""
        return (self.nonlinear_block.branch_parameters(),
                self.lf.branch_parameters())
