"""Эталонный конечно-разностный решатель для идентификации источника в 2D уравнении адвекции-diffusion.

УЧП (PDE): u_t + vx*u_x + vy*u_y - kappa*(u_xx + u_yy) = S(x,y,t).
Основное сравнение использует центральные разности для адвекции и диффузии
на обоих уровнях точности (fidelities); LF и HF отличаются только разрешением сетки.
Схема направленных разностей (upwind) сохранена как явная опция для последующего
кросс-схемного исследования, но по умолчанию в Эксперименте 1 не используется.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ExperimentConfig:
    seed: int = 7
    T: float = 0.5
    vx: float = 0.35
    vy: float = -0.15
    kappa: float = 1.0e-3
    tau: float = 0.04
    sigma_s: float = 0.10
    theta_true: tuple[float, float, float] = (1.0, 0.36, 0.63)
    n_lf: int = 21
    n_hf: int = 129
    n_obs: int = 80
    noise_fraction: float = 0.005
    snapshot_count: int = 21


def stable_dt(n: int, cfg: ExperimentConfig) -> float:
    """Вычисляет устойчивый шаг по времени для явной схемы."""
    dx = 1.0 / (n - 1)
    rate = (abs(cfg.vx) + abs(cfg.vy)) / dx + 4.0 * cfg.kappa / dx**2
    return min(0.8 / max(rate, 1e-14), cfg.tau / 20.0)


def source_numpy(x, y, t, theta, cfg: ExperimentConfig):
    """Вычисляет значение гауссова источника в точке (x, y, t)."""
    q, xs, ys = theta
    r2 = (x - xs) ** 2 + (y - ys) ** 2
    # Используем expm1 для численной стабильности при малых t
    return q * np.exp(-r2 / (2.0 * cfg.sigma_s**2)) * (-np.expm1(-t / cfg.tau))


def solve(theta, n: int, scheme: str, cfg: ExperimentConfig):
    """Явный метод Эйлера с нулевыми граничными условиями Дирихле и нулевым начальным условием."""
    if scheme not in {"central", "upwind"}:
        raise ValueError("scheme must be 'central' or 'upwind'")

    x = np.linspace(0.0, 1.0, n)
    y = x.copy()
    X, Y = np.meshgrid(x, y, indexing="ij")
    dx = x[1] - x[0]

    dt = stable_dt(n, cfg)
    steps = int(np.ceil(cfg.T / dt))
    dt = cfg.T / steps  # Корректируем dt, чтобы точно попасть в время T

    u = np.zeros((n, n), dtype=np.float64)

    # Определяем индексы шагов, на которых нужно сохранить снимки (snapshots)
    save_ids = np.unique(np.linspace(0, steps, cfg.snapshot_count).round().astype(int))
    save_set = set(save_ids.tolist())
    snapshots, times = [], []

    for k in range(steps + 1):
        t = k * dt
        if k in save_set:
            snapshots.append(u.copy())
            times.append(t)
        if k == steps:
            break

        c = u[1:-1, 1:-1]  # Внутренние точки (без границ)

        if scheme == "central":
            # Центральные разности для первых производных
            ux = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dx)
            uy = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dx)
        else:
            # Направленные разности (upwind) в зависимости от знака скорости
            ux = ((c - u[:-2, 1:-1]) / dx if cfg.vx >= 0 else (u[2:, 1:-1] - c) / dx)
            uy = ((c - u[1:-1, :-2]) / dx if cfg.vy >= 0 else (u[1:-1, 2:] - c) / dx)

        # Центральные разности для вторых производных (диффузия)
        uxx = (u[2:, 1:-1] - 2.0*c + u[:-2, 1:-1]) / dx**2
        uyy = (u[1:-1, 2:] - 2.0*c + u[1:-1, :-2]) / dx**2

        # Правая часть уравнения (источник + адвекция + диффузия)
        rhs = (-cfg.vx*ux - cfg.vy*uy + cfg.kappa*(uxx+uyy)
               + source_numpy(X[1:-1, 1:-1], Y[1:-1, 1:-1], t, theta, cfg))

        un = u.copy()
        un[1:-1, 1:-1] = c + dt*rhs
        u = un  # Граничные значения остаются строго нулевыми (Dirichlet)

    return np.asarray(times), x, y, np.asarray(snapshots)


def bilinear(field: np.ndarray, xgrid: np.ndarray, ygrid: np.ndarray,
             xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
    """Билинейная интерполяция значения поля в точке (xq, yq)."""
    dx = float(xgrid[1] - xgrid[0])
    ix = np.clip(np.floor((xq - xgrid[0]) / dx).astype(int), 0, len(xgrid)-2)
    iy = np.clip(np.floor((yq - ygrid[0]) / dx).astype(int), 0, len(ygrid)-2)
    ax = (xq - xgrid[ix]) / dx
    ay = (yq - ygrid[iy]) / dx
    return ((1-ax)*(1-ay)*field[ix, iy] + ax*(1-ay)*field[ix+1, iy]
            + (1-ax)*ay*field[ix, iy+1] + ax*ay*field[ix+1, iy+1])


def interpolate_trajectory(traj, coords: np.ndarray) -> np.ndarray:
    """Интерполирует значения траектории в произвольных точках (x, y, t)."""
    times, xg, yg, fields = traj
    out = np.empty(len(coords), dtype=np.float64)
    for k, (xq, yq, tq) in enumerate(coords):
        # Находим два ближайших временных снимка
        hi = int(np.searchsorted(times, tq, side="right"))
        hi = min(max(hi, 1), len(times)-1)
        lo = hi - 1
        if tq <= times[0]:
            lo = hi = 0
        elif tq >= times[-1]:
            lo = hi = len(times)-1

        v0 = bilinear(fields[lo], xg, yg, np.array([xq]), np.array([yq]))[0]
        if hi == lo:
            out[k] = v0
        else:
            v1 = bilinear(fields[hi], xg, yg, np.array([xq]), np.array([yq]))[0]
            a = (tq-times[lo])/(times[hi]-times[lo])
            out[k] = (1-a)*v0 + a*v1  # Линейная интерполяция по времени
    return out


def sample_hf_observations(traj, cfg: ExperimentConfig, rng: np.random.Generator):
    """Выбирает случайные зашумленные наблюдения из HF-траектории."""
    times, x, y, fields = traj
    X, Y = np.meshgrid(x, y, indexing="ij")
    valid = []
    # Берем точки только после 10% времени и не на самых краях области
    for it, t in enumerate(times):
        if t >= 0.1*cfg.T:
            ii, jj = np.where((X >= 0.05) & (X <= 0.95) & (Y >= 0.05) & (Y <= 0.95))
            valid.extend((it, i, j) for i, j in zip(ii, jj))

    if len(valid) < cfg.n_obs:
        raise RuntimeError("Not enough valid HF grid points for requested observations")

    chosen = rng.choice(len(valid), size=cfg.n_obs, replace=False)
    ids = [valid[i] for i in chosen]
    it = np.asarray([z[0] for z in ids])
    ix = np.asarray([z[1] for z in ids])
    iy = np.asarray([z[2] for z in ids])

    values = fields[it, ix, iy]
    noise_std = cfg.noise_fraction * (np.max(np.abs(fields)) + 1e-15)
    noisy = values + rng.normal(0.0, noise_std, size=len(values))
    coords = np.column_stack([x[ix], y[iy], times[it]])

    return coords.astype(np.float32), noisy.astype(np.float32), float(noise_std)


def sample_operator_query_points(theta: np.ndarray, cfg: ExperimentConfig,
                                 n_points: int, rng: np.random.Generator) -> np.ndarray:
    """Генерирует точки запроса для обучения оператора (с фокусом на траектории источника)."""
    tq = rng.uniform(0.02*cfg.T, cfg.T, n_points)
    xq = rng.uniform(0.02, 0.98, n_points)
    yq = rng.uniform(0.02, 0.98, n_points)

    # 55% точек концентрируем вокруг ожидаемой траектории адвекции источника
    near = rng.random(n_points) < 0.55
    xc = theta[1] + cfg.vx*tq
    yc = theta[2] + cfg.vy*tq
    xq[near] = np.clip(rng.normal(xc[near], 0.14), 0.02, 0.98)
    yq[near] = np.clip(rng.normal(yc[near], 0.14), 0.02, 0.98)

    return np.column_stack([xq, yq, tq]).astype(np.float32)
