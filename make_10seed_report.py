"""Агрегация результатов 10 сидов и построение диагностик mean/std."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Бэкенд без GUI, обязателен до импорта pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Ключи метрик должны совпадать с теми, что записываются в CSV
METRICS = [
    "relerr_Q", "relerr_xs", "relerr_ys", "param_relerr_mean",
    "test_RMSE", "test_MAE", "test_relL2", "test_interior_RMSE",
    "test_interior_MAE", "test_interior_relL2", "observation_RMSE",
    "PDE_residual_RMS_dimensionless", "training_seconds", "trainable_parameters",
    "lbfgsb_success", "lbfgsb_nit", "lbfgsb_nfev",
    "lf_pretraining_seconds", "joint_inverse_training_seconds",
    "lf_solver_generation_seconds", "end_to_end_method_seconds_excluding_shared_HF_data"
]

# Словарь для красивых названий методов на графиках и в отчете
METHOD_TITLES = {
    "PINN_Raissi": "PINN-Raissi",
    "M1_PI_DeepONet": "M1 PI-DeepONet"
}


def _histories(root, method):
    """Считывает историю обучения из JSON-файлов для заданного метода."""
    rows = []
    for seed_dir in sorted(root.glob("seed_*")):
        p = seed_dir / f"history_{method}.json"
        if p.exists():
            try:
                h = json.loads(p.read_text(encoding="utf-8"))
                if h:
                    rows.append((seed_dir.name, h))
            except Exception:
                pass
    return rows


def _plot_loss_bands(root, figdir):
    """Строит графики кривых потерь с полосами среднего и стандартного отклонения."""
    for method in METHOD_TITLES:
        histories = _histories(root, method)
        if not histories:
            continue

        xmax = max(max(float(r["iteration"]) for r in h) for _, h in histories)
        grid = np.linspace(0, xmax, 240)

        for field, title in [
            ("loss", "Целевая функция (Total objective)"),
            ("pde", "Среднеквадратичная невязка PDE")
        ]:
            curves = []
            for _, hist in histories:
                hist = sorted(hist, key=lambda r: float(r["iteration"]))
                x = np.asarray([float(r["iteration"]) for r in hist])
                y = np.asarray([max(float(r[field]), 1e-16) for r in hist])

                # Дубликаты итераций могут быть на границе Adam/L-BFGS; оставляем последние
                uniq, idx = np.unique(x, return_index=True)
                y = y[idx]
                if len(uniq) >= 2:
                    curves.append(np.exp(np.interp(grid, uniq, np.log(y))))

            if not curves:
                continue

            a = np.vstack(curves)
            mean = a.mean(axis=0)
            std = a.std(axis=0, ddof=1) if len(a) > 1 else np.zeros_like(mean)

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(grid, mean, label=f"среднее ({len(curves)} запусков)")
            ax.fill_between(
                grid, np.maximum(mean - std, 1e-16), mean + std,
                alpha=0.22, label="±1 станд. откл."
            )
            ax.set_yscale("log")
            ax.set_xlabel("итерация оптимизатора")
            ax.set_ylabel(field)
            ax.set_title(f"{METHOD_TITLES[method]} — {title}")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(figdir / f"{method}_{field}_mean_std.png", dpi=180)
            plt.close(fig)


def _plot_parameter_error_by_seed(df, figdir):
    """Строит диаграммы рассеяния с ошибками для заданных метрик по всем сидам."""
    metric_titles = {
        "test_interior_RMSE": "RMSE на внутреннем HF-тесте (исключая ГУ/НУ)",
        "param_relerr_mean": "Средняя относительная ошибка параметров",
        "PDE_residual_RMS_dimensionless": "Среднеквадратичная невязка PDE"
    }

    for metric, title in metric_titles.items():
        if metric not in df:
            continue

        methods = [m for m in METHOD_TITLES if m in set(df.method)]
        fig, ax = plt.subplots(figsize=(7, 4))

        for i, m in enumerate(methods):
            vals = pd.to_numeric(
                df.loc[df.method == m, metric], errors="coerce"
            ).dropna().to_numpy()

            ax.scatter(np.full(len(vals), i), vals, alpha=0.8, s=28)
            if len(vals):
                mean = float(vals.mean())
                sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                ax.errorbar(
                    i, mean, yerr=sd, fmt="D", capsize=5,
                    label=f"{METHOD_TITLES[m]} среднее ± станд. откл."
                )

        ax.set_xticks(range(len(methods)), [METHOD_TITLES[m] for m in methods])
        ax.set_ylabel(metric)
        ax.set_title(f"{title} по сидам")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figdir / f"{metric}_by_seed.png", dpi=180)
        plt.close(fig)


def _plot_mean_fields(root, figdir):
    """Строит усредненные по сидам поля: эталон, среднее предсказание и ошибка."""
    for method in METHOD_TITLES:
        preds = []
        truths = None
        axes = None

        for seed_dir in sorted(root.glob("seed_*")):
            p = seed_dir / "predictions_plot.npz"
            if not p.exists():
                continue
            with np.load(p) as z:
                if method not in z:
                    continue
                y = np.asarray(z[method], dtype=float)
                t = np.asarray(z["plot_truth"], dtype=float)
                if truths is None:
                    truths = t
                    axes = (
                        z["plot_times"].copy(),
                        z["plot_x_axis"].copy(),
                        z["plot_y_axis"].copy()
                    )
                elif y.shape != preds[0].shape:
                    continue
                preds.append(y)

        if not preds:
            continue

        pred_mean = np.mean(np.vstack(preds), axis=0)
        abs_error = np.mean(np.abs(np.vstack(preds) - truths[None, :]), axis=0)

        nt = len(axes[0])
        nx = len(axes[1])
        ny = len(axes[2])
        it = nt // 2

        ref = truths.reshape(nt, nx, ny)[it]
        pm = pred_mean.reshape(nt, nx, ny)[it]
        ae = abs_error.reshape(nt, nx, ny)[it]

        fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
        plot_titles = [
            "Эталон HF",
            f"{METHOD_TITLES[method]} среднее",
            "Средняя абсолютная ошибка"
        ]

        for ax, data, title in zip(axs, [ref, pm, ae], plot_titles):
            im = ax.imshow(
                data.T, origin="lower", extent=[0, 1, 0, 1], aspect="equal"
            )
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.82)
        fig.suptitle(
            f"Усредненное поле по {len(preds)} запускам при t={axes[0][it]:.3f}"
        )
        fig.subplots_adjust(left=0.06, right=0.90, bottom=0.12, top=0.82, wspace=0.28)
        fig.savefig(figdir / f"{method}_mean_field.png", dpi=180)
        plt.close(fig)


def main():
    """Точка входа CLI для агрегации результатов и генерации отчета."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results_10seeds")
    a = ap.parse_args()

    root = Path(a.output)
    per_path = root / "per_seed_results.csv"
    if not per_path.exists():
        raise FileNotFoundError(per_path)

    df = pd.read_csv(per_path)
    summary = []

    for method, g in df.groupby("method"):
        row = {
            "method": method,
            "n_runs": int(g.seed.nunique()) if "seed" in g else int(len(g))
        }
        for metric in METRICS:
            if metric in g:
                v = pd.to_numeric(g[metric], errors="coerce").dropna()
                row[f"{metric}_mean"] = float(v.mean()) if len(v) else np.nan
                row[f"{metric}_std"] = (
                    float(v.std(ddof=1)) if len(v) > 1
                    else (0.0 if len(v) == 1 else np.nan)
                )
        summary.append(row)

    sm = pd.DataFrame(summary)
    sm.to_csv(root / "mean_std_summary.csv", index=False)

    figdir = root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    _plot_parameter_error_by_seed(df, figdir)
    _plot_loss_bands(root, figdir)
    _plot_mean_fields(root, figdir)

    failures = (
        json.loads((root / "failures.json").read_text())
        if (root / "failures.json").exists() else []
    )

    protocol = (
        "В качестве нейросетевых методов включены только PINN-Raissi и "
        "M1 PI-DeepONet. По умолчанию LF использует центральные разности "
        "на сетке 21×21; HF — на сетке 129×129. M1 использует шесть членов "
        "потерь из Eq. (15) Howard плюс явный член HF-наблюдений для обратной "
        "идентификации источника. Метрики по полной области HF включают "
        "предписанные граничные и начальные условия; метрики по внутренней "
        "области HF исключают граничные узлы x/y и t=0. `lbfgsb_success_mean` "
        "— это эмпирическая доля сходимости оптимизатора среди завершенных "
        "запусков, а не доля сидов, выдавших какой-либо вывод."
    )

    lines = [
        "# Эксп. 1 — Сравнение PINN и PI-DeepONet (10 сидов)", "",
        "## Агрегированные результаты (среднее ± выборочное станд. отклонение)",
        sm.to_markdown(index=False, floatfmt=".5g"), "",
        "## Результаты по каждому сиду", "См. `per_seed_results.csv`.", "",
        f"## Сбои или незавершенные запуски: {len(failures)}",
        "См. `failures.json`; упавшие сиды не заменяются молча на ноль.",
        "",
        "## Протокол", protocol, "",
        "## Рисунки",
    ]

    for p in sorted(figdir.glob("*.png")):
        lines.append(f"- figures/{p.name}")

    (root / "report_10seeds.md").write_text("\n".join(lines), encoding="utf-8")
    print(root / "report_10seeds.md")


if __name__ == "__main__":
    main()
