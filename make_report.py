"""Формирование самодостаточного HTML/Markdown отчёта для PINN-Raissi и M1 PI-DeepONet."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

# Бэкенд без GUI; обязан задаваться ДО импорта pyplot, иначе на машинах без
# дисплея (или при сборке отчёта в headless-окружении) всё упадёт.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METHODS = ["M1_PI_DeepONet", "PINN_Raissi"]
TITLES = {
    "M1_PI_DeepONet": "M1 — PI-DeepONet",
    "PINN_Raissi": "PINN-Raissi",
}


def _float_fmt(v: float) -> str:
    """Форматирование чисел с плавающей точкой для HTML-таблицы."""
    return f"{v:.5g}"


def read_json(path):
    """Читает JSON-файл, если он существует, иначе возвращает пустой словарь."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fig_fields(pred, out):
    """Строит графики эталонного HF-поля и предсказаний с выделенной осью цветовой шкалы."""
    nt = len(pred["plot_times"])
    nx = len(pred["plot_x_axis"])
    ny = len(pred["plot_y_axis"])
    truth = np.asarray(pred["plot_truth"]).reshape(nt, nx, ny)
    available = [m for m in METHODS if m in pred]
    if not available:
        return

    fields = {m: np.asarray(pred[m]).reshape(nt, nx, ny) for m in available}
    rows = [("Эталон HF", None)] + [(TITLES[m], m) for m in available]
    its = sorted({max(0, nt // 4), nt // 2, min(nt - 1, 3 * nt // 4)})

    # Единый цветовой диапазон для эталона и всех предсказаний модели.
    arrays = [truth] + list(fields.values())
    vmin = min(float(a.min()) for a in arrays)
    vmax = max(float(a.max()) for a in arrays)
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    fig, axs = plt.subplots(
        len(rows), len(its),
        figsize=(4.1 * len(its), 3.2 * len(rows)),
        squeeze=False,
    )

    for r, (label, method) in enumerate(rows):
        for c, it in enumerate(its):
            field = truth[it] if method is None else fields[method][it]
            im = axs[r, c].imshow(
                field.T,
                origin="lower",
                extent=[0, 1, 0, 1],
                aspect="equal",
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
            )
            axs[r, c].set_xlabel("x")
            axs[r, c].set_ylabel("y", labelpad=1)
            if r == 0:
                axs[r, c].set_title(f"t={pred['plot_times'][it]:.3f}")

    # Подписи строк слева, цветовая шкала (colorbar) справа.
    fig.subplots_adjust(
        left=0.17, right=0.85, bottom=0.08, top=0.92,
        wspace=0.26, hspace=0.30,
    )
    for r, (label, _method) in enumerate(rows):
        pos = axs[r, 0].get_position()
        fig.text(
            0.035, pos.y0 + pos.height / 2.0, label,
            rotation=90, ha="center", va="center",
        )
    cax = fig.add_axes([0.89, 0.18, 0.020, 0.64])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Концентрация u", rotation=90, labelpad=10)

    fig.suptitle("Поля концентрации высокого уровня точности", y=0.985)
    fig.savefig(out / "field_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_errors(pred, out):
    """Строит карты абсолютной ошибки с отдельной общей осью для цветовой шкалы."""
    nt = len(pred["plot_times"])
    nx = len(pred["plot_x_axis"])
    ny = len(pred["plot_y_axis"])
    it = nt // 2
    truth = np.asarray(pred["plot_truth"]).reshape(nt, nx, ny)[it]
    available = [m for m in METHODS if m in pred]
    if not available:
        return

    errors = {
        m: np.abs(np.asarray(pred[m]).reshape(nt, nx, ny)[it] - truth)
        for m in available
    }
    vmax = max(float(e.max()) for e in errors.values())
    vmax = max(vmax, 1e-12)

    fig, axs = plt.subplots(
        1, len(errors),
        figsize=(5.0 * len(errors), 4.3),
        squeeze=False,
    )
    for ax, (method, error) in zip(axs[0], errors.items()):
        im = ax.imshow(
            error.T,
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="equal",
            vmin=0,
            vmax=vmax,
            cmap="magma",
        )
        ax.set_title(TITLES[method])
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.subplots_adjust(
        left=0.10, right=0.86, bottom=0.13, top=0.82, wspace=0.30,
    )
    cax = fig.add_axes([0.89, 0.18, 0.022, 0.60])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Абсолютная ошибка", rotation=90, labelpad=10)

    fig.suptitle(
        f"Карты абсолютной ошибки при t={pred['plot_times'][it]:.3f}",
        y=0.96,
    )
    fig.savefig(out / "absolute_error_maps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_training(out, figdir):
    """Строит кривые обучения (loss и PDE residual) и сходимость параметров."""
    histories = {m: read_json(out / f"history_{m}.json") for m in METHODS}
    histories = {m: h for m, h in histories.items() if h}
    if not histories:
        return

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for m, h in histories.items():
        it = [r["iteration"] for r in h]
        loss_vals = [max(r["loss"], 1e-14) for r in h]
        pde_vals = [max(r["pde"], 1e-14) for r in h]
        axs[0].semilogy(it, loss_vals, label=TITLES[m])
        axs[1].semilogy(it, pde_vals, label=TITLES[m])
    axs[0].set(
        xlabel="итерация оптимизатора",
        ylabel="полный loss",
        title="Целевая функция",
    )
    axs[1].set(
        xlabel="итерация оптимизатора",
        ylabel="безразмерный PDE residual (MSE)",
        title="Невязка уравнения",
    )
    for ax in axs:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "training_curves.png", dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for j, param in enumerate(("Q", "xs", "ys")):
        for m, h in histories.items():
            it = [r["iteration"] for r in h]
            vals = [r[param] for r in h]
            axs[j].plot(it, vals, label=TITLES[m])
        axs[j].set(
            xlabel="итерация",
            ylabel=param,
            title=f"Оценка {param}",
        )
        axs[j].grid(alpha=0.25)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "parameter_convergence.png", dpi=180)
    plt.close(fig)


def fig_parameter_errors(df, out):
    """Столбчатая диаграмма относительных ошибок Q, xs, ys для каждого метода."""
    if df.empty:
        return
    cols = ["relerr_Q", "relerr_xs", "relerr_ys"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df))
    w = 0.24
    for i, c in enumerate(cols):
        short = c.replace("relerr_", "")
        ax.bar(x + (i - 1) * w, df[c].astype(float), w, label=short)
    ax.set_xticks(x, [TITLES.get(m, m) for m in df["method"]])
    ax.set_ylabel("относительная ошибка параметра")
    ax.set_title("Идентификация параметров источника")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "parameter_errors.png", dpi=180)
    plt.close(fig)


def embed(path):
    """Кодирует PNG-файл в base64 data URI для встраивания прямо в HTML."""
    data = base64.b64encode(path.read_bytes()).decode()
    return "data:image/png;base64," + data


def _build_markdown_lines(table, direct):
    """Собирает тело Markdown-отчёта в виде списка строк (на русском)."""
    research_q = (
        "Сравнение однофидельной обратной PINN с составной мультифидельной "
        "PI-DeepONet для идентификации гауссова источника в двумерном "
        "уравнении адвекции-диффузии."
    )
    methods_pinn = (
        "- PINN-Raissi: полносвязная сеть с активацией tanh; обучаемые "
        "ограниченные параметры источника; оптимизация Adam, затем SciPy "
        "L-BFGS-B."
    )
    methods_m1 = (
        "- M1 PI-DeepONet: LF modified DeepONet + линейный HF DeepONet "
        "+ нелинейный HF modified DeepONet; HF-выход является суммой "
        "выходов линейного и нелинейного блоков."
    )
    methods_mpinn = "- MPINN-v2 исключён из данного сравнения."

    data_proto = (
        "LF-симуляции используют сетку 21×21; HF-симуляции — сетку 129×129. "
        "В обоих случаях применяется схема центральных разностей. Целевой "
        "источник исключён из ансамбля LF-параметрических функций. Обратное "
        "HF-обучение использует 80 зашумлённых наблюдений при "
        "noise_fraction=0.005. Чистые тестовые точки берутся из всех "
        "сохранённых HF grid-time точек, исключая точные совпадения "
        "координата-время с наблюдениями. Отдельная прореженная сетка "
        "используется только для визуализации."
    )
    objective = (
        "Для M1 сохранены шесть членов из Howard et al. (2023), Eq. (15): "
        "физика, LF-данные, регуляризация нелинейной branch-части, "
        "регуляризация LF-ветви, начальное условие и граничное условие. "
        "Поскольку данная задача является обратной с разреженными "
        "зашумлёнными HF-измерениями, добавлен седьмой, специфичный для "
        "задачи член λ_obs L_HF-observation. Тот же observation-loss "
        "используется в PINN-Raissi. Это явное расширение Eq. (15), а не "
        "утверждение, что Eq. (15) содержит наблюдательный член."
    )
    leakage = (
        "Чистые тестовые значения не используются в обучении, "
        "масштабировании, выборе гиперпараметров или ранней остановке. "
        "Масштаб концентрации вычисляется только по зашумлённым тренировочным "
        "наблюдениям. Разбиение LF train/validation производится по целым "
        "реализациям источника. Одни и те же HF-наблюдения, collocation-"
        "точки и выборки boundary/initial используются для обоих методов "
        "внутри каждого seed. L-BFGS-B использует фиксированные collocation, "
        "BC, IC и LF mini-batch точки на всём протяжении closure evaluations."
    )
    metrics_note = (
        "`test_RMSE/MAE/relL2` используют все чистые, ненаблюдаемые "
        "HF grid-time точки, включая аналитически заданные граничные и "
        "начальные условия. `test_interior_*` исключают четыре стороны "
        "границы и плоскость t=0 и являются более строгой метрикой поля "
        "вдали от навязанных ограничений."
    )
    direct_block = (
        json.dumps(direct, indent=2, ensure_ascii=False)
        if direct else "Не запускалось."
    )
    limitations = (
        "В качестве HF-модели используется конечно-разностный солвер "
        "проекта, а не WRF-Chem. Связывание с WRF-Chem, калибровка времени "
        "выполнения (планируемая дорогая HF-стоимость около 1/5 реального "
        "времени), мульти-сидовая статистика, cross-scheme сравнение "
        "(central vs upwind) и физическая валидация по данным атмосферной "
        "химии остаются задачами на будущее."
    )

    return [
        "# Эксп. 1 — PINN-Raissi vs M1 PI-DeepONet", "",
        "## Исследовательский вопрос", research_q, "",
        "## Методы", methods_pinn, methods_m1, methods_mpinn, "",
        "## Протокол данных и уровней точности", data_proto, "",
        "## Целевая функция", objective, "",
        "## Обучение и предотвращение утечек", leakage, "",
        "## Метрики (один seed)", table, "",
        metrics_note, "",
        "## Прямой HF-LS контроль (не нейросетевой метод)",
        direct_block, "",
        "## Воспроизведение", "```bash",
        "python -m pip install -r requirements.txt",
        "python run_exp1.py --seed 7 --model both --device cpu",
        "```", "",
        "## Текущие ограничения", limitations, "",
        "## Рисунки",
        "- figures/field_comparison.png",
        "- figures/absolute_error_maps.png",
        "- figures/parameter_errors.png",
        "- figures/training_curves.png",
        "- figures/parameter_convergence.png",
        "",
    ]


def _build_html_body(cfg, table_html, figs):
    """Собирает тело HTML-отчёта в виде одной строки (на русском)."""
    method_note = (
        "Eq. (15) из Howard et al. сохранена; для обратной идентификации "
        "добавлен специфичный для задачи high-fidelity наблюдательный член."
    )
    css = (
        "body{font:16px/1.55 system-ui,sans-serif;"
        "max-width:1180px;margin:35px auto;padding:0 20px;color:#222}"
        "table{border-collapse:collapse;width:100%;font-size:13px}"
        "th,td{border:1px solid #ddd;padding:7px;text-align:right}"
        "th:first-child,td:first-child{text-align:left}"
        "figure{margin:28px 0}"
        "img{max-width:100%;height:auto;border:1px solid #ddd}"
        "figcaption{color:#555;font-size:14px}"
        "pre{white-space:pre-wrap;background:#f5f5f5;padding:14px}"
    )
    cfg_json = json.dumps(cfg, indent=2, ensure_ascii=False)
    figs_html = "".join(figs)
    return (
        "<!doctype html><html lang=\"ru\"><head>"
        "<meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,"
        "initial-scale=1\">"
        "<title>PINN vs PI-DeepONet</title>"
        f"<style>{css}</style>"
        "</head><body>"
        "<h1>Эксп. 1 — PINN-Raissi vs M1 PI-DeepONet</h1>"
        f"<h2>Конфигурация</h2><pre>{cfg_json}</pre>"
        f"<h2>Метрики</h2>{table_html}"
        f"<h2>Рисунки</h2>{figs_html}"
        f"<h2>Примечание к методу</h2><p>{method_note}</p>"
        "</body></html>"
    )


def build_report(out_dir, dataset=None, device="cpu"):
    """Формирует Markdown и HTML отчёты для прогона с одним сидом."""
    out = Path(out_dir)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    metrics_path = out / "results_seed.csv"
    df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()

    pred_path = out / "predictions_plot.npz"
    if pred_path.exists():
        with np.load(pred_path) as raw:
            pred = {k: raw[k] for k in raw.files}
        fig_fields(pred, figdir)
        fig_errors(pred, figdir)

    fig_parameter_errors(df, figdir)
    fig_training(out, figdir)

    cfg = read_json(out / "training_metadata.json")
    direct = read_json(out / "direct_hf_ls_reference.json")

    if len(df):
        table = df.to_markdown(index=False, floatfmt=".5g")
    else:
        table = "Файл results_seed.csv не найден."

    lines = _build_markdown_lines(table, direct)
    (out / "report_exp1.md").write_text("\n".join(lines), encoding="utf-8")

    figs = []
    # Подписи к рисункам в HTML — на русском.
    captions = [
        (
            "field_comparison.png",
            "Эталон HF и предсказания методов.",
        ),
        (
            "absolute_error_maps.png",
            "Абсолютные ошибки поля концентрации.",
        ),
        (
            "parameter_errors.png",
            "Относительные ошибки параметров Q, xs и ys.",
        ),
        (
            "training_curves.png",
            "Целевая функция обучения и PDE residual.",
        ),
        (
            "parameter_convergence.png",
            "Оценки параметров источника в процессе оптимизации.",
        ),
    ]
    for fn, cap in captions:
        p = figdir / fn
        if p.exists():
            figs.append(
                f'<figure><img src="{embed(p)}">'
                f"<figcaption>{cap}</figcaption></figure>"
            )

    if len(df):
        table_html = df.to_html(index=False, border=0, float_format=_float_fmt)
    else:
        table_html = "<p>Результаты отсутствуют.</p>"

    html = _build_html_body(cfg, table_html, figs)
    (out / "report_exp1.html").write_text(html, encoding="utf-8")
    return out / "report_exp1.html"


def main():
    """Точка входа CLI."""
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results_exp1")
    p.add_argument("--dataset", default=None)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    print(build_report(a.output, a.dataset, a.device))


if __name__ == "__main__":
    main()
