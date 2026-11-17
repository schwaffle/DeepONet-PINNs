# Exp. 1 — PINN-Raissi vs M1 PI-DeepONet

Проект для воспроизводимого сравнения **двух** методов обратной идентификации гауссова источника в 2-D уравнении адвекции–диффузии:

1. `PINN_Raissi` — однофидельная inverse PINN.
2. `M1_PI_DeepONet` — составная мультифидельная physics-informed DeepONet по Howard et al. (2023).

`MPINN-v2` намеренно не включён в активный пайплайн, таблицы и графики.

## Физическая постановка

\[
 u_t + v_x u_x + v_y u_y - \kappa(u_{xx}+u_{yy}) = S(x,y,t),
\]

\[
 S=Q\exp\left(-\frac{(x-x_s)^2+(y-y_s)^2}{2\sigma_s^2}\right)
 (1-e^{-t/\tau}).
\]

Начальное условие: `u(x,y,0)=0`. Граничные условия Дирихле: `u=0` на всех четырёх сторонах квадрата `[0,1]^2`.

Основной fidelity-дизайн: LF central finite differences на `21×21`, HF central finite differences на `129×129`. В обеих моделях обратная задача обучается по одним и тем же 80 зашумлённым HF-наблюдениям (`noise_fraction=0.005`). Upwind кодируется в решателе для отдельного будущего cross-scheme исследования, но не входит в основной эксперимент.

## M1: структура и loss

M1 имеет три оператора:

- LF modified DeepONet, сначала warm-start обучаемый на 40 реализациях источника, затем размораживаемый при joint fine-tuning;
- линейный HF DeepONet `F_l` без активаций;
- нелинейный HF modified DeepONet `F_nl`.

\[
 u_{HF}=F_l(F_{LF}(S),x,y,t)+F_{nl}(S_{sensors},F_{LF}(S)_{subset},x,y,t).
\]

Для M1 реализованы шесть компонент Howard et al. (2023), Eq. (15):

\[
L_{Howard}=\lambda_1 L_{physics}+\lambda_2 L_{LF}
+\lambda_3 R_{nl}+\lambda_4 R_{LF}+\lambda_5 L_{IC}+\lambda_6 L_{BC}.
\]

Поскольку данная задача — **обратная идентификация** с 80 noisy HF measurements, добавляется седьмой task-specific член `lambda_obs * L_HFobs`. Это расширение постановки Howard; не следует писать, что `L_HFobs` входит в исходное Eq. (15). PINN использует ту же HF-observation loss, а также physics/IC/BC.

Параметры `Q,xs,ys` оптимизируются через sigmoid parameterization в заданных физических границах. Это обеспечивает bounds, но при насыщении sigmoid градиенты могут ухудшаться; см. parameter-convergence plot.

## Важные данные и предотвращение утечек

- LF train/validation разделяется **по целым реализациям источника**, не по отдельным точкам.
- Источник целевой обратной задачи не включается в LF-параметрический ансамбль.
- В `u_scale` используется максимум абсолютной величины noisy HF training observations. Clean HF test не используется для scaling, выбора checkpoint или early stopping.
- Полная HF test-метрика рассчитывается на всех сохранённых grid-time точках, кроме точного совпадения координата-время с обучающими наблюдениями. Дополнительно публикуется `test_interior_*`, исключающая известные boundary nodes и `t=0`, чтобы заданные нулевые условия не занижали ошибку поля.
- Отдельная структурированная подрешётка служит только для визуализации.
- Для обоих методов внутри seed совпадают HF observations, PDE collocation points, boundary points и initial points. У M1 есть дополнительная стоимость LF-предобучения; это часть исследуемого multifidelity метода и должна показываться отдельно во времени.

## Оптимизация

- Adam: collocation/BC/IC points resampled каждый epoch.
- SciPy `L-BFGS-B`: collocation/BC/IC и LF мини-батч фиксированы на всё время closure evaluation, как того требует квазиньютоновская оценка функции и градиента.
- Для `PINN_Raissi` архитектура по образцу Burgers inverse identification из репозитория Raissi: 3 inputs `(x,y,t)`, 8 скрытых слоёв по 20 `tanh`-нейронов, 1 scalar output, input normalization в `[-1,1]`, Xavier-style truncated normal и нулевые biases.

Это PyTorch-перенос архитектуры и оптимизационного протокола; это не запуск старого TensorFlow 1.x кода напрямую. Нулевые IC/BC добавлены для нашей PDE постановки.

## Установка

Рекомендуется Python 3.10 или 3.11.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Быстрая программная проверка

```bash
python smoke_test.py
```

Smoke-test создаёт маленький датасет, делает 2 Adam и 2 L-BFGS-B итерации, проверяет сохранение/загрузку `.pt` и inference. **Его численные значения не являются исследовательскими результатами.**

## Один seed — исследовательский запуск

```bash
python run_exp1.py --seed 7 --model both --output results_seed_7 --device cpu
```

Стартовая конфигурация по умолчанию (может быть изменена заранее и одинаково для обоих методов):

- Adam: 2,000 epochs;
- SciPy L-BFGS-B: maxiter 5,000;
- LF pretraining M1: 1,000 epochs;
- collocation points: 256;
- BC/IC points per Adam step: 128.

Для более близкого к протоколу Raissi длительного запуска PINN можно задать `--adam-epochs 10000 --lbfgs-max-iter 50000`. Значение maxiter для L-BFGS-B — верхняя граница, не гарантия сходимости.

## 10 seed

```bash
python run_10seeds.py --output results_10seeds --device cpu
```

Используются заранее зафиксированные seed `7,...,16`. Сохраняются per-seed результаты и `mean_std_summary.csv`. Сериализованные ошибки/неудачные запуска не заменяются нулями; см. `failures.json`.

## Дополнительный контроль HF-LS

```bash
python direct_inverse_reference.py --data results_seed_7/exp1_multifidelity_seed_7.npz --output results_seed_7 --seed 7
```

Это численная оптимизация через тот же HF solver, не третий нейросетевой метод.

## Inference

```bash
python inference.py --checkpoint results_seed_7/PINN_Raissi.pt --coords query_points.npz --output pinn_prediction.npy
python inference.py --checkpoint results_seed_7/M1_PI_DeepONet.pt --coords query_points.npz --output m1_prediction.npy
```

`query_points.npz` должен содержать массив `coords` формы `[N,3]` в физических единицах `(x,y,t)`.

## Состав проекта

- `solver.py` — PDE solver, central/upwind finite differences.
- `datasets.py` — LF ensemble, HF observations и clean test.
- `models.py` — PINN-Raissi, DeepONet, ModifiedDeepONet, composite M1.
- `train.py` — losses, Adam, SciPy L-BFGS-B, metrics and checkpoints.
- `run_exp1.py` — запуск одного seed.
- `run_10seeds.py` — 10-seed orchestration.
- `inference.py` — checkpoint loader and prediction.
- `make_report.py` — HTML/Markdown report and figures.
- `direct_inverse_reference.py` — Direct HF-LS control.
- `smoke_test.py` — software-only test.

## Границы текущей работы

HF пока моделируется central finite-difference solver, **не WRF-Chem**. До заявления о реалистичном multifidelity с WRF-Chem необходимо определить физические поля и единицы, mapping параметров источника, доступные observation operators, coupling/caching, benchmark runtime (цель — стоимость около `1/5` от реального времени, требующая эмпирического измерения), а также верификацию на атмосферных данных. Время запуска solver в текущем Exp. 1 нельзя выдавать за WRF-Chem runtime.
