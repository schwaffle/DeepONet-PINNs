# Методологические источники, использованные в реализации

1. **Howard, A. A., Perego, M., Karniadakis, G. E., & Stinis, P. (2023).** Multifidelity deep operator networks for data-driven and physics-informed problems. *Journal of Computational Physics*, 493, 112462. DOI: https://doi.org/10.1016/j.jcp.2023.112462
   - Источник для составной трехблочной архитектуры M1, суммы линейной и нелинейной коррекций, а также шести компонент функции потерь для физико-информированного мультифидельного обучения (Уравнение 15).

2. **Wang, S., Wang, H., & Perdikaris, P. (2021).** Learning the solution operator of parametric partial differential equations with physics-informed DeepONets. *Science Advances*, 7(40), eabi8605. DOI: https://doi.org/10.1126/sciadv.abi8605
   - Источник для архитектуры однофидельной физико-информированной DeepONet и базовых компонентов modified DeepONet. Репозиторий с исходным кодом: https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets

3. **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).** Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707. DOI: https://doi.org/10.1016/j.jcp.2018.10.045
   - Источник для базового метода (baseline) обратной PINN. Репозиторий с исходным кодом и пример обратной идентификации для уравнения Бюргерса: https://github.com/maziarraissi/PINNs

## Границы реализации

Код представляет собой адаптацию и перенос данных методологических идей на PyTorch для задачи обратной идентификации скалярного источника в двумерном уравнении адвекции-диффузии. Это не дословная копия репозиториев на TensorFlow 1.x или JAX. Точные детали адаптации и соответствующие оговорки см. в файлах `ARCHITECTURE_NOTES.md` и `AUDIT_REPORT.md`.
