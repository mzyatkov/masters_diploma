# Матрица доказательств для ответов комиссии

| Вопрос | Факт из кода/результатов | Метрика/артефакт | Вывод для защиты |
|---|---|---|---|
| Инфляция и скачки цен | Экономические параметры сэмплируются из распределений в MC, без явного inflation trend | `src/economic_model.py`, `configs/default.yaml` | Стохастика цен учтена, но тренд инфляции и regime-shock требуют отдельного блока |
| Условия эксплуатации | В модели используются `temperature_c`, SMART и модификатор надежности по типу | `src/data.py`, `configs/default.yaml` | Учет условий частичный (через признаки и типы), не физическая средовая модель |
| Чем измеряется эффективность | Оптимизация по `ExpectedProfit` и `CVaR`; фиксированная policy: `mean_profit`, `std_profit` | `src/monte_carlo.py`, `src/evaluation.py`, `reports/economic_metrics_fixed_policy.csv` | Эффективность decision-aware: доходность + устойчивость хвоста |
| Почему SMART | SMART поля — основной вход признаков в pipeline | `configs/default.yaml`, `README.md` | SMART выбран как тиражируемый базовый источник телеметрии |
| Учет remap | В симуляции есть превентивная/аварийная замена, отдельного `remap` action нет | `src/economic_model.py` | Remap сейчас вне scope decision layer, но возможен как расширение |
| Источник обучающей выборки | По умолчанию `csv_path: null` -> генерация синтетики; CSV режим поддерживается | `configs/default.yaml`, `src/data.py`, `README.md` | Текущий прогон воспроизводим на синтетике, для реального парка есть путь подключения |
| Сложность параметрической модели | Baseline = Weibull AFT; основная модель = CatBoost bootstrap ensemble с заданными гиперпараметрами | `src/models/baseline_survival.py`, `src/models/catboost_uncertainty.py` | Сложность контролируется конфигом и ограничением числа признаков/итераций |
| Анализ надежности прогнозов | ROC-AUC, PR-AUC, Brier, ECE, MCE + calibration plot + calibrated metrics | `src/evaluation.py`, `reports/model_metrics.csv`, `reports/calibration.png` | Надежность вероятностей анализируется явно, не только ranking quality |
| Корреляции в синтетике vs реальных данных | Генератор задает структурные зависимости через возраст/латентный lifetime/тип | `src/data.py` | Корреляции частично моделируются механизмом генерации, но не калиброваны на реальную матрицу |

## Мини-блок для устной защиты

- Текущая версия уже закрывает pipeline `данные -> прогноз -> решение -> риск`.
- Ограничения проговорены явно: `inflation/shocks`, `remap`, `synthetic-vs-real correlation`.
- Доработки реализуются эволюционно без ломки архитектуры.
