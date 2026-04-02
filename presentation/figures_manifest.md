# Инфографика для доклада: реестр и единый стиль

Документ фиксирует, какие визуализации используются в презентации, откуда они берутся, и как должны подписываться в едином стиле.

## Единый стиль подписи и легенд

- Язык подписей: русский.
- Формат подписи под рисунком: `Рисунок N - <что показано> (источник: <файл>)`.
- Обозначения моделей везде одинаковые:
  - `Weibull baseline`
  - `CatBoost ensemble (uncertainty-aware)`
- Оси для метрик:
  - `ROC-AUC`, `PR-AUC`, `Brier`, `ECE`, `MCE` (без сокращений в легендах, сокращения допустимы в заголовке).
- Оси для экономических графиков:
  - `Expected profit, RUB`
  - `CVaR_0.05, RUB`
  - `Policy variable value`

## Карта инфографики по слайдам

| Слайд | Инфографика | Источник | Статус |
|---|---|---|---|
| 1 | Общий pipeline (data -> model -> economy -> MC -> Pareto) | Рисунок-схема в презентации | Готовится в слайде |
| 2 | Неопределенные входы -> распределение прибыли | Схема в презентации | Готовится в слайде |
| 3 | Левый хвост распределения и зона CVaR | Схема в презентации | Готовится в слайде |
| 4 | Данные и целевая переменная на горизонте H | Табличная схема в презентации | Готовится в слайде |
| 5 | Baseline vs main model | Сравнительная карточка в презентации | Готовится в слайде |
| 6 | Object-wise uncertainty (3 диска: same mean, different variance) | Мини-диаграмма в презентации | Готовится в слайде |
| 7 | Метрики качества моделей | `reports/model_metrics.csv` | Готово (табличные данные) |
| 7 | ROC curve | `reports/roc.png` | При наличии файла |
| 7 | PR curve | `reports/pr.png` | При наличии файла |
| 7 | Calibration curve | `reports/calibration.png` | При наличии файла |
| 8 | Decision flow экономической политики | Схема в презентации | Готовится в слайде |
| 9 | Monte Carlo workflow (4 шага) | Блок-схема в презентации | Готовится в слайде |
| 10 | Pareto front + full population | `reports/pareto_front.csv`, `reports/pareto_population_all.csv` | Готово (табличные данные) |
| 10 | Pareto front plot | `reports/pareto_front.png` | При наличии файла |
| 10 | Pareto population plot | `reports/pareto_population_full.png` | При наличии файла |
| 11 | Три политики: risk-neutral, risk-averse, knee | `reports/optimization_highlights.json` | Готово |
| 12 | Экономика на фиксированной policy | `reports/economic_metrics_fixed_policy.csv` | Готово |

## Источники чисел для слайдов

- Метрики моделей: `reports/model_metrics.csv`
- Экономика фиксированной policy: `reports/economic_metrics_fixed_policy.csv`
- Фронт Парето: `reports/pareto_front.csv`
- Полная популяция NSGA-II: `reports/pareto_population_all.csv`
- Highlight-политики: `reports/optimization_highlights.json`

## Примечание по воспроизводимости инфографики

Если части PNG-графиков отсутствуют, их следует регенерировать полным прогоном пайплайна:

- точка входа: `scripts/run_full_pipeline.py`
- оркестрация: `src/pipeline.py`
- функции визуализации: `src/plots.py`

Для доклада достаточно использовать текущие CSV-артефакты (они уже содержат численные результаты), даже если отдельные PNG временно отсутствуют.
