# Байесовский подход к прогнозу отказов накопителей и оптимизация экономических рисков

Воспроизводимый **end-to-end** pipeline:

**Данные → модели (Weibull AFT + CatBoost-ансамбль) → экономическая симуляция → Monte Carlo → многокритериальная оптимизация (NSGA-II) → Pareto-front.**

## Требования

- Python **3.11+**
- Зависимости: `pyproject.toml`

## Установка

```bash
cd diploma_v2
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Быстрый старт (синтетика, без CSV)

```bash
python scripts/run_full_pipeline.py
```

Результаты: каталоги `outputs/` и `reports/` (метрики, графики ROC/PR/calibration, Pareto, CSV).

## Отдельные шаги

```bash
python scripts/generate_synthetic_data.py   # data/raw/synthetic_disks.csv
python scripts/train_models.py
python scripts/evaluate_models.py
python scripts/optimize_policy.py
```

## Реальный CSV

В `configs/default.yaml` укажите:

```yaml
data:
  csv_path: data/raw/your_smart.csv
```

Ожидаемые колонки (имена можно переопределить в `features:`):

- `disk_id`, `snapshot_date`, `model_type`
- `age_days`, SMART-признаки из конфига
- `failure_in_horizon` (0/1), `duration_remaining`, `event_observed`

**Допущения:** если в реальных данных другие имена — приведите их к схеме или измените `configs/default.yaml`.

## Структура и ключевые функции

| Функция | Модуль |
|--------|--------|
| `prepare_dataset` | `src/data.py` |
| `train_baseline_survival` | `src/models/baseline_survival.py` |
| `train_catboost_with_uncertainty`, `predict_with_uncertainty` | `src/models/catboost_uncertainty.py` |
| `evaluate_models` | `src/evaluation.py` |
| `simulate_policy_once` | `src/economic_model.py` |
| `run_monte_carlo_for_policy`, `compute_mean_and_cvar` | `src/monte_carlo.py` |
| `optimize_policy_with_pymoo` | `src/optimization.py` |
| `plot_pareto_front` | `src/plots.py` |
| `run_full_pipeline` | `src/pipeline.py` |

## Интерпретация «байесовского» подхода

Необязательна полная байесовская нейросеть: используется **распределение неопределённости прогноза** (ансамбль CatBoost → разброс по объектам) и **распределения экономических параметров**, совмещённые через Monte Carlo и CVaR.

## Тесты

```bash
pytest tests/ -q
```
