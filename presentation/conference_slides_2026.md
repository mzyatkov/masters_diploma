# Байесовский подход к предсказанию отказов внешних накопителей

## с оптимизацией экономических рисков

- Автор: Михаил
- Научный руководитель: Александр Стерлигов
- Формат: научная конференция, 12-15 минут
- Научная новизна (1 фраза): decision-aware связка `object-wise uncertainty -> Monte Carlo/CVaR -> Pareto policy`, а не изолированное сравнение моделей.

**Инфографика:** схема высокого уровня `данные -> прогноз -> экономика -> Monte Carlo -> Pareto`.

---

# 1. Проблема и практическая цель

- Входы в реальной эксплуатации заданы **распределениями**, а не точными числами.
- Нужна система, которая выдает не только прогноз отказа, но и **управленческое решение**.
- Цель: выбирать политику обслуживания/закупки, максимизируя прибыль с учетом риска.

**Инфографика:** блок-схема «неопределенные параметры -> решение по политике».

---

# 2. Формальная постановка под неопределенностью

- В работе **отказ** — это момент вывода диска из эксплуатации (аварийная замена/перевод в read-only контуром хранения), а не только физическое разрушение.
- Scope: не рассматриваем active scanning (`badblocks`) и программный remap как регулярную policy, потому что они дают неприемлемый I/O/CPU/RAM overhead в production.
- Экономические параметры: `purchase_cost`, `downtime_cost`, `emergency_cost`, `lead_time` и др. — случайные величины.
- Модель прогнозирования отказа также имеет неопределенность.
- Для policy `X` получаем массив `samples` прибыли из Monte Carlo.

Функции цели:

- `ExpectedProfit(X) = mean(samples)`
- `CVaR_alpha(X) = mean(worst alpha * N samples)`

**Инфографика:** левый хвост распределения прибыли и выделение CVaR-зоны.

---

# 3. Почему оптимизировать только среднее недостаточно

- Максимум среднего может приводить к «тяжелому» левому хвосту.
- Для эксплуатации важно ограничить потери в худших сценариях.
- Поэтому оптимизация ведется по двум критериям: `mean` и `CVaR`.

**Инфографика:** сравнение двух стратегий с одинаковым mean и разным CVaR.

---

# 4. Существующие решения и исследовательский gap

Что уже есть в литературе и практике:

- survival/reliability подходы (Cox/Weibull/AFT) для time-to-failure;
- ML-классификация по SMART (GBM/CatBoost/XGBoost) для `failure@H`;
- cost-sensitive maintenance и stochastic optimization в OR;
- риск-метрики (VaR/CVaR) в финтех и supply chain, реже в storage maintenance end-to-end.
- открытые эксплуатационные источники SMART-телеметрии (например, annual Backblaze drive stats) как эмпирическая база для failure analytics.

Проблема большинства работ: разрыв между точностью прогноза и управленческим решением под неопределенностью.
Важно: сравнение с литературой здесь **качественное** (позиционирование по классам подходов), а не прямой cross-paper benchmark.

Наш фокус: связать `object-wise uncertainty` прогноза с экономическим decision layer и Pareto-выбором policy.

**Инфографика:** таблица `подход -> сильные стороны -> ограничение -> что добавляем`.

---

# 5. Научный вклад и проверяемые тезисы

Исследовательский вопрос (RQ):

- как использовать вероятностный прогноз отказа с object-wise uncertainty, чтобы выбирать policy обслуживания/закупки, устойчивую по `ExpectedProfit` и `CVaR_alpha`?

Тезис T1:

- policy, выбранная по joint-критерию `mean + CVaR`, дает более устойчивый профиль хвостовых потерь, чем оптимизация только по mean.

Тезис T2:

- object-wise uncertainty полезнее глобальной confidence-оценки для экономического decision-making.

Тезис T3:

- лучшая ML-метрика не гарантирует лучшую экономику вне совместной настройки model+policy.

Практический вклад:

- воспроизводимый pipeline от данных до Pareto-front и набора интерпретируемых policy.

**Инфографика:** `гипотеза -> как проверяем -> какой артефакт`.

---

# 6. Данные и воспроизводимый pipeline

- Pipeline реализован end-to-end: `src/pipeline.py`, запуск `scripts/run_full_pipeline.py`.
- Реальных отказов в контуре МТС пока мало (облако эксплуатируется < 2 лет), поэтому синтетика используется для устойчивой валидации decision-пайплайна.
- Поддержка реального CSV и синтетики; подготовка датасета: `src/data.py`.
- Признаки и target на горизонте `H`: `src/features.py`.
- Анти-утечки: time-aware split и разделение по `disk_id`.
- Синтетическая генерация строится не только по маргинальным распределениям: добавлены зависимости во времени (цепочки состояний/табличные генераторы типа CTGAN-подхода).

**Инфографика:** таблица «disk_id, SMART, age_days, model_type -> failure@H».

---

# 7. Протокол эксперимента и валидность

- Сравнение baseline и main model проводится на одинаковом target и одинаковом split.
- Метрики прогноза: ROC-AUC, PR-AUC, Brier, ECE, MCE.
- Decision-оценка: Monte Carlo распределение прибыли, `ExpectedProfit`, `CVaR_alpha`.
- Все ключевые результаты сохраняются в `reports/` для аудита и воспроизводимости.

Минимальные допущения (явно фиксируются):

- параметрические распределения экономических величин;
- synthetic data используется для проверки связки `model -> economy -> optimization`, а внешние датасеты (например, Backblaze) — для sanity-check предсказательной силы;
- выводы интерпретируются как pipeline-level, а не как claim «универсального доминирования» одной модели.

Угрозы валидности:

- **внешняя валидность:** переносимость результатов зависит от калибровки экономических распределений под конкретный контур эксплуатации;
- **внутренняя валидность:** fixed-policy сравнение не заменяет joint-tuning модели и policy.

**Инфографика:** схема `данные/split -> метрики -> economic simulation -> optimization`.

---

# 8. Модельный стек: baseline и main model

- **Baseline:** `Weibull survival` (`src/models/baseline_survival.py`).
- **Основная модель:** `CatBoost ensemble` с неопределенностью (`src/models/catboost_uncertainty.py`).
- Фокус на 2 моделях и глубоком анализе, без «зоопарка» методов.

**Инфографика:** карточка-сравнение «роль, выход, ограничение».

---

# 9. Object-wise uncertainty в CatBoost

- Формулировка «байесовский CatBoost» в работе используется как **Bayesian approximation**, а не как полный байесовский вывод.
- Виртуальный ансамбль/bootstrapping дает практичную оценку эпистемической и алеаторной неопределенности без вычислительной цены BNN.
- Для каждого диска получаем:
  - `mean predicted p_fail`
  - оценку разброса (variance / ensemble spread)
- Это позволяет учитывать неоднородное качество прогноза по объектам.

**Инфографика:** 3 диска с одинаковым `mean p_fail`, но разным uncertainty.

---

# 10. Сравнение качества прогноза (фактические результаты)

Источник: `reports/model_metrics.csv`.


| Model             | ROC-AUC    | PR-AUC     | Brier      | ECE        | MCE        |
| ----------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| weibull_aft       | 0.9822     | 0.7630     | 0.0368     | 0.4065     | 0.7880     |
| catboost_ensemble | **0.9920** | **0.8688** | **0.0154** | **0.2091** | **0.5457** |


Вывод: CatBoost устойчиво лучше как вероятностный предиктор.

Статистическая оговорка:

- текущие значения — point estimates для данного прогона;
- для строгого статистического вывода нужны повторные запуски и bootstrap/CI по ключевым метрикам.

**Инфографика:** `reports/roc.png`, `reports/pr.png`, `reports/calibration.png` (при наличии) + таблица выше.

---

# 11. Экономическая модель и policy-переменные

Decision vector:

- `tau_replace` — порог превентивной замены,
- `safety_stock` — минимальный запас,
- `order_qty` — размер заказа.

Логика:

- если `predicted_risk > tau_replace` -> превентивная замена,
- если `stock < safety_stock` -> заказ `order_qty`,
- при аварийном отказе добавляются emergency + downtime издержки.

Явная функция прибыли:

`P = Revenue - (C_preventive + C_emergency + Penalty_downtime + C_holding)`.

Ключевая асимметрия: `Penalty_downtime` обычно на порядки выше цены превентивной замены, поэтому ложноотрицательный прогноз экономически намного дороже ложноположительного.

**Инфографика:** decision-flowchart (ветвление по порогу и складу).

---

# 12. Monte Carlo и расчет CVaR

Алгоритм для фиксированной policy:

1. Сэмплируем экономические параметры из распределений (обычный Monte Carlo, **не MCMC**).
2. Сэмплируем/учитываем uncertainty предсказаний модели.
3. Считаем прибыль в сценарии.
4. Повторяем много раз -> `samples`.
5. Считаем `mean` и `CVaR_alpha`.

Реализация: `src/monte_carlo.py`.

**Инфографика:** 5-шаговая схема + гистограмма `samples`.

---

# 13. Оптимизация mean/CVaR через NSGA-II (pymoo)

- Реализация: `src/optimization.py`.
- Оптимизируем:
  - `f1 = -ExpectedProfit`
  - `f2 = -CVaR_alpha`
- Результат: не одна точка, а множество компромиссов (Pareto-front).

Источники результатов:

- `reports/pareto_front.csv`
- `reports/pareto_population_all.csv`

**Инфографика:** `reports/pareto_front.png` и `reports/pareto_population_full.png` (при наличии).

---

# 14. Три управленческие политики из оптимизации

Источник: `reports/optimization_highlights.json`.


| Policy       | tau_replace | safety_stock | order_qty |
| ------------ | ----------- | ------------ | --------- |
| risk_neutral | 0.4187      | 3.9410       | 41.2001   |
| risk_averse  | 0.4865      | 4.8353       | 69.3642   |
| knee         | 0.4330      | 4.9205       | 46.6651   |


Интерпретация: более риск-averse политика поднимает порог и заметно увеличивает объем заказа.

**Инфографика:** bar chart по трем policy.

---

# 15. Fixed-policy проверка: до и после калибровки

Источник: `reports/economic_metrics_fixed_policy.csv`, `reports/economic_metrics_fixed_policy_realized.csv`, `reports/model_metrics.csv`.

`Monte Carlo mean/std` (стохастическая оценка):

| Model                      | mean_profit | std_profit |
| -------------------------- | ----------- | ---------- |
| weibull_aft                | 44,425.76   | 519.77     |
| catboost_ensemble          | 42,636.70   | 892.52     |
| weibull_aft_isotonic       | 43,597.52   | 390.53     |
| catboost_ensemble_isotonic | 43,385.52   | 501.50     |

`Realized policy-run` (фактический прогон policy на тестовом контуре):

| Model                      | realized_profit |
| -------------------------- | --------------- |
| weibull_aft                | 42,157.36       |
| catboost_ensemble          | 43,149.33       |
| weibull_aft_isotonic       | 42,786.13       |
| catboost_ensemble_isotonic | **43,653.75**   |

Вывод и интерпретация:

- После исправлений CatBoost (особенно с isotonic) дает лучший realized-profit на fixed-policy прогоне.
- Isotonic-калибровка повышает пригодность вероятностей для пороговых и экономических решений.
- Monte Carlo-агрегаты и realized-run показывают разные срезы, поэтому в защите важно демонстрировать оба.
- Итоговый выбор делается decision-aware: по `ExpectedProfit + CVaR` на Pareto-фронте.
- Для оффлайн планирования текущая вычислительная сложность приемлема; для SaaS потребуется ускорение.

**Инфографика:** сравнительная диаграмма `MC mean/std` vs `realized_profit`.

---

# 16. Воспроизводимость и практическая применимость

- Запуск end-to-end: `scripts/run_full_pipeline.py`.
- Оркестрация этапов: `src/pipeline.py`.
- Проверка smoke-сценария: `tests/test_pipeline_smoke.py`.
- Артефакты результатов: `outputs/`, `reports/`.

**Инфографика:** «1 команда -> модели, метрики, Pareto, highlights».

---

# 17. Выводы

- Решение воспроизводимо и прозрачно по всей цепочке: `данные -> прогноз -> policy`.
- Неопределенность встроена в decision layer через Monte Carlo и CVaR.
- Pareto-front дает управляемый выбор между доходностью и риском.
- Научный итог: показана важность совместной оптимизации `модель + policy`, а не сравнения моделей только по ROC/PR.
- Практический итог: система рекомендует **когда менять диски и как пополнять запас** под нужный риск-профиль.
- Следующий этап: макроэкономический блок (прогноз цен/инфляции), trade-off пассивного мониторинга vs активного scrubbing, и ускорение пайплайна до SaaS-формата.

**Инфографика:** 3 takeaway-блока + roadmap.

---

# 18. Список литературы (ядро)

1. Jardine, Lin, Banjevic. A review on machinery diagnostics and prognostics implementing condition-based maintenance. *Mechanical Systems and Signal Processing*, 2006.
2. Elkan. The foundations of cost-sensitive learning. *IJCAI*, 2001.
3. Koenker, Bassett. Regression quantiles. *Econometrica*, 1978.
4. Rockafellar, Uryasev. Optimization of Conditional Value-at-Risk. *Journal of Risk*, 2000.
5. Deb et al. A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE TEC*, 2002.
6. Chen, Guestrin. XGBoost: A scalable tree boosting system. *KDD*, 2016.
7. Prokhorenkova et al. CatBoost: unbiased boosting with categorical features. *NeurIPS*, 2018. URL: [https://arxiv.org/abs/1810.11363](https://arxiv.org/abs/1810.11363)
8. Backblaze. Hard Drive Test Data (SMART + failure telemetry). *Backblaze Resource Page*, 2024. URL: [https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)
9. Основные отчеты проекта: `reports/model_metrics.csv`, `reports/pareto_front.csv`, `reports/optimization_highlights.json`.

