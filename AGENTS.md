Сгенерируй полный Python-проект для дипломной темы:

"Байесовский подход к предсказанию выхода из строя внешних накопителей с учетом оптимизации экономических рисков"

Мне нужен не набор разрозненных скриптов, а воспроизводимый end-to-end pipeline:

ДАННЫЕ -> МОДЕЛЬ ПРОГНОЗА -> ЭКОНОМИЧЕСКАЯ МОДЕЛЬ -> MONTE CARLO -> ОПТИМИЗАЦИЯ mean/CVaR -> PARETO FRONT

========================
1. ОБЩАЯ ИДЕЯ РЕШЕНИЯ
========================

Реализуй систему поддержки решений для обслуживания и закупки внешних накопителей.

Система должна:
1) обучать модель прогнозирования отказа дисков,
2) выдавать не только прогноз, но и неопределенность прогноза,
3) использовать прогнозы в экономической модели,
4) учитывать неопределенность экономических параметров и неопределенность модели через Monte Carlo,
5) оптимизировать policy по двум критериям:
   - expected profit (средняя прибыль)
   - CVaR_alpha (риск в худших alpha случаях)
6) возвращать Pareto-front решений.

Важно: не распыляйся на 5+ моделей.
Нужно сделать:
- baseline модель,
- одну основную улучшенную модель,
- глубокий анализ.

========================
2. МОДЕЛИ
========================

Сделай две модели:

A. BASELINE:
- survival / Weibull baseline
- если удобно, используй lifelines (например WeibullAFTFitter)
- baseline должен предсказывать вероятность отказа в горизонте H дней

B. ОСНОВНАЯ МОДЕЛЬ:
- CatBoost как основная практическая модель
- задача: прогноз вероятности отказа диска в горизонте H дней по SMART-признакам и метаданным
- модель должна уметь оценивать uncertainty на уровне объекта (instance-wise), а не одной константой на весь датасет
- сначала попробуй использовать возможности CatBoost для uncertainty / posterior sampling
- если в текущей версии API это неудобно или нестабильно, реализуй fallback:
  ансамбль/bootstrapped CatBoost моделей -> по набору предсказаний считать mean и variance для каждого объекта
- результат основной модели для каждого объекта:
  - mean predicted failure probability
  - predictive variance или набор сэмплов/ensemble predictions

Важно: интерпретируй "байесовский подход" как принятие решений по распределениям и неопределенности, а не как обязательную реализацию сложной полной Bayesian neural network.

========================
3. ДАННЫЕ И ПРЕПРОЦЕССИНГ
========================

Сделай код так, чтобы он работал:
- либо на реальном CSV-датасете (например, SMART-данные),
- либо на синтетическом демо-датасете, если реальных данных нет.

Нужны модули:
- загрузка данных
- очистка / preprocessing
- формирование признаков
- target construction:
  бинарная цель: отказ в горизонте H дней
  и/или survival target для baseline
- train/validation/test split
- избегать утечки по времени и по disk_id:
  - по возможности сделать time-based split
  - одинаковый диск не должен оказаться и в train, и в test в некорректной форме

Если реального датасета нет, создай synthetic generator, имитирующий:
- disk_id
- model_type
- age_days
- SMART features
- failure event
- costs by disk type
- heteroscedastic uncertainty (например, дешевые диски предсказываются лучше, плохие/редкие хуже)

========================
4. МЕТРИКИ КАЧЕСТВА МОДЕЛЕЙ
========================

Сравни baseline и основную модель на одних и тех же данных.

Нужны метрики:
- ROC-AUC
- PR-AUC
- Brier score
- calibration curve + ECE/MCE (если удобно)
- для survival baseline можно добавить survival-метрики, если это не усложняет проект чрезмерно
- обязательно посчитать и экономические метрики при фиксированной policy

Сохраняй:
- таблицу сравнения моделей
- графики ROC / PR / calibration
- краткий вывод, какая модель лучше и почему

========================
5. ЭКОНОМИЧЕСКАЯ МОДЕЛЬ
========================

Нужен отдельный модуль economic_model.py.

Экономическая модель должна принимать:
- предсказания модели (mean + uncertainty или samples)
- decision variables X
- распределения экономических параметров

Распределения параметров должны задаваться в конфиге.
Например:
- purchase_cost ~ Normal / LogNormal / Triangular
- replacement_cost ~ Normal
- downtime_cost ~ Normal
- emergency_replacement_cost ~ Normal
- holding_cost ~ Normal
- salvage_value ~ Normal/Fixed
- lead_time ~ Discrete/Poisson/Fixed
- reliability modifiers by disk type

Решение (decision vector X) должно быть практически интерпретируемым.
Минимально реализуй такие decision variables:
- tau_replace: порог вероятности отказа, выше которого диск меняется превентивно
- safety_stock: минимальный запас запасных дисков
- order_qty: размер заказа при пополнении
- optional: procurement_mix по моделям дисков, если не слишком усложняет

Если procurement_mix сложно реализовать качественно, сделай сначала 3 переменные:
- tau_replace
- safety_stock
- order_qty

Экономическая логика:
- если predicted risk > tau_replace -> превентивная замена
- если диск отказал до замены -> аварийная замена + downtime cost
- если склад запасных дисков падает ниже safety_stock -> заказ order_qty
- учитывай purchase cost, holding cost, emergency cost, downtime cost
- на выходе симуляции по policy должен получаться total profit или total cost (лучше profit)

В коде сделай явную формулу прибыли и хорошо задокументируй ее.

========================
6. УЧЕТ НЕОПРЕДЕЛЕННОСТИ ЧЕРЕЗ MONTE CARLO
========================

Это центральная часть решения.

Для данной policy X:
1) много раз сэмплируются экономические параметры из распределений
2) много раз учитывается неопределенность модели:
   - либо сэмплируется вероятность отказа из предиктивного распределения объекта
   - либо берутся stochastic/ensemble predictions
3) прогоняется экономическая симуляция
4) получается массив samples прибыли/убытка

По этим samples считаются:
- expected profit = mean(samples)
- CVaR_alpha

Реализуй CVaR именно так:
- сортируем samples по возрастанию
- берем худшие alpha * N значений
- считаем их среднее
- это и есть CVaR_alpha для прибыли
(если прибыль, то худшие случаи будут самыми маленькими/наиболее отрицательными)

Сделай функцию:
compute_mean_and_cvar(samples, alpha=0.05)

========================
7. ОПТИМИЗАЦИЯ ЧЕРЕЗ PYMOO
========================

Нужен модуль optimization.py с multi-objective optimization через pymoo.

Используй, например, NSGA-II.

Оптимизировать нужно decision variables X:
- tau_replace
- safety_stock
- order_qty
- optional procurement_mix

Ограничения:
- tau_replace in [0, 1]
- safety_stock >= 0
- order_qty >= 1
- можно добавить budget constraint, если реализуется аккуратно

Функции цели:
- maximize expected profit
- maximize CVaR_alpha
Так как pymoo обычно минимизирует, можно оптимизировать:
- f1 = -expected_profit
- f2 = -CVaR_alpha

На выходе:
- Pareto-front
- набор лучших policy
- отдельно показать:
  - risk-neutral solution (максимум expected profit)
  - risk-averse solution (лучший CVaR или компромисс)
  - compromise solution (например, knee point)

Обязательно сделай визуализацию Pareto-front.

========================
8. СТРУКТУРА ПРОЕКТА
========================

Сгенерируй нормальную структуру проекта, например:

project/
  README.md
  pyproject.toml  (или requirements.txt)
  configs/
    default.yaml
  data/
    raw/
    processed/
  notebooks/
    demo.ipynb
  src/
    __init__.py
    config.py
    data.py
    features.py
    models/
      __init__.py
      baseline_survival.py
      catboost_uncertainty.py
    evaluation.py
    economic_model.py
    monte_carlo.py
    optimization.py
    plots.py
    pipeline.py
    utils.py
  scripts/
    generate_synthetic_data.py
    train_models.py
    evaluate_models.py
    optimize_policy.py
    run_full_pipeline.py
  tests/
    test_cvar.py
    test_economic_model.py
    test_pipeline_smoke.py

========================
9. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ
========================

Используй:
- python 3.11+
- pandas
- numpy
- scikit-learn
- scipy
- catboost
- lifelines
- pymoo
- matplotlib
- seaborn
- pyyaml
- tqdm

Требования к коду:
- type hints
- docstrings
- понятные имена функций
- воспроизводимость через random seed
- логирование ключевых шагов
- артефакты сохранять в outputs/ или reports/

========================
10. ЧТО ИМЕННО ДОЛЖНО БЫТЬ РЕАЛИЗОВАНО
========================

Реализуй полный рабочий код со следующими функциями/возможностями:

1) prepare_dataset(...)
2) train_baseline_survival(...)
3) train_catboost_with_uncertainty(...)
4) predict_with_uncertainty(...)
5) evaluate_models(...)
6) simulate_policy_once(...)
7) run_monte_carlo_for_policy(...)
8) compute_mean_and_cvar(...)
9) optimize_policy_with_pymoo(...)
10) plot_pareto_front(...)
11) run_full_pipeline(...)

========================
11. ВАЖНЫЕ СМЫСЛОВЫЕ ОГРАНИЧЕНИЯ
========================

Очень важно:
- не делай проект как "свалку методов"
- не добавляй лишние модели, которые не участвуют в финальном решении
- baseline + main model + глубокий анализ
- решение должно быть воспроизводимым и понятным читателю
- из кода должно быть ясно: как из данных получается решение "когда менять диск и когда закупать новые"

Также:
- uncertainty должна быть по возможности object-wise / disk-wise, а не одной общей константой на всю модель
- если API CatBoost uncertainty работает нестабильно, аккуратно реализуй fallback через bootstrapped ensemble
- если survival baseline сложно корректно довести до production-grade, сделай его как простой и понятный baseline, но основное внимание отдай основной модели и decision layer

========================
12. ФИНАЛЬНЫЙ РЕЗУЛЬТАТ
========================

Хочу получить от тебя:
1) все файлы проекта
2) README с инструкцией запуска
3) synthetic demo, который можно запустить без реального датасета
4) pipeline, который:
   - генерирует/загружает данные
   - обучает baseline и CatBoost
   - оценивает модели
   - строит economic simulation
   - считает Monte Carlo samples
   - оптимизирует policy через pymoo
   - строит Pareto-front
   - сохраняет результаты

Если где-то не хватает данных или приходится делать допущения:
- делай разумные допущения
- явно помечай их в коде и README
- не останавливай генерацию проекта

Сначала покажи:
1) дерево файлов проекта
2) краткое описание роли каждого модуля
3) затем сгенерируй содержимое файлов последовательно

Начинай.
