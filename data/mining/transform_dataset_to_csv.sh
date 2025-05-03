#!/usr/bin/env bash

# Скрипт преобразования JSON в CSV и разбиения на train/val/test
# Поддерживает пропущенные SMART-атрибуты (заполняет 0)
# Требует: jq, shuf

set -euo pipefail

# Параметры
JSON_FILE="${1:-final_dataset.json}"         # входной JSON
OUT_DIR="${2:-output}"               # каталог для CSV
TARGET="${3:-days_alive}"             # целевой признак: days_alive или cost_estimate

# Создаем выходной каталог
mkdir -p "$OUT_DIR"

# Файлы вывода
CSV_ALL="$OUT_DIR/all.csv"
CSV_TRAIN="$OUT_DIR/train.csv"
CSV_VAL="$OUT_DIR/val.csv"
CSV_TEST="$OUT_DIR/test.csv"

# Извлекаем все возможные ключи SMART across all entries
SMART_KEYS_JSON=$(jq '[.[].smart | keys[] |  select(test("\\d") | not)| select(length < 3 | not)] | unique ' "$JSON_FILE")
readarray -t SK_ARRAY < <(echo "$SMART_KEYS_JSON" | jq -r '.[]')

# Формируем заголовок CSV
BASE_FIELDS=(type mfg drive_id model days_alive mtbf errors size)
HEADER=("${BASE_FIELDS[@]}")
for key in "${SK_ARRAY[@]}"; do
  HEADER+=("$key")
done
HEADER+=(cost_estimate "$TARGET")

# Печатаем заголовок
( IFS=','; echo "${HEADER[*]}" ) > "$CSV_ALL"

# Парсим JSON, заполняем отсутствующие SMART-значения нулями,
# пропускаем колонки, чьё имя содержит цифру
jq -r --arg target "$TARGET" --argjson smart_keys "$SMART_KEYS_JSON" '
  .[] |
  [ .type
  , .mfg
  , .drive_id
  , .model
  , (.days_alive|tostring)
  , (.mtbf|tostring)
  , (.errors|tostring)
  , (.size|tostring)
  ]
  + (  # только те SMART-поля, в имени которых НЕТ цифр
      [ $smart_keys[] as $k                             # перебираем все ключи
        # | select(test("\\d") | not)               # оставляем без цифр
        | (.smart[$k] // 0) | tostring             # берём значение, либо 0
      ]
    )
  + [ (.cost_estimate|tostring)
    , (.[ $target ]|tostring)
    ]
  | @csv
' "$JSON_FILE" >> "$CSV_ALL"

# Перемешиваем (remove header) и сохраняем во временный файл
shuf --random-source=/dev/urandom <(tail -n +2 "$CSV_ALL") > "$OUT_DIR/shuffled.csv"

# Размеры для split
TOTAL=$(wc -l < "$OUT_DIR/shuffled.csv")
TRAIN_COUNT=$(( TOTAL * 70 / 100 ))
VAL_COUNT=$(( TOTAL * 15 / 100 ))
TEST_COUNT=$(( TOTAL - TRAIN_COUNT - VAL_COUNT ))

# Разбиваем
head -n "$TRAIN_COUNT" "$OUT_DIR/shuffled.csv" > "$CSV_TRAIN"
head -n $(( TRAIN_COUNT + VAL_COUNT )) "$OUT_DIR/shuffled.csv" | tail -n "$VAL_COUNT" > "$CSV_VAL"
tail -n "$TEST_COUNT" "$OUT_DIR/shuffled.csv" > "$CSV_TEST"

# Добавляем заголовок в каждый файл
for file in "$CSV_TRAIN" "$CSV_VAL" "$CSV_TEST"; do
  ( head -n1 "$CSV_ALL" && cat "$file" ) > "$file.tmp" && mv "$file.tmp" "$file"
done

echo "CSV файлы созданы в $OUT_DIR: all.csv, train.csv, val.csv, test.csv"

