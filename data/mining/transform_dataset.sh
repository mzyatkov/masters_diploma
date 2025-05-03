#!/usr/bin/bash

find SMART -depth -type d -exec bash -c '
for dir do
    # Определяем родительскую директорию и базовое имя текущей папки
    parent=$(dirname "$dir")
    base=$(basename "$dir")
    # Получаем список содержимого (может быть как файлов, так и папок)
    children=( "$dir"/* )
    # Проверяем: если в текущей директории ровно один элемент И этот элемент – директория
    if [ ${#children[@]} -ge 1 ] && [ -d "${children[0]}" ]; then
        child=$(basename "${children[0]}")
        # Если имя дочерней директории начинается с имени родительской (префикс совпадает)
        if [[ "$child" == "$base"* ]]; then
            echo "Обрабатываю: $dir => Перемещаем содержимое $child в $parent"
            mv -f "$dir"/* "$parent" && rmdir "$dir"
        fi
        if [[ "$child" == "$base" ]]; then
            echo "Обрабатываю: $dir => Перемещаем дублированное содержимое $child в $parent"
            mv -f "$dir"/* "$dir"/tmp && mv -f "$dir"/tmp/* "$parent" && rmdir "$dir"/tmp && rmdir "$dir"
        fi
    fi
done
' bash {} +

