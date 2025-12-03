import json
import re
from typing import Optional, Dict, Any, Tuple


def _fix_json_string(text: str) -> Tuple[str, list[str]]:
    """
    Внутренняя утилита: пытается аккуратно «подлечить» строку перед json.loads().
    Возвращает исправленную строку и список шагов-поправок.
    """
    steps: list[str] = []
    original = text

    # 1. Обрезаем пробелы и переводы строк
    stripped = text.strip()
    if stripped != text:
        steps.append("обрезаны пробелы/переводы строк по краям")
        text = stripped

    # 2. Заменяем одинарные кавычки на двойные (грубый, но часто полезный приём)
    replaced_quotes = text.replace("'", '"')
    if replaced_quotes != text:
        steps.append("одинарные кавычки заменены на двойные")
        text = replaced_quotes

    # 3. Убираем обратные слэши перед символами (часто бывает лишний экранирующий слэш)
    unescaped = re.sub(r'\\(.)', r'\1', text)
    if unescaped != text:
        steps.append("удалены обратные слэши перед символами")
        text = unescaped

    # 4. Добавляем фигурные скобки, если это похоже на «голый» объект без них
    if text and text[0] not in "{[" and text[-1] not in "]}" and ":" in text:
        steps.append("строка обёрнута в фигурные скобки { ... }")
        text = "{" + text + "}"

    # 5. Удаляем висячие запятые перед закрывающими скобками
    no_trailing_commas = re.sub(r',\s*([}\]])', r'\1', text)
    if no_trailing_commas != text:
        steps.append("удалены висячие запятые перед } или ]")
        text = no_trailing_commas

    # Можно добавить и другие эвристики при необходимости

    if not steps and original != text:
        steps.append("строка слегка модифицирована без явных шагов")

    return text, steps


def str_to_json(input_str: str, debug: bool = False) -> Optional[Dict[str, Any]] | Tuple[Optional[Dict[str, Any]], str]:
    """
    Преобразует строку в словарь Python, пытаясь автоматически исправить
    распространённые ошибки формата JSON.

    Логика:
    1. Сначала выполняется прямой json.loads().
    2. Если строка некорректна:
        - обрезаются пробелы и переносы;
        - заменяются одинарные кавычки на двойные;
        - убираются лишние обратные слэши;
        - при необходимости строка оборачивается в { ... };
        - удаляются висячие запятые перед } и ].
       После этого выполняется повторный json.loads().
    3. Если и после исправлений строка не парсится — возвращается None.

    :param input_str: Строка, содержащая (предполагаемый) JSON.
    :param debug:
        - False (по умолчанию): вернуть только результат (dict или None);
        - True: вернуть кортеж (result, info), где info — текст с описанием шагов и ошибок.
    :return:
        - при debug=False: dict или None;
        - при debug=True: (dict | None, str).
    """
    steps: list[str] = []

    # Первая попытка — как есть
    try:
        result = json.loads(input_str)
        if debug:
            return result, "OK: строка успешно разобрана без исправлений"
        return result
    except json.JSONDecodeError as e:
        steps.append(f"первая попытка разбора не удалась: {e}")

    # Попытка «подлечить» строку
    fixed_str, fix_steps = _fix_json_string(input_str)
    steps.extend(fix_steps)

    try:
        result = json.loads(fixed_str)
        if debug:
            info = " | ".join(steps) if steps else "успешно после незначительных исправлений"
            return result, info
        return result
    except json.JSONDecodeError as e2:
        steps.append(f"вторая (финальная) попытка разбора не удалась: {e2}")
        info = " | ".join(steps)
        if not debug:
            print(f"Error in JSON converter: Unable to parse the string as JSON: {e2}")
            return None
        return None, info


def safe_json_loads(input_str: str, debug: bool = False) -> Dict[str, Any] | Tuple[Dict[str, Any], str]:
    """
    Безопасная обёртка над str_to_json, которая гарантированно возвращает dict.

    Если str_to_json не смог распарсить строку, вместо None возвращается пустой словарь {}.

    :param input_str: Строка, содержащая (предполагаемый) JSON.
    :param debug:
        - False: вернуть только dict;
        - True: вернуть (dict, info), где info описывает ход разбора/ошибки.
    :return:
        - при debug=False: dict (пустой, если парсинг не удался);
        - при debug=True: (dict, str).
    """
    if debug:
        result, info = str_to_json(input_str, debug=True)
        if result is None:
            return {}, info + " | парсинг не удался, возвращён пустой dict"
        return result, info

    result = str_to_json(input_str, debug=False)
    if result is None:
        return {}
    return result