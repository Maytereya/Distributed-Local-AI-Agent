import json
import os
import re

import pdfplumber


def pdf_to_meili_json(pdf_path, output_json_path):
    """
    Считываем PDF-файл, извлекаем:
      - обычный текст со страниц;
      - табличные данные (если есть).
    Формируем массив объектов под формат Meilisearch.
    Каждая страница -> один элемент массива (один "документ").
    """

    with pdfplumber.open(pdf_path) as pdf:
        documents = []

        # Используем имя файла (без пути) для читабельного ID
        base_name = os.path.basename(pdf_path)
        # Заменяем неалфанумерные символы на "_"
        base_name_clean = re.sub(r'[^a-zA-Z0-9-_]', '_', base_name)

        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            # Извлекаем все таблицы на странице
            # tables -> список таблиц, каждая таблица — список строк (list of lists)
            tables = page.extract_tables() or []

            # Генерируем уникальный ID на основе имени PDF + номера страницы
            doc_id = f"{base_name_clean}_page_{page_num}"

            # Формируем объект (документ) для Meilisearch
            doc = {
                "id": doc_id,
                "title": base_name,
                "page_number": page_num,
                "keywords": text[:10],
                "content": text,
            }

            # Если таблицы есть, добавим их в поле "tables"
            # Например, список. Можно дополнительно конвертировать каждую таблицу
            # в удобный JSON-формат. Сейчас просто "как есть" (list of lists).
            if tables:
                doc["tables"] = tables

            documents.append(doc)

    # Сохраняем документы в JSON-массив
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
