from __future__ import annotations

import os
import shutil
import time
from typing import List, Literal, Optional

import gradio as gr
from gradio_pdf import PDF

from agent_logic_1 import aretrieve as retrieve
from agent_logic_1 import meilisearch_client as meilisearch
from agent_logic_2 import config as c
from agent_logic_2.id_validation import is_valid_id, sanitize_id
from converters import pdf_to_json_txt_tables_meili as pdf2json


def gr_create_collection(c_name: str):
    """
    Создает коллекцию и обновляет выпадающие списки коллекций.
    """
    if not c_name:
        gr.Warning("Введите имя коллекции", title="Предупреждение")
        return gr.update(), gr.update()

    try:
        result = retrieve.create_collection(c_name)
    except Exception as e:
        gr.Error(f"Ошибка при создании коллекции: {e}", title="Ошибка!")
        return gr.update(), gr.update()

    time.sleep(3)
    new_collections = gr_existed_collections()
    gr.Success(message=f"Коллекция {result.name} создана", title="Успешно")
    return (
        gr.update(choices=new_collections, value=c_name),
        gr.update(choices=new_collections, value=c_name),
    )


def gr_remove_collection(c_name: str):
    """
    Удаляет коллекцию по имени и обновляет выпадающие списки коллекций.
    """
    if not c_name:
        gr.Warning("Выберите коллекцию для удаления", title="Предупреждение")
        return gr.update(), gr.update()

    try:
        retrieve.remove_collection(c_name)
    except Exception as e:
        gr.Error(f"Ошибка при удалении коллекции: {e}", title="Ошибка!")
        return gr.update(), gr.update()

    new_collections = gr_existed_collections()
    gr.Success(message=f"Коллекция {c_name} удалена", title="Успешно")
    new_value = new_collections[0] if new_collections else None
    return (
        gr.update(choices=new_collections, value=new_value),
        gr.update(choices=new_collections, value=new_value),
    )


def gr_existed_collections():
    """
    Выводит имена существующих коллекций.
    :return: List of existed collection names.
    """
    try:
        chroma_service = retrieve.ChromaService(c.chroma_host, c.chroma_port)
        return chroma_service.display_collections(output_format="list")
    except Exception:
        return []


def existed_docs_in_selected_collection(selected_collection: str):
    """
    Через функционал collection.peek["metadatas"] получает названия загруженных
    документов и их страниц.
    :return: List of strings.
    """
    if not selected_collection:
        return ["Коллекция не выбрана"]
    try:
        return retrieve.handle_collection(selected_collection)
    except Exception:
        return ["Вероятно, коллекция отсутствует"]


def gr_add_to_collection(collection: str, file_path: str):
    """
    Добавляет файл в коллекцию.
    :param collection:
    :param file_path:
    :return:
    """
    if not collection:
        gr.Warning(
            "Не выбрана коллекция. Создайте или выберите коллекцию для ChromaDB.",
            title="Предупреждение",
        )
        return gr.update(value=None)

    if not file_path:
        gr.Warning("PDF не загружен, загрузите документ.", title="Предупреждение")
        return gr.update(value=None)

    try:
        retrieve.add_data(exist_collection_name=collection, upload_type="PDF", add_path=file_path, model="default")
    except Exception as e:
        gr.Error(f"Ошибка при загрузке в коллекцию: {e}", title="Ошибка!")
        return gr.update(value=None)

    gr.Success(message="Файл добавлен в коллекцию", title="Успешно")
    return gr.update(value=None)


def gr_existed_indexes():
    """
    Возвращает список существующих индексов.
    :return: List of existed Meilisearch indexes.
    """
    return meilisearch.show_list_indexes(detail_mode="uid")


def existed_docs_in_selected_index(
    selected_index: str,
    return_type: Literal["All", "All_News", "ID"],
) -> List[str] | List[List[str]]:
    """
    Возвращает список существующих документов в индексе.
    """
    if not selected_index:
        return ["Индекс не выбран"]
    if return_type == "All":
        return meilisearch.meili_list_documents(selected_index, return_type="All")
    if return_type == "All_News":
        return meilisearch.meili_list_documents(selected_index, return_type="All_News")
    return meilisearch.meili_list_documents(selected_index, return_type="ID")


def gr_add_to_index_universal(index: str, pdf_path: str, json_file: str, doc_type: str):
    """
    Универсальная функция для добавления в Meilisearch либо PDF-файла (через конвертацию),
    либо JSON-файла напрямую.
    """
    if not index:
        gr.Warning("Не выбран индекс. Укажите индекс Meilisearch", title="Предупреждение")
        return (
            gr.update(value=None),  # PDF
            gr.update(value=None),  # JSON
            gr.update(),
            gr.update(),
        )
    # Гарантируем наличие каталога для выгрузки перед любыми манипуляциями с путями.
    os.makedirs("Upload", exist_ok=True)

    if doc_type == "PDF":
        if not pdf_path:
            gr.Warning("PDF не загружен, загрузите документ", title="Предупреждение")
            return (
                gr.update(value=None),
                gr.update(value=None),
                gr.update(),
                gr.update(),
            )

        base_name = os.path.basename(pdf_path)
        base_no_ext, _ = os.path.splitext(base_name)
        # замена названия файла в подходящий формат.
        base_no_ext_clean: str = "untitled"
        if base_no_ext:
            base_no_ext_clean = sanitize_id(base_no_ext)

        json_path = f"Upload/{base_no_ext_clean}.json"
        pdf2json.pdf_to_meili_json(pdf_path, json_path)
        meili_msg = ""  # Переменная, которая сообщает об ошибках Meili
        try:
            meili_msg = meilisearch.add_doc_to_meili(json_path, index)
        except Exception as e:
            gr.Error(f"{meili_msg}. {e}", title="Ошибка")
            return (
                PDF(value=None, label="Загрузить PDF", interactive=True, scale=80),
                gr.update(value=None),  # сбрасываем JSON
                gr.update(),
                gr.update(),
            )
        time.sleep(3)
        new_list = gr_existed_indexes()
        gr.Success(f"Файл (PDF) добавлен в индекс. {meili_msg}", title="Успешно")
        return (
            PDF(value=None, label="Загрузить PDF", interactive=True, scale=80),
            gr.update(value=None),  # сбрасываем JSON
            gr.update(choices=new_list),
            gr.update(choices=new_list),
        )

    if doc_type == "JSON":
        # Обработка JSON
        if not json_file:
            gr.Warning("JSON не загружен, загрузите документ.", title="Предупреждение")
            return (
                gr.update(value=None),
                gr.update(value=None),
                gr.update(),
                gr.update(),
            )

        base_name = os.path.basename(json_file)  # "file.json"
        base_no_ext, _ = os.path.splitext(base_name)
        base_no_ext_clean: str = "untitled"
        if base_no_ext:
            base_no_ext_clean = sanitize_id(base_no_ext)
        local_json_path = f"Upload/{base_no_ext_clean}.json"
        # Копируем загруженный временный файл в свою папку
        shutil.copyfile(json_file, local_json_path)

        meili_msg = ""  # Переменная, которая сообщает об ошибках Meili

        # Индексируем в Meilisearch
        try:
            meili_msg = meilisearch.add_doc_to_meili(local_json_path, index)
        except Exception as e:
            gr.Error(f"{meili_msg}. {e}", title="Ошибка")
            return (
                gr.update(value=None),  # сбрасываем PDF
                gr.update(value=None),  # сбрасываем JSON
                gr.update(),
                gr.update(),
            )
        time.sleep(3)

        new_list = gr_existed_indexes()
        gr.Success(f"Файл (JSON) добавлен в индекс. {meili_msg}", title="Успешно")
        return (
            gr.update(value=None),  # сбрасываем PDF
            gr.update(value=None),  # сбрасываем JSON
            gr.update(choices=new_list),
            gr.update(choices=new_list),
        )

    gr.Warning("Неподдерживаемый тип документа.", title="Предупреждение")
    return (
        gr.update(value=None),
        gr.update(value=None),
        gr.update(),
        gr.update(),
    )


def gr_remove_index(index: str):
    """
    Удаляет индекс из Meilisearch и обновляет список всех индексов.

    :param index: Имя индекса, который должен быть удален из Meilisearch.
    :type index: str
    :return: Tuple containing updates for UI components with new index choices, an update
             message, and other relevant UI values.
    :rtype: tuple
    """
    meilisearch.delete_index(index)
    time.sleep(5)
    new_list = gr_existed_indexes() if gr_existed_indexes() else "Индекс отсутствует"
    gr.Success(message=f"Индекс {index} удален", title="Успешно")
    return (
        gr.update(choices=new_list, value=new_list[0] if new_list else ""),
    ) * 4


def gr_create_index(index_name: str):
    """
    Создаёт индекс в Meilisearch

    -> upload_indices_dropdown,
    -> meili_search_indexes_dropdown,
    -> meili_ind_for_cont_dropdown,
    """
    meilisearch.create_index(index_name)
    time.sleep(8)
    new_list = gr_existed_indexes()
    gr.Success(message=f"Индекс {index_name} создан", title="Успешно")
    return (
        gr.update(choices=new_list, value=index_name),
    ) * 4


def gr_rm_doc_from_index(ind_id: str, doc_id: str):
    """
    Удаляет документ из индекса Meilisearch и показывает человеко-понятное сообщение.
    НЕ обновляет UI (выпадайки/таблицы) — этим занимается обёртка rm_doc_and_refresh_all.
    """
    if not ind_id:
        gr.Warning("Не выбран индекс для удаления", title="Предупреждение")
        return

    if not doc_id:
        gr.Warning("Не выбран документ для удаления", title="Предупреждение")
        return

    # Пробуем заранее получить документ, чтобы красиво вывести заголовок
    doc = None
    try:
        doc = meilisearch.get_document_by_id(ind_id, doc_id)
    except Exception as e:
        gr.Warning(f"Не удалось получить документ перед удалением: {e}", title="Предупреждение")

    if doc:
        title = doc.get("title") or doc.get("doc_id") or doc_id
        if not title:
            content = (doc.get("content") or "").strip()
            if content:
                title = content[:60] + ("…" if len(content) > 60 else "")
    else:
        title = doc_id

    try:
        meilisearch.delete_meili_document(ind_id, doc_id)
    except Exception as e:
        gr.Error(f"Ошибка при удалении документа: {e}", title="Ошибка!")
        return

    gr.Success(
        message=f"🗑 Документ «{title}» (ID: {doc_id}) удалён из индекса '{ind_id}'.",
        title="Успешно",
    )


def validate_id_live(current: str, mode: Literal["create", "change"]):
    current = (current or "").strip()

    # 1️⃣ В режиме "Редактировать" ID не валидируем
    if mode == "change":
        return gr.update()

    # 2️⃣ Пустой ID – молча игнорируем
    if not current:
        return gr.update()

    # 3️⃣ Остальное – как раньше
    if not is_valid_id(current):
        gr.Warning("Некорректный ID. Разрешены: [a-z0-9_], длина 3–60", title="Предупреждение")
        return gr.update()

    return gr.update()


def update_docs_in_meili_index(
    index_name: str,
    output: Literal["full", "full_news", "id_only"] = "full",
    current_id: Optional[str] = None,
):
    """
    Обновляет содержимое индекса Meilisearch для UI:
    - full: возвращает (таблица, dropdown c ID)
    - full_news: возвращает модифицированную под новости таблицу, dropdown c ID)
    - id_only: возвращает только dropdown c ID
    optional current_id: какой ID сделать выбранным, если он есть в списке.
    """
    # Если индекса нет (пустой дропдаун/выпадайка) - сразу отдаем пустые значения
    if not index_name:
        empty_ids: list[str] = []
        if output == "full":
            return (
                gr.update(value=[], row_count=(1, "dynamic")),  # таблица
                gr.update(choices=empty_ids, value=""),  # dropdown
            )
        return gr.update(choices=empty_ids, value="")

    # Список документов: "All" = табличные данные,
    # "All_News" = табличные данные для новостного индекса,
    # "ID" = список ID
    full_list = existed_docs_in_selected_index(index_name, return_type="All")
    full_news_list = existed_docs_in_selected_index(index_name, return_type="All_News")
    id_list = existed_docs_in_selected_index(index_name, return_type="ID") or []

    # Какой value выбрать по умолчанию
    if current_id and current_id in id_list:
        value = current_id
    else:
        value = id_list[0] if id_list else ""

    if output == "full":
        rows_for_view = max(1, min(len(full_list), 20))
        # 1) таблица, 2) dropdown по ID
        return (
            gr.update(
                value=full_list,
                headers=["ID документа", "Заголовок", "Фрагмент"],
                col_count=(3, "fixed"),
                row_count=(rows_for_view, "dynamic"),
                max_height=420,
            ),
            gr.update(choices=id_list, value=value),
        )

    if output == "full_news":
        rows_for_view = max(1, min(len(full_news_list), 20))
        # 1) таблица, 2) dropdown по ID
        return (
            gr.update(
                value=full_news_list,
                headers=["ID документа", "Заголовок", "Срок", "Дата начала", "Дата окончания", "Фрагмент"],
                col_count=(6, "fixed"),
                row_count=(rows_for_view, "dynamic"),
                max_height=420,
            ),
            gr.update(choices=id_list, value=value),
        )

    # Только dropdown по ID
    return gr.update(choices=id_list, value=value)


def update_docs_in_chroma_collection(collection_name: str):
    """
    Функция для Chroma:
    При выборе коллекции возвращаем список документов в ней.
    """
    chroma_doc_list = retrieve.handle_collection(collection_name)
    return (
        gr.update(choices=chroma_doc_list),
        gr.update(value=chroma_doc_list),
    )


def radio_sliders_change(choice):
    """
    Обновляет интерактивное состояние всех слайдеров в зависимости от выбранной опции.

    :param choice: Строка, отображающая возможные варианты. Варианты:
        "vectorstore", "db", and "meilisearch".
    :type choice: str
    :return: Кортеж `gr.update` объектов.
    :rtype: tuple
    """
    if choice == "vectorstore":
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=True),
        )
    if choice == "db":
        return (
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=True),
        )
    if choice == "meilisearch":
        return (
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=False),
        )

    return (
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


def radio_search_engine_change(choice):
    """
    Функция для селектора выбора движка для загрузки данных. Возможны варианты
    - Chroma,
    - Meilisearch.

    collections_dropdown, index_dropdown, add_collection_button, rm_collection_button,
    add_index_button, rm_index_button, add_to_collection_button, add_to_index_button
    :param choice:
    :return:
    """
    if choice == "Meilisearch":
        return (
            gr.update(visible=False),
            gr.update(visible=True, interactive=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
        )
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def radio_type_of_upl_file_change(choice):
    """
    Конфигурирует интерфейс таким образом, чтобы загружать в MEILI либо PDF, либо JSON

    :param choice:
    :return:
    """
    if choice == "JSON":
        return (
            gr.update(visible=False),
            gr.update(visible=True),
        )
    return (
        gr.update(visible=True),
        gr.update(visible=False),
    )
