from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Dict, List, Union, Tuple, Literal, Any, Optional

import gradio as gr
from gradio_pdf import PDF

import agent_logic_2.ollama_settings as ollama_settings
from agent_logic_1 import aretrieve as retrieve
from agent_logic_1 import meilisearch_client as meilisearch
from agent_logic_2 import config as c
from agent_logic_2.benchmark_tab import gradio_benchmark as benchmark
from agent_logic_2.benchmark_tab import ollama_client as ollama
from agent_logic_2.direct_upload_meili_tab import build_blocks, TABLE_HEADERS
from agent_logic_2.id_validation import is_valid_id, sanitize_id
from agent_logic_2.prompts import load_prompt, write_prompt
from agent_logic_2.router_preprocessor import routing
from container_managenment import restart_container, system_data
from converters import pdf_to_json_txt_tables_meili as pdf2json
from whisper import whisper_dict as w
from whisper.wisper_ws_client import ws_transcribe

# Label - константы
COLLECTIONS_IN_CHROMA = "Коллекции документов Chroma DB"
INDEXES_IN_MEILI = "Индексы документов Meilisearch"
# Static files config for Gradio
STATIC_DIR = (Path(__file__).parent / "static").resolve()

# Мета-теги для превью в соцсетях
OG_IMAGE_URL = "https://ontheflyai.ru/preview/og.png"
OG_HEAD = (
    "<meta property=\"og:type\" content=\"website\" />\n"
    "<meta property=\"og:title\" content=\"Neiry.ai\" />\n"
    "<meta property=\"og:description\" "
    "content=\"Neiry.ai — чат‑бот для ваших данных. Нажмите, чтобы открыть.\" />\n"
    "<meta property=\"og:url\" content=\"https://ontheflyai.ru/\" />\n"
    "<meta property=\"og:site_name\" content=\"Neiry.ai\" />\n"
    f"<meta property=\"og:image\" content=\"{OG_IMAGE_URL}\" />\n"
    "<meta name=\"twitter:card\" content=\"summary_large_image\" />\n"
    "<meta name=\"twitter:title\" content=\"Neiry.ai\" />\n"
    "<meta name=\"twitter:description\" "
    "content=\"Neiry.ai — чат‑бот для ваших данных. Нажмите, чтобы открыть.\" />\n"
    f"<meta name=\"twitter:image\" content=\"{OG_IMAGE_URL}\" />\n"
)

# -------------------
# БЕЗОПАСНОСТЬ
# -------------------

USERNAME = c.AUTH_NAME
PASSWORD = c.AUTH_PASS
# Пока не срабатывает.
os.environ["USER_AGENT"] = "NEIRY.Agent/1.0"

ALL_MODELS = []


def check_auth(username, password):
    return username == USERNAME and password == PASSWORD


# -------------------
# Footer/Подвал
# -------------------
custom_css = """

.gradio-container footer {
    display: none !important;
}


/* Шапка с логотипом */
#logo-bar {
  display: flex !important;
  align-items: center !important;
  gap: 10px !important;
  padding: 0 !important;
  margin: 0 !important;           /* убираем нижний отступ */
  border-bottom: none !important;  /* если не нужна линия */
  line-height: 0 !important;       /* убираем «подлипание» снизу из-за baseline */
}

/* Контейнер Row с логотипом — минимальный низ */
#logo-row {
  margin-bottom: 0 !important;
  padding-bottom: 0 !important;
}

/* Само изображение: фикс. высота, без кликов, без baseline-отступа */
#brand-logo {
  height: 28px !important;
  width: auto !important;
  display: block !important;       /* убирает baseline-отступ под img */
  pointer-events: none !important; /* без взаимодействия */
  user-select: none !important;
}

/* У верхней кромки табов — убрать отступы */
#main-tabs {
  margin-top: 0 !important;
  padding-top: 0 !important;
}

/* На некоторых версиях Gradio верхняя «полка» табов — отдельный блок */
#main-tabs [data-testid="tab-nav"],
#main-tabs .tab-nav,
#main-tabs .tabs {
  margin-top: 0 !important;
  padding-top: 0 !important;
}
"""


# -------------------
# ECHOES - раздел
# -------------------

# Глобальная сессия (Gradio поддерживает per-user state)
# state = gr.State({})  # будет передаваться как дополнительный input/output


def enable_debug_logging():
    # 1) консольный логгер
    logging.basicConfig(
        level=logging.DEBUG,  # можно INFO, если шумно
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    # 2) gRPC подробности (TLS/handshake/stream)
    os.environ.setdefault("GRPC_VERBOSITY", "info")  # или debug
    os.environ.setdefault("GRPC_TRACE", "handshaker,handshake,security,transport_stream,tsi,client_channel,call_error")
    # 3) asyncio debug
    os.environ.setdefault("PYTHONASYNCIODEBUG", "1")
    try:
        loop = asyncio.get_event_loop()
        loop.set_debug(True)
    except Exception:
        pass


# ------------------------------------------------------------------------

async def echo_ai_router(message, history, session_state, ai_feed: Literal["local", "cloud"] = "local"):
    """
    Подключает роутер и стримит ответ.
    :param ai_feed: Что подключаем: локальную LLM или облачную.
    :param message: Текст запроса пользователя
    :param history: (не используется, можно убрать)
    :param session_state: словарь сессии, хранит pending и history
    :yields: два значения — текущий кусок ответа и обновлённый session_state
    """
    session_state = session_state or {}
    try:
        # routing возвращает AsyncGenerator[(partial_response, session), None]
        async for partial, session_state in routing(message, sess=session_state, ai_feed=ai_feed):
            # Каждая итерация — новое состояние и новый кусок ответа
            yield partial, session_state

    except (asyncio.CancelledError, GeneratorExit):
        # Стоп из ChatInterface: ничего не шлем в UI, просто даём отмене подняться —
        # это закроет стримовые соединения ниже по стеку (включая Ollama).
        print("Стоп в echo_ai_router ПРОИЗОШЕЛ")
        raise

    except Exception as e:
        # При ошибке тоже стримим её сразу
        yield f"⚠️ Ошибка обработки запроса в ai-router: {e}", session_state


async def chroma_echo(message: str, history: List[Dict], collection: str, threshold_value: float,
                      slider_value_n_results: int,
                      slider_value_k,
                      radio_value) -> str:
    """
    Main chroma call func. Its return the pieces of text from uploaded to chroma docs.
    :param collection: Str. Chosen collection name.
    :param message: Str. The users question.
    :param history: Obligate parameter for correct gradio executes.
    :param threshold_value:
    :param slider_value_n_results:
    :param slider_value_k:
    :param radio_value: Type of search established.
    :return: String of filtrated text from ChromaDB.
    """
    # print(f"Echo {collection=}")
    return await retrieve.main_retrieve_async(question=message, collection=collection, return_type="str",
                                              threshold=threshold_value,
                                              n_results=slider_value_n_results,
                                              k=slider_value_k,
                                              search_type=radio_value)


async def meili_echo(
        message: str,
        history: List[Dict],
        index: str,
        limit: int
) -> str:
    """
    Поддерживает прямое обращение к серверу Meilisearch.
    :param message:
    :param history:
    :param index:
    :param limit:
    :return: String - результаты поиска
    """
    search_result = meilisearch.search_meili(query=message, index_name=index, limit=limit)

    return search_result


async def universal_echo(
        message: str,
        history: List[Dict],
        radio_value: str,  # "ai-router", "gigachat", "meilisearch", "vectorstore", "db"
        threshold_value: float,
        slider_value_n_results: int,
        slider_value_k: int,
        collection: str,
        meili_index: str,
):
    if radio_value == "ai-router":
        session_state: dict = {}
        ai_feed: Literal["local", "cloud"] = "local"
        # стримим
        async for partial, session_state in echo_ai_router(message, history, session_state, ai_feed=ai_feed):
            yield partial
        # после завершения стрима — выходим
        return

    if radio_value == "gigachat":
        session_state: dict = {}
        ai_feed: Literal["local", "cloud"] = "cloud"

        # стримим
        async for partial, session_state in echo_ai_router(message, history, session_state, ai_feed=ai_feed):
            yield partial
        # после завершения стрима — выходим
        return

    elif radio_value == "meilisearch":
        result = await meili_echo(
            message=message,
            history=history,
            index=meili_index,
            limit=slider_value_k
        )
        yield result
        return

    else:
        # сюда попадём только если radio_value == "vectorstore" или "db"
        result = await chroma_echo(
            message=message,
            history=history,
            collection=collection,
            threshold_value=threshold_value,
            slider_value_n_results=slider_value_n_results,
            slider_value_k=slider_value_k,
            radio_value=radio_value
        )
        yield result
        return


# --------------------
# CHROMA DB - секция
# --------------------

def gr_create_collection(c_name: str):
    """
    Создает коллекцию и возвращает ее имя в качестве строки.
    Warning: in the next releases Chroma .name parameter will be removed!
    :param c_name: String, passed new name of the collection.
    :return: String, name of the collection.
    """
    if c_name:
        result = retrieve.create_collection(c_name)
        time.sleep(3)
        new_collections = gr_existed_collections()
        return (
            f"Коллекция {result.name} создана",
            gr.update(choices=new_collections, value=c_name, ),
            gr.update(choices=new_collections, value=c_name, ),
        )
    else:
        return (
            f"Ошибка: введите имя коллекции",
            gr.update(),
            gr.update(),
        )


def gr_remove_collection(c_name: str):
    """
    Удаляет коллекцию по имени.
    :param c_name:
    :return: String - сообщение и обновление объектов Gradio.
    """
    retrieve.remove_collection(c_name)
    new_collections = gr_existed_collections()
    return (
        gr.update(choices=new_collections, value=None),
        gr.update(choices=new_collections, ),
        "Коллекция удалена",
    )


def gr_existed_collections():
    """
    Выводит имена существующих коллекций.
    :return: List of existed collection names.
    """
    chroma_service = retrieve.ChromaService(c.chroma_host, c.chroma_port)
    return chroma_service.display_collections(output_format="list")


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
    except Exception as e:
        return ["Вероятно, коллекция отсутствует"]


def gr_add_to_collection(collection: str, file_path: str):
    """
    Добавляет файл в коллекцию.
    :param collection:
    :param file_path:
    :return:
    """
    if not collection:
        return (
            gr.update(value=None),
            "Ошибка: не выбрана Коллекция. Создайте или определите Коллекцию для ChromaDB.",
        )

    if not file_path:
        return (
            gr.update(value=None),
            "Ошибка: PDF не загружен, загрузите документ."
        )

    retrieve.add_data(exist_collection_name=collection, upload_type="PDF", add_path=file_path, model="default")
    return (
        PDF(
            value=None, label="Загрузить PDF", interactive=True, scale=80),
        "Файл добавлен в коллекцию",
    )


# ----------------
# MEILI - секция
# ----------------

def gr_existed_indexes():
    """
    Возвращает список существующих индексов.
    :return: List of existed Meilisearch indexes.
    """
    return meilisearch.show_list_indexes(detail_mode="uid")


def existed_docs_in_selected_index(selected_index: str,
                                   return_type: Literal["All", "All_News", "ID"]) -> List[str] | List[List[str]]:
    """
    Возвращает список существующих документов в индексе.
    """
    if not selected_index:
        return ["Индекс не выбран"]
    if return_type == "All":
        return meilisearch.meili_list_documents(selected_index, return_type="All")
    elif return_type == "All_News":
        return meilisearch.meili_list_documents(selected_index, return_type="All_News")
    else:
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
            gr.update()
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
                gr.update()
            )

        base_name = os.path.basename(pdf_path)
        base_no_ext, _ = os.path.splitext(base_name)
        # замена названия файла в подходящий формат.
        base_no_ext_clean: str = "untitled"
        if base_no_ext:
            base_no_ext_clean = sanitize_id(base_no_ext)

        json_path = f"Upload/{base_no_ext_clean}.json"
        pdf2json.pdf_to_meili_json(pdf_path, json_path)
        meili_msg = ''  # Переменная, которая сообщает об ошибках Meili
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


    elif doc_type == "JSON":

        # Обработка JSON

        if not json_file:
            gr.Warning("JSON не загружен, загрузите документ.", title="Предупреждение")
            return (
                gr.update(value=None),
                gr.update(value=None),
                gr.update(),
                gr.update()
            )

        base_name = os.path.basename(json_file)  # "file.json"
        base_no_ext, _ = os.path.splitext(base_name)
        base_no_ext_clean: str = "untitled"
        if base_no_ext:
            base_no_ext_clean = sanitize_id(base_no_ext)
        local_json_path = f"Upload/{base_no_ext_clean}.json"
        # Копируем загруженный временный файл в свою папку
        shutil.copyfile(json_file, local_json_path)

        meili_msg = ''  # Переменная, которая сообщает об ошибках Meili

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


    else:
        gr.Warning("Неподдерживаемый тип документа.", title="Предупреждение")
        return (
            gr.update(value=None),
            gr.update(value=None),
            gr.update(),
            gr.update()
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
    #
    time.sleep(5)
    #
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
    time.sleep(8)  #
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

    # Можно ничего не показывать на корректный ID
    # gr.Info("ID корректен", title="OK")
    return gr.update()


# ------------------------------------
# GRADIO секция функций - оберток
# ------------------------------------

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
                gr.update(value=[]),  # таблица
                gr.update(choices=empty_ids, value="")  # dropdown
            )
        else:
            return gr.update(choices=empty_ids, value="")

    # Список документов: "All" = табличные данные,
    # "All_News" = табличные данные для новостного индекса,
    # "ID" = список ID

    full_list = existed_docs_in_selected_index(index_name, return_type="All")
    full_news_list = existed_docs_in_selected_index(index_name, return_type="All_News")  # !
    id_list = existed_docs_in_selected_index(index_name, return_type="ID") or []

    # Какой value выбрать по умолчанию
    if current_id and current_id in id_list:
        value = current_id
    else:
        value = id_list[0] if id_list else ""

    if output == "full":
        # 1) таблица, 2) dropdown по ID
        return (
            gr.update(value=full_list, headers=["ID документа", "Заголовок", "Фрагмент"], col_count=(3, "fixed"), ),
            gr.update(choices=id_list, value=value),
        )

    if output == "full_news":
        # 1) таблица, 2) dropdown по ID
        return (
            gr.update(value=full_news_list,
                      headers=["ID документа", "Заголовок", "Срок", "Дата начала", "Дата окончания", "Фрагмент"],
                      col_count=(6, "fixed"), ),
            gr.update(choices=id_list, value=value),
        )
    else:
        # Только dropdown по ID
        return gr.update(choices=id_list, value=value)


def update_docs_in_chroma_collection(collection_name: str):
    """
    Функция для Chroma:
    При выборе коллекции возвращаем список документов в ней.
    """
    chroma_doc_list = retrieve.handle_collection(collection_name)
    return (
        gr.update(choices=chroma_doc_list, ),
        gr.update(value=chroma_doc_list)
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
        return (gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=False),

                gr.update(interactive=False),
                gr.update(interactive=True),
                )
    elif choice == "db":
        return (gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=True),

                gr.update(interactive=False, ),
                gr.update(interactive=True),
                )
    elif choice == "meilisearch":
        return (gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=True),

                gr.update(interactive=True, ),
                gr.update(interactive=False),
                )

    else:
        return (gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),

                gr.update(interactive=False),
                gr.update(interactive=False, ),
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
            gr.Button(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
        )
    else:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.Button(visible=False),
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
    else:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
        )


def main():
    # Инициализация при загрузке приложения options и model name
    ollama_settings.init_model_name()
    ollama_settings.init_options()
    ollama_settings.init_thinking()

    # Инициализация основных двух индексов (чтобы все поля в индексах были корректно настроены)
    client = meilisearch.connect_to_meilisearch()
    waiter = partial(meilisearch.wait_for_task_completion, client)  # фиксируем client
    meilisearch.ensure_index(client,
                             "main_index", "id",
                             meilisearch.MAIN_SETTINGS,
                             wait_fn=waiter)
    meilisearch.ensure_index(client,
                             "news",
                             "id",
                             meilisearch.NEWS_SETTINGS,
                             wait_fn=waiter)

    # Allow serving local /static files via /gradio_api/file=...
    gr.set_static_paths(paths=[STATIC_DIR])

    custom_footer = """
    <div style="
        width: 100%;
        text-align: center;
        padding: 12px 0;
        font-size: 14px;
        color: #888;
        /*border-top: 1px solid #3333;*/
        /*margin-top: 10px;*/
    ">
        © 2025 <b>neiry.ai llc.</b>
    </div>
    """
    with (gr.Blocks(css=custom_css, title="Neiry.ai", head=OG_HEAD) as blocks):
        model_state = gr.State()  # Нужно для однократной загрузки моделей из Ollama

        with gr.Row(elem_id="logo-row"):
            gr.HTML(
                "<div id='logo-bar'>"
                "<img id='brand-logo' src='/gradio_api/file=static/logo.png' alt='Логотип'>"
                "</div>"
            )

        with gr.Tabs(elem_id="main-tabs"):
            # --------------------------------------------------
            # Вкладка 1 — основной интерфейс
            # --------------------------------------------------

            with gr.Tab("\U0001F4D6 AI - ассистент"):
                chatbot = gr.Chatbot(type="messages",
                                     autoscroll=False,
                                     placeholder="<strong>🧠 ИИ - помощник</strong><br>Знает всю информацию о врачах и услугах клиники Наука",
                                     height=700,
                                     max_height=1000,
                                     label="Моя Наука")

                textbox = gr.Textbox(lines=2,
                                     max_lines=12,
                                     placeholder="Напишите свой вопрос",
                                     submit_btn=True,
                                     stop_btn=True,
                                     container=True,
                                     autoscroll=False,
                                     autofocus=False,
                                     min_width=0,
                                     scale=70,
                                     html_attributes=gr.InputHTMLAttributes(autocorrect="off", spellcheck=True)
                                     )

                radio_type_of_search = gr.Radio(["ai-router", "gigachat", "meilisearch", "vectorstore", "db", ],
                                                label="Способы поиска в базе знаний",
                                                value="ai-router",
                                                container=True,
                                                render=False,
                                                info="Выберите алгоритм поиска")

                meili_search_indexes_dropdown = gr.Dropdown(choices=gr_existed_indexes(),
                                                            label=INDEXES_IN_MEILI,
                                                            info="Выберите Индекс для поиска информации",
                                                            interactive=False,
                                                            render=False,
                                                            )
                chroma_search_collection_dropdown = gr.Dropdown(choices=gr_existed_collections(),
                                                                label=COLLECTIONS_IN_CHROMA,
                                                                info="Выберите Коллекцию для поиска информации",
                                                                interactive=False,
                                                                allow_custom_value=True,
                                                                # крайне желательно этого избежать
                                                                render=False,
                                                                )

                value_n_results_slider = gr.Slider(value=5, minimum=1, maximum=20, step=1,
                                                   label="Количество документов, включенных в выдачу",
                                                   info="Только в режиме vectorstore",
                                                   interactive=False,
                                                   render=False,
                                                   )

                thresholdvalue_slider = gr.Slider(value=0.005, minimum=0.0025, maximum=0.02, step=0.0025,
                                                  label="Порог косинусной фильтрации",
                                                  info="Только в режиме vectorstore."
                                                       "Чем выше значение, тем больше текстовых фрагментов с меньшей "
                                                       "релевантностью появится в выдаче",
                                                  interactive=False,
                                                  render=False,
                                                  )

                value_k_slider = gr.Slider(value=2, minimum=1, maximum=20, step=1,
                                           label="Количество документов, включенных в выдачу",
                                           info="Только в режимах db и meilisearch",
                                           interactive=False,
                                           render=False,
                                           )
                settings_accordion = gr.Accordion("⚙️ Настройки поиска", open=False, visible=True, render=False)

                demo = gr.ChatInterface(
                    fn=universal_echo,
                    type="messages",
                    chatbot=chatbot,
                    textbox=textbox,

                    additional_inputs_accordion=settings_accordion,

                    additional_inputs=[
                        radio_type_of_search,
                        thresholdvalue_slider,
                        value_n_results_slider,
                        value_k_slider,
                        chroma_search_collection_dropdown,
                        meili_search_indexes_dropdown,

                    ],

                    show_progress="full",

                )

                # ====== ЗАХВАТ АУДИО И РАСШИФРОВКА ======

                mic = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    streaming=False,
                    label="Микрофон",
                    interactive=True,
                    format="wav",
                    min_width=150,
                    show_download_button=False,
                    show_share_button=False,
                    editable=False,
                    show_label=False,
                    visible=True,
                    # scale=30,

                )

                async def ws_transcribe_to(audio):
                    """
                    audio -> текст от Whisper
                    :return: текст в textbox
                    """
                    if audio is None:
                        return gr.update(), gr.update()
                    text = await ws_transcribe(audio)  # функция уровнем ниже
                    return gr.update(value=text), gr.update(value=None)

            # Автотранскрипция по окончании записи и очистка по клику на крестик
            mic.change(
                fn=ws_transcribe_to,
                inputs=mic,
                outputs=[textbox, mic],
            )

            # --------------------------------------------------
            # Вкладка 2 - Upload PDF to MEILI or CHROMA DB
            # --------------------------------------------------

            with gr.Tab("\U0001F4E4 Документы"):
                with gr.Accordion(label="Загрузка готовых документов в базы знаний Meilisearch и ChromaDB",
                                  open=False, ):
                    with gr.Row():
                        radio_type_of_db = gr.Radio(["ChromaDB", "Meilisearch"],
                                                    label="Тип базы данных",
                                                    value="ChromaDB",
                                                    container=True,
                                                    # info="для добавления документа"
                                                    )

                        radio_type_of_upl_data = gr.Radio(["PDF", "JSON"],
                                                          label="Тип документа",
                                                          value="PDF",
                                                          container=True,
                                                          # info="для загрузки в MEILISEARCH",
                                                          visible=False)

                        upload_collections_dropdown = gr.Dropdown(choices=gr_existed_collections(),
                                                                  # value=None,
                                                                  allow_custom_value=True,
                                                                  filterable=True,
                                                                  label=COLLECTIONS_IN_CHROMA,
                                                                  # info="Коллекции документов",
                                                                  visible=True, )

                        upload_indices_dropdown = gr.Dropdown(choices=gr_existed_indexes(),
                                                              # value=None,
                                                              allow_custom_value=True,
                                                              filterable=True,
                                                              label=INDEXES_IN_MEILI,
                                                              # info="Индексы документов",
                                                              visible=False)

                        # Кнопки для работы с коллекциями или индексами
                        with gr.Column():
                            add_collection_button = gr.Button("✅ Добавить коллекцию",
                                                              visible=True,
                                                              size="sm",
                                                              variant="primary",
                                                              scale=10,

                                                              )
                            rm_collection_button = gr.Button("⛔ Удалить коллекцию",
                                                             visible=True,
                                                             size="sm",
                                                             variant="stop",
                                                             scale=10,
                                                             )

                            add_index_button = gr.Button("✅ Добавить индекс",
                                                         visible=False,
                                                         size="sm",
                                                         variant="primary",
                                                         scale=10,
                                                         )
                            rm_index_button = gr.Button("⛔ Удалить индекс",
                                                        visible=False,
                                                        size="sm",
                                                        variant="stop",
                                                        scale=10,
                                                        )

                    with gr.Row():
                        pdf = PDF(label="Загрузить PDF", interactive=True, scale=80)

                        # Загрузка JSON (по умолчанию невидим)
                        json_file = gr.File(
                            label="Загрузить JSON",
                            visible=False,
                            scale=80,
                            file_types=[".json"],  # или просто ["json"]
                            type="filepath"
                        )

                        with gr.Column():
                            add_to_collection_button = gr.Button("Добавить в коллекцию",
                                                                 visible=True,
                                                                 size="sm",
                                                                 variant="primary",
                                                                 )
                            add_to_index_button = gr.Button("Добавить в индекс",
                                                            visible=False,
                                                            size="sm",
                                                            variant="primary",
                                                            )

                # --------------------------------------------------------------------------------
                # Секция оформления страницы - конструктора документа
                # --------------------------------------------------------------------------------

                # единый state вместо отдельных переменных для хранения документов прямой загрузки
                meta_state = gr.State(
                    value=None)  # dict: {"doc_id": str, "index": str, "blocks": list[dict]}
                # отдельный state для хранения ID документа
                # для работы переименования в редакторе документов
                orig_doc_id_state = gr.State(value=None)

                DEFAULT_EMPTY_TABLE = [["", ""], ["", ""], ["", ""]]

                table_state = gr.State(value=DEFAULT_EMPTY_TABLE)

                with gr.Accordion(label="Форма для добавления информации в базу знаний Meilisearch",
                                  open=False, ):
                    with gr.Column():
                        with gr.Row():
                            io_radio = gr.Radio(
                                [("Создать", "create"), ("Редактировать", "change")],
                                # container=True,
                                value="create",
                                scale=20,
                                label="Выберите действие"
                            )

                            index_dropdown = gr.Dropdown(
                                choices=gr_existed_indexes(),
                                info="main_index: скрипты, news: акции/новости",
                                label="Выберите индекс Meilisearch",
                                interactive=True,
                                scale=30,
                            )
                            # -------------------------------
                            # Окно служит для ввода ID
                            # -------------------------------

                            id_input = gr.Textbox(
                                label="ID документа (латиница/цифры/нижнее подчеркивание, 3–60)",
                                info="Введите уникальный для добавления нового документа",
                                value="",
                                visible=True,
                                placeholder="например: price_list_2025 или izmeneniya_grafika_priema",
                                scale=50,
                            )

                            # _______________________________
                            # Дропдаун/выпадайка служит для выбора ID
                            # для редактирования
                            # -------------------------------

                            id_select = gr.Dropdown(
                                label="ID документа",
                                info="Выберите для редактирования документа",
                                choices=existed_docs_in_selected_index(index_dropdown.value, "ID"),
                                # value="",
                                allow_custom_value=True,
                                visible=False,
                                interactive=True,
                                scale=50,
                            )

                            with gr.Column():
                                normalize_id_btn = gr.Button("🧹 Нормализовать ID", size="sm",
                                                             variant="secondary", visible=True)
                                generate_id_from_title_btn = gr.Button("🪄 ID из заголовка", size="sm",
                                                                       variant="primary", visible=True)
                                vanish_screen_btn = gr.Button("🧹 Очистить ввод", size="sm", variant="stop",
                                                              visible=True)
                                load_doc_btn = gr.Button("⬇️ Загрузить по ID", size="sm", variant="primary",
                                                         visible=False)
                                save_doc_btn_direct = gr.Button("💾 Сохранить (обновить по ID)", size="sm",
                                                                variant="primary", visible=False)

                        # ---------------------------------------------
                        # Компилятор новостей и диапазонов
                        # действия новостей
                        # ---------------------------------------------
                        # Константа для обозначения бессрочной акции
                        FAR_FUTURE_DT = datetime(2099, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
                        #
                        INDEX_FOR_TYPE = {"static": "main_index", "news": "news"}
                        TYPE_FOR_INDEX = {"main_index": "static", "news": "news"}

                        with gr.Row():
                            doc_type_radio = gr.Radio(
                                choices=[("Скрипты", "static"), ("Новость / Акция", "news")],
                                value="static",
                                label="Тип документа",
                                scale=30,
                                interactive=False,
                                visible=True,
                            )
                            # ----------------------------------------------
                            title_input = gr.Textbox(label="Заголовок, title *", scale=70)
                        # ----------------------------------------------
                        # Редактор новостей и времени - продолжение
                        # ----------------------------------------------

                        with gr.Row(visible=False) as news_dates_row:
                            # Применено обобщение (as), к которому может применяться однотипный update
                            valid_from_dp = gr.DateTime(label="Действует с (UTC)", type="datetime", include_time=False,
                                                        timezone="UTC")  # вернет datetime
                            valid_to_dp = gr.DateTime(label="Действует по (UTC)", type="datetime", include_time=False,
                                                      timezone="UTC")
                            permanent_cb = gr.Checkbox(label="Бессрочно", value=False, visible=False)

                        def on_doc_type_change(t):
                            is_news = (t == "news")
                            return (
                                gr.update(visible=is_news),  # news_dates_row
                                gr.update(visible=is_news),  # permanent_cb
                            )

                        doc_type_radio.change(on_doc_type_change, inputs=[doc_type_radio],
                                              outputs=[news_dates_row, permanent_cb])

                        def on_index_change(idx: str):
                            """
                            Сделан, чтобы исключить выбор radio и зациклить только на индексе.
                            :param idx:
                            :return:
                            """
                            t = TYPE_FOR_INDEX.get(idx, "static")
                            is_news = (t == "news")
                            return (
                                gr.update(value=t),  # doc_type_radio
                                gr.update(visible=is_news),  # news_dates_row
                                gr.update(visible=is_news),  # permanent_cb
                            )

                        index_dropdown.change(
                            on_index_change,
                            inputs=[index_dropdown],
                            outputs=[doc_type_radio, news_dates_row, permanent_cb],
                        )

                        def to_utc(dt: datetime | None) -> datetime | None:
                            if dt is None:
                                return None
                            if dt.tzinfo is None:
                                return dt.replace(tzinfo=timezone.utc)
                            return dt.astimezone(timezone.utc)

                        def to_ts(dt: datetime) -> int:
                            return int(dt.timestamp())

                        content_input = gr.Textbox(label="Основной текст, content *", lines=20, max_lines=80)
                        keywords_input = gr.Textbox(label="Ключевые слова, keywords (через запятую)")

                        # опциональная таблица
                        table_df = gr.Dataframe(
                            label="Таблица (необязательно)",
                            visible=False,
                            headers=TABLE_HEADERS,
                            datatype="str",
                            row_count=(3, "dynamic"),
                            col_count=(len(TABLE_HEADERS), "fixed"),
                            type="array",
                            value=[["", ""], ["", ""], ["", ""]],
                            interactive=True,
                            show_fullscreen_button=True,
                        )

                        def _passthrough_table(t):
                            # t — это list[list]; чисто прокидываем в State
                            return t

                        # любое редактирование таблицы обновляет State
                        table_df.change(_passthrough_table, inputs=[table_df], outputs=[table_state])
                        # Скрыть в случае редактирования документа
                        preview_button = gr.Button("Предпросмотр блоков")
                        preview_json = gr.JSON(label="Предпросмотр JSON", visible=False)
                        #
                        # сейвим собранные блоки между кликами
                        save_button = gr.Button("Сохранить и отправить в индекс")

                    # ----------------------------
                    # Предпросмотр и сохранение
                    # ----------------------------

                    def fn_preview_json(current_doc_id, title, content, keywords, table,
                                        selected_index, doc_type, valid_from, valid_to, is_permanent,
                                        split: Literal["on", "off"] = "off"):

                        # Жесткая проверка на соответствие индекса функционалу
                        expected_type = TYPE_FOR_INDEX.get(selected_index, "static")
                        if doc_type != expected_type:
                            # жёстко приводим
                            doc_type = expected_type
                            gr.Warning("Тип документа приведён к выбранному индексу.", title="Предупреждение")

                        if not selected_index:
                            gr.Warning("Не выбран индекс", title="Предупреждение")
                            return gr.update(visible=False), None

                        try:
                            # Построение блоков (можно включить или выключить через параметр split)
                            doc_id, blocks = build_blocks(current_doc_id, title, content, keywords, table, split=split)
                        except ValueError as e:
                            gr.Error(f"{e}", title="Ошибка!")
                            return gr.update(visible=False), None

                        # валидируем даты, только если news
                        vf_utc = vt_utc = None
                        is_perm = False
                        if doc_type == "news":
                            is_perm = bool(is_permanent)
                            if is_perm:
                                vf_utc = to_utc(valid_from) or datetime.now(timezone.utc)
                                vt_utc = FAR_FUTURE_DT
                            else:
                                vf_utc = to_utc(valid_from)
                                vt_utc = to_utc(valid_to)
                                if not vf_utc or not vt_utc:
                                    gr.Warning("Укажите обе даты для новости (или отметьте 'Бессрочно')",
                                               title="Предупреждение")
                                    return gr.update(visible=False), None
                                if vf_utc > vt_utc:
                                    gr.Warning("Дата 'с' позже даты 'по'", title="Предупреждение")
                                    return gr.update(visible=False), None

                        # обогащаем каждый блок полями валидности + типом
                        if doc_type == "news":
                            for b in blocks:
                                b.update({
                                    "doc_type": "news",
                                    "valid_from": vf_utc.isoformat().replace("+00:00", "Z"),
                                    "valid_to": vt_utc.isoformat().replace("+00:00", "Z"),
                                    "from_ts": to_ts(vf_utc),
                                    "to_ts": to_ts(vt_utc),
                                    "is_permanent": is_perm,
                                })
                        else:
                            for b in blocks:
                                b.update({"doc_type": "static"})

                        meta = {
                            "doc_id": doc_id,
                            "index": selected_index,
                            "blocks": blocks
                        }

                        gr.Info(f"✅ Итого: '{doc_id}', блоков: {len(blocks)} → '{selected_index}'", title="Инфо")
                        return (
                            gr.update(visible=True, value=blocks),  # preview_json
                            meta  # meta_state
                        )

                    preview_button.click(
                        fn_preview_json,
                        inputs=[id_input, title_input, content_input, keywords_input, table_state,
                                index_dropdown, doc_type_radio, valid_from_dp, valid_to_dp, permanent_cb],
                        outputs=[preview_json, meta_state],
                    )

                    def save_and_send_to_meilisearch(meta, meili_view_index: str):
                        """
                        Сохраняет блоки в Meilisearch (режим конструктора с предпросмотром)
                        и обновляет UI:

                          - id_select в конструкторе
                          - dropdown и таблицу в секции просмотра/удаления (если индекс совпадает)
                          - очищает поля конструктора (ID, заголовок, контент, keywords)
                          - очищает и скрывает preview_json
                          - сбрасывает meta_state
                        """

                        # нет подготовленных данных
                        if not meta:
                            gr.Warning("Нет данных (сделайте Предпросмотр)", title="Предупреждение")
                            # порядок: id_select, dropdown_del, table, id_input, title, content, keywords, preview_json, permanent_cb, meta_state
                            return (
                                gr.update(),  # id_select
                                gr.update(),  # meili_content_of_index_dropdown
                                gr.update(),  # meili_indices_table
                                gr.update(),  # id_input
                                gr.update(),  # title_input
                                gr.update(),  # content_input
                                gr.update(),  # keywords_input
                                gr.update(),  # preview_json
                                gr.update(),  # cb - бессрочно
                                meta,  # meta_state оставляем как есть
                            )

                        index_name = meta.get("index")
                        blocks = meta.get("blocks") or []
                        doc_id = meta.get("doc_id")

                        if not index_name:
                            gr.Warning("Не выбран индекс", title="Предупреждение")
                            return (
                                gr.update(), gr.update(), gr.update(),
                                gr.update(), gr.update(), gr.update(), gr.update(),
                                gr.update(), gr.update(), meta,
                            )

                        if not blocks:
                            gr.Warning("Пустой массив блоков", title="Предупреждение")
                            return (
                                gr.update(), gr.update(), gr.update(),
                                gr.update(), gr.update(), gr.update(), gr.update(),
                                gr.update(), gr.update(), meta,
                            )

                        msg: str = ""
                        try:
                            msg = meilisearch.add_doc_to_meili(blocks, index_name)
                        except Exception as e:
                            gr.Error(f"{e}, сообщение от сервера Meilisearch: {msg}", title="Ошибка!")
                            return (
                                gr.update(), gr.update(), gr.update(),
                                gr.update(), gr.update(), gr.update(), gr.update(),
                                gr.update(), gr.update(), meta,
                            )

                        gr.Success(
                            f"✅ '{doc_id}', добавлено {len(blocks)} блок(ов) в '{index_name}', "
                            f"сообщение от сервера Meilisearch: {msg}",
                            title="Успешно",
                        )

                        # 1) обновляем id_select в конструкторе
                        id_select_update = update_docs_in_meili_index(
                            index_name,
                            output="id_only",
                            current_id=doc_id,
                        )

                        # 2) если выбранный в секции просмотра индекс совпадает — обновляем и её
                        if meili_view_index == index_name:
                            table_update, dropdown_update = update_docs_in_meili_index(
                                index_name,
                                output="full",
                            )
                        else:
                            table_update = gr.update()
                            dropdown_update = gr.update()

                        # 3) очищаем поля конструктора
                        id_input_update = gr.update(value="")
                        title_update = gr.update(value="")
                        content_update = gr.update(value="")
                        keywords_update = gr.update(value="")
                        permanent_cb_upd = gr.update(value=False)

                        # 4) очищаем и прячем предпросмотр
                        preview_update = gr.update(visible=False, value=None)

                        # 5) сбрасываем meta_state
                        meta_out = None

                        # порядок outputs:
                        # [id_select, meili_content_of_index_dropdown, meili_indices_table,
                        #  id_input, title_input, content_input, keywords_input,
                        #  preview_json, permanent_cb, meta_state]
                        return (
                            id_select_update,
                            dropdown_update,
                            table_update,
                            id_input_update,
                            title_update,
                            content_update,
                            keywords_update,
                            preview_update,
                            permanent_cb_upd,
                            meta_out,
                        )

                    # ------------------------------
                    # живой валидатор на каждый ввод
                    # ------------------------------

                    id_input.change(
                        validate_id_live,
                        inputs=[id_input, io_radio],  # 🆕 передаем текущий режим (редактир или создание)
                        outputs=[id_input],
                    )

                    # ------------------------------
                    # ручная нормализация
                    # ------------------------------
                    def normalize_id_click(current: str) -> dict[str, Any]:
                        cleaned = sanitize_id(current or "")
                        if cleaned:
                            gr.Info(f"ID корректен: {cleaned}", title="Инфо")
                            return gr.update(value=cleaned)

                        else:
                            gr.Warning("ID всё ещё некорректен", title="Предупреждение")
                            return gr.update()

                    normalize_id_btn.click(normalize_id_click, inputs=id_input, outputs=id_input
                                           )

                    # ------------------------
                    # генерация из заголовка
                    # ------------------------

                    def gen_id_from_title(title: str) -> dict[str, Any]:
                        cleaned = sanitize_id(title or "")

                        if cleaned:
                            gr.Success(f"ID сгенерирован: {cleaned}", title="Успешно")
                            return gr.update(value=cleaned)
                        else:
                            gr.Warning("Не удалось сгенерировать ID из заголовка", title="Предупреждение")
                            return gr.update()

                    generate_id_from_title_btn.click(gen_id_from_title, inputs=[title_input],
                                                     outputs=id_input

                                                     )

                    def vanish_all_windows():
                        gr.Info("Все окна очищены, готов к загрузке нового документа", title="Инфо")
                        return (
                            gr.update(value="create"),  # io_radio → режим "Создать"
                            "",  # id_input
                            gr.update(value=""),  # id_select: сбрасываем выбранный ID (choices не трогаем)
                            "",  # title_input
                            "",  # content_input
                            "",  # keywords_input
                            DEFAULT_EMPTY_TABLE,  # table_df
                            "static",  # doc_type_radio
                            None,  # valid_from_dp
                            None,  # valid_to_dp
                            False,  # permanent_cb
                            gr.update(visible=False, value=None),  # preview_json: спрятать и очистить
                            None,  # meta_state: сброс
                        )

                    vanish_screen_btn.click(
                        vanish_all_windows,
                        outputs=[
                            io_radio,  # 🆕 переключаем на "Создать"
                            id_input,
                            id_select,  # 🆕 очищаем выбор ID
                            title_input,
                            content_input,
                            keywords_input,
                            table_df,
                            doc_type_radio,
                            valid_from_dp,
                            valid_to_dp,
                            permanent_cb,
                            preview_json,  # 🆕 очищаем и прячем
                            meta_state,  # 🆕 сбрасываем
                        ],
                    )

                    # --------------------------------------------
                    # Секция загрузки документа для редактирования
                    # --------------------------------------------
                    def _parse_iso(x: str | None):
                        if not x: return None
                        try:
                            return datetime.fromisoformat(x.replace("Z", "+00:00")).astimezone(timezone.utc)
                        except Exception:
                            return None

                    def load_doc_into_form_by_id(index_name: str, doc_id: str):
                        """
                        Загружает документ из Meilisearch по index_name + doc_id
                        и подготавливает значения для формы редактирования.

                        ВАЖНО: выбор в выпадающем списке id_select сохраняется (value=doc_id),
                        даже если показываем предупреждения.
                        """

                        # 0. Базовая валидация входа
                        if not index_name:
                            gr.Warning("Укажите индекс", title="Предупреждение")
                            # НИЧЕГО не меняем в форме, в т.ч. не трогаем id_select
                            return (gr.update(),) * 10 + (gr.update(),)

                        if not doc_id:
                            gr.Warning("Укажите ID", title="Предупреждение")
                            return (gr.update(),) * 10 + (gr.update(),)

                        # 1. Получаем документ из Meilisearch
                        doc = meilisearch.get_document_by_id(index_name, doc_id)
                        if not doc:
                            gr.Warning("❌ Документ не найден", title="Предупреждение")
                            return (gr.update(),) * 10 + (gr.update(),)

                        # 2. Заголовок и контент (защита от None)
                        title = doc.get("title") or ""
                        content = doc.get("content") or ""

                        # 3. Нормализация типа документа с учётом индекса (миграция старых схем)
                        idx_default_type = TYPE_FOR_INDEX.get(index_name, "static")  # "news" или "static"

                        doc_type = doc.get("doc_type") or doc.get("type") or idx_default_type

                        # Миграция старых документов:
                        # - если type == "text" в индексе news → считаем документ новостью
                        # - если type == "text" в других индексах → считаем статическим
                        if doc_type == "text":
                            if idx_default_type == "news":
                                doc_type = "news"
                            else:
                                doc_type = "static"

                        # На всякий случай, если doc_type - что-то странное - откатываем к дефолту индекса
                        if doc_type not in ("news", "static"):
                            doc_type = idx_default_type

                        is_news = (doc_type == "news")

                        # 4. Даты действия новости + флаг "Бессрочно"
                        vf = _parse_iso(doc.get("valid_from"))
                        vt = _parse_iso(doc.get("valid_to"))
                        is_perm = bool(doc.get("is_permanent", False))

                        # Если это news, но дат нет и не отмечено "Бессрочно" — подсказываем,
                        # НО при этом НИЧЕГО не сбрасываем в формах, всё ниже всё равно заполняется.
                        if is_news and not (vf or vt or is_perm):
                            gr.Warning(
                                "Это документ из индекса 'news', но даты не заданы. "
                                "Укажите период или отметьте 'Бессрочно'.",
                                title="Требуются даты",
                            )

                        # 5. Нормализация keywords:
                        raw_keywords = doc.get("keywords", "")
                        if isinstance(raw_keywords, list):
                            keywords = ", ".join(str(k) for k in raw_keywords if k)
                        else:
                            keywords = str(raw_keywords or "")

                        # 6. Таблица: если нет или None - ставим дефолтную
                        table_val = doc.get("table") or DEFAULT_EMPTY_TABLE

                        # 7. Сообщение об успехе
                        gr.Success(message="✅ Документ загружен", title="Успешно")

                        # 8. Возвращаем значения для компонентов Gradio.
                        #    Порядок должен соответствовать outputs в .click:
                        #    [id_input, title_input, content_input,
                        #     valid_from_dp, valid_to_dp, permanent_cb,
                        #     keywords_input, table_df,
                        #     doc_type_radio, news_dates_row,
                        #     id_select]  ← последний - выпадайка с ID
                        return (
                            gr.update(value=doc_id),  # id_input
                            gr.update(value=title),  # title_input
                            gr.update(value=content),  # content_input

                            gr.update(value=vf),  # valid_from_dp
                            gr.update(value=vt),  # valid_to_dp
                            gr.update(value=is_perm, visible=is_news),  # permanent_cb

                            gr.update(value=keywords),  # keywords_input
                            gr.update(value=table_val),  # table_df

                            gr.update(value=doc_type),  # doc_type_radio
                            gr.update(visible=is_news),  # news_dates_row

                            # 🧷 ВАЖНО: сохраняем выбор в dropdown ID
                            gr.update(value=doc_id),  # id_select
                            gr.update(value=doc_id),  # 🆕 orig_doc_id_state
                        )

                    # Обработка события загрузки документа в форму редактирования
                    load_doc_btn.click(
                        load_doc_into_form_by_id,
                        inputs=[index_dropdown, id_select],
                        outputs=[
                            id_input,
                            title_input,
                            content_input,
                            valid_from_dp,
                            valid_to_dp,
                            permanent_cb,
                            keywords_input,
                            table_df,
                            doc_type_radio,
                            news_dates_row,
                            id_select,  # ← добавили сюда
                            orig_doc_id_state,  # 🆕
                        ],
                    )

                    #  ------------------------------------
                    # Секция сохранения документа после
                    # редактирования
                    #  ------------------------------------

                    def save_doc_by_id(
                            doc_type: Literal["static", "news", "text"],
                            index_name: str,
                            current_doc_id: str,  # текущее значение ID (пользователь мог изменить)
                            orig_doc_id: str | None,  # исходный ID документа, который был загружен для редактирования
                            title: str,
                            content: str,
                            valid_from,
                            valid_to,
                            permanent,
                            keywords: str,
                            table,
                    ):
                        """
                        Сохраняет (upsert) документ в индексе Meilisearch в режиме редактирования.

                        Поддерживает смену ID:
                          * current_doc_id — ID из интерфейса (id_select), пользователь может его изменить;
                          * orig_doc_id — исходный ID документа, загруженного для редактирования.
                            Если после сохранения current_doc_id != orig_doc_id, старый документ
                            удаляется из индекса (реализация "переименования" ID).

                        Возвращает:
                          * gr.update(...) для обновления выпадающего списка ID (id_select) —
                            в нём будет актуальный список документов, с выбранным текущим doc_id.
                        """

                        # ---------------- Базовая подготовка и валидация ----------------

                        doc_id = (current_doc_id or "").strip()
                        orig_id = (orig_doc_id or "").strip() if orig_doc_id is not None else None

                        if not index_name:
                            gr.Warning("Укажите индекс", title="Предупреждение")
                            return gr.update()

                        # валидация нового ID (при создании или переименовании)
                        if not doc_id or not is_valid_id(doc_id):
                            gr.Warning("Некорректный ID (разрешено [a-z0-9_], длина 3–60).", title="Предупреждение")
                            return gr.update()

                        if not title:
                            gr.Warning("Создайте заголовок", title="Предупреждение")
                            return gr.update()

                        if not content:
                            gr.Warning("Создайте контент", title="Предупреждение")
                            return gr.update()

                        # ---------------- Нормализация doc_type относительно индекса ----------------

                        expected_type = TYPE_FOR_INDEX.get(index_name, "static")  # "news" или "static"

                        # устаревшее значение "text" трактуем как тип индекса
                        if doc_type == "text":
                            doc_type = expected_type

                        if doc_type != expected_type:
                            doc_type = expected_type
                            gr.Warning("Тип документа приведён к выбранному индексу.", title="Предупреждение")

                        # ---------------- Базовый каркас документа ----------------

                        base_doc: dict[str, Any] = {
                            "id": doc_id,
                            "title": title or "",
                            "content": content or "",
                            "keywords": keywords or "",
                            "table": table or [],
                        }

                        # ---------------- Формирование полей для новостей ----------------

                        if doc_type == "news":
                            def _to_utc(dt):
                                if dt is None:
                                    return None
                                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

                            if permanent:
                                # бессрочная новость: нужна дата начала
                                if not valid_from:
                                    gr.Warning("Уточните дату начала", title="Предупреждение")
                                    return gr.update()
                                vf = _to_utc(valid_from)
                                vt = datetime(2099, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
                            else:
                                # ограниченный период — нужны обе даты
                                if not valid_from or not valid_to:
                                    gr.Warning("Укажите обе даты или отметьте 'Бессрочно'", title="Предупреждение")
                                    return gr.update()
                                vf, vt = _to_utc(valid_from), _to_utc(valid_to)
                                if vf > vt:
                                    gr.Warning("Дата 'с' позже даты 'по'", title="Предупреждение")
                                    return gr.update()

                            base_doc.update({
                                "doc_type": "news",
                                "valid_from": vf.isoformat().replace("+00:00", "Z"),
                                "valid_to": vt.isoformat().replace("+00:00", "Z"),
                                "from_ts": int(vf.timestamp()),
                                "to_ts": int(vt.timestamp()),
                                "is_permanent": bool(permanent),
                            })
                        else:
                            # все остальные документы — статические
                            base_doc.update({"doc_type": "static"})

                        # ---------------- Переименование ID: удаляем старый (если нужен) ----------------

                        # Флаг переименования: старый ID есть и он отличается от нового
                        is_rename = bool(orig_id) and (orig_id != doc_id)

                        if is_rename:
                            # маленький маячок, чтобы в логах было видно, что ветка переименования сработала
                            gr.Info(f"Переименование документа: старый ID = {orig_id}, новый ID = {doc_id}",
                                    title="Инфо")

                            # ВАРИАНТ: сначала удалить старый документ, потом сохранить новый
                            try:
                                print("DEBUG RENAME:", index_name, orig_id, "->", doc_id)
                                meilisearch.delete_meili_document(index_name, orig_id)
                            except Exception as e:
                                gr.Warning(
                                    f"Не удалось удалить старый документ с ID {orig_id}: {e}",
                                    title="Предупреждение",
                                )
                                # продолжаем, всё равно попробуем сохранить документ с новым ID

                        # ---------------- Сохранение в Meilisearch ----------------

                        try:
                            msg = meilisearch.upsert_document(index_name, base_doc)
                        except Exception as e:
                            gr.Error(f"Ошибка при сохранении документа: {e}", title="Ошибка!")
                            return gr.update()

                        gr.Success(f"✅ Сохранено (ответ сервера: {msg})", title="Успешно")

                        # ---------------- Обновление выпадайки ID ----------------

                        # Всегда возвращаем обновлённый список ID, с выбранным текущим doc_id
                        return update_docs_in_meili_index(index_name, output="id_only", current_id=doc_id)

                    save_doc_btn_direct.click(
                        save_doc_by_id,
                        inputs=[doc_type_radio,  # 1) doc_type
                                index_dropdown,  # 2) index_name
                                id_select,  # 3) current_doc_id (может быть изменён)
                                orig_doc_id_state,  # 4) исходный ID
                                title_input,  # 5) title
                                content_input,  # 6) content
                                valid_from_dp,  # 7) valid_from
                                valid_to_dp,  # 8) valid_to
                                permanent_cb,  # 9) permanent
                                keywords_input,  # 10) keywords
                                table_state],  # 11) table
                        outputs=[id_select],  # обновление для id_select
                    )

                # ----------------------------------------------------
                # Функционал селектора (radio) Создать/редактировать
                # ----------------------------------------------------
                def create_or_change_fn(choose: Literal["create", "change"] = "create"):
                    # Пока 2 возможных значения, но не исключено, что их будет 3: + "delete"
                    if choose == "create":
                        return (
                            gr.update(visible=True),
                            gr.update(visible=False),
                            gr.update(visible=True),
                            gr.update(visible=True),
                            gr.update(visible=True),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=True),  # preview.JSON
                            gr.update(visible=True),  # save button
                            gr.update(visible=True),  # preview button

                        )
                    else:
                        return (
                            gr.update(visible=False),
                            gr.update(visible=True),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=True),
                            gr.update(visible=True),
                            gr.update(visible=True),
                            gr.update(visible=False),  # preview.JSON
                            gr.update(visible=False),  # save button
                            gr.update(visible=False),  # preview button
                        )

                io_radio.change(create_or_change_fn, inputs=[io_radio],
                                outputs=[id_input,
                                         id_select,
                                         normalize_id_btn,
                                         generate_id_from_title_btn,
                                         vanish_screen_btn,
                                         load_doc_btn,
                                         save_doc_btn_direct,
                                         preview_json,
                                         save_button,
                                         preview_button,
                                         ])
                # Используется partial для того, чтобы передать параметр output заранее
                # Так как прямая передача параметров не из объектов Gradio невозможна
                index_dropdown.change(fn=partial(update_docs_in_meili_index, output="id_only"),
                                      inputs=index_dropdown,
                                      outputs=id_select)

                # ---------------------------------------------------
                # Секция просмотра содержимого коллекций
                # и индексов
                # ---------------------------------------------------

                # Функционал отображения даты в превью документов
                now = datetime.now()
                formatted_full = now.strftime("%A, %d %B %Y, %H:%M")  # Понедельник, 15 Декабрь 2025, 14:30

                with gr.Accordion(label="База знаний Meilisearch (просмотр и удаление)",
                                  open=True, ):

                    with gr.Row():
                        meili_ind_for_cont_dropdown = gr.Dropdown(choices=gr_existed_indexes(),
                                                                  filterable=True,
                                                                  interactive=True,
                                                                  label=INDEXES_IN_MEILI,
                                                                  # info="Выберите Индекс для просмотра",
                                                                  visible=True,
                                                                  scale=4,
                                                                  )

                        meili_content_of_index_dropdown = gr.Dropdown(
                            choices=existed_docs_in_selected_index(meili_ind_for_cont_dropdown.value, "ID"),
                            value="",
                            allow_custom_value=True,
                            label="Выбрать документ по ID для удаления",
                            visible=True,
                            interactive=True,
                            scale=4,
                        )

                        with gr.Column():
                            refresh_data_btn = gr.Button("🔄 Обновить",
                                                         size="md",
                                                         visible=True,
                                                         interactive=True,
                                                         variant="primary",
                                                         scale=2,
                                                         )
                            rm_doc_from_index_button = gr.Button("⛔ Удалить из Индекса",
                                                                 size="sm",
                                                                 visible=True,
                                                                 interactive=True,
                                                                 variant="stop",
                                                                 scale=2,
                                                                 )

                    meili_indices_table = gr.DataFrame(
                        value=existed_docs_in_selected_index(meili_ind_for_cont_dropdown.value, "All"),
                        label="Содержание выбранного Индекса" + " на сегодня: " + formatted_full,
                        headers=["ID документа", "Заголовок",
                                 "Фрагмент"],
                        row_count=(200, "dynamic"),
                        col_count=(3, "fixed"),
                        datatype="str",
                        interactive=False,
                        show_row_numbers=True,
                        show_search="filter"
                    )

                with gr.Accordion(label="База знаний Chroma DB (просмотр и удаление)", open=False, ):
                    with gr.Row():
                        chroma_coll_for_cont_dropdown = gr.Dropdown(choices=gr_existed_collections(),
                                                                    filterable=True,
                                                                    label=COLLECTIONS_IN_CHROMA,
                                                                    interactive=True,
                                                                    # info="Выберите Коллекцию для просмотра содержимого",
                                                                    visible=True,
                                                                    scale=4,
                                                                    )
                        chroma_collection_content = gr.Dropdown(
                            # choices=[],
                            choices=existed_docs_in_selected_collection(chroma_coll_for_cont_dropdown.value),
                            # value=None,
                            allow_custom_value=False,
                            label="Выбрать документ для удаления",
                            visible=True,
                            interactive=True,
                            scale=4,
                        )

                        rm_doc_from_collection_button = gr.Button("Удалить из Коллекции",
                                                                  size="sm",
                                                                  visible=True,
                                                                  interactive=False,
                                                                  variant="stop",
                                                                  scale=2
                                                                  )

                    chroma_collection_table = gr.DataFrame(
                        value=existed_docs_in_selected_collection(chroma_coll_for_cont_dropdown.value),
                        label="Содержание выбранной Коллекции",
                        headers=["Имя файла и страница", ],
                        row_count=(15, "dynamic"),
                        col_count=(1, "fixed"),
                        datatype="str",
                        interactive=False
                    )

                # ---------------------------------------------
                # Секция интерфейса чата, дополнительный код
                # Оформление и дизайн
                # ---------------------------------------------

                # ---------------------------------------------
                # Обработчики событий чат-интерфейса
                # ---------------------------------------------

                radio_type_of_search.change(fn=radio_sliders_change, inputs=radio_type_of_search,
                                            outputs=[
                                                value_n_results_slider,
                                                thresholdvalue_slider,
                                                value_k_slider,
                                                meili_search_indexes_dropdown,
                                                chroma_search_collection_dropdown,
                                            ]
                                            )

                radio_type_of_db.change(fn=radio_search_engine_change, inputs=radio_type_of_db,
                                        outputs=[
                                            upload_collections_dropdown,
                                            upload_indices_dropdown,
                                            add_collection_button,
                                            rm_collection_button,
                                            add_index_button,
                                            rm_index_button,
                                            add_to_collection_button,
                                            add_to_index_button,
                                            radio_type_of_upl_data,
                                        ]
                                        )

                radio_type_of_upl_data.change(fn=radio_type_of_upl_file_change, inputs=radio_type_of_upl_data,
                                              outputs=[
                                                  pdf,
                                                  json_file,
                                              ]
                                              )

                # --------------------------------------
                # Обработчики событий ChromaDB
                # --------------------------------------

                add_collection_button.click(
                    gr_create_collection,
                    inputs=upload_collections_dropdown,
                    outputs=[upload_collections_dropdown, chroma_search_collection_dropdown]
                )

                rm_collection_button.click(
                    gr_remove_collection,
                    inputs=upload_collections_dropdown,
                    outputs=[upload_collections_dropdown, chroma_search_collection_dropdown, ]
                )

                add_to_collection_button.click(
                    gr_add_to_collection,
                    inputs=[upload_collections_dropdown, pdf],
                    outputs=[pdf, ]
                )

                # --------------------------------------
                # Обработчики событий Meilisearch
                # --------------------------------------
                # Клик - ивент пришлось унести сюда поскольку meili_ind_for_cont_dropdown расположен низко.
                save_button.click(
                    save_and_send_to_meilisearch,
                    inputs=[meta_state, meili_ind_for_cont_dropdown],
                    outputs=[
                        id_select,  # 1
                        meili_content_of_index_dropdown,  # 2
                        meili_indices_table,  # 3
                        id_input,  # 4 — очистить ID
                        title_input,  # 5 — очистить заголовок
                        content_input,  # 6 — очистить текст
                        keywords_input,  # 7 — очистить keywords
                        preview_json,  # 8 — очистить/спрятать JSON
                        permanent_cb,  # 9 убрать галочку бессрочно, если она есть
                        meta_state,  # 10 — сбросить state
                    ],
                )

                rm_index_button.click(
                    gr_remove_index,
                    inputs=upload_indices_dropdown,
                    outputs=[
                        upload_indices_dropdown,
                        meili_search_indexes_dropdown,
                        meili_ind_for_cont_dropdown,
                        index_dropdown,

                    ]
                )

                add_index_button.click(
                    gr_create_index,
                    inputs=upload_indices_dropdown,
                    outputs=[
                        upload_indices_dropdown,
                        meili_search_indexes_dropdown,
                        meili_ind_for_cont_dropdown,
                        index_dropdown,

                    ]
                )

                # Универсальная кнопка для индексации (PDF или JSON)
                add_to_index_button.click(
                    gr_add_to_index_universal,
                    inputs=[upload_indices_dropdown, pdf, json_file, radio_type_of_upl_data],
                    outputs=[pdf, json_file, upload_indices_dropdown, meili_search_indexes_dropdown]
                )

                # ------------------------------------------
                def rm_doc_and_refresh_all(
                        index_for_delete: str,
                        doc_id_to_delete: str,
                        constructor_index: str,
                        constructor_current_id: str | None,
                ):
                    """
                    1) Удаляет документ (через gr_rm_doc_from_index).
                    2) Обновляет:
                       - dropdown "Выбрать документ по ID для удаления"
                       - таблицу содержимого индекса
                       - dropdown id_select в конструкторе (если индекс тот же).
                    """

                    # 1. Удаляем документ (сообщения об успехе/ошибке внутри)
                    gr_rm_doc_from_index(index_for_delete, doc_id_to_delete)

                    if not index_for_delete:
                        # Индекса нет — ничего не трогаем
                        return gr.update(), gr.update(), gr.update()

                    # 2. Обновляем содержимое индекса (просмотр/удаление)
                    # update_docs_in_meili_index(output="full") возвращает (table_update, dropdown_update)
                    table_update, dropdown_update = update_docs_in_meili_index(
                        index_for_delete,
                        output="full",
                    )

                    # В .click порядок outputs: [meili_content_of_index_dropdown, meili_indices_table, id_select]
                    # Поэтому dropdown_update пойдёт первым, а table_update — вторым.

                    # 3. Обновляем конструктор, если он смотрит на тот же индекс
                    if constructor_index == index_for_delete:
                        constructor_dd_update = update_docs_in_meili_index(
                            constructor_index,
                            output="id_only",
                            current_id=constructor_current_id,
                        )
                    else:
                        constructor_dd_update = gr.update()

                    return (
                        dropdown_update,  # meili_content_of_index_dropdown
                        table_update,  # meili_indices_table
                        constructor_dd_update,  # id_select (конструктор)
                    )

                rm_doc_from_index_button.click(
                    rm_doc_and_refresh_all,
                    inputs=[
                        meili_ind_for_cont_dropdown,  # индекс для удаления
                        meili_content_of_index_dropdown,  # ID документа для удаления
                        index_dropdown,  # индекс из конструктора
                        id_select,  # текущий ID в конструкторе
                    ],
                    outputs=[
                        meili_content_of_index_dropdown,  # dropdown "Выбрать документ по ID для удаления"
                        meili_indices_table,  # таблица "Содержание выбранного Индекса"
                        id_select,  # dropdown ID в конструкторе
                    ],
                )

                # ----------------------------------------------------------------
                # ЛОГИКА ОБНОВЛЕНИЯ при выборе индекса/коллекции
                # для показа документов
                # ----------------------------------------------------------------

                # При смене выбранной коллекции -> обновить список документов
                chroma_coll_for_cont_dropdown.change(
                    fn=update_docs_in_chroma_collection,
                    inputs=chroma_coll_for_cont_dropdown,
                    outputs=[chroma_collection_content,
                             chroma_collection_table]
                )

                def refresh_meili_views(
                        index_for_view: str,  # Индекс в секции просмотра/удаления
                        constructor_index: str,  # Индекс, выбранный в конструкторе документа (текущий индекс)
                        constructor_current_id: str | None,
                ):
                    """
                    Функция обновления выпадаек ID в просмотре/удалении, конструкторе
                    и таблицы документов в просмотре/удалении

                    Обновляет:
                      - таблицу и выпадайку ID в секции просмотра/удаления
                      - выпадайку ID в конструкторе (если выбран тот же индекс)
                    """

                    # если индекс в секции просмотра не выбран — ничего не ломаем
                    if not index_for_view:
                        return gr.update(), gr.update(), gr.update()

                    # 1) Обновляем таблицу + выпадайку удаления для выбранного индекса
                    if index_for_view == "news":
                        table_update, dropdown_del_update = update_docs_in_meili_index(
                            # еще одна версия обновляющей функции?
                            index_for_view,
                            output="full_news",
                        )

                    else:
                        table_update, dropdown_del_update = update_docs_in_meili_index(
                            # еще одна версия обновляющей функции?
                            index_for_view,
                            output="full",
                        )
                    # 2) Обновляем конструктор, только если он смотрит на тот же индекс
                    if constructor_index == index_for_view:
                        constructor_dd_update = update_docs_in_meili_index(
                            constructor_index,
                            output="id_only",
                            current_id=constructor_current_id,
                        )
                    else:
                        constructor_dd_update = gr.update()

                    # порядок соответствует outputs
                    return table_update, dropdown_del_update, constructor_dd_update

            refresh_data_btn.click(
                refresh_meili_views,
                inputs=[meili_ind_for_cont_dropdown,  # выпадайка выбора индекса в просмотре/удалении
                        index_dropdown,  # выпадайка выбора индекса в конструкторе документа
                        id_select  # выбор документа для редактирования в конструкторе документа
                        ],
                outputs=[
                    meili_indices_table,  # таблица просмотра всех документов с превью
                    meili_content_of_index_dropdown,  # выпадайка выбора ID в секции просмотра/удаления
                    id_select,  # выбор документа для редактирования в конструкторе документа
                ],
            )

            # При смене выбранного индекса -> обновить список документов
            meili_ind_for_cont_dropdown.change(
                fn=refresh_meili_views,
                inputs=[meili_ind_for_cont_dropdown, index_dropdown, id_select],
                outputs=[meili_indices_table, meili_content_of_index_dropdown, id_select],
            )

            # -------- FUNCTIONS SECTION ------------

            def fn_load_options(explain: bool = True) -> Union[tuple[str, str], str]:
                """
                Загружает и сериализует настройки Ollama в JSON с отступами.
                :return: JSON-форматированная строка
                    ASCII отключено, отступы есть.
                :rtype: Str
                """
                return ollama_settings.load_ollama_options(explain)

            def fn_save_options(text: str) -> str:
                """
                Сохраняет измененные настройки Ollama в файл JSON.
                """
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as e:
                    return f"❌ Ошибка преобразования JSON на уровне интерфейса: {e}"
                return ollama_settings.write_options(data)

            def fn_load_prompt(name: str, inform: bool = True) -> Union[Tuple[str, str], str]:
                return load_prompt(name, inform)

            def fn_save_prompt(name: str, text: str) -> str:

                try:
                    info = write_prompt(name, text)
                    return info
                except Exception as e:
                    return f"❌ Ошибка сохранения на уровне интерфейса: {e}"

            # ------------ ASSERT(ACCEPT/VALIDATE/CHOOSE) MAIN LLM ---------------

            def fn_load_main_model() -> List[str]:
                return [ollama_settings.read_main_model_name(False)]

            def fn_assert_main_model(name: str, ) -> str:

                try:
                    return ollama_settings.write_main_model_name(name)
                except Exception as e:
                    return f"❌ Ошибка сохранения на уровне интерфейса: {e}"

            async def reassert_main_model_dropdown(only_list: bool = False) -> Union[gr.update(), List[str]]:
                models_response = await ollama.list()
                models = sorted([m["model"] for m in models_response["models"]])
                if only_list:
                    return models
                return gr.update(choices=models, value=models[-1] if models else [])

            # ------------Секция управления контейнерами------------

            def restart() -> str:
                return restart_container.restart_ollama_container()

            # ------------------------------------------------------

            with gr.Tab("⚙️ Настройки"):
                gr.Markdown("""<h3>⚙️ Настройки нейросетей и серверов Ollama/Uvicorn</h3>""")

                with gr.Column():
                    with gr.Row():
                        status = gr.Textbox(lines=1,
                                            label="Текущий статус",
                                            submit_btn=False,
                                            container=True,
                                            autoscroll=False,
                                            interactive=True,
                                            autofocus=False,
                                            )

                    # -----------------------------------------------------
                    # OLLAMA OPTIONS Секция
                    # -----------------------------------------------------

                    with gr.Row():
                        with gr.Column():
                            main_model_selector = gr.Dropdown(
                                choices=fn_load_main_model(),
                                multiselect=False,
                                label="Выбор главной LLM",
                                interactive=True,
                                container=True,
                            )
                            think_checkbox = gr.Checkbox(label="Активировать способность рассуждать",
                                                         # info="Только для reasoning models, снижает скорость",
                                                         value=ollama_settings.init_thinking(),
                                                         container=True, )

                            reload_main_model_btn = gr.Button("⬇️ Загрузить доступные модели", size="sm", )
                            save_model_btn = gr.Button("💾 Утвердить выбранную модель", size="sm", )

                        reload_main_model_btn.click(
                            fn=reassert_main_model_dropdown,
                            inputs=[],
                            outputs=main_model_selector)

                        save_model_btn.click(fn=fn_assert_main_model,
                                             inputs=[main_model_selector],
                                             outputs=[status], )

                        think_checkbox.change(ollama_settings.write_think_status, inputs=[think_checkbox],
                                              outputs=[status])

                        json_ollama_options = gr.Code(label="📄Ollama Options",
                                                      value=fn_load_options(explain=False),
                                                      # в данном случае explain = False
                                                      language="json",
                                                      visible=True,
                                                      interactive=True,
                                                      scale=4)
                        with gr.Column():
                            load_options_btn = gr.Button("⬇️ Загрузить текущие опции", scale=20, size="sm", )
                            save_options_btn = gr.Button("💾 Сохранить новые опции", scale=20, size="sm", )
                            restart_ollama_btn = gr.Button("🔃 Перезагрузить Ollama", scale=20, size="sm",
                                                           variant="stop")

                load_options_btn.click(fn=fn_load_options,
                                       # в данном случае explain = True (умолчание), так как надо передать оповещение в статус.
                                       inputs=[],
                                       outputs=[
                                           json_ollama_options,
                                           status
                                       ],
                                       )
                save_options_btn.click(fn=fn_save_options, inputs=json_ollama_options, outputs=status)
                restart_ollama_btn.click(fn=restart, outputs=status)

                # --------------------------------------------------------
                #  Split prompt секция
                # --------------------------------------------------------
                with gr.Row():
                    with gr.Accordion(label="Split Prompt", open=False):
                        prompt_code_2 = gr.Code(
                            value=fn_load_prompt("split_prompt", False),
                            language=None,
                            label="Split Prompt section",
                            interactive=True,
                            lines=20,
                            scale=4,
                        )

                        btn_load_2 = gr.Button("🔄 Загрузить", size="md")
                        btn_save_2 = gr.Button("💾 Сохранить", size="md")

                btn_load_2.click(lambda: fn_load_prompt("split_prompt"),
                                 [], [prompt_code_2, status])
                btn_save_2.click(lambda txt: fn_save_prompt("split_prompt", txt),
                                 prompt_code_2, status)

                # --------------------------------------------------------
                #  Classificator prompt секция
                # --------------------------------------------------------
                with gr.Row():
                    with gr.Accordion(label="Classificator Prompt", open=False):
                        prompt_code_3 = gr.Code(
                            value=fn_load_prompt("classificator_prompt", False),
                            language=None,
                            label="Classificator Prompt section",
                            interactive=True,
                            lines=10,
                            scale=4,
                        )

                        btn_load_3 = gr.Button("🔄 Загрузить", size="md")
                        btn_save_3 = gr.Button("💾 Сохранить", size="md")

                btn_load_3.click(lambda: fn_load_prompt("classificator_prompt"),
                                 [], [prompt_code_3, status])
                btn_save_3.click(lambda txt: fn_save_prompt("classificator_prompt", txt),
                                 prompt_code_3, status)

                # -----------------------------------------------------
                # Final answering prompt секция
                # -----------------------------------------------------
                with gr.Row():
                    with gr.Accordion(label="Final Answer Prompt", open=False):
                        prompt_code_1 = gr.Code(
                            value=fn_load_prompt("final_answer", False),
                            language=None,
                            label="Final Answering Prompt section",
                            interactive=True,
                            lines=20,
                            scale=4,
                        )
                        btn_load_1 = gr.Button("🔄 Загрузить", size="md")
                        btn_save_1 = gr.Button("💾 Сохранить", size="md")

                btn_load_1.click(lambda: fn_load_prompt("final_answer"),
                                 [], [prompt_code_1, status])
                btn_save_1.click(lambda txt: fn_save_prompt("final_answer", txt),
                                 prompt_code_1, status)

                # ---------------------------------------
                #     CONSTANTS секция
                # ---------------------------------------
                # c1: EXAMPLES

                with gr.Row():
                    with gr.Accordion(label="Examples for Classification", open=False):
                        prompt_code_c1 = gr.Code(
                            value=fn_load_prompt("EXAMPLES", False),
                            language=None,
                            label="EXAMPLES section: примеры для классификации",
                            interactive=True,
                            lines=20,
                            scale=4,
                        )

                        btn_load_c1 = gr.Button("🔄 Загрузить", size="md")
                        btn_save_c1 = gr.Button("💾 Сохранить", size="md")

                        btn_load_c1.click(lambda: fn_load_prompt("EXAMPLES"),
                                          [], [prompt_code_c1, status])

                        btn_save_c1.click(lambda txt: fn_save_prompt("EXAMPLES", txt),
                                          prompt_code_c1, status)

                # c2: LABEL_DOC

                with gr.Row():
                    with gr.Accordion(label="Markers for Text Chunks", open=False):
                        prompt_code_c2 = gr.Code(
                            value=fn_load_prompt("LABEL_DOC", False),
                            language=None,
                            label="LABEL_DOC section: образцы маркировки распознанных текстовых сегментов",
                            interactive=True,
                            lines=10,
                            scale=4,
                        )

                        btn_load_c2 = gr.Button("🔄 Загрузить", size="md")
                        btn_save_c2 = gr.Button("💾 Сохранить", size="md")

                    btn_load_c2.click(lambda: fn_load_prompt("LABEL_DOC"),
                                      [], [prompt_code_c2, status])

                    btn_save_c2.click(lambda txt: fn_save_prompt("LABEL_DOC", txt),
                                      prompt_code_c2, status)

                # с3: LABEL_PRIORITY

                with gr.Row():
                    with gr.Accordion(label="Text Chunks Layout Priority", open=False):
                        prompt_code_c3 = gr.Code(
                            value=fn_load_prompt("LABEL_PRIORITY", False),
                            language=None,
                            label="LABEL_PRIORITY section: приоритет расположения текстовых сегментов после распознавания",
                            interactive=True,
                            lines=5,
                            scale=4,
                        )

                        btn_load_c3 = gr.Button("🔄 Загрузить", size="md")
                        btn_save_c3 = gr.Button("💾 Сохранить", size="md")

                    btn_load_c3.click(lambda: fn_load_prompt("LABEL_PRIORITY"),
                                      [], [prompt_code_c3, status])

                    btn_save_c3.click(lambda txt: fn_save_prompt("LABEL_PRIORITY", txt),
                                      prompt_code_c3, status)

                # с4: MODULES

                with gr.Row():
                    with gr.Accordion(label="MODULES: The names of agents's functions", open=False):
                        prompt_code_c4 = gr.Code(
                            value=fn_load_prompt("MODULES", False),
                            language=None,
                            label="MODULES section: имена агентских функций, ассоциированных с маркерами, только чтение",
                            interactive=False,
                            lines=5,
                            scale=4,
                        )

                        btn_load_c4 = gr.Button("🔄 Загрузить", size="md")
                        # btn_save_c4 = gr.Button("💾 Сохранить", size="md")

                    btn_load_c4.click(lambda: fn_load_prompt("MODULES"),
                                      [], [prompt_code_c3, status])

                    # btn_save_c4.click(lambda txt: fn_save_prompt("MODULES", txt),
                    #                   prompt_code_c3, status,)

                # ---------------------------------
                # Whisper general prompt секция
                # ---------------------------------

                with gr.Row():
                    with gr.Accordion(label="📖 Словарь фамилий, терминов и профессий для системы перевода речи в текст",
                                      open=False):
                        prompt_code_whisper = gr.Code(
                            value=w.load_prompt(),
                            language=None,
                            label="Whisper",
                            interactive=True,
                            lines=50,
                            scale=4,
                        )

                        with gr.Row():
                            load_btn = gr.Button("📥 Загрузить", size="sm", variant="primary")
                            save_btn = gr.Button("📤 Сохранить", size="sm", variant="secondary")
                        load_btn.click(fn=w.load_prompt, inputs=None, outputs=prompt_code_whisper)
                        save_btn.click(fn=w.save_prompt, inputs=prompt_code_whisper, )

            # ---------------------------------------
            # Вкладка 5 -- Benchmarking
            # ---------------------------------------

            with gr.Tab("🚀️ Тестирование производительности"):
                gr.Markdown("""<h3>⚙️ Тестирование производительности генеративных моделей и сервера Ollama</h3>""")
                # models_list = reassert_main_model_dropdown(True)
                with gr.Row():
                    model_selector = gr.Dropdown(
                        choices=fn_load_main_model(),
                        multiselect=True,
                        label="Выберите модели для тестирования",
                        interactive=True,
                        max_choices=2,
                    )

                    laps_slider = gr.Slider(
                        value=2, minimum=1, maximum=10, step=1,
                        label="Количество прогонов",
                        interactive=True,
                    )

                    with gr.Column():
                        refresh_models_btn = gr.Button("🔄 Загрузить/обновить список моделей", scale=20, size="md")
                        start_benchmark_btn = gr.Button("🚀 Запустить тестирование", scale=20, size="md")

                with gr.Column():
                    log_output = gr.Textbox(
                        label="Ход выполнения",
                        lines=8,
                        interactive=False,
                        autoscroll=True,
                        show_copy_button=True,
                    )

                gr.Markdown("""### ℹ️ Пояснение к метрикам
                    - **Wall Avg(s)** – среднее полное время ответа (что видит пользователь).
                    - **Eval Avg(s)** – среднее время генерации без задержек.
                    - **TPS Avg** – скорость генерации текста (Tokens Per Second).
                    - **σ** — стандартное отклонение (разброс значений).
                    """)

                result_table = gr.DataFrame(
                    headers=["Модель", "Тип", "Wall Avg (s)", "Wall σ", "Eval Avg (s)", "Eval σ", "TPS Avg", "TPS σ"],
                    row_count=(6, "dynamic"),
                    interactive=False,
                    label="Результаты замеров",
                    show_copy_button=True,
                )

                # previous_log_file = gr.File(label="📥 Загрузить старый отчёт (JSON)", file_types=[".json"])

                with gr.Row():
                    # load_prev_btn = gr.Button("📥 Загрузить старый отчёт", size="md")
                    json_view = gr.JSON(label="📄 JSON‑отчёт", visible=True, scale=3)
                    # download_log_btn = gr.File(label="📤 Скачать отчёт (JSON)", interactive=False, scale=1, height=10)

                # --- Функции ---

                async def update_dropdown():
                    models_response = await ollama.list()
                    models = sorted([m["model"] for m in models_response["models"]])
                    return gr.update(choices=models, value=models[-1] if models else [])

                async def wrapped_benchmark(models, laps, think):
                    if not models:
                        return (
                            "🟡 Сначала загрузите и выберите модели для тестирования.",
                            [],
                            {},
                            None
                        )
                    logs, table, json_data, path = await benchmark(models, laps, think)
                    return logs, table, json_data, path

                # --- Привязка кнопок ---

                refresh_models_btn.click(fn=update_dropdown, inputs=[], outputs=model_selector)

                start_benchmark_btn.click(
                    fn=wrapped_benchmark,
                    inputs=[model_selector, laps_slider, think_checkbox],
                    outputs=[log_output,
                             result_table,
                             json_view,
                             # download_log_btn,

                             ]
                )

            # --------------------------------------------
            # --- 📈Вкладка 6 Мониторинг нагрузки --------
            # --------------------------------------------

            import pandas as pd

            with gr.Tab("📈 Мониторинг нагрузки"):
                gr.Markdown("<h3>Показатели использования RAM, SSD, GPU, CPU</h3>")

                MONITORED = [
                    "bookworm-agent",
                    "ollama",
                    "whisper-gpu",
                    "meili_server",
                    "chroma_container",
                    "vosk-ru",
                    "nginx_proxy",
                ]

                THRESHOLDS = {
                    "cpu_warning": 70,
                    "cpu_critical": 90,
                    "ram_warning_mb": 96_000,
                    "ram_critical_mb": 196_000,
                    "vram_warning_free_mb": 1500,
                    "vram_critical_free_mb": 700,
                }

                # для построения графиков
                history_state = gr.State([])


                def build_monitor_view():
                    """
                    Возвращает сразу несколько “виджетов”:
                    - summary_md: человекочитаемая сводка
                    - top_ram_df: топ по RAM
                    - top_cpu_df: топ по CPU
                    - gpu_df / gpu_note
                    - details_json: полный payload
                    """
                    payload = system_data.make_human_monitor_payload(MONITORED)

                    s = payload.get("summary", {})
                    summary_md = (
                        f"### Сводка\n"
                        f"- **Процессор (CPU), суммарно по контейнерам:** {s.get('total_cpu_%_sum', 0)}%\n"
                        f"- **Оперативка (RAM), суммарно по контейнерам:** {s.get('total_ram_used_mb', 0)} MB\n"
                        f"- **Docker-сеть IN/OUT:** {s.get('total_net_in_mb', 0)} / {s.get('total_net_out_mb', 0)} MB\n"
                        f"- **Процессы (PIDs) суммарно:** {s.get('total_pids', 0)}\n\n"
                        # f"ℹ️ CPU здесь — это сумма docker CPU% по контейнерам (может быть >100% при использовании нескольких ядер)."
                    )

                    # Top consumers
                    top = payload.get("top_consumers", {})
                    by_ram = top.get("by_ram", [])
                    by_cpu = top.get("by_cpu", [])


                    top_ram_df = pd.DataFrame(by_ram) if by_ram else pd.DataFrame(
                        columns=["container", "ram_used_mb", "cpu_%"])
                    top_cpu_df = pd.DataFrame(by_cpu) if by_cpu else pd.DataFrame(
                        columns=["container", "cpu_%", "ram_used_mb"])


                    # GPU
                    gpu = payload.get("gpu", {})
                    gpu_note_ = ""
                    gpu_df = pd.DataFrame()

                    if isinstance(gpu, dict) and "gpus" in gpu:
                        gpu_df = pd.DataFrame(gpu["gpus"])
                    else:
                        gpu_note_ = gpu.get("note", "GPU данные недоступны.")

                    return summary_md, top_ram_df, top_cpu_df, gpu_df, gpu_note_, payload



                # --- UI ---
                summary_md = gr.Markdown()
                warnings_md = gr.Markdown()
                gpu_note = gr.Markdown()

                with gr.Row():
                    gpu_table = gr.Dataframe(label="Видеокарты (Общее использование / Загрузка памяти)", interactive=False, wrap=False)
                with gr.Row():
                    top_ram = gr.Dataframe(label="Топ контейнеров по загрузке оперативной памяти", interactive=False, wrap=True)
                    top_cpu = gr.Dataframe(label="Топ контейнеров по загрузке процессора", interactive=False, wrap=True)

                # графики по серверу
                cpu_plot = gr.LinePlot(x="time", y="cpu_host_%", title="Загрузка центрального процессора (CPU), %", height=260)
                ram_plot = gr.LinePlot(x="time", y="ram_mb", title="Оперативка, RAM (занятая контейнерами), MB", height=260)
                vram_plot = gr.LinePlot(x="time", y="vram_free_mb_min", title="Свободная видеопамять у самой загруженной видеокарты, MB", height=260)


                with gr.Accordion("Детальный отчет JSON", open=False):
                    details_json = gr.JSON(label="Нагрузка на сервер (по всем docker - контейнерам)", min_height=400, max_height=900)

                t = gr.Timer(1.0)

                with gr.Row():
                    btn = gr.Button("Обновить данные", size="sm", variant="secondary")
                    tick_slider = gr.Slider(
                        label="Частота обновления с шагом 0.5 сек",
                        minimum=0.5,
                        maximum=2.0,
                        step=0.5,
                        value=1.0,
                    )

                def render(payload, history):
                    # summary
                    s = payload.get("summary", {})
                    host = payload.get("host", {})
                    summary = (
                        f"### Сводка\n"
                        f"- **CPU (ядра):** {s.get('cpu_cores_used', 0)}\n"
                        f"- **CPU (% от сервера):** {s.get('cpu_host_%', 0)}% (логических CPU: {host.get('cpu_count', '?')})\n"
                        f"- **RAM (сумма контейнеров):** {s.get('total_ram_used_mb', 0)} MB\n"
                        f"- **Docker-сеть IN/OUT:** {s.get('total_net_in_mb', 0)} / {s.get('total_net_out_mb', 0)} MB\n"
                        f"- **PIDs (суммарно):** {s.get('total_pids', 0)}\n"
                    )

                    # warnings
                    warns = system_data.compute_alerts(payload, THRESHOLDS)
                    warnings_text = (
                        "### ⚠️ Предупреждения\n" + "\n".join([f"- {w}" for w in warns])
                        if warns else
                        "### ✅ Предупреждения\n- Нет превышений порогов."
                    )

                    # top consumers
                    top = payload.get("top_consumers", {})
                    by_ram = top.get("by_ram", [])
                    by_cpu = top.get("by_cpu", [])

                    top_ram_df = pd.DataFrame(by_ram) if by_ram else pd.DataFrame(
                        columns=["container", "ram_used_mb", "cpu_%"])
                    top_cpu_df = pd.DataFrame(by_cpu) if by_cpu else pd.DataFrame(
                        columns=["container", "cpu_%", "ram_used_mb"])

                    # gpu table + note
                    gpu = payload.get("gpu", {})
                    if isinstance(gpu, dict) and "gpus" in gpu:
                        gpu_df = pd.DataFrame(gpu["gpus"])
                        gpu_note_ = ""
                    else:
                        gpu_df = pd.DataFrame(
                            columns=["gpu", "name", "util_gpu_%", "vram_used_mb", "vram_free_mb", "vram_total_mb"])
                        gpu_note_ = gpu.get("note", "GPU данные недоступны.")

                    # history + df for plots
                    history = system_data.update_history(history, payload, max_points=180)
                    df = system_data.history_to_df(history)

                    # важно: df должен содержать колонки time, cpu_host_%, ram_mb, vram_free_mb_min
                    return summary, warnings_text, top_ram_df, top_cpu_df, gpu_df, gpu_note_, df, df, df, payload, history

                def tick(history):
                    payload = system_data.make_human_monitor_payload(MONITORED)
                    return render(payload, history)

                btn.click(
                    fn=tick,
                    inputs=[history_state],
                    outputs=[
                        summary_md,
                        warnings_md,
                        top_ram,
                        top_cpu,
                        gpu_table,
                        gpu_note,
                        cpu_plot,
                        ram_plot,
                        vram_plot,
                        details_json,
                        history_state
                    ],
                )

                t.tick(
                    fn=tick,
                    inputs=[history_state],
                    outputs=[
                        summary_md,
                        warnings_md,
                        top_ram,
                        top_cpu,
                        gpu_table,
                        gpu_note,
                        cpu_plot,
                        ram_plot,
                        vram_plot,
                        details_json,
                        history_state
                    ],
                )

                # --- handlers ---
                # btn.click(
                #     fn=build_monitor_view,
                #     outputs=[summary_md, top_ram, top_cpu, gpu_table, gpu_note, details_json],
                # )
                # t.tick(
                #     fn=build_monitor_view,
                #     outputs=[summary_md, top_ram, top_cpu, gpu_table, gpu_note, details_json],
                # )


                # подавляем двойное всплываение gr.Info на старте
                first_change = gr.State(True)

                def tick_slider_change(value: float, is_first: bool):
                    if is_first:
                        return value, False
                    gr.Info(f"Частота обновления данных: {value}", duration=3.0, title="Системный монитор")
                    return value, False

                tick_slider.change(
                    fn=tick_slider_change,
                    inputs=[tick_slider, first_change],
                    outputs=[t, first_change],
                )

        # -------------------------
        # Footer html реализация
        # -------------------------
        gr.HTML(custom_footer)
    blocks.queue(
        default_concurrency_limit=8,
        max_size=64)
    blocks.launch(
        server_name="0.0.0.0",
        server_port=7860,
        auth=check_auth,
        show_api=False,
        allowed_paths=[str(STATIC_DIR)],
        favicon_path=str(STATIC_DIR / "logo.png")
    )


if __name__ == "__main__":
    main()
