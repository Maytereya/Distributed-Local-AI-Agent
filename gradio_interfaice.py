from __future__ import annotations

import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Dict, List, Union, Tuple, Literal, Any

import gradio as gr
from gradio_pdf import PDF

import agent_logic_2.ollama_settings as ollama_settings
from VOSK.audio_stream_ws import flush_ws
from agent_logic_2 import config as c
from agent_logic_2.benchmark_tab import gradio_benchmark as benchmark
from agent_logic_2.benchmark_tab import ollama_client as ollama
from agent_logic_2.direct_upload_meili_tab import build_blocks, TABLE_HEADERS
from agent_logic_2.id_validation import is_valid_id, sanitize_id
from agent_logic_2.prompts import load_prompt, write_prompt
from agent_logic_2.router_preprocessor import routing
from agent_logic_pack import aretrieve3 as retrieve
from agent_logic_pack import meilisearch_client as meilisearch
# from asr_sber import asr_stream_sber, sber_flush, sber_reset_state
# from auth_sber import start_token_refresher
from container_managenment import restart_container
from converters import pdf_to_json_txt_tables_meili as pdf2json
#
from whisper.wisper_ws_client import ws_transcribe
#

# Label constants
COLLECTIONS_IN_CHROMA = "Коллекции документов Chroma DB"
INDEXES_IN_MEILI = "Индексы документов Meilisearch"
# Static files config for Gradio
STATIC_DIR = (Path(__file__).parent / "static").resolve()

# OpenGraph/Twitter preview meta tags (for messengers and social previews)
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
# SECURITY
# -------------------

USERNAME = c.AUTH_NAME
PASSWORD = c.AUTH_PASS
# Пока не срабатывает.
os.environ["USER_AGENT"] = "NEIRY.Agent/1.0"

ALL_MODELS = []


def check_auth(username, password):
    return username == USERNAME and password == PASSWORD


# -------------------
# Footer!
# -------------------
custom_css = """
/* 
   Общие стили для футера: расположим элементы 
   вертикально (flex-direction: column), 
   без лишних отступов (gap, margin, padding).
*/
footer {
    display: flex !important;           /* Чтобы мы могли управлять расположением */
    /* flex-direction: column !important;   Расположим элементы сверху вниз */
    align-items: center !important;     /* Центрируем по горизонтали */
    justify-content: center !important;
    text-align: center !important;
    gap: 0.25rem !important;            /* Небольшой зазор между строками */
    margin: 0 !important;
    padding: 3px 0 !important;          /* Можно подвинуть значение для плотности */
}

/* Убираем потенциальные точки/буллеты у "Built with Gradio" */
/* footer ul {
    list-style: none !important;
    margin: 0 !important;
    padding: 0 !important;
}*/

/* Это сам блок, где вы размещаете свою ссылку и копирайт */
#custom-footer {
    text-align: center;
    margin: 0 auto;
    padding: 0;
    color: #ccc;              /* Светло-серый цвет для текста (можно поменять) */
    font-size: 14px;
    line-height: 1.2;         /* Чуть плотнее строки */
}

/* Если хотите, чтобы только ссылка была #ccc, а текст — другим цветом,
   перенесите color в #custom-footer a { ... } */
#custom-footer a {
    text-decoration: none;
    color: #ccc;              /* Цвет ссылки */
    margin-left: 0.5rem;      /* Отступ между текстом и ссылкой */
}

/* Шапка с логотипом — без лишних отступов */
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
# ECHOES PART
# -------------------

# Глобальная сессия (Gradio поддерживает per-user state)
# state = gr.State({})  # будет передаваться как дополнительный input/output

# ЛОГИРОВАНИЕ ДЛЯ ПОИСКА ПРОБЛЕМЫ С РАСПОЗНАВАНИЕМ ОТ СБЕРБАНКА
import os, logging, sys, asyncio


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
    Обновлённая версия universal_echo — подключает роутер и стримит ответ.
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

    :param message:
    :param history:
    :param index:
    :param limit:
    :return: string результаты поиска
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
# CHROMA DB section
# --------------------

def gr_create_collection(c_name: str):
    """
    Created a collection and return its name in str.
    Warning: in the next releases Chroma .name parameter will be removed!
    :param c_name: String, passed new name of the collection.
    :return: String, name of the collection.
    """
    if c_name:
        result = retrieve.create_collection(c_name)
        time.sleep(5)
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
    retrieve.remove_collection(c_name)
    new_collections = gr_existed_collections()
    return (
        gr.update(choices=new_collections, value=None),
        gr.update(choices=new_collections, ),
        "Коллекция удалена",
    )


def gr_existed_collections():
    """

    :return: List of existed collection names.
    """
    chroma_service = retrieve.ChromaService(c.chroma_host, c.chroma_port)
    return chroma_service.display_collections(output_format="list")


def existed_docs_in_selected_collection(selected_collection: str):
    """
    Через функционал collection.peek["metadatas"] получение названия загруженных документов и их страниц
    :return: List of strings.
    """
    if not selected_collection:
        return ["Коллекция не выбрана"]
    try:
        return retrieve.handle_collection(selected_collection)
    except Exception as e:
        return ["Вероятно, коллекция отсутствует"]


def gr_add_to_collection(collection: str, file_path: str):
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
# MEILI section
# ----------------

def gr_existed_indexes():
    """

    :return: List of existed Meilisearch indexes.
    """
    return meilisearch.show_list_indexes(detail_mode="uid")


def existed_docs_in_selected_index(selected_index: str,
                                   return_type: Literal["All", "ID"]) -> List[str] | List[List[str]]:
    """
    """
    if not selected_index:
        return ["Индекс не выбран"]
    if return_type == "All":
        return meilisearch.meili_list_documents(selected_index, return_type="All")
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
        base_no_ext_clean = re.sub(r'[^a-zA-Z0-9-_]', '_', base_no_ext)

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
        time.sleep(5)
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
        base_no_ext_clean = re.sub(r'[^a-zA-Z0-9-_]', '_', base_no_ext)
        local_json_path = f"Upload/{base_no_ext_clean}.json"
        # Копируем загруженный временный файл в свою папку
        # (import shutil)
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
        time.sleep(5)

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
    Remove an index from a Meilisearch instance and update the lists of available indexes.

    The function deletes the specified index from Meilisearch, waits for a short period to
    ensure index operation consistency, and retrieves the updated list of existing indexes
    to reflect changes. If no indexes remain, the function returns a default message
    indicating the absence of indexes.

    :param index: The name of the index to be removed from Meilisearch.
    :type index: str
    :return: Tuple containing updates for UI components with new index choices, an update
             message, and other relevant UI values.
    :rtype: tuple
    """
    meilisearch.delete_index(index)
    #
    time.sleep(10)
    #
    new_list = gr_existed_indexes() if gr_existed_indexes() else "Индекс отсутствует"
    gr.Success(message=f"Индекс {index} удален", title="Успешно")
    return (
        gr.update(choices=new_list, value=new_list[0] if new_list else ""),
        gr.update(choices=new_list, value=new_list[0] if new_list else ""),
        gr.update(choices=new_list, value=new_list[0] if new_list else ""),
        gr.update(choices=new_list, value=new_list[0] if new_list else ""),
    )


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
        gr.update(choices=new_list, value=index_name),
        gr.update(choices=new_list, value=index_name),
        gr.update(choices=new_list, value=index_name),

    )


def gr_rm_doc_from_index(ind_id: str, doc_id: str):
    meilisearch.delete_meili_document(ind_id, doc_id)
    id_list = existed_docs_in_selected_index(gr_existed_indexes()[0], return_type="ID")
    full_list = existed_docs_in_selected_index(gr_existed_indexes()[0], return_type="All")
    gr.Success(message=f"Документ с ID {ind_id} удален", title="Успешно")
    return (
        gr.update(choices=id_list, ),
        gr.update(value=full_list, ),
    )


def validate_id_live(current: str) -> dict[str, Any]:
    """
    Живой валидатор для id_input: если строка уже валидна — оставляем,
    если нет — предлагаем «почищенную» версию.
    Возвращаем (value для id_input, статус).
    """
    if is_valid_id(current):
        gr.Info("ID валиден", title="Инфо")
        return gr.update(value=current)
    proposal = sanitize_id(current or "")
    # если пусто — не подменяем молча
    if proposal and proposal != current:
        gr.Info("ID нормализован автоматически", title="Инфо")
        return gr.update(value=proposal)
    gr.Warning("Некорректный ID. Разрешены: [a-z0-9_], длина 3–60", title="Предупреждение")
    return gr.update(value=current)


# ------------------------------------
# GRADIO WRAPPING functions section
# ------------------------------------

def update_docs_in_meili_index(index_name: str, output: Literal["full", "id_only"] = "full"):
    """
    Функция-обработчик для .change события:
    При выборе индекса возвращает обновлённый список документов в этом индексе
    для выпадающего списка документов (meili_content_of_index_dropdown).
    """

    full_list = existed_docs_in_selected_index(index_name, return_type="All")
    id_list = existed_docs_in_selected_index(index_name, return_type="ID")

    if output == "full":
        return (
            gr.update(value=full_list, ),
            gr.update(choices=id_list, value=(id_list[0] if id_list else "нет данных")),
        )
    else:
        return gr.update(choices=id_list, value=(id_list[0] if id_list else "нет данных"))


def update_docs_in_chroma_collection(collection_name: str):
    """
    Аналогичная функция для Chroma:
    При выборе коллекции возвращаем список документов в ней.
    """
    chroma_doc_list = retrieve.handle_collection(collection_name)
    return (
        gr.update(choices=chroma_doc_list, ),
        gr.update(value=chroma_doc_list)
    )


def txt_default():
    return f"Ожидание действий..."


def radio_sliders_change(choice):
    """
    value_n_results_slider,
    thresholdvalue_slider,
    value_k_slider,
    meili_search_indexes_dropdown,
    chroma_search_collection_dropdown,

    The search type selector that enables appropriated sliders.
    :param choice: AI Router, Vectorstore search, or Chroma DB Search.
    :return: Configurations of sliders that refine the search.
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

    # start_token_refresher()  # Получение свежего SALUT - токена от Сбера. Который действует 30 минут.
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
    # ToDo: возможно убрать!
    asr_state = gr.State(value={})  # чтобы не было первого None при отправке чанков в сбер.

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

                # ====== ЗАХВАТ АУДИО И РАСШИФРОВКА ======
                with gr.Row(visible=True):
                    asr_state = gr.State()  # хранит rec и накопленный текст
                    live_transcript = gr.Textbox(
                        label="🎙️ Живая расшифровка",
                        lines=3,
                        max_lines=3,
                        interactive=False,
                        show_copy_button=True,
                        autoscroll=True,
                        container=True,
                        visible=True,
                        scale=70,
                    )
                    mic = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        streaming=False,
                        label="Микрофон",
                        interactive=True,
                        format="wav",
                        min_width=150,
                        show_download_button=True,
                        visible=True,
                        scale=30,
                        # recording=True,
                        # stop_recording_on_silence=True,
                    )
                with gr.Row(visible=True):
                    whisper_btn = gr.Button("1️⃣ 🔊 Расшифровать", variant="primary", size="sm", scale=50, visible=True)
                    search_btn = gr.Button("2️⃣ 🔎 Передать в поиск", variant="secondary", size="sm", scale=30, visible=True)
                    # flush_btn = gr.Button("Завершить фразу", variant="stop", size="sm", scale=10, visible=False)
                    reset_asr_btn = gr.Button("3️⃣ 🗑️ Сбросить", variant="secondary", size="sm", scale=20, visible=True)
                    # Ищем поломку
                    # test_btn = gr.Button("🔊 Тест Sber: WAV", variant="secondary", size="sm", visible=False)
                    # whisper_btn = gr.Button("🔊 Расшифровать", variant="primary", size="sm", scale=10, visible=True)

                whisper_btn.click(fn=ws_transcribe, inputs=mic, outputs=live_transcript)

                # async def _test_push(asr_state):
                #     import soundfile as sf  # pip install soundfile
                #     data, sr = sf.read("data/sample.wav", dtype="float32", always_2d=False)
                #     if data.ndim == 2: data = data.mean(axis=1)
                #     # используем тот же поток:
                #     live, state = await asr_stream_sber((sr, data), asr_state or {})
                #     # и сразу EOF, чтобы увидеть финальный текст:
                #     final, state = await sber_flush(state)
                #     return (final or live), state

                # test_btn.click(fn=_test_push, inputs=[asr_state], outputs=[live_transcript, asr_state])

                # потоковое обновление текста, VOSK - версия
                # mic.stream(
                #     # fn=audio_stream.vosk_stream,
                #     fn=vosk_ws_stream,
                #     inputs=[mic, asr_state],
                #     outputs=[live_transcript, asr_state]
                # )

                # потоковое обновление текста Сбербанк - версия
                # mic.stream(fn=asr_stream_sber,
                #            inputs=[mic, asr_state],
                #            outputs=[live_transcript, asr_state])

                # Автоматически завершать фразу при остановке записи, Сбербанк - версия:
                # mic.stop_recording(
                #     fn=sber_flush,
                #     inputs=[asr_state],
                #     outputs=[live_transcript, asr_state],
                # )

                # сброс состояния без EOF
                def reset_asr(_state):
                    # аккуратно закрыть сокет, если открыт
                    return gr.update(value=""),{"text": "", "acc": bytearray(), "ws": None, "closing": False}, None

                # VOSK - версия:
                reset_asr_btn.click(reset_asr, inputs=[asr_state], outputs=[live_transcript, asr_state, mic])
                # Сбербанк - версия:
                # reset_asr_btn.click(fn=sber_reset_state, inputs=[asr_state], outputs=[live_transcript, asr_state])

                # завершить фразу и получить финал
                async def flush_click(asr_state):
                    txt, st = await flush_ws(asr_state or {})
                    return txt, st



                # VOSK - версия:
                # flush_btn.click(flush_click, inputs=[asr_state], outputs=[live_transcript, asr_state])
                # Сбербанк - версия:
                # flush_btn.click(fn=sber_flush, inputs=[asr_state], outputs=[live_transcript, asr_state])



                # ====== ГЛАВНЫЙ ИНТЕРФЕЙС ======
                chatbot = gr.Chatbot(type="messages",
                                     autoscroll=False,
                                     placeholder="<strong>Поиск по документам</strong><br>Задайте вопрос",
                                     height=700,
                                     label="Чат с ИИ Медцентра")

                textbox = gr.Textbox(lines=1,
                                     placeholder="Напишите свой вопрос",
                                     submit_btn=True,
                                     stop_btn=True,
                                     container=True,
                                     autoscroll=False,
                                     autofocus=False,
                                     html_attributes=gr.InputHTMLAttributes(autocorrect="off", spellcheck=False)
                                     )

                with gr.Column():
                    with gr.Row():
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
                settings_accordion = gr.Accordion("Настройки поиска", open=False, visible=True, render=False)

                demo = gr.ChatInterface(
                    fn=universal_echo,
                    type="messages",
                    chatbot=chatbot,  # без examples: тут они вообще не работают
                    textbox=textbox,
                    submit_btn="Отправить",  # <-- текст на кнопке отправки
                    stop_btn="⏹ Остановить",  # <-- тогда появится стоп во время стрима
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

                # Передача аудиораcшифровки в поиск AI-Search
                # Оставляем тут, иначе textbox не определится.

                def copy_to_textbox(input: str):
                    return gr.update(value=input)

                search_btn.click(
                    fn=copy_to_textbox,
                    inputs=live_transcript,
                    outputs=textbox,
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

                # ----------------------------------------------------
                # Секция оформления страницы - конструктора документа
                # ----------------------------------------------------

                # единый state вместо отдельных переменных для хранения документов прямой загрузки
                meta_state = gr.State(
                    value=None)  # dict: {"doc_id": str, "index": str, "blocks": list[dict]}
                table_state = gr.State(value=[["", ""], ["", ""], ["", ""]])

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
                            # Дропдаун служит для выбора ID
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
                            Сделан чтобы исключить выбор radio и зациклить только на индексе.
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

                    def save_and_send_to_meilisearch(meta):

                        if not meta:
                            gr.Warning("Нет данных (сделайте Предпросмотр)", title="Предупреждение")
                            return None

                        index_name = meta["index"]
                        _blocks = meta["blocks"]
                        if not index_name:
                            gr.Warning("Не выбран индекс", title="Предупреждение")
                            return None

                        if not _blocks:
                            gr.Warning("Пустой массив блоков", title="Предупреждение")
                            return None

                        msg: str = ""
                        try:
                            msg = meilisearch.add_doc_to_meili(_blocks, index_name)

                        except Exception as e:
                            gr.Error(f"{e}, сообщение от сервера Meilisearch: {msg}",
                                     title="Ошибка!")
                            return None
                        gr.Success(
                            f"✅ '{meta['doc_id']}', добавлено {len(_blocks)} блок(ов) в '{index_name}', "
                            f"сообщение от сервера  Meilisearch: {msg}", title="Успешно")
                        return None

                    save_button.click(
                        save_and_send_to_meilisearch,
                        inputs=[meta_state],

                    )
                    # ------------------------------
                    # живой валидатор на каждый ввод
                    # ------------------------------

                    id_input.change(validate_id_live, inputs=[id_input], outputs=[id_input,
                                                                                  # status_output
                                                                                  ])

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
                        return "", "", "", "", [["", ""], ["", ""], ["", ""]], "static", None, None, False

                    vanish_screen_btn.click(
                        vanish_all_windows,
                        outputs=[id_input, title_input, content_input, keywords_input, table_df,
                                 doc_type_radio, valid_from_dp, valid_to_dp, permanent_cb]
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
                        if not index_name:
                            gr.Warning("Укажите индекс", title="Предупреждение")
                            return (gr.update(),) * 10
                        if not doc_id:
                            gr.Warning("Укажите ID", title="Предупреждение")
                            return (gr.update(),) * 10

                        doc = meilisearch.get_document_by_id(index_name, doc_id)
                        if not doc:
                            gr.Warning("❌ Документ не найден", title="Предупреждение")
                            return (gr.update(),) * 10  # Хороший способ уменьшить визуальные повторения

                        # Подставляются дефолтные ключи
                        title = doc.get("title", "")
                        content = doc.get("content", "")

                        # 👉 если поля нет — берём тип из выбранного индекса
                        doc_type = doc.get("doc_type") or doc.get("type") or TYPE_FOR_INDEX.get(index_name, "static")
                        # Устаревшая маркировка поля, больше не используется, конвертируем text -> static
                        if doc_type == "text":
                            doc_type = "static"
                        is_news = (doc_type == "news")

                        #
                        vf = _parse_iso(doc.get("valid_from"))
                        vt = _parse_iso(doc.get("valid_to"))
                        is_perm = bool(doc.get("is_permanent", False))
                        # если это news, но дат нет — подсказываем пользователю
                        if is_news and not (vf or vt or is_perm):
                            gr.Warning(
                                "Это документ из индекса 'news', но даты не заданы. Укажите период или отметьте 'Бессрочно'.",
                                title="Требуются даты")

                        keywords = doc.get("keywords", "")
                        table_val = doc.get("table") or [["", ""], ["", ""], ["", ""]]

                        gr.Success(message="✅ Документ загружен", title="Успешно")
                        # Важно: поставить тип документа и корректно показать/скрыть даты

                        # Показываем даты и чекбокс только если news
                        is_news = (doc_type == "news")

                        return (
                            gr.update(value=doc_id),
                            gr.update(value=title),
                            gr.update(value=content),

                            gr.update(value=vf),  # valid_from_dp
                            gr.update(value=vt),  # valid_to_dp
                            gr.update(value=is_perm, visible=is_news),  # permanent_cb

                            gr.update(value=keywords),
                            gr.update(value=table_val),
                            gr.update(value=doc_type),  # doc_type_radio ← новый выход
                            gr.update(visible=is_news),  # news_dates_row ← новый выход
                        )

                    # Обработка события загрузки документа в форму редактирования
                    load_doc_btn.click(
                        load_doc_into_form_by_id,
                        inputs=[index_dropdown, id_select],
                        outputs=[id_input,  # 8 pcs
                                 title_input,
                                 content_input,
                                 valid_from_dp,
                                 valid_to_dp,
                                 permanent_cb,
                                 keywords_input,
                                 table_df,
                                 doc_type_radio,
                                 news_dates_row,
                                 ],
                    )

                    #  ------------------------------------
                    # Секция сохранения документа после
                    # редактирования
                    #  ------------------------------------

                    def save_doc_by_id(doc_type: Literal["static", "news", "text"], index_name: str, doc_id: str,
                                       title: str,
                                       content: str,
                                       valid_from, valid_to,
                                       permanent, keywords: str, table):
                        """
                        Важен актуальный формат исходящего документа: новость или скрипт
                        :param doc_type:
                        :param index_name:
                        :param doc_id:
                        :param title:
                        :param content:
                        :param valid_from:
                        :param valid_to:
                        :param permanent:
                        :param keywords:
                        :param table:
                        :return:
                        """
                        base_doc: dict[str, Any] = {
                            "id": doc_id,
                            "title": title or "",
                            "content": content or "",
                            "keywords": keywords or "",
                            "table": table or [],
                        }
                        # Жесткая проверка на соответствие индекса функционалу
                        expected_type = TYPE_FOR_INDEX.get(index_name, "static")
                        if doc_type != expected_type:
                            # жёстко приводим
                            doc_type = expected_type
                            gr.Warning("Тип документа приведён к выбранному индексу.", title="Предупреждение")

                        if not index_name:
                            gr.Warning("Укажите индекс", title="Предупреждение")
                            return None
                        if not doc_id or not is_valid_id(doc_id):
                            gr.Warning("Некорректный ID (разрешено [a-z0-9_], длина 3–60).", title="Предупреждение")
                            return None
                        if not title:
                            gr.Warning("Создайте заголовок", title="Предупреждение")
                            return None
                        if not content:
                            gr.Warning("Создайте контент", title="Предупреждение")
                            return None

                        # формирование документа в зависимости от его типа
                        # по умолчанию загрузится static

                        if doc_type == "news":
                            # нормализуем к UTC/ISO/ts
                            def _to_utc(dt):
                                if dt is None: return None
                                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

                            if permanent:
                                if not valid_from:
                                    gr.Warning("Уточните дату начала", title="Предупреждение");
                                    return None
                                vf = _to_utc(valid_from)
                                vt = datetime(2099, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
                            else:
                                if not valid_from or not valid_to:
                                    gr.Warning("Укажите обе даты или отметьте 'Бессрочно'", title="Предупреждение");
                                    return None
                                vf, vt = _to_utc(valid_from), _to_utc(valid_to)
                                if vf > vt:
                                    gr.Warning("Дата 'с' позже даты 'по'", title="Предупреждение");
                                    return None

                            base_doc.update({
                                "doc_type": "news",
                                "valid_from": vf.isoformat().replace("+00:00", "Z"),
                                "valid_to": vt.isoformat().replace("+00:00", "Z"),
                                "from_ts": int(vf.timestamp()),
                                "to_ts": int(vt.timestamp()),
                                "is_permanent": bool(permanent),
                            })
                        else:
                            base_doc.update({"doc_type": "static"})  # Важный момент! Возможно
                            # тут ошибка в связи с несоответствием с полем в Индексе: doc_type -> type
                            # print("Base_doc: ", base_doc)

                        msg = meilisearch.upsert_document(index_name, base_doc)

                        if msg == "OK":
                            gr.Success("✅ Сохранено", title="Успешно")
                            return None

                        else:
                            gr.Error(f"❌ {msg}", title="Ошибка!")
                            return None

                    save_doc_btn_direct.click(
                        save_doc_by_id,
                        inputs=[doc_type_radio,  # 1) doc_type
                                index_dropdown,  # 2) index_name
                                id_select,  # 3) doc_id
                                title_input,  # 4) title
                                content_input,  # 5) content
                                valid_from_dp,  # 6) valid_from
                                valid_to_dp,  # 7) valid_to
                                permanent_cb,  # 8) permanent
                                keywords_input,  # 9) keywords
                                table_state],  # 10) table
                    )

                # ToDo: Доделать с учетом date
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
                        label="Содержание выбранного Индекса",
                        headers=["ID документа", "Заголовок документа", "Фрагмент содержания"],
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

                # ---------------------------------
                # Event handlers of chat interface
                # ---------------------------------

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
                # Event handlers of ChromaDB
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
                # Event handlers of Meilisearch
                # --------------------------------------

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

                # Event handlers of universal indexation button
                # Универсальная кнопка для индексации (PDF или JSON)
                add_to_index_button.click(
                    gr_add_to_index_universal,
                    inputs=[upload_indices_dropdown, pdf, json_file, radio_type_of_upl_data],
                    outputs=[pdf, json_file, upload_indices_dropdown, meili_search_indexes_dropdown]
                )

                rm_doc_from_index_button.click(
                    gr_rm_doc_from_index,
                    inputs=[meili_ind_for_cont_dropdown, meili_content_of_index_dropdown],
                    outputs=[meili_content_of_index_dropdown, meili_indices_table]
                )

                # ----------------------------------------------------------------
                # ЛОГИКА ОБНОВЛЕНИЯ при выборе индекса/коллекции
                # для показа документов
                # ----------------------------------------------------------------

                # При смене выбранного индекса -> обновить список документов
                meili_ind_for_cont_dropdown.change(
                    fn=update_docs_in_meili_index,
                    inputs=meili_ind_for_cont_dropdown,
                    outputs=[meili_indices_table,
                             meili_content_of_index_dropdown,
                             ]
                )

                # При смене выбранной коллекции -> обновить список документов
                chroma_coll_for_cont_dropdown.change(
                    fn=update_docs_in_chroma_collection,
                    inputs=chroma_coll_for_cont_dropdown,
                    outputs=[chroma_collection_content,
                             chroma_collection_table]
                )

            refresh_data_btn.click(
                update_docs_in_meili_index,
                inputs=[meili_ind_for_cont_dropdown],
                outputs=[
                    meili_indices_table,
                    meili_content_of_index_dropdown,
                ]
            )

            # -------- FUNCTIONS SECTION ------------

            def fn_load_options(explain: bool = True) -> Union[tuple[str, str], str]:
                """
                Loads and serializes Ollama settings for requests into a JSON-formatted string with indentation.

                :return: A JSON-formatted string representation of the loaded settings
                    with ensured ASCII disabled and proper indentation.
                :rtype: Str
                """
                # return json.dumps(ollama_settings.load_settings(), ensure_ascii=False, indent=4)
                return ollama_settings.load_ollama_options(explain)

            def fn_save_options(text: str) -> str:
                """
                Saves the corrected Ollama settings to a JSON-formatted file.
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

            # ------------Container Management Section--------------

            def restart() -> str:
                return restart_container.restart_ollama_container()

            # ------------------------------------------------------

            with gr.Tab("⚙️ Настройки"):
                gr.Markdown("""<h3>⚙️ Настройки нейросетей и сервера Ollama</h3>""")

                with gr.Column():
                    status = gr.Textbox(lines=1,
                                        label="Текущий статус",
                                        submit_btn=False,
                                        container=True,
                                        autoscroll=False,
                                        interactive=True,
                                        autofocus=False,
                                        )

                    # -----------------------------------------------------
                    # OLLAMA OPTIONS SECTION
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
                #  Split prompt section
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
                #  Classificator prompt section
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
                # Final answering prompt section
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
                #     CONSTANTS SECTION
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

    # -------------------------
    # Footer html realization
    # -------------------------

    gr.HTML(
        """
        <div id="custom-footer">
            &copy; ООО "Нейри" 2025
            <a href="https://neiry-ai.ru" target="_blank">neiry-ai.ru</a>
        </div>
        """,
        visible=True
    )
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
