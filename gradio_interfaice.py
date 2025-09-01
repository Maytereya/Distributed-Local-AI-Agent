import json
import os
import re
import shutil
import time
import uuid
from typing import Dict, List, Union, Tuple
from pathlib import Path

import gradio as gr
from gradio_pdf import PDF

import agent_logic_2.ollama_settings as ollama_settings
from agent_logic_2 import config as c
from agent_logic_2.benchmark_tab import gradio_benchmark as benchmark
from agent_logic_2.benchmark_tab import ollama_client as ollama
from agent_logic_2.prompts import load_prompt, write_prompt
from agent_logic_2.router_preprocessor import routing
from agent_logic_pack import aretrieve3 as retrieve
from agent_logic_pack import meilisearch_client as meilisearch
from converters import pdf_to_json_txt_tables_meili as pdf2json

# Label constants
COLLECTIONS_IN_CHROMA = "Коллекции документов Chroma DB"
INDEXES_IN_MEILI = "Индексы документов Meilisearch"
# Static files config for Gradio
STATIC_DIR = (Path(__file__).parent / "static").resolve()

# EXAMPLES = [
#     [
#         "Запишите на прием к доктору Дразнину",  # message
#         "",  # chroma_search_collection_dropdown (не используется)
#         0.005,  # thresholdvalue_slider (заглушка)
#         5,  # value_n_results_slider (заглушка)
#         2,  # value_k_slider (заглушка)
#         "ai-router",  # radio_type_of_search
#         "",  # meili_search_indexes_dropdown (не используется)
#         {}  # state
#     ]
# ]
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


# ------------------------------------------------------------------------

async def echo_ai_router(message, history, session_state):
    """
    Обновлённая версия universal_echo — подключает роутер и стримит ответ.
    :param message: Текст запроса пользователя
    :param history: (не используется, можно убрать)
    :param session_state: словарь сессии, хранит pending и history
    :yields: два значения — текущий кусок ответа и обновлённый session_state
    """
    session_state = session_state or {}
    try:
        # routing возвращает AsyncGenerator[(partial_response, session), None]
        async for partial, session_state in routing(message, sess=session_state):
            # Каждая итерация — новое состояние и новый кусок ответа
            yield partial, session_state
    except Exception as e:
        # При ошибке тоже стримим её сразу
        yield f"⚠️ Ошибка обработки запроса в AI-router: {e}", session_state


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
        radio_value: str,  # "ai-router", "meilisearch", "vectorstore", "db"
        threshold_value: float,
        slider_value_n_results: int,
        slider_value_k: int,
        collection: str,
        meili_index: str,
):
    if radio_value == "ai-router":
        session_state: dict = {}
        # стримим
        async for partial, session_state in echo_ai_router(message, history, session_state):
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
    result = retrieve.create_collection(c_name)
    #
    time.sleep(5)
    #
    new_collections = gr_existed_collections()
    return (
        f"Коллекция {result.name} создана",
        gr.update(choices=new_collections, value=c_name, ),
        gr.update(choices=new_collections, value=c_name, ),
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


def existed_docs_in_selected_index(selected_index: str) -> list[str]:
    """

    :return:
    """
    if not selected_index:
        return ["Индекс не выбран"]
    return meilisearch.meili_list_documents(selected_index)


def gr_add_to_index_universal(index: str, pdf_path: str, json_file: str, doc_type: str):
    """
    Универсальная функция для добавления в Meilisearch либо PDF-файла (через конвертацию),
    либо JSON-файла напрямую.
    """
    if not index:
        return (
            gr.update(value=None),  # PDF
            gr.update(value=None),  # JSON
            "Ошибка: не выбран индекс. Укажите индекс Meilisearch.",
            gr.update(),
            gr.update()
        )

    if doc_type == "PDF":
        # Обработка PDF
        if not pdf_path:
            return (
                gr.update(value=None),
                gr.update(value=None),
                "Ошибка: PDF не загружен, загрузите документ.",
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
            return (
                PDF(value=None, label="Загрузить PDF", interactive=True, scale=80),
                gr.update(value=None),  # сбрасываем JSON
                f"{meili_msg}. {e}",  # Возможно, это дублирование одного и того же сообщения об ошибке
                gr.update(),
                gr.update(),
            )
        time.sleep(5)
        new_list = gr_existed_indexes()

        return (
            PDF(value=None, label="Загрузить PDF", interactive=True, scale=80),
            gr.update(value=None),  # сбрасываем JSON
            f"Файл (PDF) добавлен в индекс. {meili_msg}",
            gr.update(choices=new_list),
            gr.update(choices=new_list),
        )


    elif doc_type == "JSON":

        # Обработка JSON

        if not json_file:
            return (
                gr.update(value=None),
                gr.update(value=None),
                "Ошибка: JSON не загружен, загрузите документ.",
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
            return (
                gr.update(value=None),  # сбрасываем PDF
                gr.update(value=None),  # сбрасываем JSON
                f"{meili_msg}, {e}",  # Возможно, это дублирование одного и того же сообщения об ошибке
                gr.update(),
                gr.update(),
            )
        time.sleep(5)

        new_list = gr_existed_indexes()

        return (
            gr.update(value=None),  # сбрасываем PDF
            gr.update(value=None),  # сбрасываем JSON
            f"{meili_msg}",
            gr.update(choices=new_list),
            gr.update(choices=new_list),
        )


    else:
        return (
            gr.update(value=None),
            gr.update(value=None),
            "Неподдерживаемый тип документа.",
            gr.update(),
            gr.update()
        )


def gr_remove_index(index: str):
    """

    :param index:
    :return:

    """
    meilisearch.delete_index(index)
    #
    time.sleep(15)
    #
    new_list = gr_existed_indexes()

    return (
        gr.update(choices=new_list, value=None),
        gr.update(choices=new_list, value=None),
        gr.update(choices=new_list, value=None),
        "Индекс удален",
    )


def gr_create_index(index_name: str):
    """
    Создаёт индекс в Meilisearch
    """
    meilisearch.create_index(index_name)
    time.sleep(10)  #
    new_list = gr_existed_indexes()
    return (
        f"Индекс {index_name} создан",
        gr.update(choices=new_list, value=index_name),
        gr.update(choices=new_list, value=index_name),
    )


def gr_rm_doc_from_index():
    return None


# ------------------------------------
# GRADIO WRAPPING functions section
# ------------------------------------

def update_docs_in_meili_index(index_name: str):
    """
    Функция-обработчик для .change события:
    При выборе индекса возвращает обновлённый список документов в этом индексе
    для выпадающего списка документов (meili_content_of_index_dropdown).
    """
    meili_doc_list = existed_docs_in_selected_index(index_name)

    return (
        gr.update(choices=meili_doc_list, ),
        gr.update(value=meili_doc_list, ),
    )


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
    :param choice: AI Router, Vectorstore search or Chroma DB Search.
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
            # gr.Button(visible=True),
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
            # gr.Button(visible=False),
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
    # print("✅ |||| Загружено имя базовой LLM OLLAMA_MODEL:", ollama_settings.OLLAMA_MODEL, "|||")

    # Allow serving local /static files via /gradio_api/file=...
    gr.set_static_paths(paths=[STATIC_DIR])

    with gr.Blocks(css=custom_css, title="Neiry.ai") as blocks:
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

            with gr.Tab("\U0001F4D6 Поиск по документам"):
                chatbot = gr.Chatbot(type="messages",
                                     autoscroll=False,
                                     placeholder="<strong>Поиск по документам</strong><br>Задайте вопрос",
                                     height=700, )

                textbox = gr.Textbox(lines=1,
                                     placeholder="Напишите свой вопрос",
                                     submit_btn=True,
                                     container=True,
                                     autoscroll=False,
                                     autofocus=True)

                with gr.Column():
                    with gr.Row():
                        radio_type_of_search = gr.Radio(["ai-router", "meilisearch", "vectorstore", "db", ],
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
                        # TODO: C какой-то стати в value передается {}
                        chroma_search_collection_dropdown = gr.Dropdown(choices=gr_existed_collections(),
                                                                        label=COLLECTIONS_IN_CHROMA,
                                                                        info="Выберите Коллекцию для поиска информации",
                                                                        interactive=False,
                                                                        allow_custom_value=True,
                                                                        # крайне желательно этого избежать
                                                                        render=False,
                                                                        )

                # with gr.Row():

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

                demo = gr.ChatInterface(
                    fn=universal_echo,
                    type="messages",
                    # examples=EXAMPLES,
                    chatbot=chatbot,  # без examples тут
                    textbox=textbox,
                    additional_inputs_accordion="Настройки поиска",

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

            # --------------------------------------------------
            # Вкладка 2 - Upload PDF to MEILI or CHROMA DB
            # --------------------------------------------------

            with gr.Tab("\U0001F4E4 Загрузка готовых PDF/JSON"):
                gr.Markdown("### Загрузка и распределение документов по коллекциям ChromaDB или индексам Meilisearch")
                with gr.Row():
                    radio_type_of_db = gr.Radio(["ChromaDB", "Meilisearch"],
                                                label="Тип базы данных",
                                                value="ChromaDB",
                                                container=True,
                                                info="для добавления документа")

                    radio_type_of_upl_data = gr.Radio(["PDF", "JSON"],
                                                      label="Тип документа",
                                                      value="PDF",
                                                      container=True,
                                                      info="для загрузки в MEILISEARCH",
                                                      visible=False)

                    upload_collections_dropdown = gr.Dropdown(choices=gr_existed_collections(),
                                                              value=None,
                                                              allow_custom_value=True,
                                                              filterable=True,
                                                              label=COLLECTIONS_IN_CHROMA,
                                                              info="Коллекции документов по темам",
                                                              visible=True, )

                    upload_indices_dropdown = gr.Dropdown(choices=gr_existed_indexes(),
                                                          value=None,
                                                          allow_custom_value=True,
                                                          filterable=True,
                                                          label=INDEXES_IN_MEILI,
                                                          info="Индексы документов по темам",
                                                          visible=False)

                    # Поле для вывода текущего статуса работы с коллекциями
                    status_bar = gr.Textbox(value=txt_default,
                                            every=20.0,
                                            label="Монитор текущего статуса операции",
                                            # info="Только вывод",
                                            interactive=False, )

                    # Кнопки для работы с коллекциями или индексами
                    with gr.Column():
                        add_collection_button = gr.Button("Добавить коллекцию",
                                                          visible=True,
                                                          size="md",
                                                          )
                        rm_collection_button = gr.Button("Удалить коллекцию",
                                                         visible=True,
                                                         size="md",
                                                         )

                        # add_index_button = gr.Button("Добавить индекс", visible=False)
                        rm_index_button = gr.Button("Удалить индекс",
                                                    visible=False,
                                                    size="md",
                                                    )

                with gr.Row():
                    pdf = PDF(label="Загрузить PDF", interactive=True, scale=80)

                    # Загрузка JSON (по умолчанию невидим)
                    json_file = gr.File(
                        label="Загрузить JSON",
                        visible=False,
                        scale=80,
                        file_types=[".json"],  # или можно просто ["json"]
                        type="filepath"
                    )

                    with gr.Column():
                        add_to_collection_button = gr.Button("Добавить в коллекцию",
                                                             visible=True,
                                                             size="md",
                                                             )
                        add_to_index_button = gr.Button("Индексировать",
                                                        visible=False,
                                                        size="md",
                                                        )

                # ---------------------------------------------------
                # Секция просмотра содержимого коллекций
                # и индексов
                # ---------------------------------------------------

                gr.Markdown("### Содержание Индексов и Коллекций")

                with gr.Row():
                    with gr.Column():
                        meili_ind_for_cont_dropdown = gr.Dropdown(choices=gr_existed_indexes(),
                                                                  filterable=True,
                                                                  interactive=True,
                                                                  label=INDEXES_IN_MEILI,
                                                                  info="Выберите Индекс для просмотра содержимого",
                                                                  visible=True,
                                                                  )

                        meili_indices_table = gr.DataFrame(
                            value=existed_docs_in_selected_index(meili_ind_for_cont_dropdown.value),
                            label="Содержание выбранного Индекса",
                            headers=["Имя файла", ],
                            row_count=(15, "dynamic"),
                            col_count=(1, "fixed"),
                            datatype="str",
                            interactive=False
                        )
                        with gr.Row():
                            meili_content_of_index_dropdown = gr.Dropdown(
                                # choices=[],
                                choices=existed_docs_in_selected_index(meili_ind_for_cont_dropdown.value),
                                # value=None,
                                allow_custom_value=False,
                                label="Выбрать документ для удаления",
                                visible=True,
                                interactive=True,
                                scale=80

                            )

                            rm_doc_from_index_button = gr.Button("Удалить из Индекса",
                                                                 size="md",
                                                                 visible=True,
                                                                 interactive=False, )

                    with gr.Column():
                        chroma_coll_for_cont_dropdown = gr.Dropdown(choices=gr_existed_collections(),
                                                                    filterable=True,
                                                                    label=COLLECTIONS_IN_CHROMA,
                                                                    interactive=True,
                                                                    info="Выберите Коллекцию для просмотра содержимого",
                                                                    visible=True,
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
                        with gr.Row():
                            chroma_collection_content = gr.Dropdown(
                                # choices=[],
                                choices=existed_docs_in_selected_collection(chroma_coll_for_cont_dropdown.value),
                                # value=None,
                                allow_custom_value=False,
                                label="Выбрать документ для удаления",
                                visible=True,
                                interactive=True,
                                scale=80,
                            )

                            rm_doc_from_collection_button = gr.Button("Удалить из Коллекции",
                                                                      size="md",
                                                                      visible=True,
                                                                      interactive=False,
                                                                      )

                # ---------------------------------------------
                # Секция интерфейса чата, немного излишний код
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
                                            # add_index_button,
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
                    outputs=[status_bar, upload_collections_dropdown, chroma_search_collection_dropdown]
                )

                rm_collection_button.click(
                    gr_remove_collection,
                    inputs=upload_collections_dropdown,
                    outputs=[upload_collections_dropdown, chroma_search_collection_dropdown, status_bar, ]
                )

                add_to_collection_button.click(
                    gr_add_to_collection,
                    inputs=[upload_collections_dropdown, pdf],
                    outputs=[pdf, status_bar, ]
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
                        status_bar,
                    ]
                )

                # Event handlers of universal indexation button
                # Универсальная кнопка для индексации (PDF или JSON)
                add_to_index_button.click(
                    gr_add_to_index_universal,
                    inputs=[upload_indices_dropdown, pdf, json_file, radio_type_of_upl_data],
                    outputs=[pdf, json_file, status_bar, upload_indices_dropdown, meili_search_indexes_dropdown]
                )

                # ----------------------------------------------------------------
                # ЛОГИКА ОБНОВЛЕНИЯ при выборе индекса/коллекции
                # для показа документов
                # ----------------------------------------------------------------

                # При смене выбранного индекса -> обновить список документов
                meili_ind_for_cont_dropdown.change(
                    fn=update_docs_in_meili_index,
                    inputs=meili_ind_for_cont_dropdown,
                    outputs=[meili_content_of_index_dropdown,
                             meili_indices_table]
                )

                # При смене выбранной коллекции -> обновить список документов
                chroma_coll_for_cont_dropdown.change(
                    fn=update_docs_in_chroma_collection,
                    inputs=chroma_coll_for_cont_dropdown,
                    outputs=[chroma_collection_content,
                             chroma_collection_table]
                )

            # --------------------------------------------------
            # Вкладка 3 — конструктор документа для загрузки
            # --------------------------------------------------

            with gr.Tab("\U0001F4C4 Добавить документ в Meilisearch"):
                gr.Markdown("### Заполните форму для добавления документа в индекс Meilisearch")

                with gr.Column():
                    doc_id = str(uuid.uuid4())  # Генератор названия документа

                    def generate_new_id():
                        return str(uuid.uuid4())

                    # ----------------------------------------------------
                    # Секция оформления страницы - конструктора документа
                    # ----------------------------------------------------
                    with gr.Row():
                        index_dropdown = gr.Dropdown(
                            choices=gr_existed_indexes(),
                            label="Выберите индекс Meilisearch",
                            interactive=True,
                            scale=30,
                        )
                        id_input = gr.Textbox(label="ID документа (генерируется по умолчанию, но можно менять вручную)",
                                              value=generate_new_id,
                                              scale=50
                                              )
                        generate_id_button = gr.Button("🔄 Сгенерировать новый ID", scale=20, size="md")

                    title_input = gr.Textbox(label="Заголовок, title")
                    content_input = gr.Textbox(label="Основной текст, content", lines=20, max_lines=80)
                    keywords_input = gr.Textbox(label="Ключевые слова, keywords (через запятую)")
                    status_output = gr.Textbox(label="Статус операции", interactive=False, value=txt_default,
                                               every=10.0,
                                               container=False)
                    preview_button = gr.Button("Посмотреть получившийся JSON")
                    preview_json = gr.JSON(label="Предпросмотр JSON", visible=False)
                    save_button = gr.Button("Сохранить и отправить в индекс")

                # -----------------------------------------------------

                def new_doc_fulfilling():  # Пока не работает

                    return (
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                    )

                def fn_preview_json(current_doc_id, title, content, keywords, selected_index):

                    if not content.strip():
                        return gr.update(visible=False), "Ошибка: поле 'content' не может быть пустым"

                    if not selected_index:
                        return gr.update(visible=False), "Ошибка: не выбран индекс"

                    if not current_doc_id:
                        current_doc_id = generate_new_id()
                        id_input.value = current_doc_id

                    # Проверка уникальности ID
                    json_path = os.path.join("Upload", f"{current_doc_id}.json")
                    if os.path.exists(json_path):
                        return gr.update(visible=False), f"Ошибка: документ с ID '{current_doc_id}' уже существует."

                    doc = {
                        "id": current_doc_id,
                        "title": title or "(без заголовка)",
                        "content": content,
                        "keywords": [kw.strip() for kw in keywords.split(",") if kw.strip()],
                        "Indexes_meili": selected_index,
                    }

                    return (gr.update(visible=True,
                                      value=doc),
                            f"\u2705 Документ '{current_doc_id}' для индекса '{selected_index}' просматривается..."
                            )

                def save_and_send_to_meilisearch(current_doc_id, temporary_json, selected_index):

                    if not current_doc_id:
                        current_doc_id = generate_new_id()
                        id_input.value = current_doc_id

                    # Проверка уникальности ID
                    json_path = os.path.join("Upload", f"{current_doc_id}.json")
                    if os.path.exists(json_path):
                        return gr.update(visible=False), f"Ошибка: документ с ID '{current_doc_id}' уже существует."

                    if temporary_json:
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(temporary_json, f, ensure_ascii=False, indent=2)

                        try:
                            meilisearch.add_doc_to_meili(json_path, selected_index)
                        except Exception as e:
                            return (gr.update(),
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                    gr.update(),
                                    f"Ошибка загрузки в Meilisearch: {e}",
                                    gr.update(),
                                    gr.update(),
                                    )

                        return (gr.update(value=None),
                                gr.update(value=str(uuid.uuid4())),
                                gr.update(value=None),
                                gr.update(value=None),
                                gr.update(value=None),
                                f"\u2705 Документ успешно добавлен в индекс '{selected_index}'.",
                                gr.update(interactive=False),
                                gr.update(interactive=False),
                                )
                    else:
                        return (
                            gr.update(visible=True, ),
                            gr.update(visible=True, ),
                            gr.update(visible=True, ),
                            gr.update(visible=True, ),
                            gr.update(visible=True, ),
                            "\u274C Ошибка: документ для сохранения не передан. Сначала используйте 'Посмотреть получившийся JSON'.",
                            gr.update(visible=True, interactive=True),
                            gr.update(visible=True, interactive=True),
                        )

            # ---------------------------------------
            # Вкладка 4 -- Settings & Adjustments
            # ---------------------------------------

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

            # ------------ ASSERT(VALIDATE) MAIN LLM ---------------

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
                                interactive=True
                            )
                            reload_main_model_btn = gr.Button("🔄 Загрузить доступные модели", size="md")
                            save_model_btn = gr.Button("💾 Утвердить выбранную модель", size="md")
                            # TODO: Активация рассуждения пока не прокинута в router_preprocessor.
                            think_checkbox = gr.Checkbox(label="Активировать способность рассуждать",
                                                         # info="Только для reasoning models, снижает скорость",
                                                         value=ollama_settings.init_thinking())

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
                            load_options_btn = gr.Button("🔄 Загрузить текущие опции", scale=20, size="md")
                            save_options_btn = gr.Button("💾 Сохранить новые опции", scale=20, size="md")

                load_options_btn.click(fn=fn_load_options,
                                       # в данном случае explain = True (умолчание), так как надо передать оповещение в статус.
                                       inputs=[],
                                       outputs=[
                                           json_ollama_options,
                                           status
                                       ],
                                       )
                save_options_btn.click(fn=fn_save_options, inputs=json_ollama_options, outputs=status)

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

            # ----------------------------------
            # Event handlers for upper sections
            # ----------------------------------

            generate_id_button.click(fn=generate_new_id, inputs=[], outputs=[id_input])

            preview_button.click(
                fn=fn_preview_json,
                inputs=[id_input, title_input, content_input, keywords_input, index_dropdown],
                outputs=[preview_json,
                         status_output, ]
            )

            save_button.click(
                fn=save_and_send_to_meilisearch,
                inputs=[id_input, preview_json, index_dropdown],
                outputs=[preview_json,
                         id_input,
                         title_input,
                         content_input,
                         keywords_input,
                         status_output,
                         preview_button,
                         save_button, ]
            )

    # -------------------------
    # Footer html realization
    # -------------------------

    gr.HTML(
        """
        <div id="custom-footer">
            &copy; ООО "Нейри" 2025
            <a href="https://neiry-ai.ru" target= "_blank">neiry-ai.ru</a>
        </div>
        """,
        visible=True
    )

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
