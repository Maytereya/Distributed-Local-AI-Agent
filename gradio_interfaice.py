import os
import re
import shutil
import time
from typing import Dict
from typing import List

import gradio as gr
from gradio_pdf import PDF

from agent_logic_2 import config as c
from agent_logic_pack import aretrieve3 as retrieve
from agent_logic_pack import meilisearch_client as meilisearch
from converters import pdf_to_json_txt_tables_meili as pdf2json

# Label constants
COLLECTIONS_IN_CHROMA = "Коллекции документов Chroma DB"
INDEXES_IN_MEILI = "Индексы документов Meilisearch"
EXAMPLES = [["апатия, причины, лечение"], ["ангедония, причины, лечение"], ["акатизия, причины, лечение"],
            ["ЗНС, лечение"]]
# -------------------
# SECURITY
# -------------------

USERNAME = c.AUTH_NAME
PASSWORD = c.AUTH_PASS
# Пока не срабатывает.
os.environ["USER_AGENT"] = "NEIRY.Agent/1.0"


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
"""


# -------------------
# ECHOES PART
# -------------------

async def chroma_echo(message: str, history: List[Dict], collection: str, threshold_value: float,
                      slider_value_n_results: int,
                      slider_value_k,
                      radio_value):
    """
    Main chroma call func. Its return the pieces of text from uploaded to chroma docs.
    :param collection: Str. Chosen collection name.
    :param message: Str. The users question.
    :param history: Obligate parameter for correct gradio execute.
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


# Универсальная функция, которая проверяет radio_value и вызывает либо chroma_echo, либо meili_echo.
async def universal_echo(
        message: str,
        history: List[Dict],
        collection: str,  # Dropdown (Chroma)
        threshold_value: float,
        slider_value_n_results: int,
        slider_value_k: int,
        radio_value: str,  # "vectorstore", "db", or "meilisearch"
        meili_index: str  # Dropdown (Meilisearch)
):
    """
    Если radio_value == "meilisearch", вызываем meili_echo,
    иначе вызываем chroma_echo.
    """
    if radio_value == "meilisearch":
        # Используем slider_value_k как limit
        # Сюда добавить код про то, что нет индекса
        return await meili_echo(
            message=message,
            history=history,
            index=meili_index,
            limit=slider_value_k
        )
    else:
        # Для "vectorstore" или "db" вызываем chroma_echo
        return await chroma_echo(
            message=message,
            history=history,
            collection=collection,
            threshold_value=threshold_value,
            slider_value_n_results=slider_value_n_results,
            slider_value_k=slider_value_k,
            radio_value=radio_value
        )


# --------------------
# CHROMA DB section
# --------------------

def gr_create_collection(c_name: str):
    """
    Created collection and return its name in str.
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
    return retrieve.handle_collection(selected_collection)


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


def existed_docs_in_selected_index(selected_index: str):
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
        meilisearch.add_doc_to_meili(json_path, index)
        time.sleep(5)
        new_list = gr_existed_indexes()

        return (
            PDF(value=None, label="Загрузить PDF", interactive=True, scale=80),
            gr.update(value=None),  # сбрасываем JSON
            "Файл (PDF) добавлен в индекс",
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
        # (import shutil в начале файла)
        shutil.copyfile(json_file, local_json_path)

        # Индексируем в Meilisearch

        meilisearch.add_doc_to_meili(local_json_path, index)
        time.sleep(1)
        new_list = gr_existed_indexes()
        return (
            gr.update(value=None),  # сбрасываем PDF
            gr.update(value=None),  # сбрасываем JSON
            f"Файл (JSON) '{os.path.basename(json_file)}' добавлен в индекс '{index}'.",
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
    slider3,
    slider1,
    slider2,
    meili_search_indexes_dropdown,
    chroma_search_collection_dropdown,

    The search type selector that enables appropriated sliders.
    :param choice: Vectorstore search or Chroma DB Search.
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
    else:
        return (gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=True),

                gr.update(interactive=True),
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


# -----------------
# Chat Interface
# _________________

# with gr.Blocks() as blocks:
with gr.Blocks(css=custom_css) as blocks:
    gr.Markdown("## NEIRY.AI **bookworm**")

    chatbot = gr.Chatbot(type="messages",
                         autoscroll=True,
                         placeholder="<strong>Поиск по документам</strong><br>Задайте вопрос")

    textbox = gr.Textbox(lines=1,
                         placeholder="Напишите вопрос",
                         submit_btn=True,
                         container=True,
                         autoscroll=True,
                         autofocus=True)

    with gr.Column():
        with gr.Row():
            radio_type_of_search = gr.Radio(["vectorstore", "db", "meilisearch"],
                                            label="Способы поиска в базе знаний",
                                            value="db",
                                            container=True,
                                            info="Выберите алгоритм поиска")

            meili_search_indexes_dropdown = gr.Dropdown(choices=gr_existed_indexes(),
                                                        label=INDEXES_IN_MEILI,
                                                        info="Выберите Индекс для поиска информации",
                                                        interactive=False,
                                                        # value=None,
                                                        )

            chroma_search_collection_dropdown = gr.Dropdown(choices=gr_existed_collections(),
                                                            # filterable=True,
                                                            label=COLLECTIONS_IN_CHROMA,
                                                            info="Выберите Коллекцию для поиска информации",
                                                            interactive=True,
                                                            # value=None
                                                            )

    with gr.Row():
        slider3 = gr.Slider(value=5, minimum=1, maximum=20, step=1,
                            label="Количество документов, включенных в выдачу",
                            info="Только в режиме vectorstore",
                            interactive=True,
                            )

        slider1 = gr.Slider(value=0.005, minimum=0.0025, maximum=0.02, step=0.0025,
                            label="Порог косинусной фильтрации",
                            info="Только в режиме vectorstore."
                                 "Чем выше значение, тем больше текстовых фрагментов с меньшей "
                                 "релевантностью появится в выдаче",
                            interactive=True
                            )

        slider2 = gr.Slider(value=2, minimum=1, maximum=20, step=1,
                            label="Количество документов, включенных в выдачу",
                            info="Только в режимах db и meilisearch",
                            interactive=False,
                            )

    # ------------------------------------------
    # Upload PDF to MEILI or CHROMA DB section
    # ------------------------------------------

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
                                every=10.0,
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
    # Секция просмотра содержимого коллекций и индексов
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

    # ----------------------------
    # Секция интерфейса чата
    # ----------------------------

    with gr.Column():
        demo = gr.ChatInterface(fn=universal_echo, type="messages",
                                examples=EXAMPLES,
                                chatbot=chatbot,
                                textbox=textbox,
                                additional_inputs=[
                                    chroma_search_collection_dropdown,
                                    slider1,
                                    slider2,
                                    slider3,
                                    radio_type_of_search,
                                    meili_search_indexes_dropdown
                                ],
                                show_progress="full",

                                )

    # ----------------------------
    # Секция переключателей
    # ----------------------------

    radio_type_of_search.change(fn=radio_sliders_change, inputs=radio_type_of_search,
                                outputs=[
                                    slider3,
                                    slider1,
                                    slider2,
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
    # Кнопки работы с коллекциями ChromaDB
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
    # Кнопки работы с индексами Meilisearch
    # --------------------------------------

    # add_index_button.click(
    #     gr_create_collection,
    #     inputs=indices_dropdown,
    #     outputs=[data_upload_status, indices_dropdown, meili_search_indexes_dropdown]
    # )

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

if __name__ == "__main__":
    blocks.launch(server_name="0.0.0.0", server_port=7860, auth=check_auth, show_api=False)
