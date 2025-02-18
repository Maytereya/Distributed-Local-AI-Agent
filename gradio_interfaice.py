import gradio as gr
from gradio_pdf import PDF

from typing import Dict, Literal
from typing import List
from agent_logic_pack import aretrieve3 as retrieve
import config as c

# Label constants
COLLECTIONS_IN_CHROMA = "Коллекции документов Chroma DB"


async def echo(message: str, history: List[Dict], collection: str, threshold_value: float, slider_value_n_results: int,
               slider_value_k,
               radio_value):
    """
    Main func. Its return the pieces of text from uploaded to chroma docs.
    :param collection: Str. Chosen collection name.
    :param message: Str. The users question.
    :param history: Obligate parameter for correct gradio execute.
    :param threshold_value:
    :param slider_value_n_results:
    :param slider_value_k:
    :param radio_value: Type of search established.
    :return: String of filtrated text from ChromaDB.
    """
    print(f"Echo {collection=}")
    return await retrieve.main_retrieve_async(question=message, collection=collection, return_type="str",
                                              threshold=threshold_value,
                                              n_results=slider_value_n_results,
                                              k=slider_value_k,
                                              search_type=radio_value)


def echo_create_collection(c_name: str) -> tuple[str, gr.Dropdown, gr.Dropdown,]:
    """
    Created collection and return its name in str.
    Warning: in the next releases Chroma .name parameter will be removed!
    :param c_name: String, passed new name of the collection.
    :return: String, name of the collection.
    """
    result = retrieve.create_collection(c_name)
    return f"Коллекция {result.name} создана", gr.Dropdown(choices=existed_collections(), value=c_name,
                                                           label=COLLECTIONS_IN_CHROMA), gr.Dropdown(
        choices=existed_collections(), value=c_name,
        label=COLLECTIONS_IN_CHROMA)


def echo_remove_collection(c_name: str):
    retrieve.remove_collection(c_name)
    return (gr.Dropdown(choices=existed_collections(),
                        label=COLLECTIONS_IN_CHROMA, value=None),
            gr.Textbox(label="Информация о коллекции",
                       value="Коллекция удалена",
                       interactive=True, ), gr.Dropdown(
        choices=existed_collections(), value=c_name,
        label=COLLECTIONS_IN_CHROMA),
            gr.Dropdown(
                choices=existed_collections(),
                value=c_name,
                label=COLLECTIONS_IN_CHROMA))


def existed_collections():
    """

    :return: List of existed collection names.
    """
    chroma_service = retrieve.ChromaService(c.chroma_host, c.chroma_port)
    return chroma_service.display_collections(output_format="list")


# ?
# def select_collection(collection):
#     return f"Выбрана коллекция {collection}"


# ?
def txt_default():
    return f"Ожидание действий..."


def echo_add_to_collection(collection: str, file: str):
    retrieve.add_data(exist_collection_name=collection, upload_type="PDF", add_path=file, model="default")
    return PDF(
        value=None, label="Загрузить PDF", interactive=True, scale=80), gr.Textbox(label="Информация о коллекции",
                                                                                   interactive=False,
                                                                                   value="Файл добавлен в коллекцию"),


def radio_change(choice) -> tuple[gr.Slider, gr.Slider, gr.Slider]:
    """
    The search type selector that enables appropriated sliders.
    :param choice: Vectorstore search or Chroma DB Search.
    :return: Configurations of sliders that refine the search.
    """
    if choice == "vectorstore":
        return gr.Slider(interactive=True), gr.Slider(interactive=True), gr.Slider(interactive=False),
    else:
        return gr.Slider(interactive=False), gr.Slider(interactive=False), gr.Slider(interactive=True),


with gr.Blocks() as blocks:
    gr.Markdown("## NEIRY.AI **bookworm**")

    chatbot = gr.Chatbot(type="messages", autoscroll=True,
                         placeholder="<strong>Поиск по документам</strong><br>Задайте вопрос")

    textbox = gr.Textbox(lines=1, placeholder="напишите вопрос", submit_btn=True, container=True, autoscroll=True,
                         autofocus=True)

    with gr.Column():
        with gr.Row():
            radio = gr.Radio(["vectorstore", "db", "meilisearch"],
                             label="Способ первичного поиска", value="db", container=True,
                             info="Выберите доступный способ поиска")
            meili_indexes = gr.Dropdown(label="Индекс Meilisearch", info="Выберите Индекс для поиска информации")
            collection_to_search_in = gr.Dropdown(choices=existed_collections(),
                                                  # filterable=True,
                                                  label=COLLECTIONS_IN_CHROMA,
                                                  info="Выберите коллекцию для поиска информации")

    with gr.Row():
        slider3 = gr.Slider(value=5, minimum=1, maximum=20, step=1,
                            label="Количество фрагментов текста, включенных в выдачу, vectorstore - поиск",
                            info="Доступно в режиме vectorstore",
                            interactive=True,
                            )

        slider1 = gr.Slider(value=0.005, minimum=0.0025, maximum=0.02, step=0.0025,
                            label="Пороговое значение косинусной фильтрации",
                            info="Выбрать в диапазоне между 0.0025 и 0.02. "
                                 "Чем выше значение, тем больше текстовых фрагментов с меньшей "
                                 "релевантностью появится в выдаче",
                            interactive=True
                            )

        slider2 = gr.Slider(value=2, minimum=1, maximum=20, step=1,
                            label="Количество документов, включенных в выдачу, db - поиск",
                            info="Доступно в режиме db",
                            interactive=False,
                            )

        # Upload PDF section
    gr.Markdown("### Загрузка и распределение документов по коллекциям")
    with gr.Row():
        collection_dropdown = gr.Dropdown(choices=existed_collections(), value=None, allow_custom_value=True,
                                          filterable=True,
                                          label=COLLECTIONS_IN_CHROMA,
                                          info="Коллекции - папки с документами, классифицированными по темам")

        # Поле для вывода текущего статуса работы с коллекциями
        collection_input_txt = gr.Textbox(value=txt_default, every=10.0, label="Информация о статусе коллекции",
                                          interactive=False, )

        # Кнопки для работы с коллекциями
        with gr.Column():
            add_collection_button = gr.Button("Добавить коллекцию", )
            rm_collection_button = gr.Button("Удалить коллекцию", )

    with gr.Row():
        pdf = PDF(label="Загрузить PDF", interactive=True, scale=80)

        with gr.Column():
            # name = gr.Textbox(placeholder="Имя загруженного PDF в оперативной памяти")
            add_to_collection_button = gr.Button("Добавить в коллекцию")

    with gr.Column():
        demo = gr.ChatInterface(fn=echo, type="messages",
                                examples=[["апатия, причины, лечение"], ["ангедония, причины, лечение"],
                                          ["акатизия, причины, лечение"], ["ЗНС, лечение"]],

                                chatbot=chatbot,
                                textbox=textbox,
                                additional_inputs=[
                                    collection_to_search_in,
                                    slider1,
                                    slider2,
                                    slider3,
                                    radio,
                                ],
                                show_progress="full",

                                )

    radio.change(fn=radio_change, inputs=radio, outputs=[slider3, slider1, slider2])
    # ?
    # collection_to_search_in.change(select_collection, inputs=collection_to_search_in, outputs=[collection_to_search_in])

    add_collection_button.click(
        echo_create_collection,
        inputs=collection_dropdown,
        outputs=[collection_input_txt, collection_dropdown, collection_to_search_in]
    )
    rm_collection_button.click(
        echo_remove_collection,
        inputs=collection_dropdown,
        outputs=[collection_dropdown, collection_input_txt, collection_to_search_in]
    )
    add_to_collection_button.click(
        echo_add_to_collection,
        inputs=[collection_dropdown, pdf],
        outputs=[pdf, collection_input_txt, ]
    )

if __name__ == "__main__":
    blocks.launch()
    # upload_docs.launch()
