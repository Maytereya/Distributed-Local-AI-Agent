import gradio as gr
from gradio_pdf import PDF

from typing import Dict, Literal
from typing import List
from agent_logic_pack import aretrieve3 as retrieve
import config as c


async def echo(message: str, history: List[Dict], slider_value: float, slider_value_n_results: int, slider_value_k,
               radio_value):
    """
    Main func. Its return the pieces of text from uploaded to chroma docs.
    :param message: The users question.
    :param history: Obligate parameter for correct gradio execute.
    :param slider_value:
    :param slider_value_n_results:
    :param slider_value_k:
    :param radio_value:
    :return: String of filtrated text from ChromaDB.
    """
    return await retrieve.main_retrieve_async(question=message, return_type="str", threshold=slider_value,
                                              n_results=slider_value_n_results,
                                              k=slider_value_k,
                                              search_type=radio_value)


def echo_create_collection(c_name: str) -> tuple[str, gr.Dropdown]:
    """
    Created collection and return its name in str.
    :param c_name: String, passed new name of the collection.
    :return: String, name of the collection.
    """
    result = retrieve.create_collection(c_name)
    return result.name, gr.Dropdown(choices=existed_collections(), value=c_name,
                                    label="Коллекции документов в ChromaDB")


def echo_remove_collection(c_name: str):
    retrieve.remove_collection(c_name)
    return gr.Dropdown(choices=existed_collections(),
                       label="Коллекции документов в ChromaDB"), gr.Textbox(label="Имя коллекции выбрать/добавить",
                                                                            value="Коллекция удалена",
                                                                            interactive=True, )


def existed_collections():
    """

    :return: List of existed collection names.
    """
    chroma_service = retrieve.ChromaService(c.chroma_host, c.chroma_port)
    return chroma_service.display_collections(output_format="list")


# ?
def select_collection(collection):
    return f"Выбрано: {collection}"


def echo_add_to_collection(collection: str, file: str):
    retrieve.add_data(exist_collection_name=collection, upload_type="PDF", add_path=file, model="default")
    return gr.Textbox(label="Имя коллекции выбрать/добавить", interactive=True, value="Файл добавлен в коллекцию")


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
            radio = gr.Radio(["vectorstore", "db", ],
                             label="Способ первичного поиска", value="vectorstore", container=True)

    with gr.Row():
        slider3 = gr.Slider(value=5, minimum=1, maximum=20, step=1,
                            label="Количество чанков текста, включенных в выдачу, vectorstore - поиск",
                            info="Только в режиме поиска vectorstore",
                            interactive=True,
                            )

        slider1 = gr.Slider(value=0.005, minimum=0.0025, maximum=0.02, step=0.0025,
                            label="Пороговое значение косинусной фильтрации",
                            info="Выбрать в диапазоне между 0.0025 и 0.02. "
                                 "Чем больше значение, тем больше чанков с меньшей "
                                 "релевантностью будет в выдаче",
                            interactive=True
                            )

        slider2 = gr.Slider(value=2, minimum=1, maximum=20, step=1,
                            label="Количество документов, включенных в выдачу, db - поиск",
                            info="Только в режиме поиска db",
                            interactive=False,
                            )

        # Upload PDF section
    gr.Markdown("### Загрузка и распределение документов по коллекциям")
    with gr.Row():
        collection_dropdown = gr.Dropdown(choices=existed_collections(), value=None,
                                          label="Коллекции документов в ChromaDB")

        # Поле для ввода имени новой коллекции

        collection_input_txt = gr.Textbox(label="Имя коллекции выбрать/добавить", interactive=True, )

        # Кнопки для работы с коллекциями
        with gr.Column():
            add_collection_button = gr.Button("Добавить коллекцию", )
            rm_collection_button = gr.Button("Удалить коллекцию", )

    with gr.Row():
        pdf = PDF(label="Загрузить PDF", interactive=True, scale=80)

        with gr.Column():
            name = gr.Textbox(placeholder="Имя загруженного PDF в оперативной памяти")
            add_to_collection_button = gr.Button("Добавить в коллекцию")

    with gr.Column():
        demo = gr.ChatInterface(fn=echo, type="messages",
                                examples=[["апатия, причины, лечение"], ["ангедония, причины, лечение"],
                                          ["акатизия, причины, лечение"], ["ЗНС, лечение"]],

                                chatbot=chatbot,
                                textbox=textbox,
                                additional_inputs=[slider1,
                                                   slider2,
                                                   slider3,
                                                   radio,

                                                   ],
                                show_progress="full",

                                )

    radio.change(fn=radio_change, inputs=radio, outputs=[slider3, slider1, slider2])
    collection_dropdown.change(select_collection, inputs=collection_dropdown, outputs=[collection_input_txt])

    pdf.upload(fn=type(pdf), inputs=pdf, outputs=name)
    pdf.upload(fn=echo_add_to_collection, inputs=[collection_dropdown, pdf], outputs=collection_input_txt)

    add_collection_button.click(
        echo_create_collection,
        inputs=collection_input_txt,
        outputs=[collection_input_txt, collection_dropdown]
    )
    rm_collection_button.click(
        echo_remove_collection,
        inputs=collection_dropdown,
        outputs=[collection_dropdown, collection_input_txt]
    )
    add_to_collection_button.click(
        echo_add_to_collection,
        inputs=[collection_dropdown, pdf],
        outputs=[collection_input_txt]
    )
# upload_docs = gr.Interface(fn=..., inputs=[])

if __name__ == "__main__":
    blocks.launch()
    # upload_docs.launch()
