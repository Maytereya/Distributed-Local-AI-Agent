import gradio as gr
from gradio_pdf import PDF

from typing import Dict, Literal
from typing import List
from agent_logic_pack import aretrieve3 as retrieve
import config as c


async def echo(message: str, history: List[Dict], slider_value: float, slider_value_n_results: int, slider_value_k,
               radio_value):
    return await retrieve.main_retrieve_async(question=message, return_type="str", threshold=slider_value,
                                              n_results=slider_value_n_results,
                                              k=slider_value_k,
                                              search_type=radio_value)


def echo_create_collection(c_name: str) -> str | None:
    result = retrieve.create_collection(c_name)
    return result.name


# ?
def has_collections():
    chroma_service = retrieve.ChromaService(c.chroma_host, c.chroma_port)
    return chroma_service.display_collections(output_format="list")


testlist = [["name1", "value1"], ["name2", "value2"]]


def select_collection(collection):
    return f"Вы выбрали: {collection}"


def radio_change(choice):
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
            collection_dropdown = gr.Dropdown(choices=has_collections(), label="Коллекции в ChromaDB")
            output_text = gr.Textbox(label="Выбранная коллекция")

            # Поле для ввода имени новой коллекции
            new_collection_input = gr.Textbox(label="Имя новой коллекции")
            # Кнопка для добавления коллекции
            add_collection_button = gr.Button("Добавить коллекцию")
            # Поле вывода сообщения
            # message_output = gr.Textbox(label="Статус операции", interactive=False)

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

        pdf = PDF(label="Upload a PDF", interactive=True)
        name = gr.Textbox(placeholder="Адрес и имя PDF в оперативной памяти")

        pdf.upload(fn=lambda f: f, inputs=pdf, outputs=name)

    with gr.Column():
        demo = gr.ChatInterface(fn=echo, type="messages",
                                examples=[["апатия, причины, лечение"], ["ангедония, причины, лечение"],
                                          ["акатизия, причины, лечение"], ["ЗНС, лечение"]],
                                # title="MMR - поиск в тексте с косинусной фильтрацией",
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
        collection_dropdown.change(select_collection, inputs=collection_dropdown, outputs=output_text)

        add_collection_button.click(
            echo_create_collection,
            inputs=new_collection_input,
            outputs=[output_text, ]
        )

if __name__ == "__main__":
    blocks.launch()
