from converters import json_converter as j
import config as c
import time
from ollama import AsyncClient
from datetime import datetime

# Выбираем модель, которая будет использоваться {быстрая ll_model или медленная, но точная ll_model_big}
llm = c.ll_model

async def route(question: str):
    ollama_aclient = AsyncClient(host=c.ollama_url)
    # options make Ollama slow so far.
    # opt = Options(temperature=0, num_gpu=2, num_thread=24)
    # Получение текущей даты и времени
    current_datetime = datetime.now()


    formatted_datetime = current_datetime.strftime("%d %B %Y, %H:%M:%S") # Пример: 12 сентября 2024, 14:30:25

    prompt = (f'<|begin_of_text|><|start_header_id|>system<|end_header_id|> '
              f'Today\'s date and time: {formatted_datetime}. '
              'You are an expert at determining the appropriate method for handling a user question by selecting one of three options: querying a vectorstore, continuing a chat, or performing a web search. '
              'Use the vectorstore for questions related to medicine, medical services, psychiatry, psychopharmacology, side effects, and psychiatric treatment. '
              'If a question pertains to current events or events occurring within the next three days, use web search. '
              'If a question follows up on a previous conversation or is about prior context, choose chat. '
              'If the user enters words like "stop", "exit", "e", or similar in either Russian or English (such as "стоп", "выход"), return {"datasource": "exit"}. '
              'Do not focus strictly on keywords when identifying these topics—use your understanding of the question’s intent. '
              'Your response must be a JSON object with a single key "datasource", and it should contain no additional text, explanations, or preambles. '
              'Example: {"datasource": "web_search"} or {"datasource": "vectorstore"} or {"datasource": "chat"} or {"datasource": "exit"}. '
              'Note: carefully consider whether a web search is necessary before selecting it. '
              f'Question to route: {question}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    # async & .generate
    start_time = time.time()
    aresult = await ollama_aclient.generate(
        model=llm,
        prompt=prompt,
        # format="json",
        # options=opt,
        # keep_alive=-1,

    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Async request timing client-server is: {elapsed_time:.2f} sec")
    # print('Предыдущий результат: 3.56 секунд (LTE, MSK)')
    print(f"Eval_duration: {aresult['eval_duration'] / 1_000_000_000}")

    json_result = j.str_to_json(aresult['response'])  # Carefully check the format
    print("Router response / Ответ роутера: " + str(json_result))
    return json_result
