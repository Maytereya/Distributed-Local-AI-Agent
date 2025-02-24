from converters import json_converter as j
import config as c
import time
from ollama import AsyncClient
from datetime import datetime


async def route(question: str):
    ollama_aclient = AsyncClient(host=c.ollama_url)
    # options make Ollama slow so far.
    # opt = Options(temperature=0, num_gpu=2, num_thread=24)
    # Получение текущей даты и времени
    current_datetime = datetime.now()

    # Форматирование даты и времени
    # Пример: 12 сентября 2024, 14:30:25
    formatted_datetime = current_datetime.strftime("%d %B %Y, %H:%M:%S")

    # Вывод
    prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> '
              f'Today date and time: {formatted_datetime} '
              f'You are an expert at routing a user '
              'question to a vectorstore, chat with you or web search. '
              'Use the vectorstore for questions on medicine, medical service, side effects in psychiatry, '
              'psychopharmacology and psychiatric treatment. '
              'If you are asked questions about today '
              'events and what will happen in the next 3 days, select web-search. '
              'If the questions are about previous questions select chat.'
              'You do not need to be stringent with the keywords in the '
              'question related to these topics. '
              'Give a choice "web_search", '
              '"vectorstore" or "chat" based on the question. '
              'Return the JSON with a single key "datasource" and no preamble or '
              'explanation. Example#1: {"datasource": "web_search"}, example#2: {"datasource": "vectorstore"}, '
              'example#3: {"datasource": "chat"}.'
              'Attention: if you want to make a choice "web_search", think twice! Maybe its a mistake.'
              f'Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>')



    # async & .generate
    start_time = time.time()
    aresult = await ollama_aclient.generate(
        model=c.ll_model,
        prompt=prompt,
        # format="json",
        # options=opt,
        keep_alive=-1,

    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    # Без await не работают!
    print(f"Async request timing client-server is: {elapsed_time:.2f} sec")
    # print('Предыдущий результат: 3.56 секунд (LTE, MSK)')
    print(f"Eval_duration: {aresult['eval_duration'] / 1_000_000_000}")

    json_result = j.str_to_json(aresult['response'])  # Carefully check the format
    print("Router response: " + str(json_result))
    return json_result
