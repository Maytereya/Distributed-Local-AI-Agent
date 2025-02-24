from Deprecated import deprecated_config as c

import time
from converters import json_converter as j
from ollama import AsyncClient


# async def route1(question: str):
#     """Routing Questions to range of collections or WebSearch"""
#
#     prompt = PromptTemplate(
#         template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a
#         user question to a vectorstore or web search. Use the vectorstore for questions on LLM agents,
#         prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords
#         in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search'
#         or 'vectorstore' based on the question. Return the JSON with a single key 'datasource' and
#         no preamble or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
#         input_variables=["question"],
#     )
#
#     question_router = prompt | c.llm | JsonOutputParser()
#     return await question_router.ainvoke({"question": question})


async def route(question: str):
    ollama_aclient = AsyncClient(host=c.ollama_url)
    # options make Ollama slow so far.
    # opt = Options(temperature=0, num_gpu=2, num_thread=24)

    prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a user '
              'question to a vectorstore or web search. Use the vectorstore for questions on LLM agents, '
              'prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords in the '
              'question related to these topics. Otherwise, use web-search. Give a binary choice "web_search" or '
              '"vectorstore" based on the question. Return the JSON with a single key "datasource" and no preamble or '
              'explanation. Example#1: {"datasource": "web_search"}, example#2: {"datasource": "vectorstore"}. '
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
    print(f"Время выполнения асинхронного запроса к клиенту: {elapsed_time:.2f} секунд")
    print('Время выполнения асинхронного запроса к клиенту: 3.56 секунд (LTE, MSK)')
    print(f"Eval_duration: {aresult['eval_duration'] / 1_000_000_000}")

    json_result = j.str_to_json(aresult['response'])
    print("Grade response: " + str(json_result))

    return json_result
