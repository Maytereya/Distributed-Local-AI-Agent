from converters import json_converter as j
import config as c
from ollama import Client


def route(question: str):
    ollama_client = Client(host=c.ollama_url)

    prompt = ('<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a user '
              'question to continue chatting with you or to use web search. '
              'Choose to continue chatting for any questions NOT about the weather. '
              'All questions about weather conditions choose "web_search" '
              'Give a binary choice "web_search" or '
              '"continue_chatting" based on the question. Return the JSON with a single key "datasource" and no '
              'preamble or'
              'explanation. Example#1: {"datasource": "web_search"}, example#2: {"datasource": "continue_chatting"}. '
              f'Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>')

    # async & .generate
    aresult = ollama_client.generate(
        model=c.ll_model,
        prompt=prompt,
        keep_alive=-1,

    )

    print(f"Eval_duration: {aresult['eval_duration'] / 1_000_000_000}")

    json_result = j.str_to_json(aresult['response'])  # Carefully check the format
    print("Router response: " + str(json_result))
    return json_result
