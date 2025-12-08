import asyncio
from typing import Literal

from ollama import AsyncClient, Options

from agent_logic_2 import config as c

ollama_aclient = AsyncClient(host=c.ollama_url)
options = Options(temperature=0.9, )

# Выбор llm
llm = "ministral-3:14b-instruct-2512-fp16"

# qwen3-vl:8b-thinking-bf16
# gpt-oss:20b
# ministral-3:14b-instruct-2512-fp16

async def formulate(sentence: str, ):
    """
    Formulate a question of the user
    :param question: Сырой запрос пользователя
    :return: Обработанный запрос пользователя для облегчения поиска в векторной базе и фомулирования правильного запроса
    """

    prompt = (f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
              К тебе поступает запрос от пользователя на русском языке, составленный в свободном порядке и не всегда верный
              грамматически. 
              Твоя задача: понять о чем речь в вопросе и ответить творчески, не ограничивая себя в словах (токенах).
              Твой ответ только на РУССКОМ ЯЗЫКЕ.
              USER: {sentence}. \n\n
              <|eot_id|><|start_header_id|>assistant<|end_header_id|>""")

    aresult = await ollama_aclient.generate(
        model=llm,
        prompt=prompt,
        # format="json",
        options=options,
        keep_alive=-1,

    )

    print(f"Eval_duration of answer generation: {aresult['eval_duration'] / 1_000_000_000}")
    #
    # print("Формулировка: ")
    # print(aresult['response'])
    return aresult['response']




async def main(question: str, ):
    query = await formulate(question)
    print(query)


if __name__ == "__main__":
    # q = input("Question: ")
    hardcoded_question = """
    Расскажи о себе. 
    """
    asyncio.run(main(hardcoded_question))
