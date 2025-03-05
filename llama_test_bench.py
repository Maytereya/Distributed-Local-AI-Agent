import asyncio
from ollama import AsyncClient, Options
import re

ollama_async_client = AsyncClient(host="http://46.0.234.32:11434")
options = Options(
    temperature=0
    #     Здесь можно добавить другие опции Ollama / Llama 3.3 при желании
)

# Выбор LLM
llm = "llama3.3:70b-instruct-q8_0"


def get_cost_info(item):
    # print(f"Ищем в базе данных услугу {item}")
    return f"Ищем в базе данных услугу {item}"


# regex

def re_capture(model_response):
    # model_response = "[get_cost_info(item='Прием гинеколога')]"

    function_call_pattern = r"\[([a-zA-Z0-9_]+)\((.*)\)\]"
    match = re.match(function_call_pattern, model_response)
    if match:
        func_name = match.group(1)  # "get_cost_info"
        args_str = match.group(2)  # "item='Прием гинеколога'"

        # Теперь нужно вытащить значение аргумента item.
        # Допустим, аргумент один, имя - "item". Можно делать:
        item_match = re.match(r"item\s*=\s*'([^']+)'", args_str)
        if item_match:
            item_value = item_match.group(1)  # "Прием гинеколога"
            print("Parsed function name:", func_name)
            print("Parsed item value:", item_value)
            # => Вызов вашей функции:
            result = get_cost_info(item_value)
            print(result)
    else:
        # Обычный текст
        print(model_response)


async def formulate(question: str):
    """
    Пример демонстрации zero-shot function calling + обычный ответ.
    Если пользователь спрашивает про стоимость (cost),
    модель может сгенерировать [get_cost_info(item='...')].
    """

    # 1. Описываем в system-промпте некую виртуальную функцию "get_cost_info"
    #   Назначение: "получить информацию о цене"
    #   Параметры: item (string)
    # Модель может либо ответить напрямую, либо вызвать [get_cost_info(item="что-то")].

    system_message = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant, a call centre employee in a private hospital "Наука".
You speak Russian only. You also have the capability to call a function if the user is asking about cost/price.
Here is the function in JSON format:

[
  {{
    "name": "get_cost_info",
    "description": "Get cost info about a procedure or service",
    "parameters": {{
      "type": "dict",
      "required": ["item"],
      "properties": {{
        "item": {{
          "type": "string",
          "description": "item include name of the procedure or service"
        }}
      }}
    }}
  }}
]

If the user asks about cost, you respond ONLY with:
[get_cost_info(item='название_услуги специализация_врача фамилия_врача')]
Item must contains only the specific data: type of the procedure or service (for example: Прием, Консультация) AND 
doctor's specialization (for example: Уролог, Андролог, Гинеколог, Терапевт) AND Name of the Doctor (if it exists).
Example: [get_cost_info(item='консультация проктолог Ефремов')]
If no cost question, just answer in a normal way in Russian sentences.

<|eot_id|><|start_header_id|>user<|end_header_id|>
Вопрос: {question}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    aresult = await ollama_async_client.generate(
        model=llm,
        prompt=system_message,
        options=options,
        keep_alive=-1,
    )

    # Вывод времени ответа, если нужно
    if "eval_duration" in aresult:
        print(f"Eval_duration of answer generation: {aresult['eval_duration'] / 1_000_000_000}")

    return aresult.get('response', "")


async def main(question: str):
    response_text = await formulate(question)
    print(f"{llm} answer:\n\n{response_text}")
    re_capture(response_text)


if __name__ == "__main__":
    q = input("Ask your question: ")
    asyncio.run(main(q))
