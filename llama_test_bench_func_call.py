import asyncio
import re
from pprint import pprint

import urllib3
from ollama import AsyncClient, Options

import config as c
from nayka_api import api_nayka4 as api_call
from nayka_api import nayka_doc_registry2 as doc_reg

ollama_async_client = AsyncClient(host="http://46.0.234.32:11434")
options = Options(
    temperature=0
    #     Здесь можно добавить другие опции Ollama / Llama 3.3 при желании
)

# Выбор LLM
llm = "llama3.3:70b-instruct-q8_0"


async def formulate(llm_output: str):
    system_message_for_formulation = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
    Ты — ассистент частной многопрофильной клиники "Наука", помогающий пациентам найти подходящего врача. 
    Твоя задача - обрабатывать данные, которые поступают тебе из базы данных клиники "Наука".

    На основании поступивших данных ты должен четко сформулировать нижеследующее.
    
    - fio: Фамилия Имя и Отчество врача полностью,
    - regions: Адрес или адреса, где он ведет прием (если врач выезжает на дом, то тебе следует это указать прямо).
    - schedule: Расписание врача в виде времени начала работы (start) и окончания работы (end), а так же
     свободных окон на ближайшую неделю (slots), сгруппированное по адресам приема и датам (date).
     - specialization: На чем специализируется врач и каков его стаж работы (если такая информация представлена), 
     используй все данные, ничего не опускай.
     
    Задача обработки - сформулировать красиво, лаконично и понятно для пользователя в формате MarkDown.
    
    Если по каким-то ключам данных нет, просто напиши об этом. Но обязательно выведи те данные, которые поступили. 
    
    Примеры:
    
    Вариант 1, неполные данные:
    Белокорытцева Наталья Леонидовна.
    Врач ультразвуковой диагностики.
    Свободных окон на ближайшую неделю нет.
    Информации о специализации нет.
    
    Вариант 2, полные данные:
    Белокорытцева Наталья Леонидовна.
    Врач ультразвуковой диагностики.
    Свободные окна: 22.04.2025 в 14:00, в 15:00 и в 17:30
    Специализация: Ультразвуковая диагностика органов малого таза, печени, желчного пузыря, далее перечисляешь все, что тебе поступило.
    
    \n\n----\n\n
    Вот информация из базы данных о враче: {llm_output}
    \n\n----\n\n
    
    Внимание! Не выдумывай никаких данных, если их нет!
    Если информации о враче не поступило или тебе пришло сообщение, что врач не найден, просто передай это сообщение как есть.
    
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    aresult = await ollama_async_client.generate(
        model=llm,
        prompt=system_message_for_formulation,
        options=options,
        keep_alive=-1,
    )

    # Вывод времени ответа, если нужно
    if "eval_duration" in aresult:
        print(f"Eval_duration of answer generation: {aresult['eval_duration'] / 1_000_000_000}")

    return aresult.get('response', "")


def get_doc_info(doc_name):
    return api_call.find_doctor_schedule(doc_name)


# regex

def re_capture(model_response):
    function_call_pattern = r"\[([a-zA-Z0-9_]+)\((.*)\)\]"
    match = re.match(function_call_pattern, model_response)
    if match:
        func_name = match.group(1)  # "get_cost_info"
        args_str = match.group(2)  # "item='Прием гинеколога'"

        # Теперь нужно вытащить значение аргумента item.
        # Допустим, аргумент один, имя - "item". Можно делать:
        item_match = re.match(r"item\s*=\s*'([^']+)'", args_str)
        if item_match:
            item_value = item_match.group(1)  # "ФИО"
            print("Parsed function name:", func_name)
            print("Parsed item value:", item_value)

            # => Вызов функции:
            result = get_doc_info(item_value)
            pprint(result, width=160)
            return result
    else:
        # Обычный текст
        print(model_response)
        return model_response



client = doc_reg.NaykaClient(c.nayka_base_url,
                             auth=(c.nayka_login, c.nayka_pass))

urllib3.disable_warnings()

# Читаем весь файл в одну строку
with open("nayka_api/apidata/doctors.jsonl", "r", encoding="utf-8") as f:
    registry_of_doctors = f.read()


async def investigate(question: str):
    """
    Пример демонстрации zero-shot function calling + обычный ответ.
    Если пользователь спрашивает про врача,
    модель может сгенерировать [get_doc_info(item='...')].
    """

    # 1. Описываем в system-промпте некую виртуальную функцию "get_doc_info"
    #   Назначение: "получить информацию о враче"
    #   Параметры: item (string)
    # Модель может либо ответить напрямую, либо вызвать [get_doc_info(item="имя врача")].

    system_message = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>


Ты — ассистент частной многопрофильной клиники "Наука", помогающий пациентам найти подходящего врача. 
У тебя есть справочник с информацией о врачах, включающий:
- ФИО, 
- специализацию, 
- адреса одного или нескольких офисов клиники "Наука", в которых доктора ведут прием.
Так же о враче может быть указано, что он выезжает к пациенту на дом.
Пользователь может указать врача по фамилии, имени, направлению или по типу проблемы с которой он (пользователь) 
обратился как пациент. 

Вот список врачей, зарегистрированных в системе: 

\n\n-- Начало списка врачей --\n\n
{registry_of_doctors}
\n\n-- Окончание списка врачей --\n\n

Если ты опознал врача по вопросу пользователя, ты должен получить дополнительную информацию о его
графике работы, свободных слотах в расписании, медицинских услугах, которые он оказывает. 
Для этого нужно вызвать соответствующую функцию.

Вот эта функция в формате JSON:

[
  {{
    "name": "get_doc_info",
    "description": "Get full info about a doctor's timetable and available services provided by himself",
    "parameters": {{
      "type": "dict",
      "required": ["item"],
      "properties": {{
        "item": {{
          "type": "string",
          "description": "item includes name of the doctor"
        }}
      }}
    }}
  }}
]

Если пользователь спрашивает про врача, специализацию или услугу, 
которая может быть связана с конкретным врачом из списка врачей, предоставленного выше, ты должен ответить function call:
[get_doc_info(item='фамилия_имя_отчество_врача')]
Значение item должно содержать точные данные: 
фамилия и имя врача соответствующие таковым в списке врачей, предоставленном выше.

Ни в коем случае не придумывай имена врачей при генерации item. НО! ВАЖНО! Фамилию или имя могут сообщить тебе с ошибкой или опиской.
В этом случае попытайся найти наиболее похожую фамилию из списка выше.
Учитывай такую жизненную ситуацию и попробуй понять, кого имел в виду пользователь. Однако, не увлекайся подбором слишком сильно. 
Если спросили Белокорытову, то, скорее всего, НЕ имели в виду Белохвостикову. Тогда сообщи, что такого врача в клинике не существует.

Если вопрос о медицинской специализации или услуге, например, врач - хирург, врач УЗИ, кардиолог, терапевт - найди врача
запрашиваемой специальности в списке врачей выше и передай его в function call.

Используй только фактические фамилию, имя и отчество, которые ты видишь в списке у каждого конкретного врача.
Пример: [get_doc_info(item='Ефремов Николай Васильевич')]

Если нет упоминания о враче или медицинской услуге, отвечай обычным образом.

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
    response_text = await investigate(question)
    # print(f"{llm} answer:\n\n{response_text}")
    captured = re_capture(response_text)
    final_output = await formulate(captured)
    print(" ======================= ")
    print(f"Итоговый ответ модели: \n\n {final_output}")


if __name__ == "__main__":
    q = input("Ask your question: ")
    asyncio.run(main(q))
