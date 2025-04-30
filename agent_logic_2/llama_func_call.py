import asyncio
import re
from pprint import pprint

import urllib3
from ollama import AsyncClient, Options

import config as c
from nayka_api import api_nayka4 as api_call
from nayka_api import nayka_doc_registry2 as doc_reg

ollama_async_client = AsyncClient(c.ollama_url)
options = Options(
    temperature=0
    #     Здесь можно добавить другие опции Ollama / Llama 3.3 при желании
)

# Выбор LLM
llm = "llama3.3:70b-instruct-q8_0"


async def formulate(llm_output: str) -> str:
    """
    Сформировать человеко‑читабельный ответ (Markdown) по данным врача.
    """
    system_message_for_formulation = f"""
    [SYSTEM PROMPT]
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Ты — ассистент частной многопрофильной клиники «Наука», помогающий пациентам
    найти подходящего врача. Твоя задача — обрабатывать данные из БД клиники
    «Наука» и красиво формулировать информацию для пользователя в Markdown‑формате.

    Нужно вывести:
      • **fio** — Фамилия Имя Отчество врача полностью  
      • **regions** — адрес(а)/места приёма (если есть выезд на дом — указать)  
      • **schedule** — расписание (start, end) + «окна» (slots) на неделю,
        сгруппировав по адресам и датам  
      • **specialization** — специализация и стаж (если дан)  

    Если по какому‑то пункту данных нет — так и напиши («Расписание не указано» и т.д.).
    **Ничего не выдумывай!** Если врач не найден — просто верни это сообщение.

    [FEW‑SHOT EXAMPLES]
    Пример 1:
    INPUT:
    {{"fio": "Абрамкина Вера Александровна",
      "specialization": "Врач ультразвуковой диагностики",
      "addresses": [
        "Короткий адрес: Ленина 5 | полный адрес: г. Самара, пр. Ленина, 5"
      ]
    }}
    OUTPUT:
    Абрамкина Вера Александровна.  
    Врач – ультразвуковой диагностики.  
    Адрес приёма: г. Самара, пр. Ленина, 5.  
    Расписание не указано.

    Пример 2:
    INPUT:
    {{"fio": "Адельшина Лилия Рафаильевна",
      "specialization": "Врач терапевт",
      "addresses": [
        "Короткий адрес: Чкалова 51/1 пом 5 | полный адрес: не указан",
        "Короткий адрес: Чкалова 51/1 пом 6 | полный адрес: не указан",
        "Короткий адрес: ул. Салмышская 43/1 | полный адрес: не указан"
      ]
    }}
    OUTPUT:
    Адельшина Лилия Рафаильевна.  
    Врач – терапевт.  
    Адреса приёма:  
    - Чкалова 51/1 пом 5 (полный адрес не указан)  
    - Чкалова 51/1 пом 6 (полный адрес не указан)  
    - ул. Салмышская 43/1 (полный адрес не указан)  
    Расписание не указано.

    ----

    Вот информация из базы данных о враче:
    {llm_output}

    ----

    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    aresult = await ollama_async_client.generate(
        model=llm,
        prompt=system_message_for_formulation,
        options=options,
        keep_alive=-1,
    )

    # Опционально выводим время генерации
    if "eval_duration" in aresult:
        print(f"Eval_duration of answer generation: "
              f"{aresult['eval_duration'] / 1_000_000_000:.3f} сек")

    return aresult.get("response", "")


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
        return None
    else:
        # Обычный текст
        print(model_response)
        return model_response


client = doc_reg.NaykaClient(c.nayka_base_url,
                             auth=(c.nayka_login, c.nayka_pass))

urllib3.disable_warnings()


async def investigate(question: str):
    """
    Zero‑shot function calling + few‑shot + скрытый chain‑of‑thought.
    Модель может вызвать [get_doc_info(item='…')] или дать прямой текстовый ответ.
    """
    # Базовый system‑prompt
    system_base = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Ты — ассистент клиники «Наука», у тебя есть справочник врачей и функция get_doc_info(item: string).
Чтобы получить расписание и услуги врача, используй [get_doc_info(item='…')].
Если врач не найден — сообщай об этом текстом.

Важно:
1. При поиске врача учитывай возможные отклонения от именительного падежа и используй normalization_hint
2. Если в вопросе указана только фамилия, ищи врача по фамилии в списке
3. Если врач не найден — сообщай об этом текстом
4. Не придумывай врачей, которых нет в списке
"""

    # блок нормализации падежей
    normalization_hint = """
normalization_hint:
Если в вопросе ФИО врача стоит в косвенном падеже 
(«Адельшиной», «Иванову») или с опечаткой,
сначала переведи его в именительный (Адельшина, Иванов),
и только потом вызывай [get_doc_info(item='…')] или отвечай текстом.
Не придумывай никаких новых форм!
"""

    # Few‑shot примеры
    few_shot = """
Пример 1:
INPUT:
Вопрос: Где принимает Абрамкина Вера Александровна?
OUTPUT:
[get_doc_info(item='Абрамкина Вера Александровна')]

Пример 2:
INPUT:
Вопрос: Как называется ваша клиника?
OUTPUT:
Частная многопрофильная клиника "Наука".

Пример 3:
INPUT:
Вопрос: Расписание доктора Тен
OUTPUT:
Доктор Тен не найден в базе клиники "Наука".

Пример 4:
INPUT: 
Вопрос: специальность Щетининой
OUTPUT: 
[get_doc_info(item='Щетинина')].

"""

    # Скрытые рассуждения (chain‑of‑thought)
    cot = """
<reasoning>
1. Выделяю имя врача из вопроса.
2. Проверяю, нужно ли применять normalization_hint.
3. Сравниваю с registry_of_doctors.
4. Если совпадение есть — вызываю get_doc_info.
5. Иначе — формирую ответ «не найден» или обычный текстовый ответ.
</reasoning>
"""

    # Собираем итоговый prompt
    system_message = (
            system_base
            + normalization_hint
            + few_shot
            + cot
            + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
              f"Вопрос: {question}\n"
              "<|start_header_id|>assistant<|end_header_id|>"
    )

    # Генерируем ответ
    aresult = await ollama_async_client.generate(
        model=llm,
        prompt=system_message,
        options=options,
        keep_alive=-1,
    )

    # лог времени
    if "eval_duration" in aresult:
        print(f"Eval_duration: {aresult['eval_duration'] / 1e9:.3f}s")

    return aresult.get("response", "")


async def main(question: str):
    """Запрос → LLM → (опционально) API → форматирование."""

    response_text = await investigate(question)
    captured: str = re_capture(response_text)

    # если LLM вернула простой текст — печатаем и завершаем
    if isinstance(captured, str):
        print(" ======================= ")
        print(f"Ответ модели:\n\n{captured}")
        return

    # иначе captured — это dict/JSON с данными врача
    final_output = await formulate(captured)
    print(" ======================= ")
    print(f"Итоговый ответ модели:\n\n{final_output}")


if __name__ == "__main__":
    q = input("Ask your question: ")
    asyncio.run(main(q))
