import asyncio
import json
import logging
import os
import re
import sys

import urllib3
from ollama import AsyncClient, Options

from agent_logic_2 import config as c
from agent_logic_2.nayka_api import api_nayka4_1 as api_call
from agent_logic_2.nayka_api import nayka_doc_registry2 as doc_reg

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Добавляем корневую директорию проекта в PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

ollama_async_client = AsyncClient(c.ollama_url)
options = Options(
    temperature=0
    # Здесь можно добавить другие опции Ollama / Llama 3.3 при желании
)

# Выбор LLM
# llm = "llama3.3:latest"
llm = "llama4:scout"

# Параметры генерации
generation_options = Options(
    temperature=0.3,
    top_k=40,
    top_p=0.9,
    mirostat=1,
    mirostat_tau=5.0,
    mirostat_eta=0.1
)


async def ollama_call(prompt: str, *, max_tokens: int = 512, tries: int = 2):
    """Универсальная обертка для вызовов Ollama с повторными попытками."""
    for n in range(tries):
        try:
            return await ollama_async_client.generate(
                model=llm,
                prompt=prompt,
                options=generation_options,
                keep_alive=-1,
            )
        except TimeoutError:
            logger.warning(f"Попытка {n + 1}/{tries} истекла")
        except Exception as e:
            logger.error(f"Ошибка при вызове Ollama: {e}")
            if n == tries - 1:
                raise
    raise TimeoutError("Ollama не ответила после всех попыток")


async def formulate(llm_output: str) -> str:
    """
    Сформировать человеко‑читабельный ответ (Markdown) по данным врача.
    """
    # Если это вызов функции - возвращаем как есть
    if llm_output.startswith("[") and llm_output.endswith("]"):
        return llm_output

    system_message_for_formulation = f"""
    [SYSTEM PROMPT]
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Ты — ассистент частной многопрофильной клиники «Наука», 
    помогающий пациентам найти подходящего врача. Твоя задача — 
    обрабатывать данные из БД клиники «Наука» и красиво 
    формулировать информацию для пользователя в Markdown‑формате.

    Нужно вывести:
      • **fio** — Фамилия Имя Отчество врача полностью  
      • **regions** — адрес(а)/места приёма (если есть выезд на дом — указать)  
      • **schedule** — расписание (start, end) + «окна» (slots) на неделю,
        сгруппировав по адресам и датам  
      • **specialization** — специализация и стаж (если дан)  

    Если по какому‑то пункту данных нет — так и напиши 
    («Расписание не указано» и т.д.).
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

    # --- Встраиваем компактную версию JSON‑данных врача -----------------
    # Ограничиваем размер до 4 000 символов, чтобы не перегрузить prompt
    compact_json = json.dumps(llm_output, ensure_ascii=False)[:4000]
    # Подменяем плейсхолдер {llm_output} в system_message_for_formulation
    formatted_prompt = system_message_for_formulation.replace(
        "{llm_output}", compact_json
    )
    aresult = await ollama_call(formatted_prompt, max_tokens=250)
    # max_tokens=250 → короче ответ и быстрее генерация

    # Опционально выводим время генерации
    if "eval_duration" in aresult:
        print(
            f"Eval_duration of answer generation: "
            f"{aresult['eval_duration'] / 1_000_000_000:.3f} сек"
        )

    return aresult.get("response", "")


def get_doc_info(doc_name):
    """Получить информацию о враче по имени."""
    return api_call.find_doctor_schedule(doc_name)


def get_doctors_by_keyword(item: str):
    """
    Вернёт список врачей по ключевому слову (специальности).
    Пример item: 'кардиолог', 'узи', 'эндокринолог'.
    """
    return api_call.find_doctors_by_keyword(item)


def re_capture(model_response):
    """
    Возвращает либо строку-ответ, либо dict-данные о враче,
    либо list[dict] – список врачей по специализации.
    """
    func_call_pattern = r"\[([a-zA-Z0-9_]+)\((.*)\)\]"
    match = re.match(func_call_pattern, model_response)
    if not match:
        return model_response  # обычный текст

    func_name, args_str = match.groups()
    item_match = re.match(r"item\s*=\s*'([^']+)'", args_str)
    if not item_match:
        return None

    item_value = item_match.group(1)

    if func_name == "get_doc_info":
        result = get_doc_info(item_value)
    elif func_name == "get_doctors_by_keyword":
        result = get_doctors_by_keyword(item_value)
    else:
        result = model_response  # на всякий случай

    print("\nРезультат функции:", result)
    print("Тип результата:", type(result))

    if isinstance(result, list):
        print("\nСписок врачей:")
        for doc in result:
            print(f"- {doc}")

    return result


client = doc_reg.NaykaClient(
    c.nayka_base_url,
    auth=(c.nayka_login, c.nayka_pass)
)

urllib3.disable_warnings()


async def investigate(question: str):
    """
    Zero‑shot function calling + few‑shot + скрытый chain‑of‑thought.
    Модель может вызвать [get_doc_info(item='…')] или дать прямой 
    текстовый ответ.
    """
    print("\n=== Начало обработки вопроса ===")
    print(f"Вопрос: {question}")

    # Базовый system‑prompt
    system_base = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Ты — ассистент клиники «Наука», у тебя есть справочник врачей и 
функция get_doc_info(item: string).
Чтобы получить расписание и услуги врача, используй 
[get_doc_info(item='…')].
Если врач не найден — сообщай об этом текстом.
У тебя также есть функция get_doctors_by_keyword(item: string),
которая возвращает список врачей по специализации (item = название).

Важно:
1. Ты всегда отвечаешь только на русском языке
2. При поиске врача учитывай возможные отклонения от именительного 
   падежа и используй normalization_hint
3. Если в вопросе указана только фамилия, ищи врача по фамилии в списке
4. Если врач не найден — сообщай об этом текстом
5. Не придумывай врачей, которых нет в списке
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

Пример 5:
INPUT:
Вопрос: Какие у вас кардиологи?
OUTPUT:
[get_doctors_by_keyword(item='кардиолог')]
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

    print("\nСистемный промпт:", system_message)

    # Генерируем ответ
    aresult = await ollama_call(system_message, max_tokens=512)
    print(f"\nОтвет LLM: {aresult}")

    # лог времени
    if "eval_duration" in aresult:
        print(f"Eval_duration: {aresult['eval_duration'] / 1e9:.3f}s")

    response = aresult.get("response", "")
    print(f"\nИзвлеченный ответ: {response}")

    # Форматируем ответ
    formatted = await formulate(response)
    print(f"\nОтформатированный ответ: {formatted}")

    # Извлекаем данные
    result = re_capture(formatted)
    print(f"\nРезультат re_capture: {result}")

    # Форматируем финальный ответ
    if isinstance(result, list):
        print("\nФорматируем список врачей:")
        response = "Найдены следующие врачи:\n\n"
        for doc in result:
            print(f"Обрабатываем врача: {doc}")
            # Выводим только ФИО и специализацию
            specialization = doc.get('specialization', '-')
            if specialization and specialization != "-":
                # Берем только первую строку специализации, если их несколько
                specialization = specialization.split('\n')[0].strip()
                # Убираем лишние дефисы и пробелы
                specialization = specialization.replace('-', '').strip()
                # Если после очистки строка пустая, используем дефис
                if not specialization:
                    specialization = '-'
            else:
                specialization = '-'
            response += f"• {doc['fio']} - {specialization}\n"
        print(f"\nИтоговый ответ: {response}")
        return response
    elif isinstance(result, dict):
        print("\nФорматируем информацию о враче:")
        response = "Информация о враче:\n\n"
        response += f"• ФИО: {result['fio']}\n"
        response += f"• Специализация: {result.get('specialization', '-')}\n"
        response += f"• Регионы: {', '.join(result.get('regions', ['-']))}\n"
        print(f"\nИтоговый ответ: {response}")
        return response
    else:
        print(f"\nВозвращаем текстовый ответ: {result}")
        return str(result)


async def main(question: str):
    """Запрос → LLM → (опционально) API → форматирование."""
    try:
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
    except KeyboardInterrupt:
        print("\nОперация прервана пользователем")
    except Exception as e:
        logger.error(f"Произошла ошибка: {e}")
        raise


if __name__ == "__main__":
    q = input("Ask your question: ")
    asyncio.run(main(q))
