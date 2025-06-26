from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Tuple, AsyncGenerator, TypeAlias

from ollama import AsyncClient, Options

from agent_logic_2 import llama_func_call4 as doctor_info, config as c
from agent_logic_pack import formulate
from agent_logic_pack import meilisearch_client as meilisearch
from converters import html_cleaner

# LLM‑клиент для классификации входящих запросов
ollama = AsyncClient(c.ollama_url)
llm = c.ll_model_big
options = Options(temperature=0.1, top_k=40, top_p=0.9, repeat_penalty=1.1, stop=["<|eot_id|>"])

# Label‑ы и порядок
LABEL_PRIORITY = ["API_INFO",  # справка из API Мед.центра
                  "APPOINTMENT",  # назначение времени / запись на приём к врачу
                  "SCRIPTS"  # алгоритмы, скрипты, если - то...
                  "ESSENTIAL",  # общая информация обо всем, справочники
                  "PREPARE_FOR",  # Подготовка к анализам и прочим исследованиям

                  ]
ALLOWED = set(LABEL_PRIORITY + ["UNDEFINED"])

LABEL_DOC = """
1. API_INFO – справка из API CRM клинки: врачи, услуги, цены, расписание
2. PREPARE_FOR – правила подготовки к анализам и прочим медицинским исследованиям
3. APPOINTMENT – запись на приём к врачу
4. ESSENTIAL – общая справочная информация об услугах клиники
5. SCRIPTS – инструкции и скрипты, последовательность действий для администраторов
6. UNDEFINED – не распознан

ВАЖНО: Метка DOC_INFO недопустима! Используй API_INFO для запросов о врачах и услугах.
"""

EXAMPLES = """
INPUT: Сколько стоит приём кардиолога?            
OUTPUT: {\"labels\":[\"API_INFO\"]}

INPUT: Какое расписание работы у гинеколога?            
OUTPUT: {\"labels\":[\"API_INFO\"]}

INPUT: Как подготовиться к анализу крови?        
OUTPUT: {\"labels\":[\"PREPARE_FOR\"]}

INPUT: Запишите меня к терапевту завтра утром.    
OUTPUT: {\"labels\":[\"APPOINTMENT\"]}

INPUT: Вы плохо взяли кровь, огромный синяк!      
OUTPUT: {\"labels\":[\"SCRIPTS\"]}

INPUT: Как доехать до клиники на автобусе?           
OUTPUT: {\"labels\":[\"ESSENTIAL\"]}

INPUT: Подготовка к УЗИ и запишите к УЗИсту.      
OUTPUT: {\"labels\":[\"PREPARE_FOR\",\"APPOINTMENT\"]}

INPUT: Где принимает доктор Иванов?               
OUTPUT: {\"labels\":[\"API_INFO\"]}

INPUT: Какие услуги предоставляет клиника?        
OUTPUT: {\"labels\":[\"ESSENTIAL\"]}
"""


def _history_reveal(user: str, sess: Dict[str, Any]) -> str:
    history = sess.get("history", [])

    # Отбрасываем последний пользовательский запрос, чтобы он не дублировался в истории, передаваемой в запрос
    if history and history[-1].get("user") == user:
        history = history[:-1]

    # Печатаем скорректированный history с помощью генератора
    generator = "\n".join(
        f"{'User:' if 'user' in turn else 'Assistant:'} "
        f"{turn.get('user') or turn.get('bot') or turn.get('assistant')}"
        # нет уверенности, что вариант извлечения assistant нужен, но пока оставим.
        for turn in history
    )
    print("_history_reveal output: ", generator)
    return generator


def classificator_prompt(user: str, sess: Dict[str, Any]) -> str:
    today = datetime.now().strftime("%d %B %Y, %H:%M:%S")
    return f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Сегодня: {today}.
Ты – ассистент многопрофильной клиники «Наука» с амбулаторией, операционными, лабораторией, множеством офисов в разных городах.
Верни **только JSON** вида {{\"labels\":[…]}} (можно несколько label‑ов).
Допустимые label‑ы:\n{LABEL_DOC}\n
[EXAMPLES]\n{EXAMPLES}\n
[USER] {user}
История диалога: {_history_reveal(user, sess)}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def reformulator_prompt(text: str, sess: Dict[str, Any]) -> str:
    return f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
1. Сначала переформулируй запрос, привязывая все упоминания услуг/вопросов к фамилии врача.
2. Затем разбей на смысловые сегменты (JSON segments).


Правила:
- Если фамилия упомянута, все смежные вопросы (расписание, цена, подготовка) должны содержать ту же фамилию;
- Объединяй в один сегмент связанные запросы к одному врачу:
• Расписание + стоимость
  • Услуга + подготовка к ней
  • Любые комбинации для одного специалиста
- Разделяй сегменты, когда меняется фамилия врача или начинается общая тема.

Примеры переформулировки:
Исходно: "У Мухопад окна и прайс на УЗИ"
Этап 1: "Расписание приёмов УЗИ у Мухопад и стоимость услуг УЗИ"
Этап 2: {{"segments": ["Расписание и стоимость УЗИ у Мухопад"]}}

Исходно: "К Мухопад запись и как готовиться, а про Нурмагомедову график"
Этап 1: "Запись к Мухопад и подготовка к приёму у Мухопад. Расписание Нурмагомедовой"
Этап 2: {{"segments": ["Запись и подготовка к приёму у Мухопад", "Расписание Нурмагомедовой"]}}

ВАЖНО: Верни только JSON с полем segments, содержащим массив строк!

USER: {text}
История диалога: {_history_reveal(text, sess)}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


async def classify(text: str, sess: Dict[str, Any]) -> List[str]:
    res = await ollama.generate(model=llm,
                                prompt=classificator_prompt(text, sess),
                                options=options,
                                format="json",
                                keep_alive=-1)
    try:
        labels = [l.upper() for l in json.loads(res["response"]).get("labels", []) if l.upper() in ALLOWED]
        print("----------------- LABELS -----------------------")
        print(f"Маркировано labels: ", labels)
        print("------------------------------------------------")
        return labels or ["UNDEFINED"]
    except Exception:
        return ["UNDEFINED"]


async def split_into_segments(text: str, sess: Dict[str, Any]) -> List[str]:
    # print("\n================= SPLIT PROCESS START =================")
    # print(f"Original input: '{text}'")

    res = await ollama.generate(model=llm,
                                prompt=reformulator_prompt(text, sess),  # Добавить sess
                                options=options,
                                format="json",
                                keep_alive=-1)

    try:
        # print(f"\nRaw model response: {res['response']}")

        segments = json.loads(res["response"]).get("segments", [])
        # print(f"Raw segments: {segments}")

        splitted_segments = [s.strip() for s in segments if s.strip()]

        # print("\nProcessing results:")
        # print(f"• Segments count: {len(splitted_segments)}")
        # print(f"• Final segments: {splitted_segments}")
        # print("================= SPLIT PROCESS END =================\n")

        return splitted_segments

    except json.JSONDecodeError as e:
        print(f"\n⚠️ JSON decode error: {e}")
        print("⚠️ Returned original text as single segment")
        print("================= SPLIT PROCESS END =================\n")
        return [text]

    except Exception as e:
        print(f"\n⚠️ Unexpected error: {e}")
        print("⚠️ Returned original text as single segment")
        print("================= SPLIT PROCESS END =================\n")
        return [text]


async def final_answering(primary_request: str,
                          collected_info: str):  # Пока неясно что за тип данных будет возвращаться
    prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Ты — ассистент колл‑центра многопрофильной клиники «Наука».  
    Отвечай **всегда на русском языке**.

    Твоя задача — **дословно** вывести данные из блока DATABASE.  
    ❗ Ничего не резюмируй, не сокращай, не перефразируй. **Копируй всё строго как есть.**  
    ❗ Особенно важно: **ФИО врачей, адреса, цены и график приёма** — без изменений.

    ---

    ## Что тебе дано:
    - USER — исходный запрос клиента.  
    - DATABASE — информация, найденная в БД.

    ---

    ## Что нужно сделать:
    1. Найди в DATABASE все поля, которые **соответствуют запросу клиента**.
    2. Для каждого поля:
       - если оно есть — выведи его строго в формате ниже;
       - если его нет — напиши «нет данных».
    3. Ничего не придумывай, не добавляй от себя.
    4. Поле {{📞Заметка колл-центра}} **иногда содержит цены и скидки**. Если такие есть — **дословно выведи всё**.
    5. **Не предлагай позвонить, уточнить или проконсультироваться.** Просто выведи, что найдено.

    ---

    ## Формат ответа про врача:
    (используй ровно его, не меняй и не добавляй новых полей)

    **ФИО врача: {{• ФИО}}**  
    **Специализация кратко:** {{SPECIALIZATION_SHORT}}  
    **Специализация подробно:** {{• Специализация}}  
    **Информация для колл‑центра и стоимость приёма:** {{• 📞Заметка колл-центра}}  
    **Адрес/адреса работы:** {{• Адрес/Адреса}}  
    **График приёма:** {{• Расписание}}

    ---

    ## Формат ответа на прочие запросы (из базы знаний), в том числе о том, как подготовиться к процедуре или исследованию:
    **Информация из базы знаний:**  
    {{KNOWLEDGE_SNIPPET}}  
    (Выводи весь подходящий текст из DATABASE, если он отвечает на вопрос пользователя. Если ничего не подходит — напиши: «Релевантных данных не найдено.»)

    ---

    ## Важно:
    - Жирным делай **только подписи полей**.
    - Не используй никаких дополнительных слов или пояснений.
    - Если найдено несколько врачей — **выведи каждый блок отдельно**.
    - Если в запросе несколько тем, выводи блоки в следующем порядке:
      1. Врач
      2. Стоимость / скидки
      3. Подготовка
      4. Прочая информация из базы знаний

    ---

    ## Примеры:

    *Пример 1*  
    USER: «Сколько стоит приём Ивановой?»  
    → выведи блок про врача по формату выше.

    *Пример 2*  
    USER: «Как подготовиться к гастроскопии?»  
    → выведи блок "Информация из базы знаний".

    *Пример 3*  
    USER: «Выведи всех гастроэнтерологов на Ленина 5»  
    → выведи по одному блоку на каждого врача, принимающего по этому адресу.

    *Пример 4*  
    DATABASE:
    - Прием уролога со скидкой 25% с 01.04 по 30.06.25  
    - Первичный приём уролога КМН вместо 3500 руб за 2625 руб.  
    - [4.1.2.7] Повторный приём уролога, кмн — 2400 руб (30 минут).  
    → выведи всё как есть в {{• 📞Заметка колл-центра}}.

    *Пример 5*  
    USER: «Что делать, если пациент жалуется?»  
    → выведи блок "Информация из базы знаний".

    ---

    ## Данные:
    USER: {primary_request}  
    DATABASE: {collected_info}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    partial = ""  # накопитель
    stream = await ollama.generate(
        model=llm,
        prompt=prompt,
        options=options,
        keep_alive=-1,
        stream=True,
    )
    async for chunk in stream:
        partial += chunk["response"]
        yield partial


# ──────────────────────────────────────────────────────
# Подключаем doctor_info из llama_func_call
# ──────────────────────────────────────────────────────

async def get_doc_info_from_api(question: str, **_) -> Tuple[str, bool]:
    result = await doctor_info.investigate(question)
    return result, False


# ──────────────────────────────────────────────────────
# Подключаем MEILISEARCH
# ──────────────────────────────────────────────────────
# ToDo: Объединить все в одну функцию с соответствующим вызовом

async def preparation_for(_text: str, **__) -> Tuple[str, bool]:
    """
    Warning!
    Index meilisearch hardcoded!
    :param _text: Users' request.

    :param __: For state
    :return: Str of a text from meilisearch

    """
    extracted_keyword = await formulate.extract_keyword(_text)

    print("=============================================")
    print("Смотри, что экстрагировалось: ", extracted_keyword or "Empty")
    print("=============================================")

    collected_info = meilisearch.search_meili("preparation_docs", extracted_keyword, )

    # Очистка HTML перед подстановкой в prompt
    clean_info = html_cleaner.strip_html(collected_info)

    return clean_info, False


async def appointment_stub(_text: str, **__) -> Tuple[str, bool]:
    return "Модуль записи к врачу скоро появится. ", False


async def instructions_scripts(_text: str, **__) -> Tuple[str, bool]:
    extracted_keyword = await formulate.extract_keyword(_text)

    print("=============================================")
    print("Смотри, что экстрагировалось: ", extracted_keyword or "Empty")
    print("=============================================")

    collected_info = meilisearch.search_meili("algo_docs", extracted_keyword, )

    # Очистка HTML перед подстановкой в prompt
    clean_info = html_cleaner.strip_html(collected_info)

    return clean_info, False


async def essential_info(_text: str, **__) -> Tuple[str, bool]:
    extracted_keyword = await formulate.extract_keyword(_text)

    print("=============================================")
    print("Смотри, что экстрагировалось: ", extracted_keyword or "Empty")
    print("=============================================")

    collected_info = meilisearch.search_meili("spravka_docs", extracted_keyword, )

    # Очистка HTML перед подстановкой в prompt
    clean_info = html_cleaner.strip_html(collected_info)

    return clean_info, False


# ────────────────────────────────────────────────
# Собственно, роутер пошел. Со всеми наворотами.
# Типа оператора присваивания морж ":=" и тд
# ────────────────────────────────────────────────

# Ярлыки для вызова функций обработки данных после роутинга
# TODO: Завернуть всю поисковую часть (MEILI) в одну функцию
MODULES = {
    "API_INFO": get_doc_info_from_api,
    "APPOINTMENT": appointment_stub,
    "PREPARE_FOR": preparation_for,
    "ESSENTIAL": essential_info,
    "SCRIPTS": instructions_scripts,
}
# Дополнительные обозначения типов для понимания вывода
SessionType: TypeAlias = Dict[str, Any]
RoutingResult: TypeAlias = Tuple[str, SessionType]

# Чисто для удобства форматирования
SEGMENT_SEPARATOR = "\n\n— — —\n\n"


async def handle_pending_module(text: str, sess: SessionType) -> RoutingResult | None:
    if (pending_module := sess.get("pending")) and pending_module in MODULES:
        print("=" * 45)
        print("pending view: ", pending_module or "Empty")
        print("=" * 45)

        response, continue_pending = await MODULES[pending_module](text, session=sess)
        sess["pending"] = pending_module if continue_pending else None

        # Record interaction in history
        sess["history"].extend([
            {"user": text},
            {"bot": response}
        ])

        return response, sess
    return None


async def process_segments(text: str, sess: SessionType) -> str:
    segments = await split_into_segments(text, sess)
    responses: List[str] = []

    for idx, segment in enumerate(segments, 1):
        labels = await classify(segment, sess)
        labels.sort(key=LABEL_PRIORITY.index)

        print("=" * 45)
        print(f"PART view #{idx}: ", segment or "Empty PART")
        print("=" * 45)

        for label in labels:
            response, continue_pending = await MODULES[label](segment, session=sess)
            responses.append(response)
            if continue_pending:
                sess["pending"] = label
                print("=" * 45)
                print("need pending view: ", sess["pending"] or "Empty need")
                print("=" * 45)
                break

    return SEGMENT_SEPARATOR.join(responses)


async def routing(text: str, sess: SessionType | None = None) -> AsyncGenerator[RoutingResult, None]:
    """

    Here is variant of routing() that *yields* (partial_answer, session) pairs,
        so that the outer UI can stream them.

    :param text: Сообщение пользователя
    :param sess: словарь состояния сессии, хранит pending-модуль и историю
    :return: ответ, обновлённая сессия
    """
    # 1. Инициализируем состояние сессии обработки входящего текстового блока

    sess = sess or {}
    sess.setdefault("pending", None)
    sess.setdefault("history", [])

    # Handle pending module if exists
    if pending_result := await handle_pending_module(text, sess):
        yield pending_result

    # Process text segments
    result = await process_segments(text, sess)

    # Update history
    sess["history"].extend([
        {"user": text},
        {"bot": result}
    ])

    # Stream final response
    async for partial in final_answering(text, result):
        yield partial, sess


async def process_routing_request(query: str) -> Tuple[str, Dict[str, Any]]:
    """
    Запускает маршрутизацию и собирает все части ответа из async-генератора,
    возвращая финальную строку и итоговую сессию. Нужно чисто для тестирования данного модуля
    """
    final_response: str = ""
    final_session: dict[str, Any] = {}

    # routing возвращает AsyncGenerator[(partial_response, session), None]
    async for partial, sess in routing(query.strip()):
        # на каждой итерации приходят (partial, sess)
        final_response = partial  # перезаписываем — в итоге останется последний
        final_session = sess

    return final_response, final_session


async def main():
    async for partial, sess in routing("кардиологи клиники"):
        print(partial)  # или обновлять UI


if __name__ == "__main__":
    asyncio.run(main())
