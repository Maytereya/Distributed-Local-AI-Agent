from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Tuple, AsyncGenerator, TypeAlias, Literal

from ollama import AsyncClient, Options

from agent_logic_2 import llama_func_call as doctor_info, config as c
from agent_logic_pack import formulate
from agent_logic_pack import meilisearch_client as meilisearch
from converters import html_cleaner

# LLM‑клиент для классификации входящих запросов
ollama = AsyncClient(c.ollama_url)
llm = c.ll_model_big

_OPTIONS = {
    "conservative": {
        "temperature": 0.1,
        "top_k": 30,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "stop": ["<|eot_id|>"],
    },
    "expressive": {
        "temperature": 0.25,
        "top_k": 25,
        "top_p": 0.92,
        "repeat_penalty": 1.2,
        "stop": ["<|eot_id|>"],
    },
}


def options_set(level: Literal["conservative", "expressive"] = "conservative") -> Options:
    """
      Выбор из двух вариантов настроек генерации. Для удобства подбора.
      :param level: Conservative - изначальный вариант, expressive - вариант с чуть большей свободой.
      :return: Option for Ollama.
      """
    params = _OPTIONS.get(level, _OPTIONS["expressive"])
    return Options(**params)


# Label‑ы и порядок
LABEL_PRIORITY = ["API_INFO",  # справка из API Мед.центра
                  "APPOINTMENT",  # назначение времени / запись на приём к врачу
                  "SCRIPTS"  # алгоритмы, скрипты, если - то...

                  ]
ALLOWED = set(LABEL_PRIORITY + ["UNDEFINED"])

LABEL_DOC = """
1. API_INFO – справка из API CRM клинки: врачи, услуги, цены, расписание
2. APPOINTMENT – запись на приём к врачу
3. SCRIPTS – инструкции и скрипты, последовательность действий для администраторов
4. UNDEFINED – не распознан

ВАЖНО: Метка DOC_INFO недопустима! Используй API_INFO для запросов о врачах и услугах.
"""

EXAMPLES = """
INPUT: Сколько стоит приём кардиолога Михлик?            
OUTPUT: {\"labels\":[\"API_INFO\"]}

INPUT: Какое расписание работы у Пивоваровой?            
OUTPUT: {\"labels\":[\"API_INFO\"]}

INPUT: Как подготовиться к анализу крови?        
OUTPUT: {\"labels\":[\"SCRIPTS\"]}

INPUT: Запишите меня к терапевту завтра утром.    
OUTPUT: {\"labels\":[\"APPOINTMENT\"]}

INPUT: Вы плохо взяли кровь, огромный синяк!      
OUTPUT: {\"labels\":[\"SCRIPTS\"]}

INPUT: Как доехать до клиники на автобусе?           
OUTPUT: {\"labels\":[\"SCRIPTS\"]}

INPUT: Подготовка к УЗИ и запишите к УЗИсту.      
OUTPUT: {\"labels\":[\"SCRIPTS\",\"APPOINTMENT\"]}

INPUT: Где принимает Иванов?               
OUTPUT: {\"labels\":[\"API_INFO\"]}

"""


# ----------------------------------------------
# Функция обработки истории запросов
# и ответов на них
# ----------------------------------------------

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


# ----------------------------------------------
# Функции обработки входящих сообщений
# ----------------------------------------------

def classificator_prompt(user: str, sess: Dict[str, Any]) -> str:
    today = datetime.now().strftime("%d %B %Y, %H:%M:%S")
    return f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Сегодня: {today}.
Ты – ассистент многопрофильной клиники «Наука» с амбулаторией, операционными, лабораторией, офисами в разных городах.
Верни **только JSON** вида {{\"labels\":[…]}} (можно несколько label‑ов).
Допустимые label‑ы:\n{LABEL_DOC}\n
[EXAMPLES]\n{EXAMPLES}\n
[USER] {user}
История диалога: {_history_reveal(user, sess)}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


# TODO: Промпт требует переосмысления!

def split_prompt(text: str, sess: Dict[str, Any]) -> str:
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
Этап 1: "Расписание приемов у Мухопад и стоимость услуг ультразвуковой диагностики"
Этап 2: {{"segments": ["Расписание и стоимость ультразвуковой диагностики у Мухопад"]}}

Исходно: "К Мухопад запись и как готовиться, а про Нурмагомедову график"
Этап 1: "Запись к Мухопад и подготовка к приёму у Мухопад. Расписание Нурмагомедовой"
Этап 2: {{"segments": ["Запись и подготовка к приёму у Мухопад", "Расписание Нурмагомедовой"]}}

ВАЖНО: Верни только JSON с полем segments, содержащим массив строк!

USER: {text}
История диалога: {_history_reveal(text, sess)}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


async def split_into_segments(text: str, sess: Dict[str, Any]) -> List[str]:
    # print("\n================= SPLIT PROCESS START =================")
    # print(f"Original input: '{text}'")

    res = await ollama.generate(model=llm,
                                prompt=split_prompt(text, sess),  # Добавить sess
                                options=options_set(level="expressive"),
                                format="json",
                                keep_alive=-1)

    try:
        segments = json.loads(res["response"]).get("segments", [])
        splitted_segments = [s.strip() for s in segments if s.strip()]

        print("\nProcessing results:")
        print(f"• Segments count: {len(splitted_segments)}")
        print(f"• Final segments: {splitted_segments}")
        # print("================= SPLIT PROCESS END =================\n")

        return splitted_segments

    except json.JSONDecodeError as e:
        print(f"\n⚠️ JSON decode error: {e}")
        print("⚠️ Returned original text as single segment")
        return [text]

    except Exception as e:
        print(f"\n⚠️ Unexpected error: {e}")
        print("⚠️ Returned original text as single segment")
        return [text]


async def classify(text: str, sess: Dict[str, Any]) -> List[str]:
    res = await ollama.generate(model=llm,
                                prompt=classificator_prompt(text, sess),
                                options=options_set(level="expressive"),
                                format="json",
                                keep_alive=-1)
    try:
        labels = [l.upper() for l in json.loads(res["response"]).get("labels", []) if l.upper() in ALLOWED]
        print("----------------- LABELS -----------------------")
        print(f"Маркировано labels: ", labels)
        print("------------------------------------------------")
        return labels or ["UNDEFINED"]
    except Exception as e:
        print(f"\n<UNK> Classificator error: {e}")
        return ["UNDEFINED"]


async def final_answering(primary_request: str,
                          collected_info: str):  # Пока неясно что за тип данных будет возвращаться
    prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Ты — ассистент колл-центра клиники «Наука». Отвечай **только на русском** и **строго по шаблонам**.
        1) Определи тип запроса:
           - **Конкретный врач** — запрос содержит имя или «приём у <ФИО>».
           - **Специализация** — запрос содержит профессиональное название (например, «кардиологи»).
           - **График работы** —  запрос содержит "график работы", "расписание", "время приема", "часы приема", "когда работает"
           - **Иное** — все остальные запросы.
        
        2) Если это запрос **по конкретному врачу**:
            **ФИО врача:** {{• ФИО}}  
            **Специализация кратко:** {{compose from • Специализация}}  
            **Специализация подробно:** {{• Специализация}}  
            **Категория**: {{(Высшая/Первая/Не указано - вытащить из 📞Заметка)}} 
            **Контактный телефон:** {{(вытащить из 📞Заметка)}} 
            **Приходящий** {{(Да!/Нет! и условия из 📞Заметка)}} 
            **Стаж работы:** {{(вытащить из 📞Заметка)}}  
            **Возраcт пациентов:** {{(вытащить из 📞Заметка)}}  
            **ДМС:** {{(да/нет и условия из 📞Заметка)}}  
            **Прайс-лист:** {{(все пункты цен из 📞Заметка)}}  
            **Адрес/адреса работы:** {{• Адрес/Адреса}}
            
           ❗ Поля "Специализация_подробно" и "Прайс-лист" выводи дословно, сохраняя все маркеры и переносы строк.
            Если ответ из базы данных не содержит информации о враче, ответь "Врач не найден в базе данных".
        
        3) Если это запрос **по специализации**:
           Для каждого врача из DATABASE, у которого в {{• Специализация}} есть нужное слово:
           **ФИО врача:** {{• ФИО}}  
           **Специализация кратко:** {{составить самому}}
           (каждый врач — новый блок; если таких нет — 
           ответ: "Релевантной информации ... не найдено")
           ЕСЛИ информации из базы данных не поступило, ничего не выдумывай! Сообщи, что информации не найдено.
           
        4) Если это запрос про **график работы врача**:
            **ФИО врача:** {{• ФИО}}  
            **Специализация кратко:** {{compose from • Специализация}}  
            **График приёма с адресами работы:**  
            {{• Расписание}}
            
            ❗ **Блок {{• Расписание}} выводи дословно**, ровно как в DATABASE: 
            сохраняй «По адресу приёма…», «Окна:», отступы и переносы строк. 
            Никакой переработки или перелинковки временных слотов.
            
        
        5) Если это **прочий запрос**:
           **Информация из базы знаний:**  
           {{KNOWLEDGE_SNIPPET}} 
           ...
           Если ответ из базы данных таков: "{{KNOWLEDGE_SNIPPET}} 
           Совпадений не найдено, cформулируйте запрос иначе", то
           верни сообщение "Подходящей информации в базе знаний не найдено".
        

    ## Данные:
    
    DATABASE: {collected_info}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    USER: {primary_request}  
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    """
    partial = ""  # накопитель
    stream = await ollama.generate(
        model=llm,
        prompt=prompt,
        options=options_set(level="expressive"),
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


# ------------------------------------------------------
# Подключаем заглушку функции записи пациента
# ------------------------------------------------------
async def appointment_stub(_text: str, **__) -> Tuple[str, bool]:
    return "Модуль записи к врачу скоро появится. ", False


# ──────────────────────────────────────────────────────
# Search with MEILISEARCH function
# ──────────────────────────────────────────────────────
# ToDo: Объединить все запросы в одну функцию
# TODO: Завернуть всю поисковую часть (MEILI) в одну функцию
async def instructions_search(_text: str, **__) -> Tuple[str, bool]:
    extracted_keyword = await formulate.extract_keyword(_text, extract_type="sentence")

    print("=" * 45)
    print("Экстрагировалось: ", extracted_keyword or "Empty")
    print("=" * 45)

    collected_info = meilisearch.search_meili("spravka_docs", extracted_keyword, )

    # Очистка HTML перед подстановкой в prompt
    clean_info = html_cleaner.strip_html(collected_info)
    # Добавление маркера для лучшего распознавания LLM
    marked_info = "{KNOWLEDGE_SNIPPET}" + "\n" + clean_info
    print("=" * 45)
    print(marked_info)
    print("=" * 45)
    return marked_info, False


# ────────────────────────────────────────────────
# Основная логика роутера / Router main logic
# ────────────────────────────────────────────────

# Ярлыки для вызова функций обработки данных после роутинга

MODULES = {
    "API_INFO": get_doc_info_from_api,
    "APPOINTMENT": appointment_stub,
    "SCRIPTS": instructions_search,
}

# Дополнительные обозначения типов для понимания вывода
SessionType: TypeAlias = Dict[str, Any]
RoutingResult: TypeAlias = Tuple[str, SessionType]

# Константа для удобства форматирования
SEGMENT_SEPARATOR = "\n\n— — —\n\n"


async def handle_pending_module(text: str, sess: SessionType) -> RoutingResult | None:
    if (pending_module := sess.get("pending")) and pending_module in MODULES:
        print("=" * 45)
        print("pending look up: ", pending_module or "Empty")
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

    final_answ = SEGMENT_SEPARATOR.join(responses)
    print("=" * 45)
    print(final_answ)
    print("=" * 45)
    return final_answ


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
