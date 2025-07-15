from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Tuple, AsyncGenerator, TypeAlias, Literal

from ollama import AsyncClient

from agent_logic_2 import llama_func_call as doctor_info, config as c
from agent_logic_pack import meilisearch_client as meilisearch
from converters import html_cleaner
from ollama_settings import options_set, OLLAMA_MODEL

# LLM‑клиент для классификации входящих запросов
ollama = AsyncClient(c.ollama_url)

# Label‑ы и порядок
LABEL_PRIORITY = ["API_INFO",  # справка из API Мед.центра
                  "APPOINTMENT",  # назначение времени / запись на приём к врачу
                  "SCRIPTS"  # алгоритмы, скрипты, если - то...

                  ]
ALLOWED = set(LABEL_PRIORITY + ["UNDEFINED"])

LABEL_DOC = """
1. API_INFO – справка из API CRM клинки: врачи, услуги, цены, расписание.
2. APPOINTMENT – запись на приём к врачу.
3. SCRIPTS – инструкции и скрипты для пациентов и администраторов, база знаний, справочная информация.
4. UNDEFINED – не распознан.

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

INPUT: Подготовка к УЗИ и запишите к УЗИсту.      
OUTPUT: {\"labels\":[\"SCRIPTS\",\"APPOINTMENT\"]}

INPUT: Где принимает Иванов?               
OUTPUT: {\"labels\":[\"API_INFO\"]}

INPUT: все про кольпоскопию               
OUTPUT: {\"labels\":[\"SCRIPTS\"]}

INPUT: подготовка к вульвоскопии               
OUTPUT: {\"labels\":[\"SCRIPTS\"]}

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
    # TODO: Есть история диалога!
    return f"""
SYSTEM:
Сегодня: {today}.
Ты – ассистент многопрофильной клиники «Наука» с амбулаторией, операционными, лабораторией и офисами в разных городах.
Верни **только JSON** вида {{\"labels\":[…]}} (можно несколько label‑ов).
Допустимые label‑ы:\n{LABEL_DOC}\n
EXAMPLES: \n{EXAMPLES}\n
USER: \n{user}
HISTORY: \n {_history_reveal(user, sess)}

"""


# TODO: Промпт требует переосмысления!
def split_prompt(text: str, sess: Dict[str, Any]) -> str:
    return f"""
SYSTEM:
1. Сначала переформулируй запрос, привязывая все упоминания услуг/вопросов к фамилии врача.
2. Затем разбей на смысловые сегменты (JSON segments).

Правила:
- Если фамилия упомянута, все смежные вопросы (расписание, цена, подготовка) должны содержать ту же фамилию;
- Объединяй в один сегмент связанные запросы к одному врачу:
 • Расписание;
 • Вся известная информация о враче; 
 • Любые комбинации для одного специалиста.
- Объединяй в один сегмент связанные запросы по поводу:
 • адресов клиник и времени работы клиник;
 • медицинской услуги или подготовки к анализу/исследованию;
 • если ответ на запрос предусматривает некий алгоритм действий;
 • если ответ предусматривает информацию о медицинской процедуре, заболевании, исследовании;
- Разделяй сегменты, когда меняется фамилия врача или начинается общая тема.

Примеры переформулировки:
Исходно: "У Мухопад окна и как подготовиться к УЗИ печени"
Этап 1: "Расписание приемов у Мухопад и подготовка к ультразвуковой диагностике печени"
Этап 2: {{"segments": ["Расписание у Мухопад", "подготовка к ультразвуковой диагностике печени"]}}

Исходно: "К Мухопад запись, а про Нурмагомедову график"
Этап 1: "Запись к Мухопад и подготовка к приёму у Мухопад. Расписание Нурмагомедовой"
Этап 2: {{"segments": ["Запись на приём к Мухопад", "Расписание Нурмагомедовой"]}}

Исходно: "Вульвоскопия с кольпоскопией, как подготовиться и скажите как попасть к Нурмагомедовой"
Этап 1: "Подготовка к вульвоскопии и кольпоскопии. Расписание Нурмагомедовой"
Этап 2: {{"segments": ["Подготовка к вульвоскопии, кольпоскопии", "Расписание Нурмагомедовой"]}}

Исходно: "все что известно про кольпоскопию"
Этап 1: "кольпоскопия"
Этап 2: {{"segments": ["кольпоскопия"]}}


ВАЖНО: Верни только JSON с полем segments, содержащим массив строк!

USER: \n{text}
HISTORY: \n{_history_reveal(text, sess)}
"""


async def split_into_segments(text: str, sess: Dict[str, Any]) -> List[str]:
    """
    Важно! Функция переформулирует запрос!

    :param text: Входящий сырой запрос.
    :param sess:
    :return: Возвращает список запросов от пользователя.
    """
    # print("\n================= SPLIT PROCESS START =================")
    # print(f"Original input: '{text}'")

    res = await ollama.generate(model=OLLAMA_MODEL,
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
    """
    Классификатор сегментов входящего запроса пользователя

    :param text:
    :param sess:
    :return:
    """
    res = await ollama.generate(model=OLLAMA_MODEL,
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
        print(f"\n Classificator error: {e}")
        return ["UNDEFINED"]


async def final_answering(primary_request: str,
                          collected_info: str):  # Пока неясно что за тип данных будет возвращаться
    prompt = f"""
    SYSTEM:
    Ты — ассистент колл-центра клиники «Наука». Отвечай **только на русском** и **строго по шаблонам**.
        1) Определи тип запроса:
           - **Конкретный врач** — запрос содержит имя или «приём у <ФИО>».
           - **Специализация** — запрос содержит профессиональное название (например, «кардиологи», "урологи").
           - **График работы** —  запрос содержит "график работы", "расписание", "время приема", "часы приема", "когда работает"
           - **Иное** — все остальные запросы.
        
        2) Если это запрос **по конкретному врачу**:
        
            - поля "Специализация_подробно" и "Прайс-лист" выводи дословно,
            - если ответ из базы данных не содержит информации о враче, ответь "Врач не найден в базе данных".
            
            **Структура твоего ответа:**
            **ФИО врача:** {{ФИО}}  
            **Специализация кратко:** {{compose from Специализация}}  
            **Специализация подробно:** {{Специализация}}  
            **Категория:** {{(Высшая/Первая/Не указано - compose from 📞Заметка)}} 
            **Контактный телефон:** {{(извлеки как есть из 📞Заметка или напиши "не указан")}} 
            **Приходящий:** {{(Да!/Нет! (обязательно с восклицательным знаком) и всеми условиями compose from 📞Заметка)}} 
            **Стаж работы:** {{(compose from 📞Заметка)}}  
            **Возраcт пациентов:** {{(compose from 📞Заметка)}}  
            **ДМС:** {{(Да/Нет и условия compose from 📞Заметка)}}  
            **Прайс-лист:** {{(все пункты цен compose from 📞Заметка)}}  
            **Адрес/адреса работы:** {{Адрес/Адреса}}
            — Конец списка —
        
        3) Если это запрос по врачебной специализации и тебе передан список врачей:
            Выведи полный список всех врачей, как передано. 
            
            **Структура твоего ответа:** 
            **ФИО врача:** {{ФИО}} 
            **Специализация кратко:** {{compose from Специализация, какие услуги оказывает, не более 10 слов!}}
            **Адрес/адреса работы:** {{Адрес/Адреса}}
            (каждый врач — отдельный блок; если не найдено — "Релевантной информации не найдено")
            После последнего блока напиши — Конец списка — и НЕ начинай новую нумерацию.
           
        4) Если это запрос про **график работы врача**:
        
            **Структура твоего ответа:** 
            **ФИО врача:** {{ФИО}}  
            **Специализация кратко:** {{compose from Специализация}}  
            **График приёма с адресами работы:**{{Адрес/Адреса}}
            {{Расписание}}
            
            ❗ **Блок {{Расписание}} выводи дословно**, ровно как в DATABASE: 
            сохраняй «По адресу приёма…», «Окна:», отступы и переносы строк. 
            Никакой переработки или перелинковки временных слотов.
            Улучши отображение даты приема: вместо "2025-07-08: 08:30-13:45 * Окна: 10:00, 10:30, 12:30, 13:00, 13:30"
            напиши "**08 июля:** работает с 08:30 по 13:45 * Окна: 10:00, 10:30, 12:30, 13:00, 13:30"            
        
        5) Если это **прочий запрос**:
        
           **Структура твоего ответа:** 
           **Информация из базы знаний:**  
           {{KNOWLEDGE_SNIPPET}} 
           ...
           Если ответ из базы данных таков: 
           "{{KNOWLEDGE_SNIPPET}} 
           Совпадений не найдено, cформулируйте запрос иначе", 
           то верни сообщение: "Подходящей информации в базе знаний не найдено".
        
    DATABASE: {collected_info}
    
    USER: {primary_request}  
    
    """
    partial = ""  # накопитель
    stream = await ollama.generate(
        model=OLLAMA_MODEL,
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
# (может использовать LLM переформулировку)
# ──────────────────────────────────────────────────────
# ToDo: Объединить все запросы в одну функцию
# TODO: Завернуть всю поисковую часть (MEILI) в одну функцию
async def instructions_search(_text: str, **__) -> Tuple[str, bool]:
    """
    Для поиска нужной информации в индексе или коллекции используется переформулировка запроса пользователя
    Пока не ясно, следует ли ее делать.
    :param _text:
    :param __:
    :return: Кортеж: результат поиска и стоп - паттерн для PENDING
    """
    # Временно отключу переформулировку!
    # extracted_keyword = await formulate.extract_keyword(_text, extract_type="sentence")

    print("=" * 45)
    print("Экстрагировалось: ",
          # extracted_keyword
          _text  # шарахнем запрос напрямую без переформулировки
          or "Empty")
    print("=" * 45)

    collected_info = meilisearch.search_meili("spravka_docs",
                                              # extracted_keyword,
                                              _text,
                                              )

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
        print("pending_module content: ", pending_module or "Empty")
        print("=" * 45)

        response, continue_pending = await MODULES[pending_module](text, session=sess, )
        sess["pending"] = pending_module if continue_pending else None

        # Record interaction in history
        sess["history"].extend([
            {"user": text},
            {"bot": response}
        ])

        return response, sess
    return None

def is_possible_surname_or_specialty(segment: str) -> bool:
    """
    Возвращает True если сегмент похож на фамилию или спец-ность врача
    (использует doctor_info.repo.read_all() для ФИО и specialties)
    """
    text = segment.strip()
    if not text or len(text.split()) > 3:
        return False
    # Проверяем заглавную букву (Фамилия) или что в базе есть такая спец-ность
    docs = doctor_info.repo.read_all()
    # Проверка ФИО
    for d in docs:
        if text.lower() in d.get('fio', '').lower().split():
            return True
        if text.lower() == (d.get('specialization') or '').lower():
            return True
    return False

async def process_segments(text: str, sess: SessionType) -> str:
    segments = await split_into_segments(text, sess)
    responses: List[str] = []

    for idx, segment in enumerate(segments, 1):
        labels = await classify(segment, sess)
        main_labels = [lbl for lbl in labels if lbl in LABEL_PRIORITY]

        print("=" * 45)
        print(f"PART view #{idx}: ", segment or "Empty PART")
        print("LABELS: ", labels)
        print("=" * 45)

        # Fallback если это возможно фамилия или специальность
        if is_possible_surname_or_specialty(segment):
            print(f"  [Force doctor_info fallback] Отправляю сегмент напрямую в doctor_info: {segment}")
            response, continue_pending = await get_doc_info_from_api(segment, session=sess)
            responses.append(response)
            continue

        if not main_labels:
            print(f"  [Fallback] Отправляю сегмент напрямую в doctor_info: {segment}")
            response, continue_pending = await get_doc_info_from_api(segment, session=sess)
            responses.append(response)
            continue

        for label in main_labels:
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
    print("Финальный ответ процессора сегментов: ", final_answ)
    print("=" * 45)
    return final_answ

async def routing(text: str,
                  sess: SessionType | None = None,
                  extra_processing: Literal["direct", "processed"] = "processed",
                  ) -> AsyncGenerator[RoutingResult, None]:
    """
    Обработка входящего запроса пользователя идет в следующем направлении:
    1. split_into_segments - ПЕРЕФОРМУЛИРОВКА и разбивка на смысловые сегменты.
    2.  <- process_segments <- classify маркировка сегментов запроса пользователя.
    3. handle_pending_module и process_segments - заключительный шаг обработки модулями извлечения информации
    и передача сообщения в данную функцию.

    Here is variant of routing() that *yields* (partial_answer, session) pairs,
        so that the outer UI can stream them.

    :param extra_processing: определяется, будет ли использоваться на выходе
    постобработка входящих данных с помощью функции final_answering либо же
    данные из БД будут выводиться напрямую
    :param text: Сообщение пользователя
    :param sess: словарь состояния сессии, хранит pending-модуль и историю
    :return: ответ, обновлённая сессия
    """

    # 1. Инициализируем состояние сессии обработки входящего текстового блока

    sess = sess or {}
    sess.setdefault("pending", None)
    sess.setdefault("history", [])

    # Handle pending module if exists
    # TODO: Понять зачем вообще это тут вызывается
    if pending_result := await handle_pending_module(text, sess):
        yield pending_result

    # Process text segments
    result = await process_segments(text, sess)

    # Update history
    sess["history"].extend([
        {"user": text},
        {"bot": result}
    ])

    if extra_processing == "processed":
        async for partial in final_answering(text, result):
            yield partial, sess  # Stream final response V1 with processing by final_answering func.
    else:
        yield result, sess  # Stream final response V2 without handling by final_answering func.


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
    async for partial, sess in routing("Смирнова"):
        print(partial)  # или обновлять UI


if __name__ == "__main__":
    asyncio.run(main())