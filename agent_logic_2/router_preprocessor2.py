from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

from ollama import AsyncClient, Options

from agent_logic_2 import llama_func_call_3_1 as doctor_info, config as c
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


async def final_answering(primary_request: str, collected_info: str) -> str:
    prompt = f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Ты – ассистент колл‑центра многопрофильной клиники «Наука».
            Всегда отвечай на русском языке, выводя **все** строки из секций ниже.  
            Ничего не резюмируй, не скрывай, не перефразируй – копируй дословно.

            
            Дано:
            ◆ USER  – исходный запрос клиента.  
            ◆ DATABASE – данные, найденные в БД.
            
            ТВОЯ ЗАДАЧА (выполняй последовательно):
            1. Извлеки из DATABASE все нужные поля.  
            2. Если поле присутствует – выведи его; если отсутствует – напиши «нет данных».  
            3. Не придумывай информации, которой нет.  
            4. Не предлагай позвонить, уточнить, проконсультироваться у врача – выводи всё, что доступно.
    
            
            ### Формат ответа про врача (используй ровно его, не выдумывай новых полей!)
            
            **Информация о враче **
            
            **ФИО врача: {{NAME}}** ...
            **Общая специализация:** {{SPECIALIZATION_BRIEFLY}} ... 
            **Специализация подробно:** {{SPECIALIZATION_FULL}} ...
            **Заметка колл‑центра:** {{CALL-CENTER}} ...
            **Стоимость: ** {{PRICE}} ...
            **График приёма:** {{TIMETABLE}}...
            **Адрес/адреса работы:** {{ADDRESS}} ...
            
            ### Формат ответа про подготовку к исследованию или процедуре
            
            **Подготовка к исследованию/процедуре** {{PREPARATION}} … (полный текст, или применимый фрагмент, 
            если предоставленная тебе информация содержит как прямой ответ, так и смежные темы, не имеющие отношения к запросу пользователя.)
            Если в предоставленной информации нет ничего подходящего, сообщи, что релевантных данных не найдено.
            
            Учти!
            — Жирный шрифт только для подписи полей.  
            — Если в запросе упомянуты сразу несколько тем (врач + цена УЗИ + подготовка к УЗИ), выведи блоки один за другим.
            
            ### Примеры
            
            *Пример 1*  
            USER: «Сколько стоит приём у Ивановой?»  
            → см. формат выше.  
            
            *Пример 2*  
            USER: «Как подготовиться к гастроскопии?»  
            → выводится только блок «Подготовка…».
            
            ### Данные
            USER: {primary_request}
            
            DATABASE: {collected_info}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
    answer = await ollama.generate(
        model=llm,
        prompt=prompt,
        options=options,
        keep_alive=-1
    )
    return answer["response"]


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


async def routing(text: str, sess: Dict[str, Any] | None = None) -> Tuple[str, Dict[str, Any]]:
    """
    Главная точка входа в роутинг.
    :param text: Сообщение пользователя
    :param sess: словарь состояния сессии, хранит pending-модуль и историю
    :return: ответ, обновлённая сессия
    """
    # 1. Инициализируем состояние сессии обработки входящего текстового блока
    sess = sess or {}
    sess.setdefault("pending", None)
    sess.setdefault("history", [])

    print("Распечатка sess после .setdefault: ", sess)

    # 2. Если находим команду на продолжение — сразу обрабатываем
    if (pending := sess.get("pending")) and pending in MODULES:
        print("=============================================")
        print("pending view: ", pending or "Empty")
        print("=============================================")
        answer, need = await MODULES[pending](text, session=sess)
        sess["pending"] = pending if need else None

        # Записываем в историю запрос и ответ
        sess["history"].append({"user": text})
        sess["history"].append({"bot": answer})

        return answer, sess

    # 3. Иначе — разбиваем на сегменты и классифицируем
    segments = await split_into_segments(text, sess)
    replies: List[str] = []

    i = 0
    for part in segments:
        labels = await classify(part, sess)
        labels.sort(key=LABEL_PRIORITY.index)
        i += 1
        print("=============================================")
        print(f"PART view #{i}: ", part or "Empty PART")
        print("=============================================")

        for lab in labels:
            rep, need = await MODULES[lab](part, session=sess)
            replies.append(rep)
            if need:
                sess["pending"] = lab
                print("=============================================")
                print("need pending view: ", sess["pending"] or "Empty need")
                print("=============================================")
                break

    # 4. Собираем итоговый ответ и обновляем историю
    result = "\n\n— — —\n\n".join(replies)

    sess["history"].append({"user": text})
    sess["history"].append({"bot": result})

    # print("Результат сборки итогового ответа: \r", result)

    final = await final_answering(text, result)

    # 5. Возвращаем ответ и состояние
    return final, sess


if __name__ == "__main__":
    re, sess = asyncio.run(
        routing("""
        
        и ещё скажите, сколько стоит приём у Дразнина и в какое время он работает?
        
        """))
    print("Итоговый вывод: ____________________________________________________________________")
    print()
    print(re)
    print("sess['history']: ___________________________________________________________________")
    print()
    print(sess["history"])
