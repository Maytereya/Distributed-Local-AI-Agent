from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

from ollama import AsyncClient, Options

import config as c
from agent_logic_2 import llama_func_call as doctor_info

# LLM‑клиент для классификации
ollama = AsyncClient(c.ollama_url)
llm = "llama3.3:70b-instruct-q8_0"
options = Options(temperature=0, top_k=1, top_p=0.1, stop=["<|eot_id|>"])

# Label‑ы и порядок
LABEL_PRIORITY = ["INFO", "PREP", "APPOINTMENT", "ISSUES", "GENERAL"]
ALLOWED = set(LABEL_PRIORITY + ["UNDEFINED"])

LABEL_DOC = """
1. INFO – справка: врачи, услуги, цены, расписание
2. PREP – правила подготовки к анализам
3. APPOINTMENT – запись на приём
4. ISSUES – жалобы, претензии, угрозы
5. GENERAL – адреса, время работы, small‑talk
6. UNDEFINED – не распознан

ВАЖНО: Метка DOC_INFO недопустима! Используйте INFO для запросов о врачах и услугах.
"""

EXAMPLES = """
INPUT: Сколько стоит приём кардиолога?            
OUTPUT: {\"labels\":[\"INFO\"]}

INPUT: Как подготовиться к анализу крови?        
OUTPUT: {\"labels\":[\"PREP\"]}

INPUT: Запишите меня к терапевту завтра утром.    
OUTPUT: {\"labels\":[\"APPOINTMENT\"]}

INPUT: Вы плохо взяли кровь, огромный синяк!      
OUTPUT: {\"labels\":[\"ISSUES\"]}

INPUT: Как доехать до клиники на автобусе?           
OUTPUT: {\"labels\":[\"GENERAL\"]}

INPUT: Подготовка к УЗИ и запишите к УЗИсту.      
OUTPUT: {\"labels\":[\"PREP\",\"APPOINTMENT\"]}

INPUT: Где принимает доктор Иванов?               
OUTPUT: {\"labels\":[\"INFO\"]}

INPUT: Какие услуги предоставляет клиника?        
OUTPUT: {\"labels\":[\"INFO\"]}
"""


def _prompt(user: str) -> str:
    today = datetime.now().strftime("%d %B %Y, %H:%M:%S")
    compilation = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Сегодня: {today}. 
Ты – ассистент клиники «Наука».
Верни **только JSON** вида {{\"labels\":[…]}} (можно несколько label‑ов).
Допустимые label‑ы:\n{LABEL_DOC}\n
[EXAMPLES]\n{EXAMPLES}\n
[USER] {user}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    print("----------------- COMPILATION -----------------------")
    print(f"FIRST_prompt_function about LABELS: {compilation}")
    print("-----------------------------------------------------")
    return compilation


def _split_prompt(text: str) -> str:
    compilation = f"""
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
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    print("----------------- COMPILATION -----------------------")
    print(f"SECOND compilation_split_prompt about reformulate and splitting into segments: {compilation}")
    print("-----------------------------------------------------")
    return compilation


async def classify(text: str) -> List[str]:
    res = await ollama.generate(model=llm,
                                prompt=_prompt(text),
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


async def split_into_segments(text: str) -> List[str]:
    print("\n================= SPLIT PROCESS START =================")
    print(f"Original input: '{text}'")

    res = await ollama.generate(model=llm,
                                prompt=_split_prompt(text),
                                options=options,
                                format="json",
                                keep_alive=-1)

    try:
        print(f"\nRaw model response: {res['response']}")

        segments = json.loads(res["response"]).get("segments", [])
        print(f"Raw segments: {segments}")

        splitted_segments = [s.strip() for s in segments if s.strip()]

        print("\nProcessing results:")
        print(f"• Segments count: {len(splitted_segments)}")
        print(f"• Final segments: {splitted_segments}")
        print("================= SPLIT PROCESS END =================\n")

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


# ──────────────────────────────────────────────────────
# Подключаем doctor_info из llama_test_bench_func_call
# ──────────────────────────────────────────────────────

async def info_handle(text: str, **_) -> Tuple[str, bool]:
    raw = await doctor_info.investigate(text)
    captured = doctor_info.re_capture(raw)
    if isinstance(captured, str):
        return captured, False
    formatted = await doctor_info.formulate(captured)
    return formatted, False


# ────────────────────────────────────────────────
# Заглушки для остальных интентов
# ────────────────────────────────────────────────

async def prep_stub(_text: str, **__) -> Tuple[str, bool]:
    return "prep_stub Правила подготовки пока в разработке.", False


async def appointment_stub(_text: str, **__) -> Tuple[str, bool]:
    return "appointment_stub Модуль записи к врачу скоро появится. Сообщите дату, и мы свяжемся!", False


async def general_stub(_text: str, **__) -> Tuple[str, bool]:
    return "general_stub Клиника \"Наука\" работает ежедневно с 8:00 до 20:00. Адреса: …", False


async def issues_stub(_text: str, **__) -> Tuple[str, bool]:
    return "issues_stub Очень жаль, что возникла проблема. Ваше сообщение передано администратору.", False


# ────────────────────────────────────────────────
# Собственно, роутер пошел. Со всеми наворотами.
# Типа оператора присваивания ":=" и тд
# ────────────────────────────────────────────────

# Ярлыки для вызова функций обработки данных после роутинга
MODULES = {
    "INFO": info_handle,
    "PREP": prep_stub,
    "APPOINTMENT": appointment_stub,
    "ISSUES": issues_stub,
    "GENERAL": general_stub,
}


async def routing(text: str, sess: Dict[str, Any] | None = None) -> Tuple[str, Dict[str, Any]]:

    sess = sess or {}  # Если сессия (sess) не передана — создаётся пустая

    # Если в сессии есть "pending" — значит, предыдущий модуль запросил продолжение.
    # Тогда имидиэтли отправляем текущий текст в соответствующий модуль, не классифицируя заново.

    # pending = sess.get("pending")
    if (pending := sess.get("pending")) and pending in MODULES:  # тут присвоение и условный оператор в одном
        ans, need = await MODULES[pending](text, session=sess)
        sess["pending"] = pending if need else None
        return ans, sess

    # Разбиваем текст на смысловые сегменты
    segments = await split_into_segments(text)
    print("\nСегменты:", segments)

    replies: List[str] = []
    # Для каждого сегмента:
    for part in segments:
        # 	1.	Вызываем classify — получаем метки.
        labels = await classify(part)
        print(f"\nМетки для сегмента '{part}':", labels)
        # 	2.	Сортируем их по приоритету (INFO, PREP, …).
        labels.sort(key=LABEL_PRIORITY.index)
        # 	3.	Отправляем сегмент в соответствующий модуль:
        for lab in labels:
            rep, need = await MODULES[lab](part, session=sess)
            replies.append(rep)
            if need:
                sess["pending"] = lab
                break

    result = "\n\n— — —\n\n".join(replies)
    print("\nИтоговый ответ:", result)
    return result, sess


if __name__ == "__main__":
    asyncio.run(
        routing("""
        Хочу записаться к доктору Смирновой на завтра и узнать, 
        как подготовиться к УЗИ, если к нему вообще надо готовиться, 
        и ещё скажите, сколько стоит приём у Белохвостиковой и сколько стоит УЗИ печени
        """))

#  Условный БАГ: если передать по INFO запрос "Сколько стоит узи печени",
#  то LLama может найти рандомного УЗИста в картотеке врачей
#  и запросить информацию из Базы данных по нему.
