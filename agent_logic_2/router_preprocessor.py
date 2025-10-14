from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple, AsyncGenerator, TypeAlias, Literal, Optional

from ollama import AsyncClient

import agent_logic_2.ollama_settings as ollama_settings
from agent_logic_2 import llama_func_call as doctor_info, config as c
from agent_logic_2.llama_func_call import repo
from agent_logic_2.nayka_api.api_nayka import ensure_daily_refresh_started
from agent_logic_2.nayka_api.doctors_cc_info import get_doctors_cc_info
from agent_logic_2.prompts import load_prompt
from agent_logic_pack import meilisearch_client as meilisearch
from converters import html_cleaner

#  Initialize logging for understanding the logics of the router
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- ensure doctors repo is warmed up (file may be absent on first FILTER run) ---
async def _ensure_doctors_repo_loaded() -> None:
    try:
        # стараемся запустить планировщик ежедневного обновления (внутри активного event loop)
        try:
            ensure_daily_refresh_started()
        except Exception:
            pass
        # try fast-path
        data = repo.read_all()
        if not data:
            raise RuntimeError("doctors repo is empty")
    except Exception:
        # Fallback: обновляем репозиторий напрямую из API без участия LLM
        try:
            await repo.update(doctor_info.get_all_doctors)
        except Exception:
            pass


# LLM‑клиент для классификации входящих запросов
ollama = AsyncClient(c.ollama_url)

# -----------------------------------------------------
# СЕКЦИЯ КОНСТАНТ ЛОГИКИ РАБОТЫ РОУТЕРА
# -----------------------------------------------------

# Label‑ы и порядок в LABEL_PRIORITY:
#
# "API_INFO" - справка из API Мед.центра
# "APPOINTMENT" - назначение времени / запись на приём к врачу
# "SCRIPTS" - алгоритмы, скрипты из серии "если - то..."

LABEL_PRIORITY = load_prompt("LABEL_PRIORITY", False).split(",")
ALLOWED = set(LABEL_PRIORITY + ["UNDEFINED"])  # TODO: разобраться, не совсем понятое добавление лейбла "снаружи".
# Какой индекс MEILISEARCH открывать для конкретного лейбла:
INDEX_BY_LABEL: dict[str, str] = {
    "NEWS": "news",  # для NEWS — индекс "news"
    "SCRIPTS": "main_index",  # явно, чтоб читаемо; но по дефолту — тоже main_index
    # при необходимости добавишь другие
}
LABEL_DOC = load_prompt("LABEL_DOC", False)
RAW_MODE_MARKER = "<NO_POSTPROC>"
EXAMPLES = load_prompt("EXAMPLES", False)
ITEM_MARKER = "⊢ID:"  # маркер для компактного формата

# --- CC‑info thin process cache (по id врача) ---
_CC_INFO_MAP: Dict[int, str] | None = None


def _get_cc_info_map() -> Dict[int, str]:
    """Получает информацию о заметках call-центра по врачам."""
    global _CC_INFO_MAP
    if _CC_INFO_MAP is None:
        try:
            data = get_doctors_cc_info()
            _CC_INFO_MAP = {
                row.get('id'): row.get('callCenterInfo', '')
                for row in data if isinstance(row, dict)
            }
        except Exception:
            _CC_INFO_MAP = {}
    return _CC_INFO_MAP


def _extract_age_from_cc(cc: str) -> str:
    """Пытается вытащить возраст из заметки КЦ и вернуть нормализованную строку вида "с N лет".
    Возвращает "не указан" если не найдено.
    """
    if not isinstance(cc, str) or not cc:
        return "не указан"
    s = cc.lower()
    # Явные формулировки
    # 1) "с N(-и) лет"
    m = re.search(r"\bс\s*(\d{1,2})\s*(?:-?и)?\s*лет\b", s)
    if m:
        return f"с {m.group(1)} лет"
    # 2) "в возрасте N(-и) лет"
    m = re.search(r"\bв\s*возрасте\s*(\d{1,2})\s*(?:-?и)?\s*лет\b", s)
    if m:
        return f"с {m.group(1)} лет"
    # 3) "с возраста N(-и) лет"
    m = re.search(r"\bс\s*возраста\s*(\d{1,2})\s*(?:-?и)?\s*лет\b", s)
    if m:
        return f"с {m.group(1)} лет"
    # 4) "принимает с N(-и) лет"
    m = re.search(r"принимает\s*(пациентов\s*)?с\s*(\d{1,2})\s*(?:-?и)?\s*лет", s)
    if m:
        # номер группы может быть 1 или 2, поэтому берём первую непустую
        num = m.group(2) or m.group(1)
        return f"с {num} лет"
    # Совершеннолетние пациенты → считаем с 18 лет
    if "совершеннолет" in s:
        return "с 18 лет"
    # Иногда пишут "с 0+" или "0+"
    if "0+" in s or re.search(r"\bс\s*0\s*лет\b", s):
        return "с 0 лет"
    # Негативные формулировки дают подсказку: только взрослые → с 18 лет
    if "только взросл" in s or "взросл" in s or re.search(r"\bс\s*18\s*лет\b", s):
        return "с 18 лет"
    return "не указан"


def _bool_to_ru(v: Optional[bool]) -> str:
    return "Да" if v is True else ("Нет" if v is False else "—")


def _compact_line(d: dict) -> str:
    fio = (d.get("fio") or "").strip()
    spec = d.get("units") or d.get("specialization") or ""
    if isinstance(spec, list):
        spec = ", ".join(spec[:2])
    addr_list = d.get("regions") or []
    addr = ", ".join(addr_list[:2])

    # исходная заметка КЦ (как есть); _flag_from_cc сам приводит к lower()
    cc_text = d.get("callCenterInfo") or ""

    # нормализованные флаги через единую функцию
    arriving_flag = _flag_from_cc(cc_text, "arriving")
    dms_flag = _flag_from_cc(cc_text, "dms")
    children_flag = _flag_from_cc(cc_text, "children")

    age = _extract_age_from_cc(cc_text)

    return (
        f"{ITEM_MARKER}{d.get('id')} — {fio} — {spec} — {addr} — "
        f"ДМС: {_bool_to_ru(dms_flag)} — "
        f"Приходящий: {_bool_to_ru(arriving_flag)} — "
        f"Дети: {_bool_to_ru(children_flag)} — "
        f"Возраст: {age}"
    )


def _format_compact(matched: List[Dict[str, Any]]) -> str:
    lines = [_compact_line(d) for d in matched]
    n = len(lines)
    header = f"[SAFE_LIST]\nN={n}\nORDER=PRESERVE\nFORMAT=ECHO_ALL\n"
    return header + "\n".join(lines)


def _extract_safe_list_meta(s: str) -> Tuple[int, List[str] | None]:
    """Returns (N, ids) if [SAFE_LIST] is present, else (0, None)."""
    if "[SAFE_LIST]" not in s:
        return 0, None
    try:
        head, *_ = s.split("\n\n", 1)
        m = re.search(r"\bN=(\d+)", head)
        n = int(m.group(1)) if m else 0
        ids = re.findall(rf"{re.escape(ITEM_MARKER)}(\d+)", s)
        return n, ids
    except Exception:
        return 0, None


# =========================
# SAFE_LIST deterministic helpers
# =========================
def _parse_safe_compact_line(line: str) -> Dict[str, str]:
    """Парсит строку формата
    ⊢ID:123 — Иванов И.И. — Врач терапевт — Ленина 5 — ДМС: Да — Приходящий: Нет — Дети: Да — Возраст: с 18 лет
    и возвращает словарь полей. Не бросает исключений.
    """
    out = {"id": "", "fio": "", "spec": "", "addr": "", "dms": "—", "arriving": "—", "children": "—",
           "age": "не указан"}
    if not isinstance(line, str) or ITEM_MARKER not in line:
        return out
    try:
        head = line.split(ITEM_MARKER, 1)[1].strip()
        parts = [p.strip() for p in head.split(" — ")]
        if parts:
            out["id"] = parts[0]
        if len(parts) > 1:
            out["fio"] = parts[1]
        if len(parts) > 2:
            out["spec"] = parts[2]
        if len(parts) > 3:
            out["addr"] = parts[3]
        # Остальные метки в произвольном порядке
        tail = " — ".join(parts[4:]) if len(parts) > 4 else ""
        tl = tail.lower()
        # ДМС
        m = re.search(r"дмс:\s*(да|нет)", tl)
        if m:
            out["dms"] = "Да" if m.group(1) == "да" else "Нет"
        # Приходящий: учитываем ТОЛЬКО явные метки "да"/"нет", без эвристик
        # Если метки нет — оставляем значение по умолчанию "—" (неизвестно)
        m = re.search(r"\bприходящий:\s*(да|нет)\b", tl)
        if m:
            out["arriving"] = "Да" if m.group(1) == "да" else "Нет"
        # Дети
        m = re.search(r"дети:\s*(да|нет)", tl)
        if m:
            out["children"] = "Да" if m.group(1) == "да" else "Нет"
        # Возраст
        m = re.search(r"возраст:\s*([^—\n]+)", tail, flags=re.IGNORECASE)
        if m:
            out["age"] = m.group(1).strip()
    except Exception:
        pass
    return out


def _shorten_specialty(text: str, limit_words: int = 10) -> str:
    """Аккуратно укорачивает строку спец-сти до <= limit_words слов."""
    if not isinstance(text, str) or not text.strip():
        return "не указано"
    # иногда приходит перечисление через запятую — берём первый фрагмент
    first = text.split(";")[0].split(".")[0]
    words = [w for w in re.split(r"\s+|,\s*", first) if w]
    if len(words) <= limit_words:
        return first.strip()
    return " ".join(words[:limit_words]).strip()


def _render_numbered_list_from_safe(compact_payload: str) -> str:
    """Строит итоговый пронумерованный список в формате final_answer (без участия LLM)."""
    lines = [ln for ln in compact_payload.splitlines() if ln.strip().startswith(ITEM_MARKER)]
    blocks: List[str] = []
    for i, ln in enumerate(lines, 1):
        row = _parse_safe_compact_line(ln)
        fio = row.get("fio") or "не указано"
        spec_short = _shorten_specialty(row.get("spec") or "")
        addr = row.get("addr") or "не указан"
        arriving = row.get("arriving") or "—"
        dms = row.get("dms") or "—"
        age = row.get("age") or "не указан"
        block = (
            f"{i}. **ФИО врача:** {fio}\n"
            f"**Специализация кратко:** {spec_short}\n"
            f"**Адрес/адреса работы:** {addr}\n"
            f"**Приходящий:** {('Да!' if arriving == 'Да' else ('Нет!' if arriving == 'Нет' else 'не указан'))}\n"
            f"**Возраст пациентов:** {age}\n"
            f"**ДМС:** {('Да' if dms == 'Да' else ('Нет' if dms == 'Нет' else 'не указан'))}\n"
            "\n---\n"
        )
        blocks.append(block)
    if not blocks:
        return "Релевантной информации не найдено"
    return "\n".join(blocks) + "\n— Конец списка —"


# --- FILTER segment support (ARRIVING / CHILDREN / DMS) ---
FILTER_RE = re.compile(r'^\s*FILTER:\s*(.+)$', re.IGNORECASE)

_KEYWORDS_TRUE = {
    'dms': [
        'по дмс', 'принимает по дмс', 'принимают по дмс', 'работает по дмс', 'работают по дмс', 'есть дмс', 'дмс да'
    ],
    'children': [
        'работает с детьми', 'работают с детьми', 'принимает детей', 'принимают детей', 'детей принимает', 'детям',
        'с детьми'
    ],
    'arriving': [
        'приходящий', 'приходящие', 'разовый', 'совмещает'
    ],
}

_KEYWORDS_FALSE = {
    'dms': [
        'не по дмс', 'не принимает по дмс', 'не принимают по дмс', 'без дмс', 'дмс нет', 'нет дмс'
    ],
    'children': [
        'не работает с детьми', 'не работают с детьми', 'без детей', 'детей не принимает', 'детей не принимают',
        'только взросл'
    ],
    'arriving': [
        'не приходящий', 'не приходящие', 'неприходящий', 'неприходящие'
    ],
}


def _keyword_to_filter(segment: str) -> Dict[str, bool] | None:
    """Грубое извлечение фильтров из естественных формулировок, если LLM не вернул FILTER:"""
    if not isinstance(segment, str) or not segment.strip():
        return None
    s = segment.lower()
    out: Dict[str, bool] = {}
    # DMS — сначала проверяем отрицания, затем положительные упоминания
    # Регулярки учитывают формы "принимает/принимают", "работает/работают"
    if re.search(r"\b(не\s*(принима(ет|ют)|работа(ет|ют)|по)\s*по\s*дмс|без\s*дмс|дмс\s*нет|нет\s*дмс)\b", s):
        out['dms'] = False
    elif re.search(r"\b(принима(ет|ют)\s*по\s*дмс|работа(ет|ют)\s*по\s*дмс|по\s*дмс|дмс\s*да)\b", s):
        out['dms'] = True
    # CHILDREN
    if re.search(r"\b(не\s*работа(ет|ют)\s*с\s*детьми|без\s*детей|детей\s*не\s*принима(ет|ют)|только\s*взросл)\b", s):
        out['children'] = False
    elif any(k in s for k in _KEYWORDS_TRUE['children']):
        out['children'] = True
    # ARRIVING
    if any(k in s for k in _KEYWORDS_FALSE['arriving']):
        out['arriving'] = False
    elif any(k in s for k in _KEYWORDS_TRUE['arriving']):
        out['arriving'] = True
    # Дополнительное правило: "с N лет" → children=True (если N < 18), N>=18 → children=False
    m = re.search(r"\bс\s*(\d{1,2})\s*(?:-?и)?\s*лет\b", s)
    if m:
        try:
            n = int(m.group(1))
            if n < 18:
                out.setdefault('children', True)
            else:
                out.setdefault('children', False)
        except Exception:
            pass
    return out or None


def _safe_format(template: str, **kwargs) -> str:
    """
    Безопасно подставляет {placeholders} в шаблон, где также встречаются JSON-скобки.
    Экранирует все фигурные скобки, кроме известных плейсхолдеров из kwargs.
    """
    if not isinstance(template, str):
        return template
    # временно защищаем целевые плейсхолдеры
    protected: dict[str, str] = {}
    for k in kwargs.keys():
        protected[k] = f"<<__{k.upper()}__>>"
        template = template.replace("{" + k + "}", protected[k])
    # экранируем все остальные скобки
    template = template.replace("{", "{{").replace("}", "}}")
    # возвращаем плейсхолдеры и форматируем
    for k, marker in protected.items():
        template = template.replace(marker, "{" + k + "}")
    return template.format(**kwargs)


# --- robust JSON extraction without PCRE recursion ---

def _extract_json_object(text: str):
    """
    Достаёт первый валидный JSON-объект { ... } из текста.
    Учитывает строки и экранирование, не бросает исключений.
    """
    if not text:
        return None

    s = text
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]

        if esc:
            esc = False
            continue

        if ch == "\\":
            esc = True
            continue

        if ch == '"':
            in_str = not in_str
            continue

        if not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = s[start:i + 1]
                    try:
                        return json.loads(chunk)
                    except Exception:
                        # мягкая попытка почистить лишние переводы строк/пробелы
                        cleaned = " ".join(chunk.split())
                        try:
                            return json.loads(cleaned)
                        except Exception:
                            return None
    return None


# ----------------------------------------------
# Функция обработки истории запросов
# и ответов на них
# ----------------------------------------------

def _history_reveal(user: str, sess: Dict[str, Any]) -> str:
    """
    Processes the session history to generate a formatted string for revealing the
    interaction history. This function excludes the last user query in the history if it
    matches the current user to avoid duplication and uses a specific format for
    presentation.

    :param user: The identifier for the current user.
    :type user: str
    :param sess: The session dictionary containing interaction history. The expected key
        in the dictionary is "history" with a list of dialogue turns.
    :type sess: Dict[str, Any]
    :return: A formatted string representing the interaction history, where each turn
        is labeled as either "User:" or "Assistant:" followed by the content.
    :rtype: str
    """
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

def classificator_prompt(text_user: str, sess: Dict[str, Any]) -> str:
    today = datetime.now().strftime("%d %B %Y, %H:%M:%S")
    template = load_prompt("classificator_prompt", False)

    return _safe_format(
        template,
        today=today,
        user=text_user,
        sess=_history_reveal(text_user, sess),  # обрабатывается история диалога с пользователем
        LABEL_DOC=LABEL_DOC,
        EXAMPLES=EXAMPLES,
    )


def split_prompt(text_user: str, sess: Dict[str, Any]) -> str:
    template = load_prompt("split_prompt", False)

    return _safe_format(
        template,
        user=text_user,
        sess=_history_reveal(text_user, sess),  # обрабатывается история диалога с пользователем
    )


async def split_into_segments(text: str, sess: Dict[str, Any], think: bool = None) -> List[str]:
    """
    Важно! Функция переформулирует запрос!
    Это первая функция в каскаде обработки входящего сообщения пользователя.

    :param think: Включать ли reasoning у поддерживающих моделей
    :param text: Входящий сырой запрос.
    :param sess:
    :return: Возвращает список запросов от пользователя.
    """
    think = ollama_settings.resolve_think(think)
    # print("!!!THINK:", think)
    ollama_settings.init_model_name()
    # print("split_into_segments OLLAMA_MODEL:", ollama_settings.OLLAMA_MODEL)
    # print("split_into_segments options: ", ollama_settings.options_set())
    if not ollama_settings.OLLAMA_MODEL:
        raise ValueError("split_into_segments OLLAMA_MODEL cannot be empty")

    # Защищаемся от долгого ответа модели: ограничиваем время ожидания
    try:
        res = await asyncio.wait_for(
            ollama.generate(
                model=ollama_settings.OLLAMA_MODEL,
                prompt=split_prompt(text, sess),  # Добавляется sess
                options=ollama_settings.options_set(),
                format="json",
                keep_alive=-1,
                think=think,
            ),
            timeout=20,
        )
    except Exception as e:
        print(f"\n⚠️ split_into_segments timeout/error: {e}")
        return [text]

    try:
        obj = _extract_json_object(res.get("response"))
        segments = []
        if isinstance(obj, dict):
            raw = obj.get("segments")
            if isinstance(raw, list):
                segments = [str(s).strip() for s in raw if str(s).strip()]
        # Fallback: если модель не выполнила контракт — считаем весь текст одним сегментом
        if not segments:
            segments = [text]

        print("\nProcessing results:")
        print(f"• Segments count: {len(segments)}")
        print(f"• Final segments: {segments}")
        return segments

    except Exception as e:
        print(f"\n⚠️ Unexpected error in split_into_segments: {e}")
        print("⚠️ Returned original text as single segment")
        return [text]


async def classify(text: str, sess: Dict[str, Any], think: bool = None) -> List[str]:
    """
    Классификатор сегментов входящего запроса пользователя,
    второй этап обработки входящего сообщения пользователя.

    :param think: Включать ли reasoning у поддерживающих моделей
    :param text: Фрагмент (выделенный предыдущей функцией) входящего текста для анализа и маркировки.
    :param sess:
    :return:
    """

    # print("classify OLLAMA_MODEL:", ollama_settings.OLLAMA_MODEL)
    # print("classify options: ", ollama_settings.options_set())
    think = ollama_settings.resolve_think(think)
    # print("!!!THINK:", think)
    if not ollama_settings.OLLAMA_MODEL:
        raise ValueError("classify OLLAMA_MODEL cannot be empty")

    try:
        res = await asyncio.wait_for(
            ollama.generate(
                model=ollama_settings.OLLAMA_MODEL,
                prompt=classificator_prompt(text, sess),
                options=ollama_settings.options_set(),
                format="json",
                keep_alive=-1,
                think=think,
            ),
            timeout=20,
        )
    except Exception as e:
        print(f"\n⚠️ classify timeout/error: {e}")
        return ["UNDEFINED"]
    try:
        obj = _extract_json_object(res.get("response"))
        labels_raw: list[str] = []
        if isinstance(obj, dict):
            raw = obj.get("labels")
            if isinstance(raw, list):
                labels_raw = raw
        labels = [str(l).upper() for l in labels_raw if str(l).upper() in ALLOWED]
        # print("----------------- LABELS -----------------------")
        # print(f"Маркировано labels: ", labels)
        # print("------------------------------------------------")
        return labels or ["UNDEFINED"]
    except Exception as e:
        print(f"\n Classificator error: {e}")
        return ["UNDEFINED"]


async def final_answering(primary_request: str,
                          collected_info: str,
                          think: bool = None,
                          ):  # Пока неясно что за тип данных будет возвращаться

    template = load_prompt("final_answer", False)
    prompt = _safe_format(
        template,
        primary_request=primary_request,
        collected_info=collected_info,
    )

    # print("final_answering OLLAMA_MODEL:", ollama_settings.OLLAMA_MODEL)
    # print("final_answering options: ", ollama_settings.options_set())
    if not ollama_settings.OLLAMA_MODEL:
        raise ValueError("final_answering OLLAMA_MODEL cannot be empty")
    think = ollama_settings.resolve_think(think)
    # print("!!!THINK:", think)

    partial = ""  # накопитель
    stream = await ollama.generate(
        model=ollama_settings.OLLAMA_MODEL,
        prompt=prompt,
        options=ollama_settings.options_set(),
        keep_alive=-1,
        stream=True,
        think=think,
    )
    async for chunk in stream:
        partial += chunk["response"]
        yield partial


# ──────────────────────────────────────────────────────
# Подключаем doctor_info из llama_func_call
# ──────────────────────────────────────────────────────

async def get_doc_info_from_api(question: str, think: bool = None, **_, ) -> Tuple[str, bool]:
    think = ollama_settings.resolve_think(think)
    # print("!!!THINK:", think)
    result = await doctor_info.investigate(question, think=think)
    return result, False


# ------------------------------------------------------
# Подключаем заглушку функции записи пациента
# ------------------------------------------------------
async def appointment_stub(_text: str, think: bool = None, **__) -> Tuple[str, bool]:
    think = ollama_settings.resolve_think(think)
    # print("!!!THINK:", think)
    return "Модуль записи к врачу скоро появится. ", False


# ──────────────────────────────────────────────────────
# Search with MEILISEARCH function
# (может использовать LLM переформулировку)
# ──────────────────────────────────────────────────────
async def instructions_search(_text: str, think: bool = None, index: str = "main_index", **__) -> Tuple[str, bool]:
    """
    Для поиска нужной информации в индексе или коллекции используется переформулировка запроса пользователя
    Пока неясно, следует ли ее делать.
    :param index:
    :param think:
    :param _text:
    :param __:
    :return: Кортеж: результат поиска и стоп - паттерн для PENDING
    """
    # Временно отключу переформулировку!
    # extracted_keyword = await formulate.extract_keyword(_text, extract_type="sentence")

    # print("=" * 45)
    # print("Экстрагировалось: ",
    #       # extracted_keyword
    #       _text  # шарахнем запрос напрямую без переформулировки
    #       or "Empty")
    # print("=" * 45)

    collected_info = await asyncio.to_thread(meilisearch.search_meili, index, _text)

    # Очистка HTML перед подстановкой в prompt
    clean_info = html_cleaner.strip_html(collected_info)
    # Добавление маркера для лучшего распознавания LLM
    marked_info = "{KNOWLEDGE_SNIPPET}" + "\n" + clean_info
    # print("=" * 45)
    # print(marked_info)
    # print("=" * 45)
    return marked_info, False


# ────────────────────────────────────────────────
# Основная логика роутера / Router main logic
# ────────────────────────────────────────────────

# Ярлыки для вызова функций обработки данных после роутинга
# дополнительная секция констант для работы логики (не может быть обозначена вверху модуля, так как содержит в себе
# объявление функций для вызова

# MODULES = {
#     "API_INFO": get_doc_info_from_api,
#     "APPOINTMENT": appointment_stub,
#     "SCRIPTS": instructions_search,
# }

# Нужно промежуточное извлечение, так как строка не может содержать вызова функции
MODULES_str: str = load_prompt("MODULES", False)

# Шаблон: "КЛЮЧ": ИМЯ_ФУНКЦИИ
pattern = r'"(?P<key>[^"]+)":\s*(?P<name>\w+)'
pairs = re.findall(pattern, MODULES_str)

# Берём функции из globals() (или из MODULE) и тут уже обращение к функциям
MODULES: Dict[str, Any] = {}
for key, name in pairs:
    func = globals().get(name)
    if callable(func):
        MODULES[key] = func
    else:
        print(f"[router_preprocessor] ⚠️ MODULES: функция '{name}' для ключа '{key}' не найдена")

# Дополнительные обозначения типов для понимания вывода
# Данные по переменной sess:
SessionType: TypeAlias = Dict[str, Any]
RoutingResult: TypeAlias = Tuple[str, SessionType]

# Константа для удобства форматирования
SEGMENT_SEPARATOR = "\n\n— — —\n\n"


async def handle_pending_module(text: str, sess: SessionType, think: bool | None = None) -> RoutingResult | None:
    if (pending_module := sess.get("pending")) and pending_module in MODULES:
        print("=" * 45)
        print("pending_module content: ", pending_module or "Empty")
        print("=" * 45)

        kwargs = {"session": sess, "think": think}
        idx_name = INDEX_BY_LABEL.get(pending_module)
        if idx_name:
            kwargs["index"] = idx_name

        response, continue_pending = await MODULES[pending_module](text, **kwargs)
        sess["pending"] = pending_module if continue_pending else None

        sess["history"].extend([
            {"user": text},
            {"bot": response}
        ])

        return response, sess
    return None


async def is_possible_surname_or_specialty(segment: str) -> bool:
    """
    Возвращает True если сегмент похож на фамилию или спец-ность врача
    (использует repo.read_all() для ФИО и specialties)
    """
    text = segment.strip()
    if not text or len(text.split()) > 3:
        return False
    # ensure repo is ready (creates data file on first use)
    await _ensure_doctors_repo_loaded()
    docs = repo.read_all()
    # Проверка ФИО и специальности
    tl = text.lower()
    for d in docs:
        if tl in d.get('fio', '').lower().split():
            return True
        if tl == (d.get('specialization') or '').lower():
            return True
    return False


# --- FILTER segment support (ARRIVING / CHILDREN / DMS) ---

_DEF_TRUE = {"yes", "да", "true", "1"}
_DEF_FALSE = {"no", "нет", "false", "0"}


def _parse_filters(expr: str) -> Dict[str, bool]:
    # "ARRIVING=YES; CHILDREN=NO; DMS=YES" -> {"arriving": True, "children": False, "dms": True}
    pairs = [p.strip() for p in expr.split(';') if p.strip()]
    out: Dict[str, bool] = {}
    for p in pairs:
        if '=' not in p:
            continue
        k, v = [t.strip().lower() for t in p.split('=', 1)]
        if v in _DEF_TRUE:
            out[k] = True
        elif v in _DEF_FALSE:
            out[k] = False
    return out


def _flag_from_cc(cc: str, key: str) -> Optional[bool]:
    if not cc:
        return None
    s = cc.lower()
    if key == 'arriving':
        if 'не приход' in s or 'неприход' in s:
            return False
        if 'приходящ' in s:
            return True
        return None
    if key == 'children':
        # Явные отрицания/только взрослые/совершеннолетние
        if 'с 18 лет' in s or 'принимает с 18' in s or 'только взросл' in s or 'взросл' in s or 'совершеннолет' in s:
            return False
        # Явные указания работы с детьми
        if 'дет' in s and ('работ' in s or 'принимает' in s):
            return True
        # Возрастной признак: если указан возраст начала приёма < 18 — считаем, что работает с детьми
        m = re.search(r"\bс\s*(\d{1,2})\s*лет\b", s)
        if m:
            try:
                n = int(m.group(1))
                if n < 18:
                    return True
                else:
                    return False
            except Exception:
                pass
        return None
    if key == 'dms':
        if 'не принимает по дмс' in s or 'дмс: нет' in s or 'дмс — нет' in s or 'дмс - нет' in s:
            return False
        if 'по дмс' in s or 'дмс: да' in s or 'дмс — да' in s or 'дмс - да' in s:
            return True
        return None
    return None


def _norm_text(s: str) -> str:
    return (s or "").lower()


def _extract_specialty_terms_from_segment(segment: str, docs: List[Dict[str, Any]]) -> List[str]:
    """
    Пытается извлечь возможные ключевые слова специальности из сегмента.
    Возвращает только те токены, которые встречаются хотя бы в одной специализации/юните врача.
    """
    if not isinstance(segment, str) or not segment.strip():
        return []
    seg = _norm_text(segment)
    seg = re.sub(r"[,.!?;:()\[\]{}]", " ", seg)
    tokens = [t for t in re.split(r"\s+", seg) if t]
    # Уберём частые служебные слова и слова фильтров
    stop = set(getattr(doctor_info, 'STOP_WORDS', set())) | {
        'дмс', 'страховка', 'страховой', 'страховая', 'по', 'с', 'без',
        'дети', 'детям', 'детей', 'взрослые', 'взрослый', 'приходящий', 'приходящие', 'неприходящий',
        'принимает', 'работает', 'где', 'кто', 'список', 'нужен', 'ищу'
    }
    tokens = [t for t in tokens if t not in stop and len(t) >= 3]

    # Морфологическая нормализация (простая): множественное → единственное
    norm_tokens: List[str] = []
    for t in tokens:
        try:
            # используем имеющуюся нормализацию из doctor_info, если доступна
            if hasattr(doctor_info, 'normalize_specialty_term'):
                nt = doctor_info.normalize_specialty_term(t) or t
            else:
                nt = t[:-1] if (len(t) > 4 and (t.endswith('и') or t.endswith('ы'))) else t
        except Exception:
            nt = t
        norm_tokens.append(nt)

    # Базовые синонимы под подстроки, встречающиеся в наших данных
    synonyms = {
        'лор': 'отоларинголог',
        'узи': 'ультразвуков',  # покроет и "врач ультразвуковой диагностики"
        'узист': 'ультразвуков',
    }
    expanded_tokens: List[str] = []
    for t in norm_tokens:
        expanded_tokens.append(t)
        if t in synonyms:
            expanded_tokens.append(synonyms[t])

    if not expanded_tokens:
        return []

    # Построим текст для поиска по каждому врачу: specialization + units
    texts = []
    for d in docs:
        spec = _norm_text(d.get('specialization') or '')
        units = ", ".join(d.get('units') or [])
        texts.append(spec + " " + _norm_text(units))

    # Оставляем только те термины, которые где-то реально встречаются
    valid_terms = []
    for t in expanded_tokens:
        if any(t in txt for txt in texts):
            valid_terms.append(t)
    # Уберём дубликаты, сохранив порядок
    seen = set()
    uniq_terms = []
    for t in valid_terms:
        if t not in seen:
            seen.add(t)
            uniq_terms.append(t)
    return uniq_terms


async def _filter_doctors_via_cc_info(
        filters: Dict[str, bool],
        segment: str | None = None,
        fallback_segment: str | None = None,
) -> str:
    """Вернёт отформатированный список врачей, удовлетворяющих фильтрам, а также (если удаётся распознать)
    специальности из сегмента. Если в текущем сегменте спец‑термины не найдены — пробуем извлечь их из
    полного исходного запроса (fallback_segment).
    """
    await _ensure_doctors_repo_loaded()
    try:
        cc_by_id = _get_cc_info_map()
    except Exception as e:
        return f"Релевантной информации не найдено (ошибка загрузки заметок колл-центра: {e})."

    docs = repo.read_all()
    # Попробуем аккуратно вытащить ключевые слова специальности из сегмента
    spec_terms: List[str] = []
    if segment:
        try:
            spec_terms = _extract_specialty_terms_from_segment(segment, docs)
        except Exception:
            spec_terms = []
    # Fallback: если не нашли в текущем сегменте, попробуем во всём тексте запроса
    if not spec_terms and fallback_segment and fallback_segment != segment:
        try:
            spec_terms = _extract_specialty_terms_from_segment(fallback_segment, docs)
        except Exception:
            spec_terms = []
    require_spec = bool(spec_terms)
    matched: List[Dict[str, Any]] = []

    for d in docs:
        did = d.get('id')
        cc_text = cc_by_id.get(did, '')
        ok = True
        # 1) Специальность/направление (если распознан термин спец-сти в сегменте)
        if require_spec:
            units_text = _norm_text(" ".join(d.get('units') or []))
            spec_text = _norm_text(d.get('specialization') or '')

            def _match_in_units(term: str) -> bool:
                return term in units_text

            def _match_in_spec(term: str) -> bool:
                if not spec_text:
                    return False
                # требуем соседство с "врач" в пределах 25 символов в любую сторону
                pattern = rf"(врач[^\n\r\-,:;]{{0,25}}{re.escape(term)})|({re.escape(term)}[^\n\r\-,:;]{{0,25}}врач)"
                return re.search(pattern, spec_text) is not None

            if not (any(_match_in_units(t) for t in spec_terms) or any(_match_in_spec(t) for t in spec_terms)):
                ok = False
        # 2) Флаги FILTER (ДМС/дети/приходящий)
        for k, desired in filters.items():
            key = k.lower()
            if key.startswith('arriv') or key.startswith('приход'):
                key = 'arriving'
            elif key.startswith('child') or key.startswith('дет'):
                key = 'children'
            elif key in ('dms', 'дмс'):
                key = 'dms'
            else:
                continue
            val = _flag_from_cc(cc_text, key)
            if val is None or val != desired:
                ok = False
                break
        if ok:
            dd = dict(d)
            dd['callCenterInfo'] = cc_text or 'Нет заметок'
            matched.append(dd)

    if not matched:
        return "Релевантной информации не найдено"

    full_text = doctor_info.format_documents(matched)
    compact = _format_compact(matched)
    return compact + "\n\n[RAW_FULL]\n" + full_text


async def process_segments(text: str, sess: SessionType, think: bool | None = None) -> str:
    segments = await split_into_segments(text, sess, think)
    responses: List[str] = []

    for idx, segment in enumerate(segments, 1):
        # SAFETY NET: если LLM не вернул FILTER:, но сегмент выглядит как фильтр — обрабатываем правилами
        kw_filters = _keyword_to_filter(segment)
        if kw_filters:
            print("[KEYWORD→FILTER] parsed:", kw_filters)
            response = await _filter_doctors_via_cc_info(kw_filters, segment=segment, fallback_segment=text)
            responses.append(response)
            continue
        # 0) FILTER: ... → обрабатываем напрямую, не ломая существующие ветки
        m = FILTER_RE.match(segment)
        if m:
            flt = _parse_filters(m.group(1))
            print("[FILTER] parsed:", flt)
            response = await _filter_doctors_via_cc_info(flt, segment=segment, fallback_segment=text)
            responses.append(response)
            continue

        # Быстрый путь: если это похоже на фамилию/спец-сть — сразу идём в doctor_info,
        # чтобы не ждать классификатор LLM (который может зависнуть)
        try:
            if await is_possible_surname_or_specialty(segment):
                print(f"  [Force doctor_info fallback EARLY] Отправляю сегмент напрямую в doctor_info: {segment}")
                response, continue_pending = await get_doc_info_from_api(segment, session=sess, think=think)
                # Добавим RAW-маркер, чтобы миновать postprocessing LLM
                responses.append(RAW_MODE_MARKER + "\n" + response)
                continue
        except Exception as e:
            print(f"[EARLY fallback check error]: {e}")

        labels = await classify(segment, sess, think)
        main_labels = [lbl for lbl in labels if lbl in LABEL_PRIORITY]

        # print("=" * 45)
        # print(f"PART view #{idx}: ", segment or "Empty PART")
        # print("LABELS: ", labels)
        # print("=" * 45)

        # Fallback если это возможно фамилия или специальность (второй шанс)
        try:
            if await is_possible_surname_or_specialty(segment):
                print(f"  [Force doctor_info fallback] Отправляю сегмент напрямую в doctor_info: {segment}")
                response, continue_pending = await get_doc_info_from_api(segment, session=sess, think=think)
                responses.append(RAW_MODE_MARKER + "\n" + response)
                continue
        except Exception as e:
            print(f"[Fallback check error]: {e}")

        if not main_labels:
            print(f"  [Fallback] Отправляю сегмент напрямую в doctor_info: {segment}")
            response, continue_pending = await get_doc_info_from_api(segment, session=sess, think=think)
            responses.append(RAW_MODE_MARKER + "\n" + response)
            continue

        for label in main_labels:
            # базовые аргументы в модуль
            kwargs: Dict[str, Any] = {"session": sess, "think": think}

            # NEW: подставляем имя индекса, если задано для этого лейбла
            idx_name = INDEX_BY_LABEL.get(label)
            if idx_name:
                kwargs["index"] = idx_name
            #
            logger.info("Подставленный индекс по имени:", idx_name)
            #
            # вызов соответствующего обработчика
            response, continue_pending = await MODULES[label](segment, **kwargs)
            #
            logger.info("Ответ meilisearch при поисковом запросе:", response)
            #
            responses.append(response)

            if continue_pending:
                sess["pending"] = label
                break  # выход из цикла по main_labels, чтобы дождаться продолжения pending-модуля

    final_answ = SEGMENT_SEPARATOR.join(responses)
    # print("=" * 45)
    # print("Финальный ответ процессора сегментов: ", final_answ)
    # print("=" * 45)
    return final_answ


#  Модуль - пример для сохранения единообразия:
# async def _call_module(label: str, text: str, sess: SessionType, think: bool | None):
#     return await MODULES[label](text, session=sess, think=think)
#

async def routing(text: str,
                  sess: SessionType | None = None,
                  extra_processing: Literal["direct", "processed"] = "processed",
                  think: bool = None,
                  ) -> AsyncGenerator[RoutingResult, None]:
    """
    Обработка входящего запроса пользователя идет в следующем направлении:
    1. split_into_segments - ПЕРЕФОРМУЛИРОВКА и разбивка на смысловые сегменты.
    2.  <- process_segments <- classify маркировка сегментов запроса пользователя.
    3. handle_pending_module и process_segments - заключительный шаг обработки модулями извлечения информации
    и передача сообщения в данную функцию.

    Here is variant of routing() that *yields* (partial_answer, session) pairs,
        so that the outer UI can stream them.

    :param think: Включает Reasoning у поддерживающей его модели
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
    if pending_result := await handle_pending_module(text, sess, think=think):
        yield pending_result

    # Process text segments
    result = await process_segments(text, sess, think=think, )

    # ==== SAFE_LIST path (детерминированный рендер без изменения final_answer) ====
    N, ids = _extract_safe_list_meta(result)
    if N and ids:
        compact, raw_full = (result.split("\n\n[RAW_FULL]\n", 1) + [""])[:2] if "\n\n[RAW_FULL]\n" in result else (
            result, "")
        # Рендерим сами (без LLM), чтобы не терять позиции и не зависеть от final_answer
        formatted = _render_numbered_list_from_safe(compact)
        sess["history"].extend([{"user": text}, {"bot": formatted}])
        yield formatted, sess
        return

    # Update history
    sess["history"].extend([
        {"user": text},
        {"bot": result}
    ])

    # ⬇️ новый быстрый выход для «сырых» (готовых) результатов
    if isinstance(result, str) and (RAW_MODE_MARKER in result):
        # отдаём как есть, без постпроцесса (final_answering)
        cleaned = result.replace(RAW_MODE_MARKER, "").lstrip("\n\r ")
        yield cleaned, sess
        return

    if extra_processing == "processed":
        async for partial in final_answering(text, result, think=think):
            yield partial, sess  # Stream final response V1 with processing by final_answering func.
    else:
        yield result, sess  # Stream final response V2 without handling by final_answering func.


async def process_routing_request(query: str, think: bool | None = None) -> Tuple[str, Dict[str, Any]]:
    """
    Запускает маршрутизацию и собирает все части ответа из async-генератора,
    возвращая финальную строку и итоговую сессию. Нужно чисто для тестирования данного модуля
    """
    final_response: str = ""
    final_session: dict[str, Any] = {}

    # routing возвращает AsyncGenerator[(partial_response, session), None]
    async for partial, sess in routing(query.strip(), think=think, ):
        # на каждой итерации приходят (partial, sess)
        final_response = partial  # перезаписываем — в итоге останется последний
        final_session = sess

    return final_response, final_session


async def main():
    async for partial, sess in routing("Смирнова", think=False, ):
        print(partial)  # или обновлять UI


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]).strip() or input("Введите запрос: ").strip()
    final_resp, final_sess = asyncio.run(process_routing_request(query, think=False))
    print(final_resp)
    # print(classificator_prompt("-сообщение пользователя-",
    #                            {"history": [{"user": "-содержимое памяти-",
    #                                          "bot": "_невнятное сообщение ассистента_", }, ]}))
    # print()
    # print("ALLOWED: ", ALLOWED)
    # print()
    # print("MODULES: ", MODULES)
