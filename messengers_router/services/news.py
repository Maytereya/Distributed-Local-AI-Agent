"""Модуль домена акций клиники (NEWS-интент).

До 2026-07-10 источником был meili-индекс ``news`` с почтовыми дайджестами
(«НОВОСТИ ЗА 05.11») — пациент получал бессодержательный список. Теперь
источник — CRM ``/promotions`` (обёртка ``api_nayka.site_promotions``,
TTL-кэш справочника): настоящие названия, условия, сроки.

Фильтрация:
- по сроку: ``endDate``/``startDate`` — ISO-datetime с TZ (endDate=2026-12-30
  T20:00+00:00 = 31.12 00:00 по Самаре, т.е. «до 31 декабря»);
- по региону: ``regions`` — узлы дерева /regions (1=«Все», 2=«Самарская
  область», 3=«Самара», потомки 3 = филиалы). Бот самарский → показываем
  акции, чьи регионы пересекаются с {узел «Самара» + его предки + потомки}.
  Пенза/Нефтегорск-only скрываются (у них свои цены — показать самарцу чужую
  цену = дезинформация). Если /regions недоступен — консервативно оставляем
  только явное «Все» (id=1) и акции без регионов.

Режимы ответа (``mode`` в payload): ``list`` (какие акции есть), ``detail``
(нашли конкретную по названию/номеру), ``miss`` (честный промах поиска +
актуальный список), ``absent`` (под ``promo_by_llm``: о чём спросили, такой акции
среди действующих нет + актуальный список). Деградация источника некритична —
пустой список без принудительного handoff (прежнее поведение).

Под ``[MESSENGER_ROUTER] promo_by_llm = on`` (STEP-2026-10-29-PROMO-BY-MEANING) какая
акция отвечает на вопрос, решает LLM закрытым выбором среди действующих акций
(``PROMO_TASK``, промпт ``choose_promo_answer``), код печатает их карточки; прежде —
счёт совпадений слов (``_score_promo``): «А на первый приём есть скидка?» → акция на
витамин D, «Что входит в акцию госпитализация?» → весь список (стенд деплоя 20: 9 из 17
провалов листа «Акции и скидки» — шаблон или чужая акция). Номер из показанного списка
и «все» — как было (структурный ответ); сбой LLM и «другая тема» — прежний путь.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from agent_logic_2.nayka_api import api_nayka

from .. import llm_runtime, turn_telemetry
from .. import resilience as _R
from ..choice import Option, Status, Task, choose
from ..runtime_config import config as _cfg
from ._common import _service_fallback

if TYPE_CHECKING:
    from .core import Services


logger = logging.getLogger(__name__)

# Самара = UTC+4: endDate приходит в UTC, «до какого числа» пациенту показываем
# по местному календарю (2026-12-30T20:00Z = 31.12 00:00 местного = «до 31.12»).
_SAMARA_UTC_OFFSET = timedelta(hours=4)

# Слова запроса, не несущие названия акции («какие акции есть сейчас?» → список).
_PROMO_STOPWORDS = {
    "акция", "акции", "акцию", "акций", "акциях", "скидка", "скидки", "скидку",
    "спецпредложение", "спецпредложения", "предложение", "предложения",
    "новости", "новость", "какие", "какая", "что", "есть", "сейчас", "теперь",
    "действует", "действуют", "актуальные", "актуальная", "расскажи",
    "расскажите", "покажи", "покажите", "подскажи", "подскажите", "уточни",
    "уточните", "подробнее", "условия", "про", "для", "как", "или", "это",
    "клиника", "клинике", "клиники", "вас", "вам", "меня", "мне", "нет",
    "под", "названием", "название", "называется", "именем", "хорошо",
    "спасибо", "пожалуйста", "здравствуйте", "добрый", "день", "интересует",
    "все", "всё",
}

_ORDINAL_WORDS = {
    "первый": 1, "первая": 1, "первое": 1, "первую": 1, "первом": 1,
    "второй": 2, "вторая": 2, "второе": 2, "вторую": 2,
    "третий": 3, "третья": 3, "третье": 3, "третью": 3,
    "четвертый": 4, "четвертая": 4, "четвертую": 4,
    "пятый": 5, "пятая": 5, "пятую": 5,
    "шестой": 6, "шестая": 6, "шестую": 6,
    "седьмой": 7, "седьмая": 7, "седьмую": 7,
    "восьмой": 8, "восьмая": 8, "восьмую": 8,
}

_PROMO_LIST_LIMIT = 8

# Какая из действующих акций отвечает на вопрос пациента (`promo_by_llm`): до двух карточек; исходы — акция, такой нет,
# общий вопрос о списке, не об акциях.
PROMO_TASK = Task(
    name="promo_answer",
    max_picks=2,
    outcomes=("promo", "absent", "list", "other_topic"),
    num_predict=48,
    timeout_s=8,
    queue_timeout_ms=3000,
)
_PROMO_OPTION_TEXT = 240  # знаков текста акции в варианте выбора: условия обычно в начале
# Выбранную акцию подтверждает отдельный короткий вопрос с полным текстом акции: чужая акция — дезинформация (карточка 29.10:
# «Почему нет сил» → «Доктор Шубин», бонусы новым клиентам → подарочные сертификаты).
PROMO_CHECK_TASK = Task(name="promo_check", outcomes=("yes", "no"), num_predict=24, timeout_s=8, queue_timeout_ms=3000)
_PROMO_CHECK_TEXT = 900  # знаков текста акции в проверке


def promo_by_llm_on() -> bool:
    """`[MESSENGER_ROUTER] promo_by_llm = on`: акцию под вопрос выбирает LLM по смыслу, не счёт совпадений слов."""

    return str(getattr(_cfg, "MR_PROMO_BY_LLM", "off") or "").strip().lower() == "on"

# Выбор акции из показанного списка: «2», «про 3», «расскажи номер 5», «вторую». Одно определение
# (STEP-2026-10-16-ONE-PHRASING): по нему классификатор продолжает NEWS (`classifier._promo_followup_entities`),
# а здесь номер из цифр — группа 1 (`_resolve_pick_index`). Порядковые слова номер берут из `_ORDINAL_WORDS`.
_PROMO_PICK_TEXT_RE = re.compile(
    r"^\s*(?:расскажи(?:те)?\s+)?(?:про\s+|номер\s+)?"
    r"(?:(\d{1,2})|перв\w+|втор\w+|трет\w+|четв[её]рт\w+|пят\w+|шест\w+|седьм\w+|восьм\w+)"
    r"\s*[!.,?]*\s*$",
    re.I,
)


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower().replace("ё", "е")).strip()


def _promo_tokens(query: str) -> list[str]:
    """Содержательные токены запроса (без стоп-слов и коротышей)."""

    tokens = re.findall(r"[a-zа-яе0-9]{3,}", _norm(query))
    return [t for t in tokens if t not in _PROMO_STOPWORDS]


_PROMO_KEYWORD_RE = re.compile(r"\b(?:акци\w*|скидк\w*|спецпредложени\w*|новост\w*)\b", re.I)


def _promo_name_candidate(query: str, *, from_followup: bool) -> str:
    """Кандидат НАЗВАНИЯ акции из запроса — класс-фикс против вежливой обвязки.

    Прод-диалог 10.07: «Хорошо, какие акции сейчас есть и скидки?» уходил в
    «Не нашёл акцию по запросу „Хорошо, …“» — «хорошо» не стоп-слово, и весь
    вопрос считался поиском. Правило класса: названием считается только то,
    что стоит ПОСЛЕ слова «акция/скидка/…» («акция почему нет сил» → «почему
    нет сил»); всё до него — обвязка. Слова «акция» нет (follow-up после
    списка) → кандидат = вся реплика. Пустой кандидат = запрос списка.

    :param query: реплика пользователя
    :param from_followup: реплика пришла из promo-follow-up правила
    :return: кандидат названия (может быть пустым)
    """

    raw = str(query or "").strip()
    matches = list(_PROMO_KEYWORD_RE.finditer(raw))
    if matches:
        return raw[matches[-1].end():].strip(" \t«»\"'.,!?—-:;")
    return raw if from_followup else ""


def _token_matches(token: str, haystack: str) -> bool:
    """Токен матчится подстрокой или общим префиксом ≥5 (падежи: витамином→витамин)."""

    if token in haystack:
        return True
    if len(token) < 5:
        return False
    prefix = token[:5]
    return any(w.startswith(prefix) for w in haystack.split())


def _parse_promo_dt(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _promo_is_active(promo: dict[str, Any], now_utc: datetime) -> bool:
    end = _parse_promo_dt(promo.get("endDate"))
    if end is not None and now_utc > end:
        return False
    start = _parse_promo_dt(promo.get("startDate"))
    if start is not None and now_utc < start:
        return False
    return True


def promo_end_display(promo: dict[str, Any]) -> str:
    """«31.12.2026» по самарскому календарю; пусто, если endDate нет/кривой."""

    end = _parse_promo_dt(promo.get("endDate"))
    if end is None:
        return ""
    return (end + _SAMARA_UTC_OFFSET).strftime("%d.%m.%Y")


def _samara_relevant_region_ids(regions: list[Any]) -> set[int] | None:
    """Узел «Самара» + предки (Самарская область, Все) + потомки (филиалы).

    :param regions: живое дерево /regions (id/parent/name)
    :return: множество релевантных id или None, если узел «Самара» не найден
             (деградация — caller фильтрует консервативно)
    """

    nodes = [r for r in regions if isinstance(r, dict) and r.get("id") is not None]
    if not nodes:
        return None
    by_id = {r["id"]: r for r in nodes}
    children: dict[Any, list[Any]] = {}
    for r in nodes:
        children.setdefault(r.get("parent"), []).append(r["id"])

    samara_ids = [r["id"] for r in nodes if _norm(r.get("name")) == "самара"]
    if not samara_ids:
        return None

    relevant: set[int] = set()
    for sid in samara_ids:
        relevant.add(sid)
        cursor = by_id.get(sid, {}).get("parent")
        hops = 0
        while cursor is not None and hops < 10:  # предки; гард от цикла в данных
            relevant.add(cursor)
            cursor = by_id.get(cursor, {}).get("parent")
            hops += 1
        stack = list(children.get(sid, []))
        while stack:  # потомки (филиалы Самары)
            node = stack.pop()
            if node in relevant:
                continue
            relevant.add(node)
            stack.extend(children.get(node, []))
    return relevant


def _promo_in_samara(promo: dict[str, Any], relevant: set[int] | None) -> bool:
    promo_regions = [r for r in (promo.get("regions") or []) if r is not None]
    if not promo_regions:
        return True  # без привязки = для всех
    if relevant is None:
        # /regions недоступен: консервативно пропускаем только явное «Все» (id=1)
        return 1 in promo_regions
    return any(r in relevant for r in promo_regions)


def _promo_public_fields(promo: dict[str, Any]) -> dict[str, Any]:
    text = str(promo.get("text") or "").strip()
    # CRM вклеивает в текст кнопку сайта «Подробнее» — пациенту это мусорный хвост.
    text = re.sub(r"\s*подробнее\W*$", "", text, flags=re.I)
    return {
        "id": promo.get("id"),
        "title": str(promo.get("title") or "").strip(),
        "subtitle": str(promo.get("subtitle") or "").strip(),
        "text": text,
        "end_display": promo_end_display(promo),
        "is_analysis": bool(promo.get("isAnalysis")),
        "is_doctor_service": bool(promo.get("isDoctorService")),
    }


def _score_promo(tokens: list[str], promo: dict[str, Any]) -> tuple[int, int]:
    """(титульные попадания, текстовые попадания) содержательных токенов."""

    title = _norm(promo.get("title")) + " " + _norm(promo.get("subtitle"))
    body = _norm(promo.get("text"))
    title_hits = sum(1 for t in tokens if _token_matches(t, title))
    text_hits = sum(1 for t in tokens if _token_matches(t, body))
    return title_hits, text_hits


def _resolve_pick_index(query: str, entities: dict[str, Any]) -> int | None:
    """Номер акции из follow-up: «2», «про вторую», promo_pick_index из правила."""

    pick = entities.get("promo_pick_index")
    if isinstance(pick, int) and pick > 0:
        return pick
    m = _PROMO_PICK_TEXT_RE.fullmatch(str(query or ""))
    if m and m.group(1):
        return int(m.group(1))
    for word, idx in _ORDINAL_WORDS.items():
        if re.search(rf"\b{word}\b", _norm(query)):
            return idx
    return None


PROMOTIONS_UNAVAILABLE_TEXT = "Не получилось загрузить всю информацию об акциях — передаю оператору."


def _promotions_unavailable(entities: dict[str, Any]) -> dict[str, Any]:
    """Сбой акций или справочника филиалов — перевод (ход (д), решение владельца 18.10)."""

    payload = _service_fallback(
        note="news source unavailable",
        handoff_message=PROMOTIONS_UNAVAILABLE_TEXT,
        entities=entities,
        extra={"news": [], "mode": "list"},
    )
    return _R.mark_degraded(payload, upstream="promotions", failure_mode=_R.FM_EXCEPTION, fallback_used=False)


async def news_info(self: "Services", query: str, entities: dict[str, Any]) -> dict[str, Any]:
    """Актуальные акции клиники из CRM /promotions: список / поиск / детали.

    :param self: экземпляр сервисного слоя
    :param query: текст запроса пользователя
    :param entities: NLU-сущности (+ promo_query/promo_pick_index из follow-up правила)
    :return: payload с ``news`` (список публичных полей акций), ``mode`` и
             ``entities_used``
    """

    _ = self
    try:
        promos_raw = await asyncio.to_thread(api_nayka.site_promotions, realtime=True)
    except Exception:
        _R.log_degraded(upstream="promotions", failure_mode=_R.FM_EXCEPTION, fallback_used=False)
        return _promotions_unavailable(entities)
    try:
        # Дерево регионов через общий TTL-кэш справочников (шарится с шапкой
        # расписания) — не ходим в CRM за /regions на каждый промо-ход.
        regions = await asyncio.to_thread(api_nayka.site_regions_cached, realtime=True)
    except Exception:
        regions = []
    if not regions:
        # Справочник филиалов пуст — это сбой (`site_regions` при сбое отдаёт []): акции, привязанные к филиалам
        # региона, отбросились бы молча — «акций нет» (DATA-14).
        _R.log_degraded(upstream="regions", failure_mode=_R.FM_EMPTY_DATA, fallback_used=False)
        return _promotions_unavailable(entities)

    now_utc = datetime.now(timezone.utc)
    relevant = _samara_relevant_region_ids(regions if isinstance(regions, list) else [])
    active = [
        p for p in (promos_raw if isinstance(promos_raw, list) else [])
        if isinstance(p, dict) and _promo_is_active(p, now_utc) and _promo_in_samara(p, relevant)
    ]
    # Дедуп по нормализованному названию: CRM ведёт дубли («Социальная скидка»
    # заведена дважды под разные регионы) — пациенту показываем одну.
    seen_titles: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for p in active:
        key = _norm(p.get("title"))
        if key and key in seen_titles:
            continue
        seen_titles.add(key)
        deduped.append(p)
    active = deduped

    # ВАЖНО: promo_query валиден только если выдан правилом ДЛЯ ЭТОГО хода
    # (маркер promo_query_for == живой текст). Merge оркестратора переносит
    # entities в состояние диалога, и протухший promo_query перебивал живые
    # запросы следующих ходов (прод-баг 10.07: после «все» все запросы про
    # акции отдавали один и тот же список). Валидный promo_query приоритетен:
    # вопросная ветка кладёт туда НАЗВАНИЕ карточки при вопросе «до какого
    # числа действует?».
    promo_query = str(entities.get("promo_query") or "")
    if promo_query and str(entities.get("promo_query_for") or "") != str(query or ""):
        promo_query = ""  # протухшее из merge прошлых ходов
    effective_query = promo_query or str(query or "")

    # «все» / «покажи все акции» — полный список без капа, БЕЗ токен-поиска
    # (прод-баг 10.07: «все» уходило в поиск и матчило «всех» в тексте
    # «Социальной скидки» → список из одной акции вместо полного).
    if re.search(r"\bвс[её]\b", _norm(effective_query)):
        return {
            "news": [_promo_public_fields(p) for p in active],
            "mode": "list",
            "total_active": len(active),
            "entities_used": entities,
        }
    list_limit = _PROMO_LIST_LIMIT

    pick = _resolve_pick_index(effective_query, entities)
    if pick is not None:
        stashed = entities.get("_promo_context")
        titles = stashed.get("titles") if isinstance(stashed, dict) else None
        if isinstance(titles, list) and 1 <= pick <= len(titles):
            wanted = _norm(titles[pick - 1])
            for p in active:
                if _norm(p.get("title")) == wanted:
                    return {
                        "news": [_promo_public_fields(p)],
                        "mode": "detail",
                        "entities_used": entities,
                    }
        # Номер без валидного контекста — отдаём список (ниже) с подсказкой.

    if promo_by_llm_on():
        by_meaning = await _promo_by_meaning(effective_query, active, entities)
        if by_meaning is not None:
            return by_meaning

    from_followup = bool(entities.get("promo_query"))
    candidate = _promo_name_candidate(effective_query, from_followup=from_followup)
    tokens = _promo_tokens(candidate)
    if tokens:
        scored = sorted(
            ((_score_promo(tokens, p), i, p) for i, p in enumerate(active)),
            key=lambda x: (-x[0][0], -x[0][1], x[1]),
        )
        best_title, best_text = scored[0][0] if scored else (0, 0)
        if best_title > 0:
            top = [p for (t, x), _i, p in scored if t == best_title and best_title > 0][:2]
            return {
                "news": [_promo_public_fields(p) for p in top],
                "mode": "detail",
                "entities_used": entities,
            }
        if best_text > 0:
            hits = [p for (t, x), _i, p in scored if x > 0][:_PROMO_LIST_LIMIT]
            return {
                "news": [_promo_public_fields(p) for p in hits],
                "mode": "list",
                "entities_used": entities,
            }
        return {
            "news": [_promo_public_fields(p) for p in active[:list_limit]],
            "mode": "miss",
            "query_echo": candidate.strip()[:80],
            "total_active": len(active),
            "entities_used": entities,
        }

    return {
        "news": [_promo_public_fields(p) for p in active[:list_limit]],
        "mode": "list",
        "total_active": len(active),
        "entities_used": entities,
    }


def _promo_option(promo: dict[str, Any]) -> str:
    """Вариант выбора: название, подзаголовок, срок и начало текста акции — как ведёт клиника в CRM."""

    fields = _promo_public_fields(promo)
    body = " ".join(fields["text"].split())[:_PROMO_OPTION_TEXT]
    head = " — ".join(x for x in (fields["title"], fields["subtitle"]) if x)
    until = f" (до {fields['end_display']})" if fields["end_display"] else ""
    return f"{head}{until}: {body}" if body else f"{head}{until}"


async def _promo_by_meaning(query: str, active: list[dict[str, Any]], entities: dict[str, Any]) -> dict[str, Any] | None:
    """Какая акция отвечает на вопрос — выбор LLM среди действующих (`PROMO_TASK`); None — прежний путь (не об акциях, сбой
    LLM, выбор без акции), и решение тогда лимит хода не тратит."""

    if not active or not str(query or "").strip():
        return None
    spent = llm_runtime.turn_decisions_spent()
    try:
        choice = await choose(PROMO_TASK, query, [Option(id=str(i), text=_promo_option(p)) for i, p in enumerate(active, 1)])
    except Exception:  # новый слой: его сбой не ломает ход — прежний путь
        logger.warning("promo_by_llm_failed", exc_info=True)
        choice = None
    outcome = choice.outcome if choice is not None and choice.status is Status.OUTCOME else ""
    picks = [active[int(i) - 1] for i in (choice.ids if choice is not None else ()) if str(i).isdigit() and 1 <= int(i) <= len(active)]
    turn_telemetry.note(promo_by_llm=outcome or "none", promo_by_llm_picks=len(picks))
    if outcome == "promo" and picks:
        picks = [p for p in picks if await _promo_answers(query, p)]
        turn_telemetry.note(promo_by_llm_checked=len(picks))
    if outcome == "promo" and picks:
        return {"news": [_promo_public_fields(p) for p in picks], "mode": "detail", "entities_used": entities}
    if outcome in ("absent", "list"):
        return {
            "news": [_promo_public_fields(p) for p in active[:_PROMO_LIST_LIMIT]],
            "mode": outcome,
            "total_active": len(active),
            "entities_used": entities,
        }
    if llm_runtime.turn_decisions_spent() > spent and llm_runtime.refund_turn_decision():
        turn_telemetry.note(tu_promo_by_llm_refund=1)
    return None


async def _promo_answers(query: str, promo: dict[str, Any]) -> bool:
    """Отвечает ли выбранная акция на вопрос — отдельный вопрос LLM с полным текстом акции (`PROMO_CHECK_TASK`); сбой — нет:
    неподтверждённую акцию не печатаем."""

    fields = _promo_public_fields(promo)
    card = "\n".join(x for x in (fields["title"], fields["subtitle"], " ".join(fields["text"].split())[:_PROMO_CHECK_TEXT]) if x)
    try:
        choice = await choose(PROMO_CHECK_TASK, query, [], context=card)
    except Exception:  # новый слой: его сбой не ломает ход — акция не подтверждена
        logger.warning("promo_check_failed", exc_info=True)
        return False
    return choice.status is Status.OUTCOME and choice.outcome == "yes"
