"""Аудит «одна формулировка» (STEP-2026-10-16-ONE-PHRASING): одно определение вместо копий, поведение не меняется.

Источник — аудит 14.10 (`docs/HANDOFF_2026-10-14_cloud_L14.md`, «Правила, привязанные к конкретной формулировке»):
одно и то же правило распознавания жило в двух местах, правку одной копии вторая не видит.

Здесь характеризация групп, которые сводятся к одному определению (старые копии заморожены и сравниваются с
новыми на одном наборе реплик до и после правки), и закрепление «как есть» групп, которые НЕ сводятся, потому что
копии различаются по поведению (решение владельца):

1. Выбор акции из показанного списка — «2», «про 3», «расскажи номер 5». Узнавали две копии:
   `classifier._PROMO_PICK_TEXT_RE` (продолжить NEWS) и регулярка-литерал в `news._resolve_pick_index` (номер из
   цифр). Цифровая ветка — та же регулярка; сведено к одной (`news._PROMO_PICK_TEXT_RE`, номер — группа).
2. «Нет» на вопрос бота — `policies._NO_RE` и `policies._CONFIRM_NO_EXACT` в `_is_negative_text`: регулярка — копия
   множества (те же 10 фраз, множество проверяется после нормализации и шире по знакам и кавычкам). Сведено к
   множеству. Единственное расхождение — девять архаичных церковнославянских глифов U+1C80–U+1C88, которые `re.I`
   приравнивает к «в», «т», «о»… а `str.lower` нет («неᲄ»): пациент их не набирает; закреплено тестом.

Не сводится (копии различаются, закреплено как есть):
- порядковые акции: классификатор пускает основы `втор\\w+`, а `news._ORDINAL_WORDS` знает только часть форм —
  «вторым», «первой», «пятое», «про третьей» продолжают NEWS, но номер не находится → «Не нашёл акцию…»;
- «все» в акциях: классификатор — вся реплика (`покажи(те) все (акции)`), сервис — слово «все» где угодно;
- «да»: `_YES_RE` узнаёт «наверное», множество `_CONFIRM_YES_EXACT` — нет (и знает «согласна»);
- «ладно» / «нет»: мягкая пауза записи (`_APPOINTMENT_SOFT_PAUSE_EXACT`) против «да»/«нет» на подтверждении;
- «только город»: решение владельца 15.10 — не объединять (здесь только закреплено расхождение).
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import classifier, flow_policy, policies
from messengers_router.appointment_flow_guard import is_appointment_soft_pause_request
from messengers_router.mess_types import AppointmentPhase, DialogState, SessionState
from messengers_router.recovery_policy import contextual_reply_kind
from messengers_router.services import _prices_helpers, news
from tests.test_price_one_or_many_class import _DEFAULT, _DEPLOY
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура модуля

# ============================================================================ 1. выбор акции

# --------------------------------------------------------------------------- старые определения (9c2e7db), не править

_OLD_PROMO_FOLLOWUP_TEXT_RE = re.compile(
    r"^\s*(?:уточни(?:те)?|подробнее|расскажи(?:те)?(?:\s+подробнее)?|условия(?:\s+акции)?|давай(?:те)?"
    r"|(?:покажи(?:те)?\s+)?вс[её](?:\s+акции)?)\s*[!.,?]*\s*$",
    re.I,
)
_OLD_PROMO_PICK_TEXT_RE = re.compile(
    r"^\s*(?:расскажи(?:те)?\s+)?(?:про\s+|номер\s+)?"
    r"(?:\d{1,2}|перв\w+|втор\w+|трет\w+|четв[её]рт\w+|пят\w+|шест\w+|седьм\w+|восьм\w+)"
    r"\s*[!.,?]*\s*$",
    re.I,
)
_OLD_PROMO_QUESTION_TEXT_RE = re.compile(
    r"\b(?:до\s+какого|когда\s+заканчива|какой\s+срок|сколько\s+(?:ещё\s+|еще\s+)?действ"
    r"|срок\w*\s+действ|какие\s+услови|что\s+за\s+услови|кому\s+подход|для\s+кого)\w*",
    re.I,
)
_OLD_ORDINAL_WORDS = {
    "первый": 1, "первая": 1, "первое": 1, "первую": 1, "первом": 1,
    "второй": 2, "вторая": 2, "второе": 2, "вторую": 2,
    "третий": 3, "третья": 3, "третье": 3, "третью": 3,
    "четвертый": 4, "четвертая": 4, "четвертую": 4,
    "пятый": 5, "пятая": 5, "пятую": 5,
    "шестой": 6, "шестая": 6, "шестую": 6,
    "седьмой": 7, "седьмая": 7, "седьмую": 7,
    "восьмой": 8, "восьмая": 8, "восьмую": 8,
}


def _old_norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower().replace("ё", "е")).strip()


def _old_resolve_pick_index(query: str, entities: dict[str, Any]) -> int | None:
    pick = entities.get("promo_pick_index")
    if isinstance(pick, int) and pick > 0:
        return pick
    m = re.fullmatch(
        r"\s*(?:расскажи(?:те)?\s+)?(?:про\s+|номер\s+)?(\d{1,2})\s*[!.,?]*\s*",
        str(query or ""),
        flags=re.I,
    )
    if m:
        return int(m.group(1))
    for word, idx in _OLD_ORDINAL_WORDS.items():
        if re.search(rf"\b{word}\b", _old_norm(query)):
            return idx
    return None


def _old_promo_followup_entities(text: str, last_entities: dict[str, Any]) -> dict[str, Any] | None:
    ctx = last_entities.get("_promo_context")
    titles = ctx.get("titles") if isinstance(ctx, dict) else None
    if not isinstance(titles, list) or not titles:
        return None
    if str(last_entities.get("_last_label") or "").strip().upper() != "NEWS":
        return None
    raw = str(text or "").strip()
    if not raw or len(raw) > 60:
        return None
    ctx_mode = str(ctx.get("mode") or "") if isinstance(ctx, dict) else ""
    passthrough = {
        "promo_query": raw,
        "promo_query_for": raw,
        "_promo_context": {"titles": [str(t) for t in titles]},
    }
    if _OLD_PROMO_FOLLOWUP_TEXT_RE.fullmatch(raw) or _OLD_PROMO_PICK_TEXT_RE.fullmatch(raw):
        return passthrough
    if ctx_mode == "detail" and len(titles) == 1 and _OLD_PROMO_QUESTION_TEXT_RE.search(raw):
        return {
            "promo_query": str(titles[0]),
            "promo_query_for": raw,
            "_promo_context": {"titles": [str(titles[0])], "mode": "detail"},
        }
    tokens = re.findall(r"[a-zа-яе0-9]{3,}", raw.lower().replace("ё", "е"))
    if tokens and len(tokens) <= 4:
        for title in titles:
            norm_title = str(title).lower().replace("ё", "е")
            if all(tok in norm_title for tok in tokens):
                return passthrough
    return None


# --------------------------------------------------------------------------- реплики

_PICK_CORES = (
    # цифры: одна-две узнаются, три — нет; не-ASCII цифры (`\d` в Юникоде) — тоже цифры
    "0", "1", "2", "7", "9", "10", "12", "99", "100", "123", "02", "２", "٣", "²",
    # обвязка номера
    "про 2", "номер 3", "расскажи про 5", "расскажите номер 12", "расскажи 7", "расскажите 8", "про  2", "номер2",
    "про2", "расскажи2", "№2", "2й", "2-я", "#2", "2.", "2)", "2 акция", "акция 2", "вторая акция", "давай 2",
    # порядковые: формы из словаря сервиса и вне его
    "первый", "первая", "первое", "первую", "первом", "первой", "первым",
    "второй", "вторая", "второе", "вторую", "вторым", "второму", "втор",
    "третий", "третья", "третье", "третью", "третьей", "четвертый", "четвёртую", "четвертое",
    "пятый", "пятое", "пятой", "шестой", "седьмую", "восьмая", "восьмом", "девятая",
    "про вторую", "про третьей", "номер первый", "расскажи про первую", "расскажите про восьмую",
    "давай вторую", "расскажи о второй", "а вторая?", "вторую пожалуйста",
    # смесь цифр и порядковых
    "перв2", "пятый2", "второй2", "втор2",
    # соседи из других веток правила
    "все", "всё", "все акции", "покажите всё", "а все?", "давайте все", "уточни", "подробнее", "а подробнее?",
    "условия акции", "до какого числа действует", "какие условия", "чекап", "витамин", "сколько стоит чекап",
    # латиница внутри, чужое
    "vtoraya", "2nd", "two", "вcе", "да", "нет", "",
)
_PADS = ("", " ", "\t", "\n", " ", " ", "​")
_TAILS = ("", "!", "?", ".", ",", "...", "!?", " !", "!!!", ")", "…", ";")
_CASES = (str, str.upper, str.capitalize)


def _pick_replies() -> list[Any]:
    out: list[Any] = [None, ""]
    for core in _PICK_CORES:
        for case in _CASES:
            for pad in _PADS:
                for tail in _TAILS:
                    out.append(f"{pad}{case(core)}{tail}{pad}")
    return out


PICK_REPLIES = _pick_replies()

_TITLES = ["Чекап для мужчин", "Витамин D со скидкой", "Ферритин и железо", "Приём терапевта"]
PROMO_CONTEXTS: dict[str, dict[str, Any]] = {
    "list_after_news": {"_last_label": "NEWS", "_promo_context": {"titles": _TITLES, "mode": "list"}},
    "detail_after_news": {"_last_label": "NEWS", "_promo_context": {"titles": _TITLES[:1], "mode": "detail"}},
    "list_other_label": {"_last_label": "PRICE", "_promo_context": {"titles": _TITLES, "mode": "list"}},
    "no_context": {"_last_label": "NEWS"},
}

# Что узнаётся сейчас (до и после — одинаково): продолжает ли классификатор NEWS и какой номер берёт сервис.
PICK_AUDIT: dict[str, tuple[bool, int | None]] = {
    "2": (True, 2),
    "Про 2!": (True, 2),
    "расскажи номер 3": (True, 3),
    "12": (True, 12),
    "123": (False, None),
    "вторую": (True, 2),
    "четвёртую": (True, 4),
    # расходятся копии «порядковое»: классификатор пускает, сервис номер не находит → «Не нашёл акцию…»
    "вторым": (True, None),
    "первой": (True, None),
    "пятое": (True, None),
    "про третьей": (True, None),
    # пробелы аудита 14.10 (п. 9): правило классификатора их не пускает (сервис номер бы нашёл — поиском слова)
    "давай вторую": (False, 2),
    "расскажи о второй": (False, 2),
    "а подробнее?": (False, None),
}


# --------------------------------------------------------------------------- эквивалентность по местам вызова


@pytest.mark.parametrize("context", sorted(PROMO_CONTEXTS))
def test_promo_classifier_site_recognizes_exactly_as_before(context):
    """Классификатор (`rule_news_promo_followup`): новое = старое на каждой реплике в каждом контексте."""

    entities = PROMO_CONTEXTS[context]
    diff = [r for r in PICK_REPLIES if classifier._promo_followup_entities(r, entities) != _old_promo_followup_entities(r, entities)]
    assert diff == []


@pytest.mark.parametrize("entities", [{}, {"promo_pick_index": 3}, {"promo_pick_index": 0}, {"promo_pick_index": "2"}])
def test_promo_service_pick_index_exactly_as_before(entities):
    """Сервис акций (`news._resolve_pick_index`): номер — тот же на каждой реплике."""

    diff = [r for r in PICK_REPLIES if news._resolve_pick_index(r, entities) != _old_resolve_pick_index(r, entities)]
    assert diff == []
    picked = {r for r in PICK_REPLIES if _old_resolve_pick_index(r, {}) is not None}
    assert picked and len(picked) < len(PICK_REPLIES)  # набор проверяет обе стороны


def test_promo_pick_on_unicode_spaces_as_before():
    spaces = [chr(i) for i in range(sys.maxunicode + 1) if chr(i).isspace() or re.fullmatch(r"\s", chr(i))]
    entities = PROMO_CONTEXTS["list_after_news"]
    for space in spaces:
        for reply in (f"{space}2", f"2{space}", f"про{space}2", f"номер{space}{space}3!{space}", f"{space}вторую{space}"):
            assert news._resolve_pick_index(reply, {}) == _old_resolve_pick_index(reply, {}), repr(reply)
            assert classifier._promo_followup_entities(reply, entities) == _old_promo_followup_entities(reply, entities), repr(reply)


def test_classifier_has_no_copy_of_the_promo_pick_rule(monkeypatch):
    """Одно определение: классификатор спрашивает сервис акций, своей регулярки выбора у него нет."""

    entities = PROMO_CONTEXTS["list_after_news"]
    assert not hasattr(classifier, "_PROMO_PICK_TEXT_RE")
    assert classifier._promo_followup_entities("абракадабра", entities) is None
    monkeypatch.setattr(news, "_PROMO_PICK_TEXT_RE", re.compile("абракадабра|(7)"))
    assert classifier._promo_followup_entities("абракадабра", entities) is not None
    assert classifier._promo_followup_entities("2", entities) is None
    assert news._resolve_pick_index("7", {}) == 7 and news._resolve_pick_index("2", {}) is None


def test_promo_audit_cases_are_pinned_as_is():
    entities = PROMO_CONTEXTS["list_after_news"]
    got = {r: (classifier._promo_followup_entities(r, entities) is not None, news._resolve_pick_index(r, {})) for r in PICK_AUDIT}
    assert got == PICK_AUDIT


# --------------------------------------------------------------------------- через эндпоинт, два хода

_SETS = {"default": _DEFAULT, "deploy": _DEPLOY}
_PROMOS = [
    {"id": 11 + i, "title": title, "subtitle": "", "text": f"Описание акции {i + 1}.", "endDate": None, "startDate": None, "regions": [1]}
    for i, title in enumerate(_TITLES)
]
_REGIONS = [
    {"id": 1, "name": "Все", "parent": None},
    {"id": 2, "name": "Самарская область", "parent": 1},
    {"id": 3, "name": "Самара", "parent": 2},
]


@pytest.fixture(scope="module")
def promo_world(mini_prod_world, tmp_path_factory) -> Path:  # noqa: F811
    root = tmp_path_factory.mktemp("one_phrasing") / "world"
    shutil.copytree(mini_prod_world, root)
    (root / "captured").mkdir(exist_ok=True)
    (root / "captured" / "promotions.json").write_text(json.dumps(_PROMOS, ensure_ascii=False), encoding="utf-8")
    (root / "captured" / "regions.json").write_text(json.dumps(_REGIONS, ensure_ascii=False), encoding="utf-8")
    return root


def _dialog(turns: list[str], session_id: str, seed: SessionState | None = None) -> list[dict[str, Any]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    async def run() -> list[dict[str, Any]]:
        memory, services = MemoryStore(), Services()
        if seed is not None:
            seed.session_id = session_id
            memory.set(seed)
        out = []
        for text in turns:
            req = MessengerGenerateRequest(session_id=session_id, text=text, debug=True, button_id="")
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            decision = ((body.get("state_update") or {}).get("debug") or {}).get("decision") or {}
            out.append({
                "text": body.get("text", ""),
                "label": decision.get("label"),
                "flags": set(decision.get("flags") or []),
                "handoff": bool(body.get("handoff")),
            })
        return out

    return asyncio.run(run())


def _play(turns: list[str], config: str, monkeypatch, world_dir: Path, seed: SessionState | None = None) -> list[dict[str, Any]]:
    from messengers_router.runtime_config import config as cfg

    for key, value in _SETS[config].items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    with ev_world.FrozenWorld(world_dir, "day") as world:
        record = _dialog(turns, f"one-phrasing-{config}-{abs(hash((tuple(turns), seed is None)))}", seed)
        assert list(world.net.attempts) == []
    return record


@pytest.mark.parametrize("config", sorted(_SETS))
@pytest.mark.parametrize(
    ("reply", "card"),
    [("2", _TITLES[1]), ("Про 2!", _TITLES[1]), ("расскажи номер 3", _TITLES[2]), ("про вторую", _TITLES[1]), ("4", _TITLES[3])],
)
def test_promo_pick_through_endpoint_opens_the_card(reply, card, config, promo_world, monkeypatch):
    """Класс: после списка акций номер в любой узнаваемой форме открывает карточку этой акции (правило, без LLM)."""

    first, second = _play(["какие акции есть", reply], config, monkeypatch, promo_world)
    assert first["label"] == "NEWS" and all(t in first["text"] for t in _TITLES)
    assert second["label"] == "NEWS"
    assert "rule_news_promo_followup" in second["flags"]
    assert second["text"].startswith(f"Акция «{card}»")


@pytest.mark.parametrize("config", sorted(_SETS))
def test_promo_ordinal_outside_service_forms_is_a_miss_as_is(config, promo_world, monkeypatch):
    """Расхождение копий «порядковое» (решение владельца): «вторым» продолжает NEWS, но сервис номер не находит."""

    second = _play(["какие акции есть", "вторым"], config, monkeypatch, promo_world)[1]
    assert "rule_news_promo_followup" in second["flags"]
    assert second["text"].startswith("Не нашёл акцию по запросу «вторым»")


# ============================================================================ 2. «нет» на вопрос бота

_OLD_NO_RE = re.compile(
    r"^\s*(нет|неа|не\s+подтверждаю|не\s+подтверждаем|неверно|не\s+верно|не\s+надо|не\s+нужно|"
    r"неправильно|не\s+правильно)\s*[!.,?;:]?\s*$",
    re.I,
)
_OLD_CONFIRM_NO_EXACT = {
    "нет", "неа", "неверно", "не верно", "неправильно", "не правильно", "не надо", "не нужно", "не подтверждаю",
    "не подтверждаем",
}


def _old_is_negative_text(text: str) -> bool:
    raw = str(text or "")
    if _OLD_NO_RE.match(raw):
        return True
    norm = policies._normalize_confirmation_text(raw)
    if not norm:
        return False
    if norm in _OLD_CONFIRM_NO_EXACT:
        return True
    if norm.startswith("не подтвержда"):
        return True
    return False


_NO_CORES = (
    *sorted(_OLD_CONFIRM_NO_EXACT),
    "не  надо", "не\tнадо", "не\nверно", "не нужно", "нет нет", "нет, спасибо", "нет, не надо", "да нет", "неа нет",
    "не подтверждаю запись", "не подтвержда", "не подтверждаю!!", "нёт", "нет)", "(нет)", "«нет»", "\"не надо\"",
    "нетт", "н е т", "неверное", "не верю", "не", "ни", "да", "ага", "ладно", "наверное", "согласна", "отмена",
    "нет?", "нет.", "нет!?", "нет;", "нет:", "net", "no", "",
)
_NO_TAILS = ("", "!", "?", ".", ",", ";", ":", "!!", "?!", "...", ")", " !", "! ", "!\n", "…")


def _no_replies() -> list[Any]:
    out: list[Any] = [None, ""]
    for core in _NO_CORES:
        for case in (str, str.upper, str.capitalize, str.title):
            for pad in _PADS:
                for tail in _NO_TAILS:
                    out.append(f"{pad}{case(core)}{tail}{pad}")
    return out


NO_REPLIES = _no_replies()

# Глифы, которые `re.I` приравнивает к строчной кириллице, а `str.lower` — нет (U+1C80–U+1C88: «ᲀ»≈«в», «ᲂ»≈«о»,
# «ᲄ»/«ᲅ»≈«т»…). Пациент их не набирает; это единственное место, где `_NO_RE` видел больше множества.
_FOLD_ONLY_GLYPHS = frozenset(chr(c) for c in range(0x1C80, 0x1C89))


def test_negative_text_exactly_as_before():
    """`_is_negative_text` (через него — `contextual_reply_kind`, подтверждение записи, все вопросы бота «да/нет»):
    новое = старое на каждой реплике набора."""

    diff = [r for r in NO_REPLIES if policies._is_negative_text(r) != _old_is_negative_text(r)]
    assert diff == []
    negative = {r for r in NO_REPLIES if _old_is_negative_text(r)}
    assert negative and len(negative) < len(NO_REPLIES)


def test_negative_text_on_unicode_spaces_and_case_as_before():
    spaces = [chr(i) for i in range(sys.maxunicode + 1) if chr(i).isspace() or re.fullmatch(r"\s", chr(i))]
    for space in spaces:
        for reply in (f"{space}нет", f"нет{space}", f"не{space}надо", f"НЕ{space}{space}ВЕРНО!{space}", f"нет!{space}"):
            assert policies._is_negative_text(reply) == _old_is_negative_text(reply), repr(reply)


def test_no_rule_casefold_differs_from_lower_only_on_archaic_glyphs():
    """Где `re.I` (старая регулярка) видит букву фразы «нет», а `str.lower` (множество) — нет: только U+1C80–U+1C88."""

    letters = sorted({ch for phrase in _OLD_CONFIRM_NO_EXACT for ch in phrase if ch != " "})
    any_letter = re.compile("[" + "".join(letters) + "]", re.I)
    fold_only = {
        chr(c)
        for c in range(sys.maxunicode + 1)
        if any_letter.fullmatch(chr(c)) and chr(c).lower().replace("ё", "е") not in letters
    }
    assert fold_only <= _FOLD_ONLY_GLYPHS


def test_policies_has_no_regex_copy_of_no():
    """Одно определение «нет»: регулярки-копии множества нет; архаичный глиф («неᲄ») больше не «нет» — единственное
    известное расхождение со старым кодом (пациент такого не набирает)."""

    assert not hasattr(policies, "_NO_RE")
    assert _old_is_negative_text("неᲄ") and not policies._is_negative_text("неᲄ")
    assert all(policies._is_negative_text(phrase) for phrase in policies._CONFIRM_NO_EXACT)


NO_AUDIT: dict[str, tuple[str, str]] = {
    # реплика: (contextual_reply_kind, appointment_confirmation_transition)
    "нет": ("no", "no"),
    "Нет!": ("no", "no"),
    "не надо": ("no", "no"),
    "НЕ  ВЕРНО.": ("no", "no"),
    "«нет»": ("no", "no"),
    "не подтверждаю запись": ("no", "no"),
    "нет, спасибо": ("other", "other"),
    "нет, не надо": ("other", "other"),
    "да нет": ("other", "other"),
    "наверное": ("yes", "other"),
}


def test_no_audit_cases_are_pinned_as_is():
    got = {r: (contextual_reply_kind(r), policies.appointment_confirmation_transition(r)) for r in NO_AUDIT}
    assert got == NO_AUDIT


def _confirm_step() -> SessionState:
    return SessionState(
        session_id="one-phrasing-confirm",
        history=[
            {"role": "user", "text": "запишите к терапевту на 7 октября в 10"},
            {"role": "assistant", "text": "Подтвердите запись, пожалуйста: ответьте «да» или «нет»."},
        ],
        last_entities={
            "city": "Самара",
            "appointment_flow_active": True,
            "appointment_confirm_pending": True,
            "specialty": "терапевт",
            "branch_name": "пр. Ленина, 5",
            "date_from": "2026-10-07",
            "time_from": "10:00",
            "patient_name": "Иванов Иван",
        },
        dialog=DialogState(label="APPOINTMENT", phase=AppointmentPhase.CONFIRM),
    )


@pytest.mark.parametrize("config", sorted(_SETS))
@pytest.mark.parametrize("reply", ["нет", "Нет!", "не надо", "НЕ  ВЕРНО.", "неправильно", "«нет»"])
def test_no_on_appointment_confirm_through_endpoint(reply, config, mini_prod_world, monkeypatch):  # noqa: F811
    """Класс: «нет» в любой узнаваемой форме на «Подтвердите запись» — отказ от времени (переспрос даты), не заявка."""

    (turn,) = _play([reply], config, monkeypatch, mini_prod_world, _confirm_step())
    assert turn["label"] == "APPOINTMENT"
    assert "appointment_confirm_no" in turn["flags"]
    assert turn["handoff"] is False
    assert "уточните новую дату и время" in turn["text"]


# ============================================================================ 3. не сводится — закреплено как есть


def test_yes_copies_differ_as_is():
    """`_YES_RE` и `_CONFIRM_YES_EXACT` — не копии: «наверное» только в регулярке, «согласна» — только в множестве."""

    assert policies._YES_RE.match("наверное") and "наверное" not in policies._CONFIRM_YES_EXACT
    assert not policies._YES_RE.match("согласна") and "согласна" in policies._CONFIRM_YES_EXACT


@pytest.mark.parametrize(("reply", "soft_pause", "kind"), [("ладно", True, "yes"), ("нет", True, "no"), ("Нет!", True, "no")])
def test_soft_pause_and_yes_no_differ_as_is(reply, soft_pause, kind):
    """«ладно» / «нет»: мягкая пауза при сборе данных записи и «да» / «нет» на вопросе бота — разные смыслы по шагу."""

    assert is_appointment_soft_pause_request(reply) is soft_pause
    assert contextual_reply_kind(reply) == kind


@pytest.mark.parametrize(
    ("reply", "price_tool", "flow"),
    [("Самара", True, True), ("г. Самара", True, True), ("в Самаре", False, True), ("Самара пожалуйста", False, True),
     ("в Тольятти", False, True), ("я из Самары", False, False), ("ферритин", True, False)],
)
def test_city_only_copies_differ_as_is(reply, price_tool, flow):
    """«Только город» — решение владельца 15.10: не объединять. Две копии расходятся (закреплено как есть)."""

    assert _prices_helpers._is_city_only_reply(reply) is price_tool
    assert flow_policy._is_city_only_reply(reply) is flow
