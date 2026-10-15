"""«Показать все» — одно определение для инструмента цены и классификатора (D5, 14.10).

Было: две копии одного правила — `_prices_helpers._PRICE_SHOW_ALL_RE` (инструмент цены раскрывает весь список
семейства) и регулярка-литерал в `classifier._is_price_family_show_all_followup` (классификатор продолжает PRICE после
показанного списка семейства). Аудит 14.10 (`docs/HANDOFF_2026-10-14_cloud_L14.md`, п. 8): копии разъезжаются —
правку одной вторая не увидит, и бот перестанет понимать собственную подсказку «напишите: "все"».

Стало: реплику узнаёт одно определение — `_prices_helpers._is_price_show_all_request` (рядом с подсказкой);
классификатор проверяет только контекст и зовёт его. Поведение не меняется: старые определения зафиксированы здесь
копией (как в `c55c02c`) и сравниваются с новыми для каждого места вызова на наборе реплик шире аудита — пробелы
Юникода (все 29 символов, которые `str.strip` и `\\s` считают пробелом), знаки, регистр, ё, города, латиница,
цифры, пустая строка, None.

Чего здесь нет: расширения узнавания («покажите все», «а все?», «давайте все», «показать всё» не узнаются ни до, ни
после) — это изменение поведения: только под выключателем и по решению владельца. Набор закреплён как есть.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import classifier
from messengers_router.services import _prices_helpers
from tests.test_price_one_or_many_class import _DEFAULT, _DEPLOY
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура модуля

# --------------------------------------------------------------------------- старые определения (c55c02c), не править

_OLD_PRICE_SHOW_ALL_RE = re.compile(
    r"^\s*(?:все|всё|покажи\s+все|показать\s+все|все\s+варианты|все\s+услуги|все\s+анализы)\s*[!.,?]*\s*$",
    re.I,
)


def _old_is_price_show_all_request(query_text: str) -> bool:
    return bool(_OLD_PRICE_SHOW_ALL_RE.fullmatch(str(query_text or "").strip()))


def _old_is_price_family_show_all_followup(text: str, last_entities: dict[str, Any]) -> bool:
    ctx = last_entities.get("_price_family_context")
    if not isinstance(ctx, dict) or not isinstance(ctx.get("family_variants"), list):
        return False
    if str(last_entities.get("_last_label") or "").strip().upper() != "PRICE":
        return False
    return bool(
        re.fullmatch(
            r"\s*(?:все|всё|покажи\s+все|показать\s+все|все\s+варианты|все\s+услуги|все\s+анализы)\s*[!.,?]*\s*",
            str(text or ""),
            flags=re.I,
        )
    )


# --------------------------------------------------------------------------- реплики

# Кейсы аудита и задачи D5 — узнаётся ли реплика сейчас (до и после — одинаково).
AUDIT: dict[str, bool] = {
    "все": True,
    "покажи все": True,
    "все варианты": True,
    "покажите все": False,
    "а все?": False,
    "давайте все": False,
    "показать всё": False,  # «всё» узнаётся только одно, после глагола — только «все»
    "ещё": False,
    "в Самаре": False,
    "я из Самары": False,
    "Самара": False,
    "Тольятти": False,
    "в Тольятти пожалуйста": False,
    "Самара, ул. Победы": False,
    "": False,
    "vse": False,
    "show all": False,
    "12345": False,
}

_CORES = (
    # узнаются
    "все", "всё", "покажи все", "показать все", "все варианты", "все услуги", "все анализы",
    # соседи, которые не узнаются
    "покажи всё", "показать всё", "покажите все", "покажите всё", "всё варианты", "все вариант", "все цены",
    "давай все", "давайте все", "а все", "и все", "да все", "хочу все", "можно все", "все давайте", "все пожалуйста",
    "покажи мне все", "покажи все варианты", "все анализы крови", "анализы все", "все-все", "все все", "все 8",
    "еще", "ещё", "остальные", "покажи остальные", "весь список", "полный список",
    "вcе", "bce", "всe",  # латинские буквы внутри
    "ферритин", "сколько стоит оак", "да", "нет", "1", "10",
    "в Самаре", "я из Самары", "Самара", "г. Самара", "город Самара", "Тольятти", "в Тольятти пожалуйста",
    "Самара, ул. Победы", "vse", "all", "show all", "12345",
)
_PADS = ("", " ", "  ", "\t", "\n", "\u00a0", "\u2003", "\u3000", "\x1c", "\u200b", "\ufeff")
_TAILS = ("", "!", "?", ".", ",", "...", "!?", "?!", " !", "! ", " ! ", "!!!", ")", " :)", "…", ";", "!\n", ".\u00a0", "\u00a0!")
_CASES = (str, str.upper, str.capitalize, str.title)


def _replies() -> list[Any]:
    out: list[Any] = [None, "", " ", "\n", "\u00a0", *AUDIT]
    for core in _CORES:
        for case in _CASES:
            for pad in _PADS:
                for tail in _TAILS:
                    out.append(f"{pad}{case(core)}{tail}{pad}")
    # внутренний разделитель многословных
    for sep in (" ", "  ", "\t", "\n", "\u00a0", "\u2003", "\u200b", "-", ", "):
        out += [f"покажи{sep}все", f"показать{sep}все", f"все{sep}варианты", f"все{sep}услуги", f"все{sep}анализы"]
    return out


REPLIES = _replies()

# Контексты места вызова классификатора: показан ли список семейства и была ли метка цены.
_FAMILY = {"service_name": "гепатит", "family_variants": [{"serviceName": "Гепатит A", "cost": 300}], "visible_limit": 10}
CONTEXTS: dict[str, dict[str, Any]] = {
    "family_after_price": {"_last_label": "PRICE", "_price_family_context": _FAMILY},
    "label_padded_lower": {"_last_label": " price ", "_price_family_context": _FAMILY},
    "variants_empty_list": {"_last_label": "PRICE", "_price_family_context": {"family_variants": []}},
    "other_label": {"_last_label": "PREPARE", "_price_family_context": _FAMILY},
    "no_label": {"_price_family_context": _FAMILY},
    "no_family": {"_last_label": "PRICE"},
    "family_not_dict": {"_last_label": "PRICE", "_price_family_context": ["x"]},
    "variants_not_list": {"_last_label": "PRICE", "_price_family_context": {"family_variants": "x"}},
}


def _unicode_spaces() -> list[str]:
    """Все символы, которые считает пробелом `str.strip` или `\\s` (сейчас — 29, множества равны)."""

    space_re = re.compile(r"\s")
    return [chr(i) for i in range(sys.maxunicode + 1) if chr(i).isspace() or space_re.fullmatch(chr(i))]


# --------------------------------------------------------------------------- эквивалентность по местам вызова


def test_price_tool_site_recognizes_exactly_as_before():
    """Инструмент цены (`price_info`, `service_bundle_info`): новое определение = старое на каждой реплике."""

    diff = [r for r in REPLIES if _prices_helpers._is_price_show_all_request(r) != _old_is_price_show_all_request(r)]
    assert diff == []
    recognized = {r for r in REPLIES if _old_is_price_show_all_request(r)}
    assert recognized and len(recognized) < len(REPLIES)  # набор проверяет обе стороны, а не одну


@pytest.mark.parametrize("context", sorted(CONTEXTS))
def test_classifier_site_recognizes_exactly_as_before(context):
    """Классификатор (`rule_price_family_show_all`): новое = старое на каждой реплике в каждом контексте."""

    entities = CONTEXTS[context]
    diff = [
        r
        for r in REPLIES
        if classifier._is_price_family_show_all_followup(r, entities) != _old_is_price_family_show_all_followup(r, entities)
    ]
    assert diff == []


def test_both_sites_agree_after_a_shown_family():
    """После показанного списка семейства оба места узнают одно и то же множество реплик."""

    entities = CONTEXTS["family_after_price"]
    assert [r for r in REPLIES if classifier._is_price_family_show_all_followup(r, entities) != _prices_helpers._is_price_show_all_request(r)] == []


def test_unicode_spaces_around_and_inside_as_before():
    """Старое классификатора — `\\s*` без `strip`, старое инструмента — `strip` и `^…$`: на каждом пробеле Юникода
    новое определение совпадает с обоими."""

    spaces = _unicode_spaces()
    assert len(spaces) >= 25
    entities = CONTEXTS["family_after_price"]
    for space in spaces:
        for reply in (f"{space}все", f"все{space}", f"{space}все!{space}", f"покажи{space}все", f"все{space}{space}?", f"все!{space}!"):
            new = _prices_helpers._is_price_show_all_request(reply)
            assert new == _old_is_price_show_all_request(reply), repr(reply)
            assert classifier._is_price_family_show_all_followup(reply, entities) == _old_is_price_family_show_all_followup(reply, entities) == new, repr(reply)


def test_audit_cases_are_pinned_as_is():
    """Что узнаётся сейчас (до и после — одинаково). Расширение — под выключателем, решение владельца."""

    assert {reply: _prices_helpers._is_price_show_all_request(reply) for reply in AUDIT} == AUDIT


def test_classifier_has_no_copy_of_the_rule(monkeypatch):
    """Одно определение: классификатор спрашивает инструмент цены, своей регулярки у него нет."""

    entities = CONTEXTS["family_after_price"]
    assert not classifier._is_price_family_show_all_followup("абракадабра", entities)
    monkeypatch.setattr(_prices_helpers, "_is_price_show_all_request", lambda text: text == "абракадабра")
    assert classifier._is_price_family_show_all_followup("абракадабра", entities)
    assert not classifier._is_price_family_show_all_followup("все", entities)


# --------------------------------------------------------------------------- через эндпоинт, два хода

_SETS = {"default": _DEFAULT, "deploy": _DEPLOY}
_FAMILY_QUESTION = "сколько стоит общий анализ крови"
_NUMBERED = re.compile(r"^\d+\. (?P<name>.+?) — \d[\d\s]* руб", re.M)
_HINT = re.compile(r"найдено еще (?P<rest>\d+) вариантов\. Чтобы показать их, напишите: \"(?P<word>[^\"]+)\"")
_AS_THE_HINT_SAYS = "<слово из подсказки бота>"


def _dialog(turns: list[str], session_id: str) -> list[dict[str, Any]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    async def run() -> list[dict[str, Any]]:
        memory, services = MemoryStore(), Services()
        out = []
        for text in turns:
            if text == _AS_THE_HINT_SAYS:
                hint = _HINT.search(out[-1]["text"])
                assert hint, out[-1]["text"]
                text = hint.group("word")
            req = MessengerGenerateRequest(session_id=session_id, text=text, debug=True, button_id="")
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            decision = ((body.get("state_update") or {}).get("debug") or {}).get("decision") or {}
            out.append({"text": body.get("text", ""), "label": decision.get("label"), "flags": set(decision.get("flags") or [])})
        return out

    return asyncio.run(run())


def _play(turns: list[str], config: str, monkeypatch, world_dir) -> list[dict[str, Any]]:
    from messengers_router.runtime_config import config as cfg

    for key, value in _SETS[config].items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    with ev_world.FrozenWorld(world_dir, "day") as world:
        record = _dialog(turns, f"show-all-{config}-{abs(hash(tuple(turns)))}")
        assert list(world.net.attempts) == []
    return record


@pytest.mark.parametrize("config", sorted(_SETS))
@pytest.mark.parametrize("reply", [_AS_THE_HINT_SAYS, "Всё!", " покажи все ", "показать все.", "все варианты", "ВСЕ АНАЛИЗЫ", "все услуги?"])
def test_show_all_reply_expands_the_shown_family(reply, config, mini_prod_world, monkeypatch):  # noqa: F811
    """Класс: после списка семейства с подсказкой «напишите: "все"» реплика «показать все» в любой узнаваемой форме
    (в том числе слово ровно из подсказки) раскрывает весь список — метка PRICE правилом, все строки с ценой."""

    first, second = _play([_FAMILY_QUESTION, reply], config, monkeypatch, mini_prod_world)
    hint = _HINT.search(first["text"])
    assert hint, first["text"]
    shown = [m.group("name") for m in _NUMBERED.finditer(first["text"])]
    total = len(shown) + int(hint.group("rest"))
    assert shown and total > len(shown)

    assert second["label"] == "PRICE"
    assert "rule_price_family_show_all" in second["flags"]
    expanded = [m.group("name") for m in _NUMBERED.finditer(second["text"])]
    assert len(expanded) == total
    assert set(shown) <= set(expanded)
    assert not _HINT.search(second["text"])


@pytest.mark.parametrize("config", sorted(_SETS))
@pytest.mark.parametrize("turns", [["все"], ["сколько стоит ферритин", "все"]], ids=["first_turn", "after_one_price"])
def test_show_all_rule_needs_a_shown_family(turns, config, mini_prod_world, monkeypatch):  # noqa: F811
    """Без показанного списка семейства правило «показать все» не срабатывает (одна цена — не список)."""

    record = _play(turns, config, monkeypatch, mini_prod_world)
    assert "rule_price_family_show_all" not in record[-1]["flags"]
