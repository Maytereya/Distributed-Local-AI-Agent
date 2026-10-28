"""Класс: сильное совпадение цены не теряет различающую букву или цифру (BUG-2026-10-27-STRONG-DROPS-LETTER).

Прод с деплоя 18 (`price_by_parse = on`): сильная фраза отвечает ценой без выбора строки LLM. Поиск синонима МИС выбрасывает
предлоги и у реплики, и у ключа словаря — среди них одиночные «с», «в», «а», «и», «у». Синонимы клиники «гепатит в» (HBsAg) и
«гепатит с» (Анти-HCV) становились одной цепочкой «гепатит»: на «гепатит с», «гепатит а» и «гепатит» пациент читал цену гепатита
В, хотя LLM работала и не спрашивалась (офлайн 26.10, прайс 08.10 + словарь 03.10; здесь — тот же ход через эндпоинт).

Инварианты под `strong_family_letter = on` — через обработчик эндпоинта `messenger_generate_once`, конфиг прода деплоя 18, оракул
разметки и выбора строки, на синтетике (все буквенные семейства живого прайса) и на выдержке живого прайса 08.10:
- пациент не читает цену строки, названной другой буквой того же семейства, чем он написал (кириллица и латиница одно);
- синоним клиники со своей буквой — сильный: «гепатит с» → Анти-HCV без выбора строки LLM;
- слово семейства без буквы («гепатит») синонимом с буквой не становится — строку выбирает LLM или список;
- встречно: «гепатит в» → HBsAg остаётся сильным; каждое название прайса остаётся сильным;
- цифры и составные различители (ВПЧ 16/18, тип 1/2, Т3/Т4, B12, IgG/IgM) поиск не теряет — и без фикса (граница класса);
- off — как было (контраст: чужая буква на сильном пути).

Свип по всему живому прайсу и словарю — `scripts/sweep_strong_family_letter.py`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.services._strong_match import _letter_families, _names_other_letter, family_letters, strong_rows
from tests.test_price_stage2_turns import Case, _Oracle, _Down
from tests.test_price_turns_golden import FIXTURE, mini_prod_world, run_dialog  # noqa: F401 — фикстура

# Выключатели прода деплоя 18 (26.10), которые читает код ветки (конфиг прогонов стенда d18p2_*).
PROD_D18: dict[str, Any] = {
    "MR_TURN_UNDERSTANDING_PRICE": "on",
    "MR_TURN_UNDERSTANDING_ONE": "on",
    "MR_TURN_SPANS_LLM": "on",
    "MR_CHOOSE_SYMPTOM": "on",
    "MR_CHOOSE_CACHE_TTL_S": 0,
    "MR_PARSE_FAIL_VARIANTS": 3,
    "MR_TOPIC_OVERRIDE_LLM": "off",
    "MR_PARSE_FRAGMENT_WINDOWS": "off",
    "MR_PARSE_COMPLETE_WINDOWS": "off",
    "MR_PENDING_REPLY": "on",
    "MR_PRICE_CANDIDATES_BY_MEANING": "on",
    "MR_BASKET_SCORED_CANDIDATES": "on",
    "MR_PREPARE_VS_BOOKING_LLM": "on",
    "MR_PRICE_OPTION_MIS_NOTES": "on",
    "MR_BASKET_WEAK_PHRASE_JUDGE": "on",
    "MR_PENDING_BUTTONS": "on",
    "MR_ROLES_CATALOG_LLM": "on",
    "MR_CATALOG_CONFIRM_LLM": "on",
    "MR_TOPIC_BY_LLM": "rule_test_assist,rule_doctor_info_specialty,rule_schedule,rule_analysis_lookup_ambiguous,"
    "rule_website_help,rule_address_work_hours,rule_test_result,rule_prepare",
    "MR_P1_FACET_NONE_OFFER": "on",
    "MR_PRICE_BUDGET_CLARIFY": "on",
    "MR_DOCTORS_BY_SERVICE_DATA": "shadow",
    "MR_PREPARE_GENERAL_MEMO_LLM": "on",
    "MR_BUTTON_PASS_RULES": "on",
    "MR_PRICE_GENERAL_CLARIFY": "on",
    "MR_SYNONYM_WORDFORMS": "on",
    "MR_PRICE_SELECT_RULES": "general",
    "MR_PREPARE_BY_PARSE": "on",
    "MR_GROUNDER_GENERIC_BY_LLM": "on",
    "MR_BIOMATERIAL_BY_MIS": "on",
    "MR_TEST_ASSIST_FROM_DATA": "on",
    "MR_PRICE_BY_PARSE": "on",
    "MR_BOOKING_TARGET_SLOT_LLM": "on",
    "MR_BUTTON_PRICE_SCOPE": "doctor",
    "MR_LLM_DECISION_TURN_SECONDS": 25,
    "MR_NAMES_DOCTOR_CATALOG": "on",
    "MR_NAMES_DOCTOR_NOISE": "on",
    "MR_NAMES_DOCTOR_FOLLOWUP": "on",
    "MR_NAMES_RESULT_SURNAME": "on",
    "MR_NAMES_PATIENT_FIO": "on",
    "MR_NAMES_PATIENT_RULE": "on",
    "MR_NAMES_PATIENT_FILL": "on",
    "MR_BASKET_CALC": "on",
    "MR_ADDRESS_BRANCH_FLAGS": "on",
    "MR_ADDRESS_FOLLOWUP_LLM": "on",
    # Оракул отвечает только разметке и выбору строки; сбои прочих мест не должны размыкать предохранитель для них.
    "MR_LLM_DECISION_BREAKER_FAILURES": 0,
}

# Строки семейств — названия живого прайса 08.10 (буква: название). В выдержке «мини-прод» из них только В и С гепатита.
FAMILIES: dict[str, dict[str, str]] = {
    "гепатит": {
        "а": "Гепатит А -  Анти-HAV IgM",
        "в": "Гепатит В - HBsAg",
        "с": "Гепатит С - Анти-HCV суммарные",
        "d": "Гепатит D – anti-HDV суммарные",
        "е": "Гепатит Е- Анти-HEV IgG",
        "g": "Вирус гепатита G [кач.]",
    },
    "витамин": {
        "а": "Витамин А (ретинол)",
        "с": "Витамин C (аскорбиновая кислота)",
        "д": "Витамин D, 25-гидрокси (кальциферол), ВЭЖХ-МС/МС - метод масс-спектрометрия",
        "е": "Витамин Е (токоферол)",
    },
    "протеин": {"с": "Протеин С", "s": "Протеин S"},
    "стрептококк группы": {
        "а": "Посев на бета-гемолитический стрептококк группы А (Streptococcus pyogenes)",
        "в": "Стрептококк группы В (Streptococcus group B, S.agalactiae)",
    },
}
# Латинское и кириллическое написание одной буквы (как `_fold_family_letter`); д/d, г/g — по звучанию.
_SPELLINGS = {"а": ("а", "a"), "в": ("в", "b"), "с": ("с", "c"), "е": ("е", "e"), "д": ("д", "d"), "d": ("d", "д"), "g": ("g", "г"), "s": ("s",)}
# Синонимы клиники — как в живом словаре 03.10: «<слово> <буква>» кириллицей, на строку своей буквы.
_LIVE_SYNONYMS = {"гепатит в": ("Гепатит В - HBsAg",), "гепатит с": ("Гепатит С - Анти-HCV суммарные",)}


def _synonyms() -> dict[str, tuple[str, ...]]:
    return {f"{word} {letter}": (name,) for word, rows in FAMILIES.items() for letter, name in rows.items()}


def _patch_synonyms(monkeypatch, synonyms: dict[str, tuple[str, ...]]) -> None:
    from messengers_router.services import _biomaterial as _bio

    hermetic = _bio._vocabularies

    def _with_synonyms():
        vocab = hermetic()
        vocab.synonym_candidates = dict(synonyms)
        vocab.synonym_to_service = {key: names[0] for key, names in synonyms.items() if len(names) == 1}
        return vocab

    monkeypatch.setattr(_bio, "_vocabularies", _with_synonyms)


@pytest.fixture(scope="module")
def families_world(tmp_path_factory) -> Path:
    """Мир «мини-прод» плюс строки всех буквенных семейств (синтетика из названий живого прайса)."""

    source = tmp_path_factory.mktemp("families_src")
    shutil.copytree(FIXTURE, source, dirs_exist_ok=True)
    price = next((source / "price_by_region").glob("*.jsonl"))
    rows = [json.loads(line) for line in price.read_text(encoding="utf-8").splitlines() if line.strip()]
    have = {row["serviceName"] for row in rows}
    next_id = max(int(row["serviceId"]) for row in rows) + 1
    for family in FAMILIES.values():
        for name in family.values():
            if name in have:
                continue
            rows.append(
                {"serviceId": next_id, "priceUnitId": 141, "serviceHomecode": str(90000 + next_id), "serviceName": name,
                 "deadline": "1-2", "cost": 1000.0 + next_id % 997, "promotion": None}
            )
            next_id += 1
    price.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return ev_world.build_world(source, "20261008", tmp_path_factory.mktemp("families") / "world")


def _turn(world_dir: Path, question: str, phrase: str, picks: tuple[str, ...] | None, mode: str, switch: str, monkeypatch):
    from messengers_router.runtime_config import config as cfg

    for key, value in {**PROD_D18, "MR_STRONG_FAMILY_LETTER": switch}.items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    oracle = _Oracle(Case((question,), (phrase,), rows=picks)) if mode == "llm" else _Down()
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", oracle)
    monkeypatch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
    with ev_world.FrozenWorld(world_dir, "day") as world:
        record = run_dialog([question], f"family-letter-{mode}-{switch}-{question}")[0]
        attempts = list(world.net.attempts)
    assert attempts == []
    assert record["text"].strip()
    return record["text"], list(getattr(oracle, "asked", []))


def _foreign(text: str, word: str, letter: str) -> list[str]:
    """Названия строк семейства `word` другой буквы, которые пациент прочитал."""

    return [name for other, name in FAMILIES[word].items() if other != letter and name in text]


# --------------------------------------------------------------------------- синтетика: все буквенные семейства, через эндпоинт

_LETTER_CASES = [(word, letter, spelled) for word, rows in FAMILIES.items() for letter in rows for spelled in _SPELLINGS[letter]]


@pytest.mark.parametrize("mode", ["llm", "down"])
@pytest.mark.parametrize("word,letter,spelled", _LETTER_CASES, ids=[f"{w}-{s}" for w, _l, s in _LETTER_CASES])
def test_patient_never_reads_a_price_of_another_letter_under_on(word, letter, spelled, mode, families_world, monkeypatch):
    _patch_synonyms(monkeypatch, _synonyms())
    phrase = f"{word} {spelled}"
    own = FAMILIES[word][letter]
    text, _asked = _turn(families_world, f"сколько стоит {phrase}", phrase, (own,), mode, "on", monkeypatch)
    assert _foreign(text, word, letter) == [], text
    if mode == "llm":
        assert own in text, text


@pytest.mark.parametrize("word", sorted(FAMILIES))
def test_clinic_synonym_with_its_letter_is_strong_without_row_choice_under_on(word, families_world, monkeypatch):
    """«<слово> <буква>» кириллицей — синоним клиники: строка своей буквы без выбора строки LLM (он недоступен и не спрошен) —
    поиск не склеивает синоним с соседним по семейству."""

    _patch_synonyms(monkeypatch, _synonyms())
    for letter, own in FAMILIES[word].items():
        phrase = f"{word} {letter}"
        text, asked = _turn(families_world, f"сколько стоит {phrase}", phrase, None, "llm", "on", monkeypatch)
        assert own in text and _foreign(text, word, letter) == [] and "rows" not in asked, (phrase, text, asked)


@pytest.mark.parametrize("word", sorted(FAMILIES))
def test_family_word_without_a_letter_is_not_strong_under_on(word, families_world, monkeypatch):
    """Синоним с буквой не делает слово без буквы сильным: строку выбирает LLM (её спросили), без LLM — не одна цена семейства."""

    _patch_synonyms(monkeypatch, _synonyms())
    _text, asked = _turn(families_world, f"сколько стоит {word}", word, (), "llm", "on", monkeypatch)
    assert "rows" in asked
    text, _asked = _turn(families_world, f"сколько стоит {word}", word, None, "down", "on", monkeypatch)
    assert len([name for name in FAMILIES[word].values() if name in text]) != 1, text


# --------------------------------------------------------------------------- живой прайс: выдержка 08.10 и словарь 03.10

_LIVE_CASES = [
    # вопрос, фраза разметки, начало названия строки, которую выбирает LLM; чужая строка
    ("сколько стоит гепатит с", "гепатит с", ("Гепатит С",), "Гепатит В - HBsAg"),
    ("сколько стоит анализ на гепатит с", "гепатит с", ("Гепатит С",), "Гепатит В - HBsAg"),
    ("сколько стоит гепатит c", "гепатит c", ("Гепатит С",), "Гепатит В - HBsAg"),
    ("сколько стоит гепатит а", "гепатит а", (), "Гепатит В - HBsAg"),
    ("сколько стоит гепатит в", "гепатит в", ("Гепатит В - HBsAg",), "Гепатит С - Анти-HCV суммарные"),
]


@pytest.mark.parametrize("order", ["clinic", "reversed"])
@pytest.mark.parametrize("mode", ["llm", "down"])
@pytest.mark.parametrize("question,phrase,picks,foreign", _LIVE_CASES, ids=[c[0] for c in _LIVE_CASES])
def test_live_excerpt_no_foreign_hepatitis_price_under_on(question, phrase, picks, foreign, mode, order, mini_prod_world, monkeypatch):  # noqa: F811
    synonyms = dict(_LIVE_SYNONYMS if order == "clinic" else reversed(list(_LIVE_SYNONYMS.items())))
    _patch_synonyms(monkeypatch, synonyms)
    text, _asked = _turn(mini_prod_world, question, phrase, picks, mode, "on", monkeypatch)
    assert foreign not in text, text


def test_live_excerpt_hepatitis_c_is_strong_by_clinic_synonym_under_on(mini_prod_world, monkeypatch):  # noqa: F811
    _patch_synonyms(monkeypatch, _LIVE_SYNONYMS)
    text, asked = _turn(mini_prod_world, "сколько стоит гепатит с", "гепатит с", ("Гепатит С",), "llm", "on", monkeypatch)
    assert "Гепатит С - Анти-HCV суммарные — 440 руб." in text and "rows" not in asked, (text, asked)


def test_live_excerpt_hepatitis_b_stays_strong_under_on(mini_prod_world, monkeypatch):  # noqa: F811
    _patch_synonyms(monkeypatch, _LIVE_SYNONYMS)
    text, asked = _turn(mini_prod_world, "сколько стоит гепатит в", "гепатит в", None, "llm", "on", monkeypatch)
    assert "Гепатит В - HBsAg — 355 руб." in text and "rows" not in asked, (text, asked)


@pytest.mark.parametrize("question,phrase", [("сколько стоит гепатит с", "гепатит с"), ("сколько стоит гепатит а", "гепатит а"), ("сколько стоит гепатит", "гепатит")])
def test_off_reads_hepatitis_b_on_the_strong_path_as_before(question, phrase, mini_prod_world, monkeypatch):  # noqa: F811
    """Контраст: путь off — как было, чужая буква на сильном пути и при работающей LLM (её не спрашивают)."""

    _patch_synonyms(monkeypatch, _LIVE_SYNONYMS)
    text, asked = _turn(mini_prod_world, question, phrase, ("Гепатит С",), "llm", "off", monkeypatch)
    assert "Гепатит В - HBsAg — 355 руб." in text and "rows" not in asked, (text, asked)


# --------------------------------------------------------------------------- `strong_rows` на прайсе: встречно и граница класса


def _switch(monkeypatch, value: str) -> None:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_PRICE_BY_PARSE", "on", raising=False)
    monkeypatch.setattr(cfg, "MR_STRONG_FAMILY_LETTER", value, raising=False)


def _live_rows() -> list[dict[str, Any]]:
    from agent_logic_2.nayka_api import api_price

    rows = [row for row in api_price.load_price_by_region(3) if isinstance(row, dict)]
    if len(rows) < 1000:
        pytest.skip("нет локального среза прайса (ловушка CLAUDE.md): живой прайс — на Mac с apidata")
    return rows


def test_live_price_families_are_letter_families_only():
    """Семейства — из названий прайса; предлог перед кириллическим словом («Глюкоза в моче») буквой не считается."""

    families = _letter_families(_live_rows())
    assert families["гепати"] >= {"a", "b", "c", "d", "e"} and families["витами"] >= {"a", "c", "d", "e"}
    assert families["протеи"] == {"c", "s"}
    assert not {"глюкоз", "железо", "кортиз", "пролак", "белок"} & set(families)


def test_every_live_price_name_stays_strong_under_on(monkeypatch):
    rows = _live_rows()
    _switch(monkeypatch, "on")
    names = sorted({str(row.get("serviceName") or "").strip() for row in rows} - {""})
    assert [name for name in names if not strong_rows(name, rows)] == []


_CYRILLIC = {"a": "а", "b": "в", "c": "с", "e": "е", "d": "д", "g": "г", "k": "к"}


def test_live_price_letter_sweep_never_strong_to_another_letter_under_on(monkeypatch):
    """Свип по живому прайсу: каждое слово семейства с каждой его буквой (обе раскладки) не сильно строке другой буквы. Синонимы —
    как их пишет клиника: «<слово> <буква кириллицей>» на все строки своей буквы."""

    rows = _live_rows()
    families = _letter_families(rows)
    names = sorted({str(row.get("serviceName") or "") for row in rows} - {""})
    words = {word for name in names for word in name.lower().replace("-", " ").split() if word.isalpha() and word[:6] in families}
    by_letter: dict[tuple[str, str], list[str]] = {}
    for name in names:
        for stem_letter in family_letters(name, families):
            by_letter.setdefault(stem_letter, []).append(name)
    _patch_synonyms(
        monkeypatch,
        {f"{word} {_CYRILLIC.get(letter, letter)}": tuple(found) for (stem, letter), found in by_letter.items() for word in words if word[:6] == stem},
    )
    _switch(monkeypatch, "on")
    bad = []
    for word in sorted(words):
        for letter in sorted(families[word[:6]]):
            for spelled in {letter, _CYRILLIC.get(letter, letter)}:
                phrase = f"{word} {spelled}"
                bad += [(phrase, row.get("serviceName")) for row in strong_rows(phrase, rows) if _names_other_letter(str(row.get("serviceName")), family_letters(phrase, families), families)]
    assert len(words) >= 8 and bad == []


_DESIGNATORS = {
    "впч 16": "ВПЧ 16 тип [кол.]",
    "впч 18": "ВПЧ 18 тип [кол.]",
    "впг 1": "Вирус простого герпеса 1 типа IgG",
    "впг 2": "Вирус простого герпеса 2 типа IgG",
    "т3": "сТ3 (FT3) Свободный трийодтиронин",
    "т4": "сТ4 (FT4) Свободный тироксин",
    "витамин в12": "Витамин B12",
    "витамин в6": "Витамин B6 (пиридоксаль-5-фосфат)",
    "хламидии igg": "Chlamydia trachomatis IgG",
    "хламидии igm": "Chlamydia trachomatis IgM",
}


@pytest.mark.parametrize("switch", ["off", "on"])
@pytest.mark.parametrize("phrase", sorted(_DESIGNATORS))
def test_digits_and_compound_designators_are_not_lost(phrase, switch, monkeypatch):
    """Граница класса: цифра или составной различитель вне синонима — не обвязка, синоним не засчитывается (и без фикса)."""

    rows = [{"serviceId": n, "serviceHomecode": str(7000 + n), "serviceName": name, "cost": 100.0} for n, name in enumerate(_DESIGNATORS.values())]
    _patch_synonyms(monkeypatch, {key: (name,) for key, name in _DESIGNATORS.items()})
    _switch(monkeypatch, switch)
    for asked in (phrase, f"анализ на {phrase}"):
        assert [row["serviceName"] for row in strong_rows(asked, rows)] == [_DESIGNATORS[phrase]], asked
