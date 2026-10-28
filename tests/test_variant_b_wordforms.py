"""Вариант B L-13 называет чужую строку, когда верная записана другой словоформой (STEP-2026-10-16-T7-1-2, находка).

Симптом (LLM недоступна): «экспресс-тест на сифилис IgG и IgM» → «нашёл похожую услугу: «Кровь на сифилис (реакция
микропреципитации…)»», хотя в прайсе есть «Экспресс-тест для определения сифилиса (антитела IgG и IgM к Treponema
pallidum)». Корень: `_price_select_llm._whole_word_names` берёт самое редкое слово пациента и строки, где оно есть ЦЕЛИКОМ;
«сифилис» целиком есть только в чужой строке (в верной — «сифилиса»), поэтому оно и редкое, и верная строка отпадает.

Фикс (выключатель `variant_b_wordforms` снят 28.10 — on на проде с 18.10): словоформа различающего слова —
одно слово (`russian_nlu.tokens_share_stem`, порог 6, только слова из одних букв); строка со словоформой добавляется, если
покрывает слов пациента строго больше любой строки со словом целиком; порядок — по числу слов пациента. Встречный свип
по 398 названиям мини-прайса («сколько стоит» + название без скобок): изменилось 6, ни одно попадание и ни одна первая
строка не потеряны (сифилис — верная первой; «chlamydia trachomatis igg» — строки IgG раньше IgA/IgM; «повторный приём
оториноларинголога N категория» — строки семейства вместо общего «Приём (осмотр…) повторный»).

Инвариант класса: при недоступной LLM вариант B называет первой строку, которая покрывает больше слов пациента, —
словоформа не делает чужую строку «единственной»; слова, которого в прайсе нет вовсе, — названий нет («гастроскопию» не
«Гастрин»); служебные токены кода («витамин_a» / «витамин_b12») не сливаются.
"""

from __future__ import annotations

import json
import re

import pytest

from evalbench import world as ev_world
from tests.test_price_one_or_many_class import _configure
from tests.test_price_turns_golden import FIXTURE, mini_prod_world, run_dialog  # noqa: F401 — фикстура модуля

_SYPHILIS = "сколько стоит экспресс-тест на сифилис IgG и IgM"
_SYPHILIS_PHRASE = "экспресс-тест на сифилис IgG и IgM"  # фраза услуги в реплике — разметка LLM
_RIGHT = "Экспресс-тест для определения сифилиса (антитела IgG и IgM к Treponema pallidum)"
_WRONG = "Кровь на сифилис (реакция микропреципитации с кардиолипиновым антигеном,антитела IgG и IgM к Treponema pallidum)"


def _rows() -> list[dict]:
    path = FIXTURE / "price_by_region" / "price_region_3_20261008.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _names(question: str) -> tuple[str, ...]:
    from messengers_router.services import _price_select_llm as PS
    from messengers_router.services._prices_helpers import _price_query_tokens

    rows = _rows()
    phrase = str(PS._extract_price_service_from_query(question) or "")
    words = [t for t in _price_query_tokens(question) if len(t) >= 3]
    pool = PS._build_pool([question, PS._apply_service_synonyms(question), phrase], rows, per_text=PS._PER_QUERY_LIMIT, fallback_texts=words)
    pool += [row for row in rows if PS._row_name(row) == _WRONG]  # как в ходе: строка правил — среди кандидатов
    return PS._whole_word_names(question, pool, rows)


def test_the_row_covering_the_patient_words_comes_first(mini_prod_world, monkeypatch):  # noqa: F811
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        after = _names(_SYPHILIS)
    assert after[0] == _RIGHT and _WRONG in after  # словоформа: верная первой, неоднозначность — списком (без неё — чужая единственная)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("сколько стоит гастроскопию", ()),  # слова нет в прайсе — названий нет, не «Гастрин»
        ("сколько стоит витамин а", ("Витамин А (ретинол)",)),  # «витамин_a» не сливается с «витамин_b12»
    ],
)
def test_boundaries_of_the_class(question, expected, mini_prod_world):  # noqa: F811
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        assert _names(question)[: len(expected) or None] == expected


def test_through_the_endpoint_with_the_llm_down_two_turns(mini_prod_world, monkeypatch):  # noqa: F811
    """Через эндпоинт, набор прода, выбор строки LLM недоступен: вариант B — названия без цен; под `on` первой — верная строка;
    второй ход «а сколько?» не называет цены строки, которой пациент не выбирал. Разметка фраз — оракул (`turn_spans_llm = on`,
    прод с деплоя 17): при лежащей LLM целиком цена — только реплике-названию, иначе «Уточните» (`price_by_parse = on`), и до
    варианта B ход не доходит."""

    from messengers_router import llm_runtime
    from tests.spans_oracle import SpansOracle, keep_llm_breaker_closed

    _configure(monkeypatch, "deploy")
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", SpansOracle((_SYPHILIS_PHRASE,)))
    keep_llm_breaker_closed(monkeypatch)
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        first, second = run_dialog([_SYPHILIS, "а сколько?"], "variant-b-on")
    assert not re.search(r"\d\s*руб", first["text"] + second["text"]), (first["text"], second["text"])
    assert first["text"].index(_RIGHT) < first["text"].index(_WRONG), first["text"]


# --------------------------------------------------------------------------- общее слово (BUG-2026-10-17-VARIANT-B-COMMON-WORD)
#
# Живой прайс 16.10 (dev 867, LLM недоступна): «стоимость консультации врача» — под off перевод, под on — «нашёл похожие
# услуги» и пять приёмов хирургов (против решения 25.09: цену приёма без специальности не называть). «консультации»
# дословно есть в одном названии (и потому — самое редкое слово), а со словоформой «консультация» — в 163: редкость —
# артефакт формы, слово общее. Словоформы добавляют строки, только если слово вместе с ними различает — есть не больше
# чем в `_VARIANT_B_LIMIT` названиях прайса (список покажет их все); механика по данным прайса, без слов в коде.
# В мини-прайсе «консультации» дословно нет — добавлена одна такая строка, как на живом прайсе.

_COMMON = "стоимость консультации врача"
_RARE_FORM_ROW = {
    "serviceId": 990001,
    "priceUnitId": 137,
    "serviceHomecode": "99.9.1",
    "serviceName": "Консультации психолога групповые",
    "deadline": " ",
    "cost": 1500.0,
    "promotion": None,
}


@pytest.fixture(scope="module")
def rare_form_world(tmp_path_factory):
    import shutil

    from tests.test_price_turns_golden import FIXTURE as SNAPSHOTS

    root = tmp_path_factory.mktemp("rare_form")
    shutil.copytree(SNAPSHOTS, root / "snapshots")
    with (root / "snapshots" / "price_by_region" / "price_region_3_20261008.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_RARE_FORM_ROW, ensure_ascii=False) + "\n")
    return ev_world.build_world(root / "snapshots", "20261008", root / "world")


def _common_names(rows: list[dict]) -> tuple[str, ...]:
    from messengers_router.services import _price_select_llm as PS
    from messengers_router.services._prices_helpers import _price_query_tokens

    phrase = str(PS._extract_price_service_from_query(_COMMON) or "")
    words = [t for t in _price_query_tokens(_COMMON) if len(t) >= 3]
    pool = PS._build_pool([_COMMON, PS._apply_service_synonyms(_COMMON), phrase], rows, per_text=PS._PER_QUERY_LIMIT, fallback_texts=words)
    return PS._whole_word_names(_COMMON, pool, rows)


def test_a_word_rare_only_as_a_word_form_names_nothing(rare_form_world, monkeypatch):
    """Ключ варианта B — «консультации» (дословно одно название вне кандидатов), со словоформой — общее слово: строк
    нет; без условия «различает» были бы приёмы (что и было 16.10)."""

    from messengers_router.services import _price_select_llm as PS

    rows = [*_rows(), _RARE_FORM_ROW]
    with ev_world.FrozenWorld(rare_form_world, "day"):
        assert PS._token_df(rows).get("консультации") == 1
        assert PS._wordform_family_size("консультации", rows) > PS._VARIANT_B_LIMIT
        assert _common_names(rows) == ()
        monkeypatch.setattr(PS, "_wordform_family_size", lambda *_args: 0)  # как до 17.10: словоформы без условия
        assert _common_names(rows) and all(name.startswith("Прием") for name in _common_names(rows))


def test_the_distinguishing_word_form_still_comes_first(mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный: «сифилис» со словоформой — в двух названиях (различает) — верная строка первой, как раньше."""

    from messengers_router.services import _price_select_llm as PS

    with ev_world.FrozenWorld(mini_prod_world, "day"):
        assert PS._wordform_family_size("сифилис", _rows()) <= PS._VARIANT_B_LIMIT
        assert _names(_SYPHILIS)[0] == _RIGHT


def test_common_word_through_the_endpoint_two_turns(rare_form_world, monkeypatch):
    """Через эндпоинт, набор прода, LLM недоступна: «стоимость консультации врача» — не список приёмов и без цены;
    второй ход «а сколько?» — без цены строки, которой пациент не видел."""

    _configure(monkeypatch, "deploy")
    with ev_world.FrozenWorld(rare_form_world, "day"):
        first, second = run_dialog([_COMMON, "а сколько?"], "variant-b-common")
    assert "хирург" not in first["text"].lower() and "похожие услуги" not in first["text"], first["text"]
    assert not re.search(r"\d\s*руб", first["text"] + second["text"]), (first["text"], second["text"])
