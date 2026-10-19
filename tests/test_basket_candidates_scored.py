"""BUG-2026-10-17-BASKET-HOME-VISIT-ROW: в корзине строка «Выезд на дом», которой пациент не называл.

Свип 16.10 (g0702): «Сколько стоит анализ на вич спит гепатит в с» → ВИЧ, HBsAg и «По услуге «Выезд на дом (г. Самара)
(зона №1)»: 1 000 руб.». Корень (19.10, офлайн): обрывки реплики («с», «в с») становятся позициями корзины, а их
кандидаты для выбора LLM даёт ранжирование прайса; совпадения слов у обрывка нет, и ранжирование отдаёт запасной поиск
подстрокой (`_score_price_rows`, `score = 0`): «с» входит почти в любое название, кандидаты — первые строки прайса. На
проде среди них оказалась строка выезда на дом, и LLM корзины её выбрала.

Под `[MESSENGER_ROUTER] basket_scored_candidates = on` (умолчание off — как было): кандидаты позиции — только строки,
у которых ранжирование нашло совпадение слов; у обрывка без слов прайса кандидатов нет — «Не распознал», как у прода
до деплоя 5. Инвариант: строка в ответе корзины — среди кандидатов, совпавших со словами своей позиции.

Мир — `mini_prod` плюс строка «Выезд на дом …» первой в прайсе (как запасной поиск видит прод); LLM корзины —
подделка-«противник»: выбирает выезд на дом, если его предлагают, и строку по слову позиции иначе.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import choice, llm_runtime
from messengers_router.services import _basket_select as B
from messengers_router.services import _prices_helpers as H

_TESTS = Path(__file__).resolve().parent
_HOME_VISIT = "Выезд на дом (г. Самара) (зона №1)"
_BASKET_MARK = "ответить на вопрос о ценах нескольких услуг сразу"
_QUESTIONS = ["Сколько стоит анализ на вич спит гепатит в с", "сколько стоит вич, гепатит в, с"]


@pytest.fixture(scope="module")
def home_visit_world(tmp_path_factory) -> Path:
    src = tmp_path_factory.mktemp("hv_src")
    shutil.copytree(_TESTS / "fixtures" / "mini_prod" / "snapshots", src, dirs_exist_ok=True)
    price = src / "price_by_region" / "price_region_3_20261008.jsonl"
    row = {"serviceId": 990100, "priceUnitId": 155, "serviceHomecode": "HV1", "serviceName": _HOME_VISIT, "deadline": " ", "cost": 1000.0}
    price.write_text(json.dumps(row, ensure_ascii=False) + "\n" + price.read_text(encoding="utf-8"), encoding="utf-8")
    return ev_world.build_world(src, "20261008", tmp_path_factory.mktemp("hv") / "world")


class _AdversaryBasket:
    """LLM корзины: если среди вариантов есть выезд на дом — выбирает его и по варианту на позицию; разбор и прочее —
    «недоступна»."""

    def __init__(self) -> None:
        self.offered_home_visit = False

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        if _BASKET_MARK not in f"{system}\n{prompt}":
            raise ConnectionError("тест: LLM недоступна для этого места")
        picks, seen_positions = [], set()
        for line in prompt.splitlines():
            m = re.match(r"^(\d+)\. (.+?) → (.+) — ", line)
            if not m:
                continue
            number, position, name = int(m.group(1)), m.group(2), m.group(3)
            if name == _HOME_VISIT:
                self.offered_home_visit = True
                picks.append(number)
            elif position not in seen_positions:
                seen_positions.add(position)
                picks.append(number)
        return GenerateResponse(model="m", response=json.dumps({"picks": picks}), done=True)


def _turn(world_dir, monkeypatch, question: str, switch: str) -> tuple[str, bool]:
    from messengers_router.runtime_config import config as cfg
    from tests.test_price_turns_golden import run_dialog

    llm = _AdversaryBasket()
    with ev_world.FrozenWorld(world_dir, "day"), monkeypatch.context() as patch:
        patch.setattr(cfg, "MR_BASKET_SCORED_CANDIDATES", switch, raising=False)
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        choice._CACHE.clear()
        [record] = run_dialog([question], f"hv-{switch}-{question}")
    return record["text"], llm.offered_home_visit


@pytest.mark.parametrize("question", _QUESTIONS)
def test_fragment_without_price_words_gets_no_letter_candidates(question, home_visit_world, monkeypatch):
    on, offered_on = _turn(home_visit_world, monkeypatch, question, "on")

    assert not offered_on, "строку без общих слов с позицией LLM корзины предлагать нельзя"
    assert "Выезд на дом" not in on, on


def _phrases_basket(world_dir, monkeypatch, switch: str) -> tuple[list[dict], list[str], bool]:
    """Корзина из фраз разбора, как у g0702 при решённом разборе: «вич» (сильная), «гепатит в» и обрывок «с» (слабые)."""

    import asyncio

    from messengers_router.runtime_config import config as cfg
    from messengers_router.turn_parse import ParsedPhrase
    from messengers_router.services._strong_match import strong_rows

    reply = "Сколько стоит анализ на вич спит гепатит в с"
    llm = _AdversaryBasket()
    with ev_world.FrozenWorld(world_dir, "day"), monkeypatch.context() as patch:
        patch.setattr(cfg, "MR_BASKET_SCORED_CANDIDATES", switch, raising=False)
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        choice._CACHE.clear()
        rows = H._catalog_rows(None)
        phrases = [
            ParsedPhrase("вич", reply.index("вич"), reply.index("вич") + 3, "strong", rows=tuple(strong_rows("вич", rows))),
            ParsedPhrase("гепатит в", reply.index("гепатит"), reply.index("гепатит") + 9, "weak"),
            ParsedPhrase("с", len(reply) - 1, len(reply), "weak"),
        ]
        positions, not_found = asyncio.run(B.basket_from_phrases(reply, phrases, (), rows))
    return positions, not_found, llm.offered_home_visit


def test_off_is_as_before_and_reproduces_the_defect(home_visit_world, monkeypatch):
    positions, _not_found, offered = _phrases_basket(home_visit_world, monkeypatch, "off")

    assert offered, "как было: обрывок «с» получал кандидатом строку выезда (запасной поиск подстрокой)"
    with ev_world.FrozenWorld(home_visit_world, "day"):
        letters = [B._row_name(r) for r in B._candidates({"service_name": "с", "fragment": "с", "prices": []}, H._catalog_rows(None))]
    assert letters[0] == _HOME_VISIT, letters  # первые строки прайса, в которых есть буква «с»


def test_on_the_fragment_is_not_recognized_instead_of_a_foreign_row(home_visit_world, monkeypatch):
    positions, not_found, offered = _phrases_basket(home_visit_world, monkeypatch, "on")

    assert not offered
    assert not any(_HOME_VISIT in B._row_name(row) for p in positions for row in p.get("prices") or []), positions
    assert "с" in not_found, not_found  # честное «Не распознал: с»
    assert any("ВИЧ" in B._row_name(row) for p in positions for row in p.get("prices") or [])


def test_scored_candidates_keep_real_matches(home_visit_world, monkeypatch):
    from messengers_router.runtime_config import config as cfg

    with ev_world.FrozenWorld(home_visit_world, "day"):
        rows = H._catalog_rows(None)
        for switch in ("off", "on"):
            monkeypatch.setattr(cfg, "MR_BASKET_SCORED_CANDIDATES", switch, raising=False)
            names = [B._row_name(r) for r in B._candidates({"service_name": "гепатит в", "fragment": "гепатит в", "prices": []}, rows)]
            assert any("Гепатит В" in name for name in names), (switch, names)
        monkeypatch.setattr(cfg, "MR_BASKET_SCORED_CANDIDATES", "on", raising=False)
        assert B._candidates({"service_name": "с", "fragment": "с", "prices": []}, rows) == []
