"""П4 (прод 10.08): бот не знал собственные филиалы вне черты города.

Симптом: «Может ли сдать спермограмму в южном городе» → дамп восемнадцати
филиалов Самары. Для бота пациент был «без города»: `match_city` возвращает None
для «Придорожный», «Южный город», «в южном городе», «Мехзавод».

Хуже (найдено при разборе): «пр-т Николаевский» — адрес филиала в Южном городе —
`match_city` резолвит в ГОРОД «Николаевский» из справочника городов, ранний гард
роутера считает это другим городом и уводит к оператору «Сейчас могу помочь
только по Самаре». Собственный филиал клиники выглядел чужим городом.

Уточнение из плана подтвердилось: `_has_explicit_non_samara_regions` тут ни при
чём — реальная точка решения это ранний гард `match_city` + `_is_samara_city`
в `route_patient_message_stream`.

Данные (узел id=3 «Самара», 32 филиала): тридцать в черте города, три вне —
`пос.Придорожный («Южный город»), пр-т Николаевский, д. 2`,
`п.Мехзавод, квартал 10, д.13`, `ул.Челышевская, 3 (ЮГ-2)`.

Инвариант (класс): место из самарского периметра обслуживаем как Самару (к
оператору как «другой город» НЕ отправляем); при явном упоминании места со своим
филиалом подставляем этот филиал; вне периметра (Тольятти, Ульяновск, Оренбург,
Саратов) — прежний честный отказ. Словарь мест строится ИЗ ДАННЫХ филиалов;
спутники — закрытый список по решению владельца №5.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from messengers_router.services import _samara_perimeter as sp

_SNAPSHOT = Path(__file__).resolve().parents[1] / "agent_logic_2" / "nayka_api" / "apidata" / "samara_branches_snapshot.json"


@pytest.fixture(scope="module")
def branches() -> list[dict]:
    if not _SNAPSHOT.exists():
        pytest.skip("снапшот самарских филиалов недоступен в этом чекауте")
    rows = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
    return [r for r in rows if isinstance(r, dict)]


# --- Словарь мест строится ИЗ ДАННЫХ ---------------------------------------

def test_places_derived_from_branch_data(branches):
    """Ни одно место не захардкожено: Придорожный/Южный город/Мехзавод/ЮГ-2
    извлекаются из полей `name` и `addressForSite` самих филиалов."""
    places = sp.perimeter_places(branches)
    for expected in ("придорожный", "южный город", "мехзавод", "юг-2"):
        assert expected in places, (expected, sorted(places))


def test_owner_satellites_present(branches):
    """Спутники — закрытый список по решению владельца №5 (14.08)."""
    places = sp.perimeter_places(branches)
    for expected in ("новокуйбышевск", "кинель", "смышляевка", "стройкерамика"):
        assert expected in places, expected


def test_samara_itself_is_not_a_perimeter_place(branches):
    """«Самара» — сам город, не отдельное место периметра."""
    assert "самара" not in sp.perimeter_places(branches)


# --- Распознавание в реплике ----------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Может ли сдать спермограмму в южном городе", "южный город"),
        ("южный город", "южный город"),
        ("в Придорожном", "придорожный"),
        ("Мехзавод", "мехзавод"),
        ("пр-т Николаевский, д. 2", "николаевский"),
        ("ЮГ-2", "юг-2"),
        ("в Новокуйбышевске", "новокуйбышевск"),
        ("в Кинеле", "кинель"),
        ("в Смышляевке", "смышляевка"),
        ("Стройкерамика", "стройкерамика"),
    ],
    ids=[
        "prod_10_08", "bare", "pridorozhny", "mehzavod", "nikolaevsky_ave",
        "yug2", "novokuibyshevsk", "kinel", "smyshlyaevka", "stroykeramika",
    ],
)
def test_perimeter_places_recognised_in_word_forms(text, expected, branches):
    assert sp.match_perimeter_place(text, branches) == expected


@pytest.mark.parametrize(
    "text",
    [
        "в Тольятти",
        "Ульяновск",
        "в Оренбурге",
        "Саратов",
        "в Самаре",
        "сколько стоит ОАК",
        "адреса филиалов",
        # Ложные срабатывания коротких основ:
        "сходил в кино",
        "южная улица",
    ],
    ids=[
        "tolyatti", "ulyanovsk", "orenburg", "saratov", "samara",
        "price", "addresses", "kino_not_kinel", "yuzhnaya_street",
    ],
)
def test_outside_perimeter_and_noise_not_matched(text, branches):
    assert sp.match_perimeter_place(text, branches) is None


# --- Филиал для места ------------------------------------------------------

@pytest.mark.parametrize(
    "place, fragment",
    [
        ("южный город", "Николаевский"),
        ("придорожный", "Николаевский"),
        ("мехзавод", "Мехзавод"),
        ("юг-2", "ЮГ-2"),
    ],
    ids=["yuzhny", "pridorozhny", "mehzavod", "yug2"],
)
def test_branch_resolved_for_place_with_own_branch(place, fragment, branches):
    """При явном упоминании места отвечаем КОНКРЕТНЫМ филиалом, а не дампом
    восемнадцати адресов Самары."""
    name = sp.branch_name_for_place(place, branches)
    assert name and fragment.lower() in name.lower(), (place, name)


@pytest.mark.parametrize("place", ["новокуйбышевск", "кинель", "смышляевка", "стройкерамика"])
def test_satellites_have_no_own_branch(place, branches):
    """Спутники своих филиалов не имеют — обслуживаем как Самару в целом,
    филиал не подставляем."""
    assert sp.branch_name_for_place(place, branches) is None


# --- Границы ---------------------------------------------------------------

def test_empty_and_missing_data_are_safe():
    """Без данных филиалов остаются только спутники (они не из данных, а из
    решения владельца) — места филиалов пропадают, падений нет."""
    assert sp.match_perimeter_place("", []) is None
    assert sp.match_perimeter_place("в Новокуйбышевске", []) == "новокуйбышевск"
    assert sp.match_perimeter_place("южный город", []) is None
    assert sp.branch_name_for_place("", []) is None


# --- E2E: ранний city-gate роутера ----------------------------------------

def _run_stream_once(user_text: str, state, services, memory):
    import asyncio

    from messengers_router import router as router_mod

    async def _collect():
        out = []
        async for env in router_mod.patient_routing_stream(user_text, state, services, memory):
            out.append(env)
        return out

    return asyncio.run(_collect())


def test_branch_address_no_longer_reads_as_foreign_city(monkeypatch):
    """«пр-т Николаевский» — адрес собственного филиала в Южном городе, а не
    город «Николаевский» из справочника. Ранний гард обязан пропустить реплику
    дальше, а не увести к оператору «только по Самаре»."""
    from messengers_router.memory import MemoryStore
    from messengers_router.mess_types import ResponseEnvelope, SessionState
    from messengers_router.services import Services

    async def fake_run_pipeline(*args, **kwargs):
        _ = args, kwargs
        yield ResponseEnvelope(text="прошли дальше", handoff=False)

    monkeypatch.setattr("messengers_router.orchestrator.run_pipeline", fake_run_pipeline)

    services = Services()
    services.ensure_background_refresh_started = lambda: None
    out = _run_stream_once(
        "хочу сдать анализы на пр-т Николаевский, д. 2",
        SessionState(session_id="p4-nikolaevsky"),
        services,
        MemoryStore(),
    )

    assert out, "ответа нет"
    assert "только по Самаре" not in out[0].text, out[0].text


def test_outside_perimeter_city_still_refused(monkeypatch):
    """Анти-over-trigger: вне периметра прежний честный отказ сохраняется."""
    from messengers_router.memory import MemoryStore
    from messengers_router.mess_types import SessionState
    from messengers_router.services import Services

    async def fail_run_pipeline(*args, **kwargs):
        _ = args, kwargs
        raise AssertionError("city-gate должен сработать ДО оркестратора")

    monkeypatch.setattr("messengers_router.orchestrator.run_pipeline", fail_run_pipeline)

    services = Services()
    services.ensure_background_refresh_started = lambda: None
    out = _run_stream_once(
        "хочу записаться в Оренбурге",
        SessionState(session_id="p4-orenburg"),
        services,
        MemoryStore(),
    )

    assert len(out) == 1
    assert out[0].handoff is True
    assert "только по Самаре" in out[0].text


@pytest.mark.parametrize(
    "value, expected",
    [
        ("Николаевский", True),   # пр-т Николаевский — адрес нашего филиала
        ("Тольятти", False),
        ("Ульяновск", False),
        ("Оренбург", False),
        ("Саратов", False),
        ("Спутник", False),
        ("", False),
    ],
    ids=["nikolaevsky", "tolyatti", "ulyanovsk", "orenburg", "saratov", "sputnik", "empty"],
)
def test_branch_address_word_predicate(value, expected, branches):
    """Отдельная проверка: «город» из справочника, который на деле слово из
    НАШЕГО адреса, чужим городом не считается. Проверка не зависит от источника
    среза филиалов (живой снапшот или seed) — слово есть в обоих."""
    assert sp.is_branch_address_word(value, branches) is expected


def test_word_stem_keeps_short_roots_intact():
    """Основа не режется до ложных совпадений: «кино» ≠ «Кинель»."""
    assert sp._word_stem("кино") == "кино"
    assert sp._word_stem("кинель") == sp._word_stem("кинеле")
    assert sp._word_stem("южном") == sp._word_stem("южный")
