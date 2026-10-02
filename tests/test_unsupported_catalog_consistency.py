"""Класс `unsupported_list_denies_existing_specialist` (29.09).

Бот отвечал «Данные врачи не ведут приём» на любой вопрос про офтальмолога, хотя
в клинике есть подразделение «Врач-офтальмолог». Офтальмолог стоял в
`_UNSUPPORTED_SPECIALIST_PATTERNS` с тех пор, когда его в клинике не было:
список «у нас такого нет» захардкожен и с данными клиники не сверялся.

Инвариант: специалиста, который есть в ТЕКУЩЕМ справочнике врачей клиники, бот
не объявляет отсутствующим.

Источник правды — живой справочник, а не карта канонизации: карта хранит всех,
кто КОГДА-ЛИБО был в МИС. «Челюстно-лицевой хирург» есть в карте с июня, но
сейчас такого врача в клинике нет, и отказ по нему верен.

Поэтому проверка по живому справочнику — @pytest.mark.live: её вердикт зависит
от данных клиники. В гейте — решение владельца 29.09 по офтальмологу.
"""

import glob
import json
from pathlib import Path

import pytest

from messengers_router.policies import detect_unsupported_catalog
from messengers_router.services._unit_canonicalisation import canonicalise_unit

_REPO = Path(__file__).resolve().parents[1]


def _live_units() -> set[str]:
    files = sorted(
        glob.glob(str(_REPO / "agent_logic_2/nayka_api/apidata/doctors_*.jsonl"))
        + glob.glob(str(_REPO / "app_data/nayka_api/apidata/doctors_*.jsonl"))
    )
    if not files:
        pytest.skip("справочника врачей в этом checkout нет")
    units: set[str] = set()
    with open(files[-1], encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                units.update(u.strip() for u in row.get("units") or [] if isinstance(u, str) and u.strip())
    return units


@pytest.mark.live
def test_specialist_present_in_clinic_is_never_denied():
    denied = []
    for unit in sorted(_live_units()):
        for spec in canonicalise_unit(unit) or ():
            match = detect_unsupported_catalog(f"есть ли у вас {spec}")
            if match is not None and match.kind == "unsupported_specialist":
                denied.append((unit, spec, match.canonical_name))
    assert not denied, f"бот объявляет отсутствующими врачей, которые есть в клинике: {denied}"


@pytest.mark.parametrize(
    "text",
    ["есть ли у вас офтальмолог", "сколько стоит приём офтальмолога", "запишите к офтальмологу"],
)
def test_ophthalmologist_is_not_denied(text: str):
    # Решение владельца 29.09: офтальмолог в клинике есть («Врач-офтальмолог»).
    match = detect_unsupported_catalog(text)
    assert match is None or match.kind != "unsupported_specialist", match


# --- Класс `stoplist_refuses_lab_test` (02.10) --------------------------------
#
# «хочу сделать анализ на гепатит с», «можно сделать тест на грипп» → «клиника не
# оказывает данную услугу». Стоп-лист прививок ловил «сделать/поставить» рядом с
# названием болезни, а анализы на эти инфекции в прайсе есть. Инвариант: анализ,
# который есть в прайсе, стоп-лист не объявляет недоступным; запрос самой прививки
# — по-прежнему отказ (решение владельца: прививок в клинике нет).
# См. BUG-2026-10-02-STOPLIST-REFUSES-LAB-TEST.

_INFECTION_WORDS = ("гепатит", "грипп", "кори", "корь", "краснух", "паротит", "ветрян", "полиомиелит", "пневмокок")


def _infection_test_names() -> list[str]:
    from agent_logic_2.nayka_api import api_price
    from messengers_router.services._prices_helpers import SAMARA_PRICE_REGION_ID

    rows = [r for r in api_price.load_price_by_region(SAMARA_PRICE_REGION_ID) if isinstance(r, dict)]
    assert len(rows) > 1000, "нужен живой прайс региона"
    names = []
    for row in rows:
        name = str(row.get("serviceName") or "")
        low = name.lower()
        if any(word in low for word in _INFECTION_WORDS) and "вакцин" not in low and "привив" not in low:
            names.append(name)
    return names


def test_lab_test_for_infection_in_price_is_not_refused():
    names = _infection_test_names()
    assert len(names) >= 10, f"мало анализов на инфекции в прайсе ({len(names)}) — выборка не о том"
    refused = []
    for name in names:
        words = " ".join(name.split()[:3])
        for phrase in (f"хочу сделать анализ на {words}", f"можно сделать тест {words}", f"поставить анализ {words}"):
            match = detect_unsupported_catalog(phrase)
            if match is not None:
                refused.append((phrase, match.canonical_name))
    assert not refused, f"анализ из прайса объявлен недоступным в {len(refused)} фразах: {refused[:5]}"


@pytest.mark.parametrize(
    "text",
    ["хочу сделать анализ на гепатит с", "можно сделать тест на грипп", "сделать анализ на корь"],
)
def test_reported_lab_tests_are_not_refused(text: str):
    assert detect_unsupported_catalog(text) is None


@pytest.mark.parametrize(
    "text",
    ["сделать прививку от гриппа", "поставить прививку от кори", "вакцина от гепатита в", "сколько стоит вакцинация от клеща"],
)
def test_vaccination_requests_are_still_refused(text: str):
    match = detect_unsupported_catalog(text)
    assert match is not None and match.canonical_name == "вакцинация", match
