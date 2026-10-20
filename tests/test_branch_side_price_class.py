"""Побочный вопрос о цене внутри записи — цена услуги записи, не слов реплики (BUG-2026-10-20-BRANCH-SIDE-PRICE-FOREIGN).

Стенд `pa_branch_new_topic`, `pa_branch_price_side_then_pick` под `pending_buttons = on` (деплой 11, 170e352): «хочу
записаться к урологу» → «Какой филиал вам удобен?» → «а сколько стоит приём?» → «Прием … врача-хирурга — 3 000 руб.».
Побочный ответ (п. 2 облака 19.10) шёл обычным путём цены по словам реплики: «приём» без специальности находил первый
попавшийся приём; «а цена?» — сырое слово записи «Урологу» каталог по буквам уводил в «Послеоперационную консультацию»
(«Неврологу» — в «Невролиз», «Гинекологу» — в «Курс гинекологический»).

Инвариант класса: побочный ценовой вопрос внутри записи без своей услуги спрашивает цену услуги записи (специальность из
собранного), а не ищет по словам реплики; строки не той специальности, сбой, цели нет — без цены. Собранное записи цело,
вопрос о филиале звучит снова. Тесты — через обработчик эндпоинта; LLM — подделка настоящего типа (выбор строки прайса
берёт первого кандидата — как прод-LLM, когда кандидаты только чужие).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from evalbench import world as ev_world
from messengers_router import expects
from tests.test_booking_slots_by_meaning import _LLM
from tests.test_pending_buttons_class import _CLASSIFIER_MARK, _OFF, _ON, _PENDING_MARK, _classifier, _collected, _in_world

_TESTS = Path(__file__).resolve().parent
_PRICE_MARK = "ответить пациенту на вопрос о цене"
_BRANCH_TAIL = "Какой филиал вам удобен? Или написать список врачей по этой услуге?"

# Основа приёма каждой специальности в названиях строк прайса и в подразделениях врачей.
_STEMS = ("уролог", "гинеколог", "невролог", "хирург")
_SPECIALTIES = {"урологу": "уролог", "гинекологу": "гинеколог", "неврологу": "невролог"}
_REPLIES = ("а сколько стоит приём?", "а цена?", "сколько это стоит")
# Строки настоящего прайса, куда по буквам уходили слова записи и «приём» (BUG-2026-09-25-SPECIALTY-HITS-PROCEDURE, свип
# 20.10 на мире 20261008): приём хирурга; «Урологу», «Неврологу», «Гинекологу» — не приём.
_TRAPS = ("Послеоперационная консультация уролога", "Невролиз и декомпрессия нерва", "Курс гинекологический (хронические состояния)")
_ADDED_ROWS = (
    ("Прием (осмотр, консультация) врача-хирурга", 3000.0),
    ("Прием (осмотр, консультация) врача-хирурга (флеболога)", 3000.0),
    (_TRAPS[0], 1300.0),
    (_TRAPS[1], 30000.0),
    (_TRAPS[2], 39600.0),
)


@pytest.fixture(scope="module")
def side_price_world(tmp_path_factory) -> Path:
    """Мир: прайс `mini_prod` + приём хирурга (на настоящем прайсе «приём» по словам находит его первым), выдержка врачей
    и цены услуг своей специальности у врачей (без них ответ о цене приёма — перевод, и побочного ответа нет)."""

    src = side_price_source(tmp_path_factory.mktemp("side_price_src"))
    return ev_world.build_world(src, "20261008", tmp_path_factory.mktemp("side_price") / "world")


def side_price_source(src: Path, doctor=lambda doctor: doctor) -> Path:
    """Срезы мира `side_price_world` в каталоге `src`; `doctor` правит строку врача до цен врачей (мир другого теста)."""

    shutil.copytree(_TESTS / "fixtures" / "mini_prod" / "snapshots", src, dirs_exist_ok=True)
    excerpt = (_TESTS / "fixtures" / "doctors_excerpt" / "doctors_20261008.jsonl").read_text(encoding="utf-8").splitlines()
    (src / "doctors_20261008.jsonl").write_text(
        "".join(json.dumps(doctor(json.loads(line)), ensure_ascii=False) + "\n" for line in excerpt if line.strip()), encoding="utf-8"
    )
    price_path = src / "price_by_region" / "price_region_3_20261008.jsonl"
    rows = [json.loads(line) for line in price_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    template = next(row for row in rows if row["serviceName"] == "Прием (осмотр, консультация) врача-уролога")
    for offset, (name, cost) in enumerate(_ADDED_ROWS):
        rows.insert(offset, {**template, "serviceId": 990001 + offset, "serviceHomecode": f"99.1.{offset + 1}", "serviceName": name, "cost": cost})
    price_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    doctors = [json.loads(line) for line in (src / "doctors_20261008.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    doctor_prices = []
    for doctor in doctors:
        units = " ".join(str(unit) for unit in doctor.get("units") or ()).lower()
        for stem in _STEMS:
            if stem not in units:
                continue
            for row in rows:
                if stem in row["serviceName"].lower():
                    doctor_prices.append(
                        {
                            "doctorId": doctor["id"],
                            "regionId": (doctor.get("region_ids") or [None])[0],
                            "regionName": (doctor.get("regions") or [""])[0],
                            "fio": doctor["fio"],
                            "serviceName": row["serviceName"],
                            "serviceId": None,
                            "cost": row["cost"],
                            "priceUnitId": None,
                            "serviceHomecode": row["serviceHomecode"],
                            "deadline": None,
                        }
                    )
    (src / "doctor_prices").mkdir()
    (src / "doctor_prices" / "doctor_prices_20261008.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in doctor_prices), encoding="utf-8"
    )
    return src


def _llm(book: str) -> _LLM:
    return _LLM(
        {
            _PENDING_MARK: json.dumps({"picks": [], "outcome": "none"}),  # «ничего из этого» — вопрос о вопросе бота
            _CLASSIFIER_MARK: lambda prompt: _classifier("APPOINTMENT") if book in prompt else _classifier("PRICE"),
            _PRICE_MARK: json.dumps({"picks": [1]}),
        }
    )


def _foreign(text: str, own: str) -> list[str]:
    low = text.lower()
    return [stem for stem in _STEMS if stem != own and stem in low] + [trap for trap in _TRAPS if trap.lower() in low]


@pytest.mark.parametrize("reply", _REPLIES)
@pytest.mark.parametrize("target", sorted(_SPECIALTIES))
def test_side_price_question_is_about_the_booking_service(target, reply, side_price_world, monkeypatch):
    own, book = _SPECIALTIES[target], f"хочу записаться к {target}"
    asked, side, picked = _in_world(side_price_world, monkeypatch, [book, reply, "1"], switches=_ON, llm=_llm(book))

    assert asked["text"].endswith(_BRANCH_TAIL), asked["text"]
    assert "appointment_branch_side_question" in side["flags"] and side["label"] == "PRICE", side
    assert "руб." in side["text"] and own in side["text"].lower(), side["text"]  # цена приёма своей специальности
    assert not _foreign(side["text"], own), side["text"]  # ни одной чужой строки
    assert side["text"].endswith("\n\n" + _BRANCH_TAIL), side["text"]
    assert _collected(side["state"]) == _collected(asked["state"])  # собранное записи цело
    assert side["state"][expects.EXPECTS_KEY]["options"] == asked["state"][expects.EXPECTS_KEY]["options"]
    assert picked["state"].get("branch_name") == asked["state"][expects.EXPECTS_KEY]["options"][0], picked


@pytest.mark.parametrize("target", sorted(_SPECIALTIES))
def test_side_price_question_without_the_model_decision_is_as_before(target, side_price_world, monkeypatch):
    own, book = _SPECIALTIES[target], f"хочу записаться к {target}"
    llm = _LLM({_CLASSIFIER_MARK: lambda prompt: _classifier("APPOINTMENT") if book in prompt else _classifier("PRICE")})
    on = _in_world(side_price_world, monkeypatch, [book, "а сколько стоит приём?"], switches=_ON, llm=llm)[-1]
    off = _in_world(side_price_world, monkeypatch, [book, "а сколько стоит приём?"], switches=_OFF, llm=llm)[-1]

    assert on["text"] == off["text"] and "appointment_branch_side_question" not in on["flags"]
    assert not _foreign(on["text"], own), on["text"]


def test_side_price_question_naming_its_own_service_is_searched_by_its_words(side_price_world, monkeypatch):
    """Своя услуга в реплике — путь цены по её словам, как раньше: «приём гинеколога» внутри записи к урологу."""

    book = "хочу записаться к урологу"
    side = _in_world(side_price_world, monkeypatch, [book, "а сколько стоит приём гинеколога?"], switches=_ON, llm=_llm(book))[-1]

    assert "appointment_branch_side_question" in side["flags"], side
    assert "гинеколог" in side["text"].lower() and not _foreign(side["text"], "гинеколог"), side["text"]


def test_booking_price_service_of_the_collected():
    """Услуга цены записи — из собранного: специальность — приём специальности, услуга — она сама, цели нет — пусто."""

    from messengers_router.router import _booking_price_service

    assert _booking_price_service({"specialty": "уролог", "service_name": "Урологу"}) == ("прием уролог", "уролог")
    assert _booking_price_service({"specialty": "невролог"}) == ("прием невролог", "невролог")
    assert _booking_price_service({"service_name": "УЗИ почек"}) == ("УЗИ почек", "")
    assert _booking_price_service({}) == ("", "")
