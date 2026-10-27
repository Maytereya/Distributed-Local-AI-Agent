"""Стенд замеров (`evalbench`): прибор сам должен быть исправен.

Инварианты класса «прибор врёт молча» (мастер-хэндофф §4 (а), замер самого прибора):
- каждый промпт бота распознаётся своим местом вызова, ни одна пара шаблонов не путается —
  иначе правка промпта тихо переводит вызов в чужое место и adversary отвечает не туда;
- у каждого живого места есть ответ adversary настоящего формата;
- мир — копия: сверка манифеста ловит удаление и правку срезов;
- сеть закрыта: попытка выйти наружу падает сразу и записывается;
- цены в тексте сверяются с прайсом, а не с кодом цен бота.
"""

from __future__ import annotations

import asyncio
import json
import re
import socket
import time
from pathlib import Path

import pytest

from evalbench import llm as ev_llm
from evalbench import world as ev_world
from evalbench.detectors import Templates, classify_turn
from evalbench.prices import PriceCatalog, multi_coverage, price_verdict

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "messengers_router" / "prompts"
# Загружаются, но в LLM не уходят (разведка 09.10): ответ adversary им не нужен.
DEAD_PROMPTS = {"recovery_patient", "messenger_final_answer"}


def _templates() -> dict[str, str]:
    from messengers_router import prompt_registry

    return {p.stem: prompt_registry.load_prompt_text(p.stem) for p in sorted(PROMPTS.glob("*.txt"))}


def _filled(template: str) -> str:
    return re.sub(r"<<[A-Z0-9_]+>>", "сколько стоит общий анализ крови", template)


def test_every_prompt_is_recognised_as_its_own_site():
    templates = _templates()
    index = ev_llm.SiteIndex.from_templates(templates)
    assert not index.ambiguous_pairs()
    for key, text in templates.items():
        assert index.site_of(_filled(text)) == key, key


def test_unknown_prompt_is_not_silently_assigned():
    index = ev_llm.SiteIndex.from_templates(_templates())
    assert index.site_of("Произвольный промпт, которого нет среди шаблонов бота.") == "unknown"
    assert index.site_of(ev_llm.WARM_UP_PROMPT) == "warm_up"


@pytest.mark.parametrize("key", sorted(p.stem for p in PROMPTS.glob("*.txt") if p.stem not in DEAD_PROMPTS))
def test_adversary_has_parsable_answer_for_every_live_site(key):
    text = _filled(_templates()[key])
    answer = ev_llm.adversary_answer(key, text)
    assert answer, key
    if key in {"classifier_patient", "classifier_refine_patient", "price_service_select", "prepare_document_select",
               "prepare_same_memo", "price_kind_ambiguous", "renderer_critic_patient_alignment"} or key.startswith("choose_"):
        json.loads(answer)


def test_adversary_never_raises_urgency():
    """Монотонная безопасность: adversary только понижает срочность — замер полноты срочного честный."""

    for prompt in ("задыхаюсь", "боль в груди", "не хочу жить"):
        assert json.loads(ev_llm.adversary_answer("classifier_patient", prompt))["label"] != "URGENT"
    assert ev_llm.adversary_answer("symptom_specialists", "задыхаюсь") == "НЕТ"


def test_adversary_choice_picks_the_last_option_and_never_an_outcome():
    # Модуль выбора (ход (в)): худший правдоподобный ответ — последний вариант; срочность не поднимает.
    numbered = "Инструкция.\n- правило в инструкции\n\nСообщение пациента: «оак»\n\nВарианты:\n1. ОАК — 500 руб.\n2. ОАМ — 300 руб."
    named = "Инструкция.\n\nСообщение пациента: «болит сердце»\n\nВрачи:\n- терапевт\n- кардиолог"
    empty = "Инструкция.\n\nСообщение пациента: «оак»\n\nВарианты:\n(вариантов нет)"
    assert json.loads(ev_llm.adversary_answer("choose_demo", numbered))["picks"] == [2]
    assert json.loads(ev_llm.adversary_answer("choose_demo", named))["picks"] == ["кардиолог"]
    assert json.loads(ev_llm.adversary_answer("choose_demo", empty))["picks"] == []
    assert {json.loads(ev_llm.adversary_answer("choose_demo", p))["outcome"] for p in (numbered, named, empty)} == {"none"}


def test_adversary_price_select_picks_last_pool_row():
    prompt = "Вопрос\n1. ОАК — 500 руб.\n2. Глюкоза — 300 руб.\n3. ТТГ — 600 руб.\nОтветь только JSON: {\"match\": [номера]}"
    assert json.loads(ev_llm.adversary_answer("price_service_select", prompt)) == {"match": [3], "search_terms": []}


def test_client_modes_return_real_ollama_types_and_count_calls():
    from ollama import GenerateResponse

    index = ev_llm.SiteIndex.from_templates({"service_normalizer": _templates()["service_normalizer"]})
    prompt = _filled(_templates()["service_normalizer"])

    adversary = ev_llm.HarnessClient(mode="adversary", sites=index)
    res = asyncio.run(adversary.generate(model="m", prompt=prompt))
    assert isinstance(res, GenerateResponse) and res.response == "Общий анализ крови"

    down = ev_llm.HarnessClient(mode="down", sites=index)
    with pytest.raises(ConnectionError):
        asyncio.run(down.generate(model="m", prompt=prompt))
    with pytest.raises(ConnectionError):
        asyncio.run(down.generate(model="m", prompt="чужой промпт без шаблона"))
    assert [c.outcome for c in down.take_turn_calls()] == ["down", "unknown_site"]


def test_down_timeout_hangs_until_the_call_site_timeout():
    """`down_timeout` с `hang_s` — зависшая LLM: вызов ждёт тайм-аут места вызова (исход `cancelled`), без него — тайм-аут
    после зависания; `hang_s = 0` — тайм-аут сразу, как было (BUG-2026-10-27-CLASSIFIER-NO-TIMEOUT: время хода off/on)."""

    index = ev_llm.SiteIndex.from_templates({"service_normalizer": _templates()["service_normalizer"]})
    prompt = _filled(_templates()["service_normalizer"])

    async def call(client, site_timeout: float) -> float:
        started = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(client.generate(model="m", prompt=prompt), site_timeout)
        return time.monotonic() - started

    hung = ev_llm.HarnessClient(mode="down_timeout", sites=index, hang_s=10)
    assert 0.05 <= asyncio.run(call(hung, 0.05)) < 1
    short = ev_llm.HarnessClient(mode="down_timeout", sites=index, hang_s=0.05)
    assert asyncio.run(call(short, 5)) >= 0.05
    instant = ev_llm.HarnessClient(mode="down_timeout", sites=index)
    assert asyncio.run(call(instant, 5)) < 0.05
    assert [c.outcome for c in hung.take_turn_calls() + short.take_turn_calls() + instant.take_turn_calls()] == ["cancelled", "timeout", "timeout"]


def test_replay_serves_recorded_answer_and_counts_miss(tmp_path):
    index = ev_llm.SiteIndex.from_templates({"service_normalizer": _templates()["service_normalizer"]})
    prompt = _filled(_templates()["service_normalizer"])
    cache = ev_llm.ResponseCache(tmp_path / "cache.jsonl")
    cache.put(ev_llm.ResponseCache.key("m", prompt, None, False), "глюкоза", "service_normalizer")

    replay = ev_llm.HarnessClient(mode="replay", sites=index, cache=ev_llm.ResponseCache(tmp_path / "cache.jsonl"))
    assert asyncio.run(replay.generate(model="m", prompt=prompt)).response == "глюкоза"
    with pytest.raises(ConnectionError):
        asyncio.run(replay.generate(model="m", prompt=prompt + " другое"))
    assert [c.outcome for c in replay.take_turn_calls()] == ["cache_hit", "replay_miss"]


_ROLE_TEMPLATE = (
    "Инструкция выбора для проверки стенда: выбери номер позиции из списка ниже.\n"
    "<<DATA>>\n"
    "Сообщение пациента: «<<QUESTION>>»\n"
)


def test_site_of_role_call_is_found_by_its_system_instruction():
    # Ход (в): инструкция — в роли system, в `prompt` только данные хода; место вызова — по обоим.
    index = ev_llm.SiteIndex.from_templates({"choose_demo": _ROLE_TEMPLATE})
    client = ev_llm.HarnessClient(mode="down", sites=index)
    system = _ROLE_TEMPLATE.split("<<DATA>>")[0].strip()
    for kwargs in ({"system": system, "prompt": "Сообщение пациента: «оак»"}, {"prompt": "Сообщение пациента: «оак»"}):
        with pytest.raises(ConnectionError):
            asyncio.run(client.generate(model="m", **kwargs))
    assert [(c.site, c.outcome) for c in client.take_turn_calls()] == [("choose_demo", "down"), ("unknown", "unknown_site")]


def test_cache_key_of_recorded_answers_is_unchanged_and_system_is_part_of_new_keys():
    import hashlib

    old = hashlib.sha256(json.dumps(["m", "json", False, "p"], ensure_ascii=False).encode("utf-8")).hexdigest()
    assert ev_llm.ResponseCache.key("m", "p", "json", False) == old
    assert ev_llm.ResponseCache.key("m", "p", "json", False, system="S1") != ev_llm.ResponseCache.key("m", "p", "json", False, system="S2")
    schema = {"type": "object", "properties": {"a": {}, "b": {}}}
    same = {"properties": {"b": {}, "a": {}}, "type": "object"}
    assert ev_llm.ResponseCache.key("m", "p", schema, False) == ev_llm.ResponseCache.key("m", "p", same, False)


def test_role_call_is_replayed_and_passed_to_the_real_client(tmp_path):
    index = ev_llm.SiteIndex.from_templates({"choose_demo": _ROLE_TEMPLATE})
    system = _ROLE_TEMPLATE.split("<<DATA>>")[0].strip()
    cache = ev_llm.ResponseCache(tmp_path / "cache.jsonl")
    cache.put(ev_llm.ResponseCache.key("m", "данные", {"type": "object"}, False, system=system), '{"picks": [1]}', "choose_demo")
    replay = ev_llm.HarnessClient(mode="replay", sites=index, cache=ev_llm.ResponseCache(tmp_path / "cache.jsonl"))
    assert asyncio.run(replay.generate(model="m", prompt="данные", system=system, format={"type": "object"})).response == '{"picks": [1]}'

    seen: list[dict] = []

    class _Real:
        async def generate(self, **kwargs):
            seen.append(kwargs)
            from ollama import GenerateResponse

            return GenerateResponse(model="m", response='{"picks": []}', done=True)

    live = ev_llm.HarnessClient(mode="real", sites=index, real=_Real())
    asyncio.run(live.generate(model="m", prompt="данные", system=system, format={"type": "object"}))
    assert seen[0]["system"] == system and seen[0]["prompt"] == "данные"


# --------------------------------------------------------------------------- мир


def _apidata(tmp_path: Path) -> Path:
    src = tmp_path / "apidata"
    (src / "price_by_region").mkdir(parents=True)
    (src / "service_info").mkdir()
    (src / "price_by_region" / "price_region_3_20261007.jsonl").write_text('{"a":1}\n')
    (src / "price_by_region" / "price_region_3_20261008.jsonl").write_text('{"a":2}\n')
    (src / "price_by_region" / "price_region_3_20261009.jsonl").write_text('{"a":3}\n')
    (src / "service_info" / "service_info_20261003.jsonl").write_text('{"s":1}\n')
    (src / "service_info" / "service_info_20261008.jsonl").write_text("")  # пустой срез не годится
    (src / "doctors_20261008.jsonl").write_text('{"d":1}\n')
    return src


def test_world_takes_latest_nonempty_snapshot_not_after_date(tmp_path):
    names = sorted(p.name for p in ev_world.select_snapshots(_apidata(tmp_path), "20261008"))
    assert names == ["doctors_20261008.jsonl", "price_region_3_20261008.jsonl", "service_info_20261003.jsonl"]


def test_world_manifest_catches_deleted_changed_and_extra_files(tmp_path):
    root = ev_world.build_world(_apidata(tmp_path), "20261008", tmp_path / "world")
    assert ev_world.verify_world(root) == []
    (root / "apidata" / "doctors_20261008.jsonl").unlink()
    (root / "apidata" / "service_info" / "service_info_20261003.jsonl").write_text('{"s":2}\n')
    (root / "apidata" / "junk.jsonl").write_text("x")
    problems = ev_world.verify_world(root)
    assert any(p.startswith("удалён") for p in problems)
    assert any(p.startswith("изменён") for p in problems)
    assert any(p.startswith("лишний") for p in problems)


def test_world_clock_profiles():
    day = ev_world.frozen_now_utc("20261008", "day")
    night = ev_world.frozen_now_utc("20261008", "night")
    assert day.astimezone(ev_world.SAMARA).hour == 12
    samara_night = night.astimezone(ev_world.SAMARA)
    assert (samara_night.day, samara_night.hour) == (9, 1)
    frozen_dt, frozen_date = ev_world.frozen_clock(night)
    assert frozen_dt.now(ev_world.SAMARA).strftime("%Y%m%d") == "20261009"
    assert frozen_dt.now().strftime("%Y%m%d") == "20261008"  # часы машины — UTC, как в контейнере
    assert frozen_date.today().isoformat() == "2026-10-08"


def test_netguard_blocks_and_records_but_allows_listed_host():
    guard = ev_world.NetGuard(allow_hosts=frozenset({"llm.example"}))
    guard.install()
    try:
        with pytest.raises(ConnectionRefusedError):
            socket.create_connection(("10.255.255.1", 80), timeout=1)
        with pytest.raises(socket.gaierror):
            socket.getaddrinfo("mis.example", 80)
        assert guard._allowed("llm.example") and guard._allowed("127.0.0.1")
    finally:
        guard.uninstall()
    assert guard.attempts == ["10.255.255.1:80", "mis.example (DNS)"]


# --------------------------------------------------------------------------- цены и исходы

_ROWS = [
    {"serviceId": 1, "serviceName": "Общий анализ крови", "cost": 500.0},
    {"serviceId": 2, "serviceName": "Общий анализ крови с лейкоформулой", "cost": 750.0},
    {"serviceId": 3, "serviceName": "Глюкоза", "cost": 300.0},
]


def test_price_pairs_match_unsourced_and_unattributed():
    catalog = PriceCatalog(_ROWS, doctor_costs={2700}, promo_costs={3710})
    pairs = catalog.pairs(
        "Общий анализ крови с лейкоформулой — 750 руб.\n"
        "Глюкоза — 1 990 руб.\n"
        "Иванов Иван Иванович (Невролог) — 2 700 руб.;\n"
        "2. Петрова Анна Сергеевна — 2 700 руб.; расписание временно недоступно.\n"
        "Анализы перед операцией 3710 ₽ вместо 5300 ₽\n"
        "Стоимость этой услуги — 300 руб."
    )
    assert [(p.status, p.service_id) for p in pairs] == [
        ("matched", 2), ("unsourced", None), ("doctor", None), ("doctor", None), ("promo", None), ("unsourced", None), ("unattributed", None),
    ]


def test_price_verdicts():
    catalog = PriceCatalog(_ROWS)
    one_right = catalog.pairs("Общий анализ крови — 500 руб.")
    one_wrong = catalog.pairs("Глюкоза — 300 руб.")
    listing = catalog.pairs("1. Общий анализ крови — 500 руб.\n2. Глюкоза — 300 руб.")
    single = {"kind": "single", "core": [1], "ok": [2], "ambiguous": False}
    assert price_verdict(one_right, single) == "correct"
    assert price_verdict(one_wrong, single) == "foreign"
    assert price_verdict(listing, {"kind": "family", "core": [1], "ok": [], "ambiguous": True}) == "list_hit"
    assert price_verdict(listing, {"kind": "single", "core": [2], "ok": [], "ambiguous": False}) == "list_miss"
    assert price_verdict(one_right, {"kind": "family", "core": [1, 2], "ok": [], "ambiguous": True}) == "single_for_ambiguous"
    assert price_verdict(one_right, {"kind": "absent", "core": [], "ok": [], "ambiguous": False}) == "foreign"
    assert price_verdict([], single) == "no_price"
    assert price_verdict(one_right, None) is None
    assert price_verdict(one_right, {"kind": "context_dependent", "core": [], "ok": []}) is None


def test_multi_coverage_counts_named_list_positions():
    """Ответ одной услугой на вопрос о двух — не «список с нужной», а покрытие 1 из 2."""

    catalog = PriceCatalog(_ROWS)
    label = {"kind": "multi", "core": [1, 3], "ok": [2], "ambiguous": False}
    assert multi_coverage(catalog.pairs("Общий анализ крови — 500 руб."), label) == (1, 1, 2, 0)
    both = catalog.pairs("1. Общий анализ крови — 500 руб.\n2. Глюкоза — 300 руб.")
    assert multi_coverage(both, label) == (2, 2, 2, 0)
    # Строка вне разметки внутри списка видна отдельно, а не растворяется в «списке с нужной».
    only_glucose = {"kind": "multi", "core": [3], "ok": [], "ambiguous": False}
    assert multi_coverage(both, only_glucose) == (1, 1, 1, 1)
    assert multi_coverage(both, {"kind": "single", "core": [1], "ok": []}) is None


def test_basket_total_and_take_lines():
    """Корзина-калькулятор (`basket_calc`): итог — `total`, только если сходится с напечатанными позициями (выдуманный итог —
    «без источника»); строки перечня взятия — `take`, в услуги ответа (покрытие, вне разметки) не входят."""

    rows = [*_ROWS, {"serviceId": 9, "serviceName": "Взятие крови из вены", "cost": 195.0}]
    catalog = PriceCatalog(rows)
    head = (
        "По услуге «оак»:\n1. Общий анализ крови — 500 руб.\n2. Общий анализ крови с лейкоформулой — 750 руб.\n\n"
        "По услуге «Глюкоза»:\n1. Глюкоза — 300 руб.\n\n"
    )
    takes = "Взятие биоматериала оплачивается отдельно:\n— Взятие крови из вены — 195 руб.\nЕсли нужно, помогу."
    ok = catalog.pairs(head + "Итого: от 800 руб. до 1 050 руб. — зависит от варианта «оак».\n" + takes)
    assert [(p.status, p.amount) for p in ok if p.status != "matched"] == [("total", 800), ("total", 1050), ("take", 195)]
    label = {"kind": "multi", "core": [1, 3], "ok": [2], "ambiguous": False}
    assert multi_coverage(ok, label) == (3, 2, 2, 0)  # взятие — не «услуга вне разметки»
    wrong = catalog.pairs(head + "Итого: 1 050 руб.\n" + takes)
    assert [p.status for p in wrong if p.amount == 1050] == ["unsourced"]
    exact = catalog.pairs("По услуге «А»:\n1. Глюкоза — 300 руб.\n\nПо услуге «Б»:\n1. Общий анализ крови — 500 руб.\n\nИтого: 800 руб.")
    assert [p.status for p in exact] == ["matched", "matched", "total"]
    # Строка взятия вне перечня (пациент назвал её позицией) — обычная услуга ответа.
    assert [p.status for p in catalog.pairs("1. Взятие крови из вены — 195 руб.")] == ["matched"]


def test_run_cfg_override_parsing():
    from evalbench.run import _parse_cfg

    assert _parse_cfg("MR_TURN_UNDERSTANDING_PRICE=on") == ("MR_TURN_UNDERSTANDING_PRICE", "on")
    assert _parse_cfg("MR_TURN_TELEMETRY=false") == ("MR_TURN_TELEMETRY", False)
    assert _parse_cfg("MR_DOCTORS_TOP_N=5") == ("MR_DOCTORS_TOP_N", 5)
    with pytest.raises(ValueError):
        _parse_cfg("без_знака")


def _body(text, *, handoff=False, label=None, flags=(), error=None):
    debug = {"decision": {"label": label, "flags": list(flags), "source": "x"}}
    if error:
        debug["route_error"] = error
    return {"text": text, "handoff": handoff, "state_update": {"debug": debug}}


def test_turn_outcomes_from_envelope_debug_and_session_state():
    tpl = Templates.from_bot()
    catalog = PriceCatalog(_ROWS)
    urgent = classify_turn(_body("Соединяю с оператором.\n\n" + tpl.hours_note, handoff=True, label="URGENT"), {}, tpl, catalog)
    assert urgent.outcome == "urgent"
    manual = classify_turn(_body(tpl.handoff["manual_operator"], handoff=True, label="OTHER"), {}, tpl, catalog)
    assert (manual.outcome, manual.handoff_reason) == ("handoff", "manual_operator")
    tech = classify_turn(_body(tpl.handoff["service_error"], handoff=True), {}, tpl, catalog)
    assert tech.outcome == "handoff_tech"
    offer = classify_turn(_body("Что-то. Перевести?"), {"_operator_offer_pending": True}, tpl, catalog)
    assert offer.outcome == "operator_offer"
    price = classify_turn(_body("Глюкоза — 300 руб."), {}, tpl, catalog)
    assert price.outcome == "price" and price.price_pairs[0].service_id == 3
    nothing = classify_turn(_body("Не нашёл актуальную стоимость для «ХХХ»."), {}, tpl, catalog)
    assert nothing.outcome == "no_info"
    broken = classify_turn(_body("", handoff=True, error="boom"), {}, tpl, catalog)
    assert broken.outcome == "tech_error"


def test_world_keeps_turn_telemetry_out_of_the_real_journal(tmp_path):
    """Стенд на версии с телеметрией: строки хода — в память прогона, настоящий журнал не трогается."""

    from messengers_router import turn_telemetry

    root = ev_world.build_world(_apidata(tmp_path), "20261008", tmp_path / "world")
    real_emit = turn_telemetry._emit
    with ev_world.FrozenWorld(root, "day") as world:
        token = turn_telemetry.begin("once", session_id="s")
        turn_telemetry.note(answer_source="prebuilt")
        turn_telemetry.finish(token, handoff=False, chars=3)
        row = world.take_telemetry()
    assert row["answer_source"] == "prebuilt" and world.take_telemetry() is None
    assert turn_telemetry._emit is real_emit


def test_world_pins_missing_today_snapshot_and_keeps_live_mis_offline(tmp_path):
    """Нет среза «на сегодня» — последний непустой, без похода в МИС; живые методы МИС — отказ без сети."""

    import requests

    from agent_logic_2.nayka_api import api_nayka, api_service_info

    root = ev_world.build_world(_apidata(tmp_path), "20261008", tmp_path / "world")
    with ev_world.FrozenWorld(root, "night") as world:  # ночь: «сегодня» по Самаре — 20261009
        assert api_service_info.service_info_path().name == "service_info_20261003.jsonl"
        assert api_service_info.service_info_path("20261001").name == "service_info_20261001.jsonl"
        with pytest.raises(requests.ConnectionError):
            api_nayka._session_get("http://mis/api/v1/site/doctors")
        assert world.unavailable_calls == ["v1/site/doctors"] and world.net.attempts == []


def test_world_saves_synonym_journal_before_its_temp_dir_is_gone(tmp_path):
    """Журнал синонимов прогона — копией в каталог прогона: временный каталог мира удаляется на выходе,
    а журнал смотрят глазами (13.10: «Цена анализа» в журнале не видела ни одна метрика стенда)."""

    from messengers_router.services import _price_select_llm

    root = ev_world.build_world(_apidata(tmp_path), "20261008", tmp_path / "world")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with ev_world.FrozenWorld(root, "day") as world:
        assert world.save_journal(run_dir / "empty.jsonl") == 0 and world.journal_tail(0) == []
        journal = _price_select_llm._journal_path()
        journal.write_text('{"phrase": "ферритин"}\n{"phrase": "ггтп"}\n', encoding="utf-8")
        assert world.journal_tail(1) == ['{"phrase": "ггтп"}'] and world.journal_tail(2) == []
        assert world.save_journal(run_dir / "synonym_journal.jsonl") == 2
    assert not journal.exists()
    assert (run_dir / "empty.jsonl").read_text(encoding="utf-8") == ""
    assert (run_dir / "synonym_journal.jsonl").read_text(encoding="utf-8").splitlines() == ['{"phrase": "ферритин"}', '{"phrase": "ггтп"}']


def test_merge_joins_synonym_journals_and_sums_their_lines(tmp_path, monkeypatch):
    """Склейка частей: журнал — строки всех частей, счётчик строк — сумма, а не первая часть."""

    from evalbench import merge as ev_merge

    monkeypatch.setattr(ev_merge, "PRIVATE_ROOT", tmp_path)
    for k, phrase in ((1, "ферритин"), (2, "ггтп")):
        part = tmp_path / "runs" / f"r.part{k}of2"
        part.mkdir(parents=True)
        header = {"name": part.name, "sets": ["g.jsonl"], "cases": 1, "done": 1, "cfg": {}, "valid": True, "synonym_journal_lines": 1}
        (part / "run.json").write_text(json.dumps(header), encoding="utf-8")
        (part / "turns.jsonl").write_text(json.dumps({"set": "g", "id": str(k)}) + "\n", encoding="utf-8")
        (part / "synonym_journal.jsonl").write_text(json.dumps({"phrase": phrase}, ensure_ascii=False) + "\n", encoding="utf-8")
    out = ev_merge.merge("r")
    head = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert head["synonym_journal_lines"] == 2 and head["valid"] is True
    assert [json.loads(line)["phrase"] for line in (out / "synonym_journal.jsonl").open(encoding="utf-8")] == ["ферритин", "ггтп"]


# --------------------------------------------------------------------------- симптом → врач

from evalbench.specialties import SpecialtyMatcher, specialty_verdict  # noqa: E402

_ROLES = ["терапевт", "мануальный терапевт", "уролог", "уролог-андролог", "хирург, флеболог", "оториноларинголог", "акушер-гинеколог", "невролог"]


def test_specialties_named_with_cases_aliases_and_longest_match():
    m = SpecialtyMatcher(_ROLES)
    assert m.named("обращаются к специалистам нашей клиники: невролог, терапевт.") == ["невролог", "терапевт"]
    assert m.named("Запишитесь к неврологу или к мануальному терапевту") == ["невролог", "мануальный терапевт"]
    assert m.named("1. Иванов Иван Иванович (Лор) — 2 000 руб.") == ["оториноларинголог"]
    assert m.named("к урологу-андрологу") == ["уролог-андролог"]
    assert m.named("Флеболог принимает на Ленина") == ["флеболог"]
    assert m.named("гинекологу") == ["акушер-гинеколог"]
    assert m.named("Сколько стоит УЗИ?") == []


def test_specialty_verdicts():
    core, ok = ["невролог"], ["терапевт"]
    assert specialty_verdict(["невролог"], core, ok) == "right"
    assert specialty_verdict(["терапевт"], core, ok) == "acceptable"
    assert specialty_verdict(["невролог", "уролог"], core, ok) == "partial"
    assert specialty_verdict(["уролог"], core, ok) == "wrong"
    assert specialty_verdict([], core, ok) == "none"
    combined = ["сердечно-сосудистый хирург, флеболог, лимфолог"]
    assert specialty_verdict(["флеболог", "лимфолог"], ["флеболог"], [], combined) == "right"
    assert specialty_verdict(["флеболог", "лимфолог"], ["флеболог"], []) == "partial"


def test_cancelled_live_call_is_recorded():
    """Вызов, оборванный тайм-аутом места вызова, попадает в счётчик (09.10: зависшие ходы были «без LLM»)."""

    class _Hang:
        async def generate(self, **_kw):
            await asyncio.sleep(10)

    index = ev_llm.SiteIndex.from_templates({"service_normalizer": _templates()["service_normalizer"]})
    client = ev_llm.HarnessClient(mode="real", sites=index, real=_Hang())
    prompt = _filled(_templates()["service_normalizer"])

    async def call():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(client.generate(model="m", prompt=prompt), 0.05)

    asyncio.run(call())
    assert [c.outcome for c in client.take_turn_calls()] == ["cancelled"]


def test_cache_keeps_first_answer_and_separates_generation_options(tmp_path):
    """Дубли — первая запись (её видел бейслайн); ответ при других параметрах генерации не подходит;
    записи без параметров (до 10.10) подходят всегда."""

    path = tmp_path / "cache.jsonl"
    key = ev_llm.ResponseCache.key("m", "p", None, False)
    ev_llm.ResponseCache(path).put(key, "при 0,2", "s", '{"temperature": 0.2}')
    ev_llm.ResponseCache(path).put(key, "второй при 0,2", "s", '{"temperature": 0.2}')
    cache = ev_llm.ResponseCache(path)
    assert cache.get(key, '{"temperature": 0.2}') == "при 0,2"
    assert cache.get(key, '{"temperature": 0}') is None

    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(json.dumps({"key": key, "site": "s", "response": "старый"}, ensure_ascii=False) + "\n")
    assert ev_llm.ResponseCache(legacy).get(key, '{"temperature": 0}') == "старый"

    client = ev_llm.HarnessClient(mode="adversary", sites=ev_llm.SiteIndex.from_templates({}), default_options={"temperature": 0.2})
    assert client._options_fingerprint({"temperature": 0.2}) == ""
    assert client._options_fingerprint({"temperature": 0, "num_predict": 24}) != ""


# --------------------------------------------------------------------------- карточка


def test_card_counts_differing_answers_and_choice_slo():
    from evalbench import card

    before = [{"set": "g", "id": "1", "answer": "Цена 500 руб."}, {"set": "g", "id": "2", "answer": "Соединяю с оператором."}]
    after = [{"set": "g", "id": "1", "answer": "Цена 500 руб. "}, {"set": "g", "id": "2", "answer": "Другой ответ"}, {"set": "g", "id": "3", "answer": "x"}]
    assert card.answer_diff(before, after) == (1, 2, ["2"])

    turn = {"id": "1", "outcome": "answer", "handoff": False, "urgent": False, "ms": 10}
    turns = [
        {**turn, "telemetry": {"choose": [{"task": "price_service", "status": "picked", "ms": 900}, {"task": "price_service", "status": "picked", "ms": 0, "cached": True}]}},
        {**turn, "id": "2", "telemetry": {"choose": [{"task": "price_service", "status": "unavailable", "ms": 3000}]}},
        {**turn, "id": "3", "telemetry": None},
    ]
    m = card.metrics(turns, {})
    assert (m["choose_n"], m["choose_unavailable"], m["choose_invalid"], m["choose_p95_ms"]) == (3, 1, 0, 3000)
    # Решение тени (L-15b, шаг 4) на ответ не влияло — в решения хода не входит.
    shadow = {"task": "turn_phrases", "status": "unavailable", "ms": 5000, "shadow": True}
    turns[0]["telemetry"]["choose"].append(shadow)
    m = card.metrics(turns, {})
    assert (m["choose_n"], m["choose_unavailable"], m["choose_p95_ms"]) == (3, 1, 3000)


# --------------------------------------------------------------------------- служебный текст в ответе

_LEAKED = {
    "field_label": "По запросу нашёл варианты:\nФормат: По услуге «ТТГ (TSH) тиреотропный гормон»:\n1. ТТГ (TSH) тиреотропный гормон — 400 руб.",
    "double_dot": "1. ТТГ (TSH) тиреотропный гормон — 400 руб..",
    "placeholder": "Стоимость услуги — <<PRICE>>. Записаться можно {branch_name}.",
    "null_value": "Срок готовности «Ферритин» — None.",
    "empty_name": "По услуге «» нашёл следующее:",
    "prompt": 'АЛГОРИТМ: шаг 1. {"label": "PRICE"}',
}
_CLEAN = (
    "Операционная гистероскопия — 20 000 руб. Формат: дневной стационар. Адрес: г. Самара, пр. Ленина, 5.",
    "Памятка клиники «Ферритин»:\nКровь сдается натощак.. Пациент должен находиться в покое.",  # опечатка МИС после слова
    "Срок готовности «Ферритин» — 1–2 дня. Подробнее: https://example.org/nan-none-null/soglasie",
    "2. Иванов Иван Иванович (Невролог) — 2 700 руб.; расписание временно недоступно…",
    "Сдавать натощак, т.е. после 8–12 часов голода... Nonetheless — по назначению врача.",
)


@pytest.mark.parametrize("kind", sorted(_LEAKED))
def test_service_text_detector_names_the_kind_that_leaked(kind):
    from evalbench.detectors import service_text

    assert kind in service_text(_LEAKED[kind])


def test_service_text_detector_is_silent_on_clean_answers_and_bot_templates():
    """Встречный свип: настоящие тексты бота (шаблон «не понял», часы операторов, формат оказания) и данные МИС с их
    опечатками — не служебный текст; утечка промпта (`leak`) — те же определения, что у детектора."""

    from evalbench.detectors import _LEAK_RE, service_text
    from messengers_router import policies
    from messengers_router.text_templates import LOW_CONF_CLARIFY_TEXT, price_names_without_prices_text

    for text in (*_CLEAN, LOW_CONF_CLARIFY_TEXT, policies.OPERATOR_HOURS_NOTE, price_names_without_prices_text("оак", ["ОАК", "ОАМ"])):
        assert service_text(text) == [], text
    fixture = ROOT / "tests" / "fixtures" / "mini_prod" / "snapshots" / "price_by_region" / "price_region_3_20261008.jsonl"
    names = [json.loads(line)["serviceName"] for line in fixture.open(encoding="utf-8") if line.strip()]
    assert len(names) > 300 and [name for name in names if service_text(f"{name} — 500 руб.")] == []
    for leak in ("АЛГОРИТМ", "<<QUESTION>>", "context_action", '"label": "PRICE"', "Label: PRICE"):
        kinds = service_text(leak)
        assert _LEAK_RE.search(leak) and kinds and set(kinds) <= {"prompt", "placeholder"}, leak


def test_card_counts_service_text_turns_by_kind():
    from evalbench import card

    turn = {"outcome": "price", "handoff": False, "urgent": False, "ms": 10}
    turns = [
        {**turn, "id": "1", "answer": _LEAKED["field_label"]},
        {**turn, "id": "2", "answer": _LEAKED["double_dot"]},
        {**turn, "id": "3", "answer": _CLEAN[0]},
    ]
    m = card.metrics(turns, {})
    assert (m["service_text"], m["service_text_kinds"]) == (2, {"field_label": 1, "double_dot": 1})
    text = card.render([({"name": "r", "code_sha": "x", "mode": "down", "cases": 3, "done": 3, "valid": True}, m)])
    assert "| Служебный текст в ответе | 2 |" in text and "double_dot 1, field_label 1" in text


# --------------------------------------------------------------------------- диалоги из нескольких ходов


def test_dialogs_run_turns_in_one_session_and_record_shown_context(tmp_path, monkeypatch):
    """Второй ход видит первый (одна сессия), на ход — контекст после него и строки журнала (решение 13.10:
    эффект «контекста = показанного» и утечки контекста виден только во втором ходе)."""

    import io

    from evalbench import dialogs as ev_dialogs

    fixture = ROOT / "tests" / "fixtures" / "mini_prod" / "snapshots"
    root = ev_world.build_world(fixture, "20261008", tmp_path / "world")

    class _NoCalls:
        def take_turn_calls(self):
            return []

    out, seen = io.StringIO(), []
    # Первый ход — голое название: мир стенда здесь без LLM (`_NoCalls`), и под `price_by_parse = on` (прод с 26.10) цену без LLM
    # получает только реплика-название (решение 22.10 (а)); «сколько стоит ферритин» — «Уточните название» (до 27.10 был он).
    dialog = {"id": "d1", "set": "t", "expect": "подготовка к ферритину", "turns": ["ферритин", "а как подготовиться?"]}
    with ev_world.FrozenWorld(root, "day") as world:
        stats = asyncio.run(
            ev_dialogs._run_dialogs(
                [dialog], client=_NoCalls(), world=world, tpl=Templates.from_bot(), catalog=PriceCatalog.from_world(root / "apidata"),
                llm_mode="hybrid", run_name="t", out=out, progress=seen.append,
            )
        )
        attempts = list(world.net.attempts)
    assert attempts == [] and (stats["done"], stats["turns"], stats["stopped"]) == (1, 2, None)
    row = json.loads(out.getvalue())
    assert (row["id"], row["expect"]) == ("d1", "подготовка к ферритину") and len(seen) == 1
    first, second = row["turns"]
    assert first["outcome"] == "price" and "Ферритин" in first["answer"]
    assert first["state"][ev_dialogs.SHOWN_KEY] == ["Ферритин"] and first["state"]["service_name"] == "Ферритин"
    assert second["text"] == "а как подготовиться?" and second["answer"].strip()
    assert second["state"].get("service_name") == "Ферритин"  # контекст пришёл из первого хода той же сессии
    assert isinstance(first["journal"], list) and isinstance(second["journal"], list)


_DIALOG_SETS = ("two_turn_l14.jsonl", "two_turn_synthetic.jsonl")


def test_dialog_sets_are_well_formed(tmp_path):
    from evalbench import dialogs as ev_dialogs

    dialogs = ev_dialogs.load_dialogs([ROOT / "evalbench" / "sets" / name for name in _DIALOG_SETS])
    assert len({d["id"] for d in dialogs}) == len(dialogs)  # наборы гоняют вместе: id не повторяются и между ними
    assert all(len(d["turns"]) >= 2 and d.get("expect") and d.get("tags") for d in dialogs)
    assert all(set(d.get("about_shown") or ()) <= set(range(2, len(d["turns"]) + 1)) for d in dialogs)
    texts = [ev_dialogs._turn(turn)[0] for d in dialogs for turn in d["turns"]]
    assert not [t for t in texts if re.search(r"\d{5,}|@|\+7", t)]  # синтетика: ни телефонов, ни почты, ни номеров
    assert [d["id"] for d in ev_dialogs.load_dialogs([ROOT / "evalbench" / "sets" / "two_turn_l14.jsonl"], ids={"oak_prepare"})] == ["oak_prepare"]
    broken = tmp_path / "broken.jsonl"
    broken.write_text(json.dumps({"id": "x", "turns": []}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ev_dialogs.load_dialogs([broken])
    assert ev_dialogs._turn({"text": "ферритин", "button": "menu.price.test"}) == ("ферритин", "menu.price.test")
    assert ev_dialogs._turn("ферритин") == ("ферритин", "")


def test_dialogs_show_puts_turns_side_by_side_and_marks_differences():
    from evalbench import dialogs as ev_dialogs

    def run(name, second_answer, shown):
        header = {"name": name, "code_sha": "abc", "mode": "real", "done": 1, "dialogs": 1, "live_calls": 3, "valid": True, "cfg": {}}
        turns = [
            {"text": "сколько стоит глюкоза", "answer": "Глюкоза — 300 руб.", "outcome": "price", "ms": 10, "state": {"_shown_services": shown}, "llm_calls": [], "journal": []},
            {"text": "а цена?", "answer": second_answer, "outcome": "price", "ms": 20, "state": {}, "llm_calls": [{"site": "price_service_select", "outcome": "ok", "ms": 1500}],
             "journal": [{"phrase": "а", "found": False, "services": []}]},
        ]
        return header, {"g": {"id": "g", "expect": "только «Глюкоза»", "turns": turns}}

    text = ev_dialogs.render_dialogs([run("off", "Аллерген а1 — 900 руб.", []), run("on", "Глюкоза — 300 руб.", ["Глюкоза"])])
    assert "## g — ждём: только «Глюкоза»" in text and "**2. «а цена?»**" in text
    assert "[A] price" in text and "[B] price" in text and "показано: «Глюкоза»" in text
    assert "журнал: «а» found=False" in text and "price_service_select ok 1.5 с" in text
    assert text.count("**ответы различаются**") == 1 and "**Ходов с разным ответом:** 1" in text
    single = ev_dialogs.render_dialogs([run("on", "Глюкоза — 300 руб.", ["Глюкоза"])])
    assert "[A]" not in single and "различаются" not in single


# --------------------------------------------------------------------------- метрики диалогов


def _dturn(text, answer, outcome="price", **extra):
    return {"text": text, "answer": answer, "outcome": outcome, "ms": 10, "handoff": False, "urgent": False, "llm_calls": [], "journal": [], **extra}


_GLUCOSE = "По услуге «Глюкоза» нашёл следующее:\n1) Розничная цена: Глюкоза — 300 руб."


def test_dialog_marks_context_shown_class():
    """Ход без своей услуги: цены и названия — только показанные раньше (с ценой или без: срок, варианты без цен);
    слова вопроса — не название услуги. Ход со своей услугой и первый ход не проверяются."""

    from evalbench.dialogs import mark_dialog

    catalog = PriceCatalog(_ROWS)
    clean = mark_dialog([_dturn("сколько стоит глюкоза", _GLUCOSE), _dturn("а цена?", _GLUCOSE)], [2], catalog)
    assert [m.problems() for m in clean] == [[], []] and clean[1].about_shown
    letter = mark_dialog([_dturn("сколько стоит глюкоза", _GLUCOSE), _dturn("а цена?", "По услуге «а» нашёл:\n1. Общий анализ крови — 500 руб.")], [2], catalog)
    assert (letter[1].unseen_prices, letter[1].echo) == (["Общий анализ крови — 500"], ["а"])
    prep = mark_dialog([_dturn("сколько стоит глюкоза", _GLUCOSE), _dturn("а как подготовиться?", "Памятки по подготовке к «а как подготовиться» у меня нет.", "no_info")], [2], catalog)
    assert prep[1].echo == ["а как подготовиться"] and not prep[1].unseen_prices
    other = mark_dialog([_dturn("сколько стоит глюкоза", _GLUCOSE), _dturn("а натощак?", "Подходят памятки: «Общий анализ крови».", "memo")], [2], catalog)
    assert other[1].unseen_names == ["общий анализ крови"] and not other[1].echo
    named = mark_dialog([_dturn("сколько готовится глюкоза", "Срок готовности «Глюкоза» — 1 день.", "answer"), _dturn("сколько стоит?", _GLUCOSE)], [2], catalog)
    assert named[1].problems() == []  # название показано без цены — его цена не «непоказанное»
    unshown = mark_dialog([_dturn("сколько стоит гастроскопия", "Не нашёл стоимость для «гастроскопия».", "no_info"), _dturn("сколько стоит?", _GLUCOSE)], [2], catalog)
    assert unshown[1].unseen_prices == ["Глюкоза — 300"]  # утечка L-13: ничего не показано, а цена есть
    own = mark_dialog([_dturn("сколько стоит глюкоза", _GLUCOSE), _dturn("а оак?", "Общий анализ крови — 500 руб.")], [], catalog)
    assert own[1].problems() == [] and not own[1].about_shown
    free = mark_dialog([_dturn("сколько стоит глюкоза", _GLUCOSE), _dturn("а цена?", "Стоимость — 300 руб., а со скидкой 250 руб.", "answer")], [2], catalog)
    assert free[1].unseen_prices == ["250"]  # сумма без строки прайса: показанная — не нарушение, новая — нарушение
    ask = "Отменить текущий процесс записи? Ответьте «да» или «нет»."
    options = mark_dialog([_dturn("хочу записаться", "Какой филиал удобен?", "clarify"), _dturn("ладно", ask, "answer"), _dturn("да, отмените", ask, "answer")], [2, 3], catalog)
    assert [m.echo for m in options] == [[], [], []] and options[2].loop == "question"  # варианты ответа бота — не эхо
    first = mark_dialog([_dturn("хочу отменить или перенести запись", "Хотите отменить или перенести запись?", "clarify"), _dturn("да, отмените", ask, "answer")], [2], catalog)
    assert first[1].echo == []  # и когда бот предлагает варианты впервые
    names = mark_dialog([_dturn("сколько стоит глюкоза", _GLUCOSE), _dturn("а цена?", "По запросу «а» нашёл похожую услугу: «Глюкоза».", "answer")], [2], catalog)
    assert names[1].echo == ["а"] and not names[1].unseen_names


def test_dialog_marks_loops_but_not_a_repeated_answer_to_a_repeated_question():
    from evalbench.dialogs import bot_questions, loop_kind

    offer = "Не нашёл такую услугу. Перевести на оператора?"
    assert loop_kind(_dturn("сколько стоит х", offer, "operator_offer"), _dturn("конечно, переведите", "Хорошо. " + offer, "operator_offer")) == "question"
    ask = "Скажите, пожалуйста, название услуги/анализа — я уточню стоимость."
    assert loop_kind(_dturn("сколько стоит?", ask, "clarify"), _dturn("а цена?", ask, "clarify")) == "repeat"
    assert loop_kind(_dturn("сколько стоит глюкоза", _GLUCOSE), _dturn("сколько стоит?", _GLUCOSE)) is None
    assert loop_kind(_dturn("сколько стоит?", ask, "clarify"), _dturn("а цена?", ask + " Перевести на оператора?", "operator_offer")) is None
    from messengers_router.text_templates import LOW_CONF_CLARIFY_TEXT

    assert bot_questions(LOW_CONF_CLARIFY_TEXT) == set()  # примеры вопросов в кавычках — не вопрос бота
    assert bot_questions("Подтвердите запись: терапевт, 15:00. Верно? Если нет — напишите время.") == {"верно?"}


def _dialog_run(name, second, *, sets=()):
    header = {"name": name, "code_sha": "abc", "mode": "down", "done": 2, "dialogs": 2, "live_calls": 0, "valid": True, "cfg": {}, "sets": list(sets),
              "world": {"date": "19000101", "profile": "day"}}
    rows = {
        "glucose_a": {"id": "glucose_a", "set": "t", "expect": "только «Глюкоза»", "tags": ["context_shown", "price_repeat"], "about_shown": [2],
                      "turns": [_dturn("сколько стоит глюкоза", _GLUCOSE), _dturn("а цена?", second)]},
        "offer_yes": {"id": "offer_yes", "set": "t", "expect": "перевод", "tags": ["yes_no", "operator_offer"], "about_shown": [2],
                      "turns": [_dturn("сколько стоит х", "Не нашёл. Перевести на оператора?", "operator_offer"),
                                _dturn("конечно, переведите", "Перевести на оператора?", "operator_offer", journal=[{"phrase": "х"}])]},
    }
    return header, rows


def test_dialogs_card_counts_two_runs_without_utterances():
    from evalbench import dialogs as ev_dialogs

    catalog = PriceCatalog(_ROWS)
    runs = [_dialog_run("off", "По услуге «а» нашёл:\n1. Общий анализ крови — 500 руб.."), _dialog_run("on", _GLUCOSE)]
    loaded = [(header, ev_dialogs.dialog_metrics(rows, ev_dialogs.run_marks(rows, catalog))) for header, rows in runs]
    off, on = (m for _h, m in loaded)
    assert (off["unseen_price"], off["echo"], off["service_text"], off["loop_question"], off["about_shown"]) == (1, 1, 1, 1, 2)
    assert (on["unseen_price"], on["echo"], on["service_text"], on["loop_question"], on["journal"]) == (0, 0, 0, 1, 1)
    assert off["by_tag"]["price_repeat"] == {"dialogs": 1, "service_text": 1, "unseen_price": 1, "unseen_name": 0, "echo": 1, "loop": 0}
    assert off["ids"] == {"service_text": ["glucose_a"], "unseen_price": ["glucose_a"], "echo": ["glucose_a"], "loop": ["offer_yes"]}
    text = ev_dialogs.render_card(loaded, ev_dialogs._answers_differ(runs))
    assert "| Метрика | off | on | Δ |" in text and "|   цена непоказанного (ходов) | 1 | 0 | -1 |" in text
    assert "| price_repeat | 1 / 1 | 1 / 0 | 1 / 0 | 0 / 0 | 1 / 0 | 0 / 0 |" in text
    assert "**Ходов с разным ответом:** 1 из 4 общих (диалоги: glucose_a)" in text
    assert "- цена непоказанного: A: glucose_a; B: —" in text and "double_dot 1" in text
    for said in ("сколько стоит глюкоза", "а цена?", "конечно, переведите"):  # карточка — без реплик пациентов
        assert said not in text


def test_dialogs_show_has_summary_and_marks_turns(tmp_path):
    """`show` — сводка прогона и отметки нарушений у ходов; разметка набора — из текущего файла (без перепрогона)."""

    from evalbench import dialogs as ev_dialogs

    header, rows = _dialog_run("off", "По услуге «а» нашёл:\n1. Общий анализ крови — 500 руб.")
    current = tmp_path / "t.jsonl"
    current.write_text(json.dumps({"id": "glucose_a", "turns": ["x", "y"], "expect": "e", "tags": ["new_class"], "about_shown": []}) + "\n", encoding="utf-8")
    text = ev_dialogs.render_dialogs([(header, rows)], catalogs=[PriceCatalog(_ROWS)])
    assert "- служебный текст 0; ходов без своей услуги 2: цена непоказанного 1, название непоказанного 0, слова вопроса 1; петли 1" in text
    assert "  - **нарушение:** цена непоказанного: Общий анализ крови — 500" in text and "  - **нарушение:** слова вопроса названием услуги: «а»" in text
    assert "  - **нарушение:** петля: тот же вопрос бота, что ходом раньше" in text and "классы: context_shown, price_repeat; без своей услуги — ходы 2" in text
    ev_dialogs.refresh_annotations({**header, "sets": [str(current)]}, rows)
    assert (rows["glucose_a"]["tags"], rows["glucose_a"]["about_shown"]) == (["new_class"], [])
    assert rows["offer_yes"]["tags"] == ["yes_no", "operator_offer"]  # диалога нет в файле — разметка прогона


def test_concurrent_turns_get_their_own_telemetry():
    """Параллельные ходы (`run --concurrency ≥ 2`): строки телеметрии копятся вперемешку, ход забирает свою по хэшу сессии
    (DEBT-2026-10-22-STAND-CONCURRENCY-TELEMETRY: до 23.10 у всех ходов параллельного прогона телеметрия была пустой)."""

    import hashlib

    from evalbench.world import session_rows

    def sess(sid: str) -> str:
        return hashlib.sha256(sid.encode("utf-8")).hexdigest()[:8]

    rows = [
        {"sess": sess("bench-a"), "tid": "1", "tu_parse_status": "picked"},
        {"sess": sess("bench-b"), "tid": "2", "tu_parse_status": "none_fit"},
        {"sess": sess("bench-a"), "tid": "3", "tu_parse_status": "single"},
    ]
    mine, rest = session_rows(rows, "bench-b")
    assert [r["tid"] for r in mine] == ["2"] and [r["tid"] for r in rest] == ["1", "3"]
    mine, rest = session_rows(rest, "bench-a")
    assert [r["tid"] for r in mine] == ["1", "3"] and rest == []


# --------------------------------------------------------------------------- успех по кнопкам (метрика «рабочий бот», 25.10)


def test_button_captions_are_the_gateway_screen_table():
    from evalbench import buttons
    from tests.test_button_actions import GATEWAY_V1_TITLES

    assert buttons.GATEWAY_CAPTIONS == GATEWAY_V1_TITLES and set(buttons.LEAF_TOPIC) == set(GATEWAY_V1_TITLES)


def test_press_case_sends_button_then_question_and_handoff_button_stops(tmp_path):
    """Набор кнопок идёт как у пациента: нажатие листа (подпись в `text`), затем вопрос без `button_id` в той же сессии.
    Одним запросом кнопка-тема `slot` вопрос не читает — ответом была бы подсказка кнопки, а не цена."""

    import io

    from evalbench import run as ev_run

    fixture = ROOT / "tests" / "fixtures" / "mini_prod" / "snapshots"
    root = ev_world.build_world(fixture, "20261008", tmp_path / "world")

    class _NoCalls:
        def take_turn_calls(self):
            return []

    cases = [
        {"id": "p1", "set": "b", "question": "ферритин", "button": "menu.price.test", "press": True},
        {"id": "p2", "set": "b", "question": "когда будет готов ферритин", "button": "menu.results.eta", "press": True},
        {"id": "p3", "set": "b", "question": "ферритин", "button": "menu.price.test"},
    ]
    out = io.StringIO()
    with ev_world.FrozenWorld(root, "day") as world:
        stats = asyncio.run(
            ev_run._run_cases(
                cases, client=_NoCalls(), world=world, tpl=Templates.from_bot(), catalog=PriceCatalog.from_world(root / "apidata"),
                llm_mode="hybrid", run_name="t", out=out, progress=lambda _m: None,
            )
        )
        attempts = list(world.net.attempts)
    assert attempts == [] and stats["done"] == 3
    priced, eta, one_request = (json.loads(line) for line in out.getvalue().splitlines())
    assert priced["press"]["question_sent"] and priced["press"]["outcome"] != "price" and priced["outcome"] == "price"
    assert "Ферритин" in priced["answer"] and priced["state"]["service_name"] == "Ферритин"
    assert eta["press"]["handoff"] and not eta["press"]["question_sent"] and eta["handoff"]
    assert one_request["press"] is None and one_request["outcome"] != "price"


def _btn_turn(tid: str, answer: str, **kw):
    return {"id": tid, "answer": answer, "outcome": kw.pop("outcome", "answer"), "price_pairs": kw.pop("pairs", []), "press": {"question_sent": True}, **kw}


def test_button_verdicts_price_by_labels_rest_by_judge_on_this_answer():
    from evalbench import buttons

    pair = {"amount": 530, "service_id": 7, "name": "Ферритин", "status": "matched"}
    label = {"kind": "single", "core": [7], "ok": [], "ambiguous": False}
    case = {"id": "x", "button": "menu.price.test", "press": True}
    assert buttons.turn_verdict(_btn_turn("x", "Ферритин — 530 руб.", pairs=[pair]), case, label, {}) == ("success", None)
    assert buttons.turn_verdict(_btn_turn("x", "Глюкоза — 530 руб.", pairs=[{**pair, "service_id": 8}]), case, label, {}) == ("fail", "цена: чужая цена")
    assert buttons.turn_verdict(_btn_turn("x", "Перевести?", outcome="operator_offer"), case, label, {}) == ("fail", "цена: перевод вместо цены")
    assert buttons.turn_verdict(_btn_turn("x", "Такой услуги нет.", outcome="no_info"), case, {**label, "kind": "absent"}, {})[0] == "success"
    assert buttons.turn_verdict(_btn_turn("x", "x"), case, {**label, "kind": "context_dependent"}, {}) == ("excluded", "без контекста")
    assert buttons.turn_verdict(_btn_turn("x", "Цена <<PRICE>>"), case, label, {}) == ("fail", "служебный текст")
    assert buttons.turn_verdict(_btn_turn("x", "", outcome="tech_error"), case, label, {}) == ("fail", "сбой")
    doctor_line = {"amount": 3000, "service_id": None, "name": None, "status": "doctor"}
    assert buttons.turn_verdict(_btn_turn("x", "Иванов И. И. (Уролог) — 3 000 руб.", pairs=[doctor_line]), case, label, {}) == ("unjudged", None)

    addr = {"id": "a", "button": "menu.address", "press": True}
    answer = "ул. Солнечная, 53 — будни 07:00–18:30"
    judged = {("a", buttons.answer_sha(answer)): {"verdict": "success", "answer_sha": buttons.answer_sha(answer)}}
    assert buttons.turn_verdict(_btn_turn("a", answer), addr, None, judged) == ("success", None)
    assert buttons.turn_verdict(_btn_turn("a", answer + " Другой текст."), addr, None, judged) == ("unjudged", None)  # ответ изменился
    assert buttons.turn_verdict(_btn_turn("a", answer), {**addr, "exclude": "другой город"}, None, judged) == ("excluded", "другой город")

    eta = {"id": "e", "button": "menu.results.eta", "press": True}
    stopped = {"id": "e", "answer": "соединяю", "press": {"question_sent": False, "handoff": True}}
    assert buttons.turn_verdict(stopped, eta, None, {}) == ("by_design", None)
    assert buttons.turn_verdict({**stopped, "id": "a"}, addr, None, {}) == ("fail", "перевод на нажатии")


def test_button_card_section_counts_per_leaf_without_utterances():
    from evalbench import buttons

    cases = {
        "g1": {"id": "g1", "button": "menu.price.test", "press": True, "question": "СЕКРЕТНАЯ РЕПЛИКА"},
        "g2": {"id": "g2", "button": "menu.address", "press": True, "question": "СЕКРЕТНАЯ РЕПЛИКА"},
        "g3": {"id": "g3", "button": "menu.address", "press": True, "question": "СЕКРЕТНАЯ РЕПЛИКА", "exclude": "другой город"},
    }
    labels = {"g1": {"kind": "single", "core": [7], "ok": []}}
    pair = {"amount": 530, "service_id": 7, "name": "Ферритин", "status": "matched"}
    before = [_btn_turn("g1", "нет", outcome="no_info"), _btn_turn("g2", "все филиалы"), _btn_turn("g3", "x")]
    after = [_btn_turn("g1", "Ферритин — 530 руб.", pairs=[pair]), _btn_turn("g2", "все филиалы"), _btn_turn("g3", "x")]
    judged = {("g2", buttons.answer_sha("все филиалы")): {"verdict": "fail", "klass": "шаблон вместо ответа", "answer_sha": buttons.answer_sha("все филиалы")}}
    b0 = buttons.button_metrics(before, cases, labels, judged)
    b1 = buttons.button_metrics(after, cases, labels, judged)
    assert b0["total"] == {"fail": 2} and b1["total"] == {"success": 1, "fail": 1} and b1["excluded"] == {"другой город": 1}
    assert b1["classes"] == {"шаблон вместо ответа": ["g2"]}
    text = "\n".join(buttons.render_buttons([({"name": "d17"}, b0), ({"name": "d18"}, b1)]))
    assert "## Успех по кнопкам" in text and "**50% (1/2)**" in text and "Цена анализа (`menu.price.test`)" in text
    assert "стало лучше:** 1 (g1)" in text and "СЕКРЕТНАЯ" not in text and "все филиалы" not in text
    assert buttons.button_metrics(after, {}, labels, judged) is None  # прогон не по набору кнопок — раздела нет


def test_run_with_unreachable_live_llm_is_not_valid():
    """Обрыв связи с LLM посреди прогона (VPN) — прогон не мерил живую LLM: `valid=false` (25.10, `bkt_on`: 39 из 44 живых
    вызовов упали, а прогон был действителен). Тайм-аут ответа модели — поведение бота, прогон не портит."""

    from collections import Counter

    from evalbench.run import llm_unreachable

    assert llm_unreachable(Counter({"ok": 5, "error:ConnectTimeout": 15, "error:ConnectionError": 24})) == 39
    assert llm_unreachable(Counter({"ok": 5, "timeout": 2, "cancelled": 1, "cache_hit": 9})) == 0


def test_price_pairs_see_the_overview_card_format():
    """Обзор голого названия печатает «Название»: и под ним «— цена: N руб.» — сумма связывается с заголовком блока
    (разбор 95 цен 26.10: в `d13_svc` 5 таких ответов детектор считал «без цены»); вне блока — как было."""

    catalog = PriceCatalog(_ROWS)
    pairs = catalog.pairs(
        "«Глюкоза»:\n— цена: 300 руб.\n— срок готовности: 1 день\n— где: в лабораторных филиалах\n\n"
        "Стоимость этой услуги — 300 руб.\n"
        "«Общий анализ крови»:\nСправка: 500 руб."
    )
    assert [(p.status, p.service_id) for p in pairs] == [("matched", 3), ("unattributed", None), ("unattributed", None)]
