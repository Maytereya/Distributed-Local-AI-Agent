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
