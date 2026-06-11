"""S1 (Диалог #232): частичный /regions больше НЕ схлопывается на одну «Гагарину 64».

Раньше при частичном живом /regions (неполная parent-иерархия → производный город
не выводится) выживала только «Гагарина 64» (единственная с «Самара» в name), и бот
отдавал один филиал. Теперь единый источник `resolve_samara_branches` делает fallback:
live(здоровый) → last-good снапшот → seed (32 филиала). Worst case — полный
статический список по типу услуги, а не одинокая/неверная Гагарина.

Инвариант (класс): walk-in по анализам/ЭКГ при ЧАСТИЧНОМ живом /regions отдаёт ПОЛНЫЙ
список филиалов соответствующего типа из fallback, не один. Если и fallback пуст (нет
seed/снапшота) — честный degraded (final defense, мой guard). Здоровый live и
конкретный branch-запрос — без изменений.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router.services import Services
from messengers_router.services import _samara_branches as SB


def run(coro):
    return asyncio.run(coro)


def _samara(idx, addr, **flags):
    f = {"analysis": True}
    f.update(flags)
    return {"id": idx, "name": f"Самара {addr}", "addressForSite": f"г. Самара, {addr}", "city": "Самара", **f}


def _mock_regions(monkeypatch, svc, rows):
    async def fake():
        return rows
    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake)


@pytest.mark.parametrize(
    "query,frag,min_n",
    [("сдать анализы", "анализы", 10), ("где сдать ЭКГ", "ЭКГ", 5)],
    ids=["analysis", "ecg"],
)
def test_partial_regions_falls_back_to_full_list(monkeypatch, query, frag, min_n):
    """Частичный /regions (1 филиал) → полный fallback-список, НЕ одна Гагарина."""
    svc = Services()
    _mock_regions(monkeypatch, svc, [_samara(1, "ул.Гагарина, 64", ecg=True)])
    res = run(svc.address_info(query, {"service_name": frag}))
    addrs = res.get("addresses") or []
    assert len(addrs) >= min_n, f"{query!r}: fallback не дал полный список ({len(addrs)})"
    assert res.get("handoff_reason") != "tech_unavailable"
    assert not (len(addrs) == 1 and "гагарина" in addrs[0].lower()), "схлопнулось на одну Гагарину"


def test_partial_regions_usi_falls_back_to_usi_list(monkeypatch):
    """Приоритет №2: УЗИ при частичном /regions → именно УЗИ-филиалы из fallback (seed usi=4),
    а НЕ врачебно-несамарская утечка doctors-cache (Выезд на дом / Пирогова / Б.Садовая —
    класс BUG-2026-06-04-04, который без гейта перехватывал УЗИ на деградации)."""
    svc = Services()
    _mock_regions(monkeypatch, svc, [_samara(1, "ул.Гагарина, 64")])  # схлопывание (без usi)
    res = run(svc.address_info("где сделать УЗИ", {"service_name": "узи"}))
    addrs = res.get("addresses") or []
    joined = " | ".join(addrs).lower()
    assert "выезд на дом" not in joined and "пирогова" not in joined and "садовая, 139" not in joined, (
        f"УЗИ дал врачебно-несамарскую утечку doctors-cache: {addrs}"
    )
    assert len(addrs) <= 6, f"УЗИ должно дать ~4 филиала с usi-флагом, не 17 врачебных: {addrs}"
    assert any(
        any(k in a.lower() for k in ("ленина, 5", "кирова, 223", "ново-садовая, 180", "победы, 83"))
        for a in addrs
    ), f"в выдаче нет УЗИ-филиалов: {addrs}"


def test_partial_regions_general_address_falls_back_to_full_list(monkeypatch):
    """Приоритет №2: общий адрес/график (без услуги) при частичном /regions → полный список."""
    svc = Services()
    _mock_regions(monkeypatch, svc, [_samara(1, "ул.Гагарина, 64")])  # схлопывание
    for query in ("график работы филиалов", "адреса филиалов"):
        res = run(svc.address_info(query, {}))
        addrs = res.get("addresses") or []
        assert len(addrs) >= 10, f"{query!r}: general-address fallback не дал полный список ({len(addrs)})"
        assert not (len(addrs) == 1 and "гагарина" in addrs[0].lower()), query


def test_healthy_regions_usi_used_directly(monkeypatch):
    """Регресс: здоровый live с usi-флагами → live-фильтр (НЕ fallback на seed)."""
    svc = Services()
    rows = [_samara(i, f"ул. Тест {i}", usi=(i < 3)) for i in range(12)]
    _mock_regions(monkeypatch, svc, rows)
    res = run(svc.address_info("где сделать УЗИ", {"service_name": "узи"}))
    addrs = res.get("addresses") or []
    assert len(addrs) == 3, f"здоровый live должен дать 3 usi, не seed-fallback: {addrs}"


def test_healthy_regions_used_directly(monkeypatch):
    """Здоровый live (≥порога) → используется как есть."""
    svc = Services()
    _mock_regions(monkeypatch, svc, [_samara(i, f"ул. Тест {i}") for i in range(12)])
    res = run(svc.address_info("сдать анализы", {"service_name": "анализы"}))
    assert len(res.get("addresses") or []) >= 10
    assert res.get("handoff_reason") != "tech_unavailable"


def test_partial_and_no_fallback_is_degraded(monkeypatch):
    """Final defense: частичный live + нет seed/снапшота → честный degraded, не Гагарина."""
    svc = Services()
    _mock_regions(monkeypatch, svc, [_samara(1, "ул.Гагарина, 64")])
    monkeypatch.setattr(SB, "load_seed", lambda: [])
    monkeypatch.setattr(SB, "load_snapshot", lambda: [])
    res = run(svc.address_info("сдать анализы", {"service_name": "анализы"}))
    assert (res.get("addresses") or []) == []
    assert res.get("handoff_reason") == "tech_unavailable"


def test_specific_branch_query_keeps_branch(monkeypatch):
    """Регресс: конкретный филиал («на Гагарина») → отдаём Гагарину, не теряем."""
    svc = Services()
    _mock_regions(monkeypatch, svc, [_samara(1, "ул.Гагарина, 64"), _samara(2, "пр. Ленина, 5")])
    res = run(svc.address_info("анализы на Гагарина", {"service_name": "анализы", "branch": "Гагарина"}))
    addrs = res.get("addresses") or []
    assert any("гагарина" in a.lower() for a in addrs), f"branch-запрос потерял филиал: {addrs}"


def test_seed_is_healthy_sanity():
    """Гард окружения: seed-снапшот валиден и правдоподобен (≥порога analysis-филиалов)."""
    seed = SB.load_seed()
    assert SB.is_healthy_samara(seed), "seed unhealthy — проверь nonbookable_points.json"
    assert SB._analysis_count(seed) >= 25


def test_seed_carries_verified_usi_and_doctor_flags():
    """Seed обогащён usi/doctorService (сверено с живым /regions, пробой 2026-06-11).

    Без этих флагов cold-start fallback для УЗИ/приёма отдавал бы 0 филиалов
    (раньше usi/doctorService хардкодились False). Класс-гард: маппинг
    has_usi/has_doctor → usi/doctorService не теряется при правках load_seed.
    """
    seed = SB.load_seed()
    usi = [r for r in seed if r.get("usi")]
    doctor = [r for r in seed if r.get("doctorService")]
    assert len(usi) == 4, f"usi филиалов в seed: {len(usi)} (ожидалось 4)"
    assert len(doctor) >= 9, f"doctorService филиалов в seed: {len(doctor)} (ожидалось ≥9)"
    # УЗИ-филиалы — авторитетный набор из BUG-2026-06-02-04 / пробоя
    usi_addr = " ".join(r.get("addressForSite", "") for r in usi)
    for need in ("Ленина, 5", "Победы, 83", "Кирова, 223", "Ново-Садовая, 180А"):
        assert need in usi_addr, f"УЗИ-филиал отсутствует в seed: {need}"


def test_snapshot_path_isolated_from_real_apidata():
    """Класс-инвариант (BUG-2026-06-11-01): тест-гейт самодостаточен.

    S1-persist пишет last-good снапшот на диск из настоящего `_ensure_regions_loaded`.
    Если снапшот-путь указывает в РЕАЛЬНЫЙ apidata, тест, мокающий здоровый /regions,
    заражает fallback другого теста (тот читает чужие 2-3 филиала вместо 31 seed).
    conftest изолирует путь в tmp — проверяем, что он ВНЕ реального кэш-каталога.
    """
    from agent_logic_2.nayka_api import cache_paths

    real_dir = cache_paths.resolve_cache_data_dir().resolve()
    snap = SB._snapshot_path()
    assert snap is not None
    assert real_dir not in snap.resolve().parents, (
        f"снапшот-путь {snap} внутри реального apidata {real_dir} — "
        "conftest-изоляция не активна, гейт зависит от состояния диска"
    )


def test_persist_then_load_roundtrip_in_isolation():
    """Механизм снапшота цел: persist здорового среза → load его читает (в изоляции)."""
    rows = [_samara(i, f"ул. Тест {i}") for i in range(5)]
    SB.persist_snapshot(rows)
    loaded = SB.load_snapshot()
    assert SB.is_healthy_samara(loaded)
    assert len(loaded) == 5
