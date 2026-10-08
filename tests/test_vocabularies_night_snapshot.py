"""Ночью словари МИС не пустеют (DATA-1, ревью 05.10; BUG-2026-10-08-NIGHT-EMPTY-VOCAB).

После самарской полуночи файла `service_info_<сегодня>.jsonl` нет до цикла 08:20. `_cache_key`
смотрел только его и отдавал пустые словари, не дойдя до отката на последний непустой срез:
свип по 214 синонимам — ночью 65% теряли цену, 5% получали чужую. Гейт этого не видел:
conftest прикрепляет срезы и глушит синонимы, поэтому здесь оба прикрепления сняты явно.

Инвариант класса: нет среза «на сегодня» (или он пуст) → словари строятся из последнего
непустого, без скачивания в обработке вопроса; появился свежий — берётся он; срезов нет совсем —
пустые словари без исключения.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_logic_2.nayka_api import api_service_info
from messengers_router.services import _biomaterial as _bio

# Настоящий `_vocabularies` — до autouse-подмены `_hermetic_mis_synonyms` (она глушит синонимы).
_REAL_VOCABULARIES = _bio._vocabularies
_TODAY = "20261008"


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def night(monkeypatch, tmp_path):
    """Каталог срезов во временной папке, «сегодня» = 08.10, скачивание запрещено."""

    def path(date=None):
        return tmp_path / f"service_info_{date or _TODAY}.jsonl"

    def no_download(*_args, **_kwargs):
        raise AssertionError("словари не должны качать срез в обработке вопроса (DATA-7)")

    monkeypatch.setattr(api_service_info, "service_info_path", path)  # снимает прикрепление conftest
    monkeypatch.setattr(api_service_info, "update_service_info", no_download)
    monkeypatch.setattr(api_service_info, "load_service_info", no_download)
    monkeypatch.setattr(_bio, "_vocabularies", _REAL_VOCABULARIES)  # снимает глушилку синонимов
    monkeypatch.setattr(_bio, "_VOCAB_CACHE", {})
    return tmp_path


def test_no_today_snapshot_uses_latest_nonempty(night):
    _write(night / "service_info_20261007.jsonl", [{"serviceName": "Ферритин", "serviceSynonyms": "железо депо", "biomatNames": "кровь"}])

    vocab = _bio._vocabularies()

    assert vocab.synonym_to_service == {"железо депо": "Ферритин"}
    assert vocab.biomat_tokens


def test_empty_today_snapshot_falls_back_to_yesterday(night):
    (night / f"service_info_{_TODAY}.jsonl").write_text("", encoding="utf-8")
    _write(night / "service_info_20261007.jsonl", [{"serviceName": "Ферритин", "serviceSynonyms": "железо депо"}])

    assert _bio._vocabularies().synonym_to_service == {"железо депо": "Ферритин"}


def test_fresh_snapshot_replaces_yesterday_when_it_appears(night):
    _write(night / "service_info_20261007.jsonl", [{"serviceName": "Ферритин", "serviceSynonyms": "железо депо"}])
    assert "железо депо" in _bio._vocabularies().synonym_to_service

    _write(night / f"service_info_{_TODAY}.jsonl", [{"serviceName": "Ферритин", "serviceSynonyms": "ферритин сыворотки"}])

    assert _bio._vocabularies().synonym_to_service == {"ферритин сыворотки": "Ферритин"}


def test_no_snapshots_at_all_gives_empty_vocab_without_error(night):
    vocab = _bio._vocabularies()

    assert vocab.synonym_to_service == {}
    assert vocab.biomat_tokens == frozenset()
