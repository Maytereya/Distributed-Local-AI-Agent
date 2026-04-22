"""Локальный порт для doctor-name matching функций.

Изолирует прямой импорт `agent_logic_2.doctor_name_matching` от core-слоя.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_logic_2.doctor_name_matching import (
    extract_doctor_name_candidate as _extract_doctor_name_candidate,
    resolve_schedule_surname as _resolve_schedule_surname,
    surname_variants as _surname_variants,
)
from agent_logic_2.nayka_api.api_nayka import (
    find_existing_doctors_file as _find_existing_doctors_file,
    load_doctors_data as _load_doctors_data,
)
from .russian_nlu import normalize_ru

_DOCTOR_CACHE_SIGNATURE: tuple[str, int] | None = None
_DOCTOR_SURNAMES_MAP: dict[str, str] = {}
_DOCTOR_FIO_MAP: dict[str, str] = {}


def _normalize_doctor_key(value: str | None) -> str:
    """Нормализует ФИО/фамилию врача для сопоставления с локальным кэшем."""
    return " ".join(normalize_ru(value).split())


def _doctor_cache_signature(file: Path | None) -> tuple[str, int] | None:
    """Возвращает сигнатуру файла doctors-кэша для безопасного in-memory кэша."""
    if file is None:
        return None
    try:
        stat = file.stat()
    except OSError:
        return None
    return str(file), int(stat.st_mtime_ns)


def _ensure_local_doctors_index_loaded() -> tuple[dict[str, str], dict[str, str]]:
    """Загружает и кэширует индекс врачей из локального `doctors_*.jsonl`."""
    global _DOCTOR_CACHE_SIGNATURE, _DOCTOR_SURNAMES_MAP, _DOCTOR_FIO_MAP

    doctors_file = _find_existing_doctors_file()
    signature = _doctor_cache_signature(doctors_file)
    if signature is None:
        _DOCTOR_CACHE_SIGNATURE = None
        _DOCTOR_SURNAMES_MAP = {}
        _DOCTOR_FIO_MAP = {}
        return _DOCTOR_SURNAMES_MAP, _DOCTOR_FIO_MAP

    if _DOCTOR_CACHE_SIGNATURE == signature:
        return _DOCTOR_SURNAMES_MAP, _DOCTOR_FIO_MAP

    surnames_map: dict[str, str] = {}
    fio_map: dict[str, str] = {}
    try:
        doctors = _load_doctors_data(doctors_file)
    except OSError:
        doctors = []

    for doc in doctors:
        if not isinstance(doc, dict):
            continue
        fio = str(doc.get("fio") or "").strip()
        if not fio:
            continue
        fio_key = _normalize_doctor_key(fio)
        if fio_key:
            fio_map.setdefault(fio_key, fio)
        surname = str(fio.split()[0] or "").strip()
        surname_key = _normalize_doctor_key(surname)
        if surname_key:
            surnames_map.setdefault(surname_key, surname)

    _DOCTOR_CACHE_SIGNATURE = signature
    _DOCTOR_SURNAMES_MAP = surnames_map
    _DOCTOR_FIO_MAP = fio_map
    return _DOCTOR_SURNAMES_MAP, _DOCTOR_FIO_MAP


def extract_doctor_name_candidate(text: str, *, prefer_schedule: bool = False) -> str | None:
    return _extract_doctor_name_candidate(text, prefer_schedule=prefer_schedule)


def resolve_schedule_surname(value: str | None, doctors: list[dict[str, Any]]) -> str | None:
    return _resolve_schedule_surname(value, doctors)


def surname_variants(value: str) -> list[str]:
    return _surname_variants(value)


def resolve_cached_doctor_name_candidate(text: str, *, prefer_schedule: bool = False) -> str | None:
    """
    Извлекает кандидата на фамилию врача только если он подтверждается локальным doctors-кэшем.

    :param text: исходный текст пользователя
    :param prefer_schedule: использовать schedule-режим извлечения кандидата
    :return: каноническая фамилия врача или None
    """
    candidate = _extract_doctor_name_candidate(text, prefer_schedule=prefer_schedule)
    if not candidate:
        return None

    surnames_map, fio_map = _ensure_local_doctors_index_loaded()
    if not surnames_map and not fio_map:
        return None

    norm = _normalize_doctor_key(candidate)
    if not norm:
        return None

    exact_fio = fio_map.get(norm)
    if exact_fio:
        return str(exact_fio.split()[0] or "").strip() or None

    exact_surname = surnames_map.get(norm)
    if exact_surname:
        return exact_surname

    for variant in _surname_variants(candidate):
        matched = surnames_map.get(_normalize_doctor_key(variant))
        if matched:
            return matched

    return None
