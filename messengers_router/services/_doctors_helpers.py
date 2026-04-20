"""Doctor/specialty/service helpers (Stage 20 cluster 6).

Pure helpers for doctor matching, specialty classification, service-token
normalisation, and schedule slot extraction. Moved out of
``services_legacy.py`` behind a re-export shim.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from ..doctor_name_port import (
    extract_doctor_name_candidate,
    surname_variants,
)
from ..russian_nlu import normalize_ru
from ..service_phrase import extract_service_phrase
from ..specialty_parser import (
    ENDOSCOPY_SERVICE_RE as _ENDOSCOPY_SERVICE_RE,
    SPECIALTY_CANONICAL as _SPECIALTY_CANONICAL,
    UZI_QUERY_RE as _UZI_QUERY_RE,
    extract_specialties_from_text as _shared_extract_specialties_from_text,
    extract_specialty_from_text as _shared_extract_specialty_from_text,
    matches_specialty_terms as _shared_matches_specialty_terms,
    specialty_equivalent as _shared_specialty_equivalent,
    specialty_terms as _shared_specialty_terms,
)
from ._common import (
    _dedupe_str,
    _normalise_catalog_text,
    _normalise_input,
)


# ---------------------------------------------------------------------------
# Regex patterns and constant sets
# ---------------------------------------------------------------------------

_FIO_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё\-]{2,}")
_UZI_LINE_RE = re.compile(r"\b(узи|ультразвук\w*|ультразвуков\w*)\b", re.I)
_UZI_FALSE_POSITIVE_RE = re.compile(
    r"\b(под\s+контролем\s+узи|во\s+время\s+консультативн\w*\s+при(е|ё)м\w*|"
    r"в\s+рамках\s+при(е|ё)м\w*|интерпретац\w*|разъяснен\w*)\b",
    re.I,
)
_UZI_ROLE_HINT_RE = re.compile(
    r"\b(узист\w*|врач\w*\s+узи|врач\w*\s+ультразвуков\w*\s+диагностик\w*|"
    r"каки\w*\s+узист\w*|каки\w*\s+врач\w*\s+узи)\b",
    re.I,
)
_UZI_PROCEDURE_HINT_RE = re.compile(
    r"\b(сдела\w*|дела\w*|провест\w*|процедур\w*|исследован\w*|"
    r"брюшн\w*|щитовид\w*|мал\w*\s+таз\w*|молочн\w*|почек|печен\w*|сердц\w*|сосуд\w*)\b",
    re.I,
)
_SPECIALTY_PRIORITY_SURNAMES: dict[str, tuple[str, ...]] = {
    # Бизнес-приоритет списка хирургов в выдаче.
    "хирург": ("тюрин", "джарар", "алимназаров", "губский"),
}
_SERVICE_FILTER_STOPWORDS = {
    "хочу",
    "нужно",
    "надо",
    "можно",
    "сделать",
    "пройти",
    "провести",
    "выполняет",
    "выполняют",
    "делает",
    "делают",
    "какой",
    "какие",
    "врач",
    "врачи",
    "доктор",
    "доктора",
    "процедура",
    "процедуры",
    "услуга",
    "услуги",
    "исследование",
    "исследования",
}
_CATALOG_DOCTOR_STOPWORDS = {
    "запишите",
    "записать",
    "записаться",
    "расписание",
    "расписание",
    "врач",
    "доктор",
    "специалист",
    "прием",
    "приеме",
    "приём",
    "приёме",
    "да",
    "нет",
    "пожалуйста",
    "будьте",
    "добры",
}
_CATALOG_SERVICE_LEADIN_RE = re.compile(
    r"^\s*(?:(?:пожалуйста|будьте\s+добры|подскажите|скажите|мне)\s+)?"
    r"(?:(?:запишите|записать|записаться|можно|хочу|нужно|надо)\s+)?"
    r"(?:(?:на|к)\s+)?",
    re.I,
)
_CATALOG_SERVICE_TRAILING_TIME_RE = re.compile(
    r"\b(?:сегодня|завтра|послезавтра|\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?|\d{1,2}:\d{2})\b.*$",
    re.I,
)
_CATALOG_WORD_RE = re.compile(r"[a-zа-яё0-9\-]+", re.I)
_CATALOG_SERVICE_SIGNAL_RE = re.compile(
    r"\b(услуг\w*|процедур\w*|исследован\w*|анализ\w*|сда[тч]\w*|"
    r"узи|экг|холтер|мрт|кт|фгдс|фкс|эндоскоп\w*|гастроскоп\w*|кольпоскоп\w*|"
    r"колоноскоп\w*|рентген\w*|флюорограф\w*|биопс\w*|пункц\w*|"
    r"при[её]м\w*|консультац\w*|операц\w*|липид\w*|холестерин\w*|оак|оам)\b",
    re.I,
)
_CATALOG_SERVICE_STOPWORDS = _SERVICE_FILTER_STOPWORDS | {
    "мне",
    "бы",
    "пож",
    "пожалуйста",
    "будьте",
    "добры",
    "здравствуйте",
    "добрый",
    "день",
    "запишите",
    "записаться",
    "запись",
    "к",
    "на",
    "в",
    "во",
    "по",
    "из",
    "могу",
    "можете",
    "покажите",
    "подскажите",
    "скажите",
}
_SCHEDULE_SPECIALTY_TOKENS = set(_SPECIALTY_CANONICAL) | {"узи", "узист", "экг", "мрт", "кт", "фгдс", "фкс"}


# ---------------------------------------------------------------------------
# Helpers (order preserved from services_legacy.py for diff readability)
# ---------------------------------------------------------------------------


def _is_schedule_no_slots_text(payload: Any) -> bool:
    """
    Определяет текстовый ответ Nayka API, когда врач найден, но свободных слотов нет.

    :param payload: ответ из find_doctor_schedule
    :return: True, если это кейс отсутствия свободных слотов, а не отсутствия врача
    """

    if not isinstance(payload, str):
        return False
    norm = _normalise_input(payload)
    return "свободных слотов нет" in norm


def _schedule_payload_matches_doctor(data: Any, doctor_name: str) -> bool:
    """
    Проверяет, что payload расписания действительно относится к нужному врачу.

    Защищает от ответов API, где по фамилии может вернуться "общий" список
    других врачей (ложный позитив на первом непустом list).

    :param data: ответ find_doctor_schedule
    :param doctor_name: ожидаемая фамилия/ФИО
    :return: True, если в payload есть совпадающий врач
    """

    target = str(doctor_name or "").strip()
    if not target or not isinstance(data, list):
        return False
    for row in data:
        if not isinstance(row, dict):
            continue
        row_fio = str(row.get("fio") or "").strip()
        if not row_fio:
            continue
        if _doctor_matches_fio(row_fio, target, resolved_surname=target):
            return True
    return False


def _doctor_catalog_query_candidates(raw_text_or_name: str) -> list[str]:
    raw = str(raw_text_or_name or "").strip()
    if not raw:
        return []
    candidate = extract_doctor_name_candidate(raw, prefer_schedule=True)
    probes = _dedupe_str([candidate or "", raw], max_items=4)
    out: list[str] = []
    for probe in probes:
        norm = _normalise_catalog_text(probe)
        if not norm:
            continue
        tokens = [
            token
            for token in _CATALOG_WORD_RE.findall(norm)
            if len(token) >= 3 and token not in _CATALOG_DOCTOR_STOPWORDS
        ]
        if not tokens:
            continue
        out.append(tokens[0])
        if len(tokens) > 1:
            out.append(" ".join(tokens[:2]))
    return _dedupe_str(out, max_items=6)


def _service_catalog_query_candidates(raw_text_or_name: str, *, current_service_name: str = "") -> list[str]:
    raw = str(raw_text_or_name or "").strip()
    if not raw and not current_service_name:
        return []
    service_phrase = extract_service_phrase(raw) if raw else None
    stripped = _CATALOG_SERVICE_LEADIN_RE.sub("", raw).strip()
    stripped = _CATALOG_SERVICE_TRAILING_TIME_RE.sub("", stripped).strip(" ,.;:-")
    base_candidates = [service_phrase or "", stripped, raw, str(current_service_name or "")]
    candidates = _dedupe_str(base_candidates, max_items=8)

    out: list[str] = []
    for item in candidates:
        norm = _normalise_catalog_text(item)
        if not norm:
            continue
        has_service_signal = bool(service_phrase and item == service_phrase) or bool(_CATALOG_SERVICE_SIGNAL_RE.search(item))
        tokens = [token for token in _CATALOG_WORD_RE.findall(norm) if len(token) >= 2]
        core_tokens = [token for token in tokens if token not in _CATALOG_SERVICE_STOPWORDS]
        if not has_service_signal and not core_tokens:
            continue
        if core_tokens:
            out.append(" ".join(core_tokens[:8]))
        if has_service_signal:
            out.append(item)
    return _dedupe_str(out, max_items=8)


def _extract_specialty_from_text(text: str) -> str:
    """
    Извлекает каноническую специальность из пользовательского текста.

    Сначала пытается найти составную специальность через role-синонимы
    (`травматолог ортопед`, `уролог андролог` и т.п.), затем использует
    legacy-regex fallback.

    :param text: исходный текст пользователя
    :return: каноническая специальность или пустая строка
    """

    return _shared_extract_specialty_from_text(text or "")


def _procedure_query_role_specialty(text: str) -> str:
    """
    Возвращает ролевую специальность для процедурного запроса.

    Пример:
    - "фгдс", "эндоскопия", "колоноскопия" -> "эндоскопист"

    :param text: текст запроса или service_name
    :return: каноническая специальность или пустая строка
    """
    if _ENDOSCOPY_SERVICE_RE.search(text or ""):
        return "эндоскопист"
    return ""


def _looks_like_schedule_specialty_token(value: str) -> bool:
    """
    Проверяет, является ли токен названием специальности/исследования,
    а не фамилией врача.

    :param value: кандидат на фамилию
    :return: True, если это specialty-like токен
    """
    norm = _normalise_input(value)
    if not norm:
        return False
    return norm in _SCHEDULE_SPECIALTY_TOKENS


def _is_uzi_query_text(text: str) -> bool:
    return bool(_UZI_QUERY_RE.search(text or ""))


def _split_spec_lines(spec_text: str) -> list[str]:
    return [ln.strip(" \t•-") for ln in str(spec_text or "").splitlines() if ln.strip()]


def _matches_uzi_doctor_profile(doc: dict[str, Any]) -> bool:
    spec_text = str(doc.get("specialization") or "")
    if not spec_text:
        return False

    units_text = " ".join(str(x or "") for x in (doc.get("units") or []))
    units_norm = _normalise_input(units_text)
    if "ультразвук" in units_norm or re.search(r"\bузи\b", units_norm):
        return True

    for raw_line in _split_spec_lines(spec_text):
        line = _normalise_input(raw_line)
        if not line or not _UZI_LINE_RE.search(line):
            continue
        if _UZI_FALSE_POSITIVE_RE.search(line):
            continue
        if "врач ультразвуковой диагностики" in line or "ультразвуков" in line:
            return True
        if line.startswith("узи "):
            return True
    return False


def _specialty_terms(specialty: str) -> tuple[str, ...]:
    """
    Возвращает нормализованные термины специальности для role-матчинга.

    :param specialty: каноническая специальность (например, "хирург", "лор", "узи")
    :return: кортеж терминов/синонимов для подстрочного поиска
    """
    return _shared_specialty_terms(specialty)


def _specialty_norm(value: str) -> str:
    return _normalise_input(value)


def _specialty_equivalent(left: str, right: str) -> bool:
    """
    Проверяет эквивалентность двух обозначений специальности.

    Пример: "лор" ~= "оториноларинголог".

    :param left: первая специальность
    :param right: вторая специальность
    :return: True, если обозначения эквивалентны
    """

    return _shared_specialty_equivalent(left, right)


def _extract_specialties_from_text(text: str) -> tuple[str, ...]:
    """
    Извлекает все распознанные специальности из произвольного текста.

    :param text: исходный текст
    :return: кортеж нормализованных специальностей
    """

    return _shared_extract_specialties_from_text(text)


def _is_direct_specialty_text_match(text: str, specialty: str) -> bool:
    """
    Строго проверяет соответствие текста конкретной специальности.

    Важно для прямых запросов по врачу:
    - "терапевт" не должен матчиться на "гирудотерапевт";
    - гибриды вида "кардиолог-ревматолог" не должны попадать в чистый запрос
      "кардиолог".

    :param text: текст для проверки (serviceName/unit_name)
    :param specialty: целевая специальность
    :return: True для чистого соответствия специальности
    """

    target = _specialty_norm(specialty)
    if not target:
        return False
    found = _extract_specialties_from_text(text)
    if not found:
        return False
    if not any(_specialty_equivalent(spec, target) for spec in found):
        return False
    for spec in found:
        if not _specialty_equivalent(spec, target):
            return False
    return True


def _doctor_matches_primary_specialty(doc: dict[str, Any], specialty: str) -> bool:
    """
    Проверяет, что врач относится к специальности именно по primary/main профилю.

    :param doc: карточка врача
    :param specialty: целевая специальность
    :return: True, если есть релевантный main-unit для этой специальности
    """

    main_units = _collect_role_unit_names(doc, main_value=True)
    if main_units:
        return any(_is_direct_specialty_text_match(unit_name, specialty) for unit_name in main_units)
    # Legacy fallback: если main-структуры нет, используем любой unit-level match.
    return _doctor_role_specialty_match_level(doc, specialty) >= 1


def _matches_specialty_terms(text: str, specialty: str) -> bool:
    """
    Проверяет совпадение текста с role-терминами специальности.

    :param text: произвольный текст (подразделение/специализация)
    :param specialty: искомая специальность
    :return: True, если найдено совпадение по одному из терминов
    """
    return _shared_matches_specialty_terms(text, specialty)


def _collect_role_unit_names(doc: dict[str, Any], *, main_value: bool) -> list[str]:
    """
    Возвращает список названий подразделений врача по признаку main.

    :param doc: карточка врача из doctors.jsonl
    :param main_value: True для main=true, False для main=false fallback
    :return: уникализированный список unit names
    """
    out: list[str] = []
    unit_links = doc.get("unit_links") or []
    if isinstance(unit_links, list):
        for raw_link in unit_links:
            if not isinstance(raw_link, dict):
                continue
            if bool(raw_link.get("main")) != main_value:
                continue
            unit_name = str(raw_link.get("company_unit_name") or "").strip()
            if unit_name and unit_name not in out:
                out.append(unit_name)
    if main_value:
        # Поддержка старого формата кэша, где main-характеристика уже агрегирована в main_units.
        for raw in (doc.get("main_units") or []):
            unit_name = str(raw or "").strip()
            if unit_name and unit_name not in out:
                out.append(unit_name)
    if not main_value:
        # Для legacy-кэшей без unit_links/main берем units как fallback-связи.
        main_units_norm = {
            _normalise_input(str(x or ""))
            for x in (doc.get("main_units") or [])
            if str(x or "").strip()
        }
        for raw in (doc.get("units") or []):
            unit_name = str(raw or "").strip()
            if main_units_norm and _normalise_input(unit_name) in main_units_norm:
                continue
            if unit_name and unit_name not in out:
                out.append(unit_name)
    return out


def _doctor_role_specialty_match_level(doc: dict[str, Any], specialty: str) -> int:
    """
    Матчинг ролевого запроса по специальности с приоритетом main-полей.

    Уровни:
    - 2: найдено совпадение в unit name с main=true
    - 1: найдено совпадение в unit name с main=false / legacy units
    - 0: совпадений нет

    :param doc: карточка врача
    :param specialty: каноническая специальность (например, "хирург")
    :return: целочисленный приоритет совпадения
    """
    spec_norm = _normalise_input(specialty)
    if not spec_norm:
        return 0

    main_true_units = _collect_role_unit_names(doc, main_value=True)
    if any(_matches_specialty_terms(unit_name, spec_norm) for unit_name in main_true_units):
        return 2

    main_false_units = _collect_role_unit_names(doc, main_value=False)
    if any(_matches_specialty_terms(unit_name, spec_norm) for unit_name in main_false_units):
        return 1

    return 0


def _doctor_main_payload(doc: dict[str, Any]) -> tuple[list[str], list[str], bool]:
    """
    Извлекает main-поля врача из нового и старого формата кэша.

    :param doc: карточка врача из doctors.jsonl
    :return: (main_units, main_specializations, has_any_main_links)
    """
    main_units: list[str] = []
    for raw in (doc.get("main_units") or []):
        value = str(raw or "").strip()
        if value and value not in main_units:
            main_units.append(value)

    main_specs: list[str] = []
    for raw in (doc.get("main_specializations") or []):
        value = str(raw or "").strip()
        if value and value not in main_specs:
            main_specs.append(value)

    has_main_links = False
    unit_links = doc.get("unit_links") or []
    if isinstance(unit_links, list):
        for raw_link in unit_links:
            if not isinstance(raw_link, dict):
                continue
            if not bool(raw_link.get("main")):
                continue
            has_main_links = True
            unit_name = str(raw_link.get("company_unit_name") or "").strip()
            link_spec = str(raw_link.get("specialization") or "").strip()
            if unit_name and unit_name not in main_units:
                main_units.append(unit_name)
            if link_spec and link_spec not in main_specs:
                main_specs.append(link_spec)

    if main_units or main_specs:
        has_main_links = True
    return main_units, main_specs, has_main_links


def _iter_unit_link_specs(doc: dict[str, Any]) -> list[tuple[str, str, bool]]:
    """
    Возвращает список описаний из unit_links в виде (unit_name, specialization, main_flag).

    :param doc: карточка врача
    :return: список описаний по связям подразделений
    """
    out: list[tuple[str, str, bool]] = []
    unit_links = doc.get("unit_links") or []
    if not isinstance(unit_links, list):
        return out
    for raw_link in unit_links:
        if not isinstance(raw_link, dict):
            continue
        unit_name = str(raw_link.get("company_unit_name") or "").strip()
        link_spec = str(raw_link.get("specialization") or "").strip()
        if not link_spec:
            continue
        out.append((unit_name, link_spec, bool(raw_link.get("main"))))
    return out


def _specialization_matches_specialty(unit_name: str, link_spec: str, specialty: str) -> bool:
    """
    Проверяет, относится ли specialization-блок к нужной специальности.

    :param unit_name: имя подразделения врача (company_unit_name)
    :param link_spec: текст specialization для связи
    :param specialty: целевая специальность
    :return: True, если specialization релевантен специальности
    """
    spec_norm = _normalise_input(specialty)
    if not spec_norm:
        return False
    # В ролевом режиме (по специальности) опираемся именно на unit_name.
    # Иначе длинный текст specialization может содержать "чужие" термины
    # и подмешивать нерелевантные блоки описания.
    return _matches_specialty_terms(unit_name, spec_norm)


def _pick_display_specialization(
    doc: dict[str, Any],
    *,
    preferred_specialty: str = "",
    preferred_service: str = "",
) -> str:
    """
    Выбирает описание врача для UI без «перепутанных» блоков специализации.

    Приоритет:
    1) main=true + совпадение с запрошенной специальностью/услугой
    2) main=false + совпадение с запрошенной специальностью/услугой
    3) любой main=true specialization
    4) любой main=false specialization
    5) top-level specialization из кэша

    :param doc: карточка врача
    :param preferred_specialty: специальность из запроса (если есть)
    :param preferred_service: услуга/процедура из запроса (если есть)
    :return: выбранный текст specialization
    """
    spec_norm = _normalise_input(preferred_specialty)
    service_norm = _normalise_input(preferred_service)
    link_specs = _iter_unit_link_specs(doc)

    main_true_matched: list[str] = []
    main_false_matched: list[str] = []
    main_true_any: list[str] = []
    main_false_any: list[str] = []

    for unit_name, link_spec, is_main in link_specs:
        if is_main:
            if link_spec not in main_true_any:
                main_true_any.append(link_spec)
        else:
            if link_spec not in main_false_any:
                main_false_any.append(link_spec)

        is_match = False
        if spec_norm and _specialization_matches_specialty(unit_name, link_spec, spec_norm):
            is_match = True
        if not is_match and service_norm:
            # Для процедурных запросов match по тексту specialization.
            if _doctor_matches_service({"specialization": link_spec, "unit_links": [], "main_specializations": []}, service_norm):
                is_match = True

        if is_match:
            if is_main:
                if link_spec not in main_true_matched:
                    main_true_matched.append(link_spec)
            else:
                if link_spec not in main_false_matched:
                    main_false_matched.append(link_spec)

    if main_true_matched:
        return main_true_matched[0]
    if main_false_matched:
        return main_false_matched[0]

    _, main_specs, _ = _doctor_main_payload(doc)
    if main_specs:
        return main_specs[0]
    if main_true_any:
        return main_true_any[0]
    if main_false_any:
        return main_false_any[0]
    return str(doc.get("specialization") or "")


def _is_role_specialty_query(query_text: str, specialty: str) -> bool:
    """
    Определяет, является ли запрос ролевым (по специальности), а не процедурным.

    Логика:
    - Для большинства специальностей считаем запрос ролевым.
    - Для УЗИ различаем:
      - role: "какие узисты", "врач узи";
      - procedure: "узи брюшной полости", "сделать узи ...".

    :param query_text: текст запроса пользователя
    :param specialty: распознанная специальность
    :return: True для ролевого сценария, False для процедурного
    """
    spec_norm = _normalise_input(specialty)
    query_norm = _normalise_input(query_text)
    if not spec_norm:
        return False
    if spec_norm != "узи":
        return True
    if _UZI_ROLE_HINT_RE.search(query_norm) and not _UZI_PROCEDURE_HINT_RE.search(query_norm):
        return True
    return False


def _doctor_matches_specialty(doc: dict[str, Any], specialty: str, query_text: str) -> bool:
    """
    Проверяет соответствие врача специальности с учетом нового поля main.

    Правило:
    - role-запрос (например, "какие хирурги"): сначала матчим по main=true;
      если main-связей у врача нет, используем fallback по старому текстовому профилю.
    - процедурный запрос (например, "узи брюшной полости"): используем старый путь
      по specialization/эвристикам процедуры.

    :param doc: карточка врача
    :param specialty: искомая специальность
    :param query_text: исходный запрос пользователя
    :return: True, если врач подходит под фильтр
    """
    spec_norm = _normalise_input(specialty)
    if not spec_norm:
        return False

    role_query = _is_role_specialty_query(query_text, spec_norm)
    if spec_norm == "узи" and not role_query:
        return _matches_uzi_doctor_profile(doc)

    if role_query:
        return _doctor_role_specialty_match_level(doc, spec_norm) > 0

    fio = _normalise_input(str(doc.get("fio", "")))
    spec_text = _normalise_input(str(doc.get("specialization", "")))
    regions = " ".join([_normalise_input(str(x)) for x in (doc.get("regions") or []) if str(x).strip()])
    units = " ".join([_normalise_input(str(x)) for x in (doc.get("units") or []) if str(x).strip()])
    hay = " | ".join([fio, spec_text, regions, units])
    if spec_norm == "узи":
        return _matches_uzi_doctor_profile(doc)
    if spec_norm in hay:
        return True
    return _matches_specialty_terms(hay, spec_norm)


def _stem_service_token(token: str) -> str:
    """
    Упрощенный стемминг русских слов для match процедур (без NLP-библиотек).

    :param token: токен услуги
    :return: укороченный вариант токена
    """
    t = normalize_ru(token)
    if len(t) < 5:
        return t
    endings = (
        "иями",
        "ями",
        "ами",
        "иями",
        "ией",
        "ия",
        "ие",
        "ию",
        "ии",
        "ой",
        "ей",
        "ом",
        "ем",
        "ах",
        "ях",
        "ам",
        "ям",
        "ый",
        "ий",
        "ая",
        "ое",
        "ые",
        "ую",
        "ого",
        "ему",
        "ым",
        "им",
        "у",
        "а",
        "я",
    )
    for suffix in endings:
        if t.endswith(suffix) and len(t) - len(suffix) >= 4:
            return t[: -len(suffix)]
    return t


def _service_tokens(service_name: str) -> list[str]:
    """
    Нормализует service_name в информативные токены.

    :param service_name: название услуги/процедуры
    :return: список токенов для поиска в специализации врача
    """
    raw_tokens = re.findall(r"[a-zа-яё0-9]{2,}", _normalise_input(service_name))
    out: list[str] = []
    for tok in raw_tokens:
        t = normalize_ru(tok)
        if t in _SERVICE_FILTER_STOPWORDS:
            continue
        if len(t) < 3:
            continue
        if t not in out:
            out.append(t)
    return out


def _doctor_matches_service(doc: dict[str, Any], service_name: str) -> bool:
    """
    Проверяет, выполняет ли врач конкретную процедуру/услугу.

    В этом фильтре используем только профильные текстовые поля врача
    (specialization и link-level specialization), т.к. запрос процедурный.

    :param doc: карточка врача
    :param service_name: название услуги от NLU/эвристики
    :return: True, если в профиле врача найдено совпадение по услуге
    """
    tokens = _service_tokens(service_name)
    if not tokens:
        return False

    parts: list[str] = [str(doc.get("specialization") or "")]
    for raw in (doc.get("main_specializations") or []):
        parts.append(str(raw or ""))
    for raw_link in (doc.get("unit_links") or []):
        if isinstance(raw_link, dict):
            parts.append(str(raw_link.get("specialization") or ""))
    hay = _normalise_input(" ".join(parts))
    if not hay:
        return False

    matched = 0
    for tok in tokens:
        stem = _stem_service_token(tok)
        if tok in hay or (stem and stem in hay):
            matched += 1

    if len(tokens) == 1:
        return matched >= 1
    return matched >= min(len(tokens), 2)


def _iter_slot_datetimes(schedule: dict[str, Any]) -> list[datetime]:
    out: list[datetime] = []
    if not isinstance(schedule, dict):
        return out
    for days in schedule.values():
        if not isinstance(days, list):
            continue
        for day in days:
            if not isinstance(day, dict):
                continue
            day_date = str(day.get("date") or day.get("curDate") or "").strip()
            if not day_date:
                continue
            slots = day.get("slots") or []
            if not isinstance(slots, list):
                continue
            for slot in slots:
                t = str(slot or "").strip()
                if len(t) < 5:
                    continue
                try:
                    out.append(datetime.fromisoformat(f"{day_date}T{t[:5]}:00"))
                except Exception:
                    continue
    return out


def _fio_tokens(text: str) -> list[str]:
    return [t.lower() for t in _FIO_TOKEN_RE.findall(str(text or ""))]


def _doctor_matches_fio(fio: str, doctor_query: str, resolved_surname: str | None = None) -> bool:
    """
    Проверка фамилии/ФИО ТОЛЬКО по fio врача.
    Не ищем по specialization, чтобы "Ким" не матчился на "хроническим".
    """
    tokens = _fio_tokens(fio)
    if not tokens:
        return False

    if resolved_surname:
        target = _normalise_input(resolved_surname)
        return bool(target and tokens and _normalise_input(tokens[0]) == target)

    candidates: list[str] = []
    q_tokens = _fio_tokens(doctor_query)
    if q_tokens:
        candidates.extend([v for v in surname_variants(q_tokens[0]) if v])

    normalized = sorted({ _normalise_input(x) for x in candidates if len(_normalise_input(x)) >= 2 }, key=len, reverse=True)
    if not normalized:
        return False

    for token in tokens:
        for candidate in normalized:
            if token.startswith(candidate):
                return True
    return False


def _specialty_label_for_doctor(doc: dict[str, Any], *, preferred_specialty: str = "") -> str:
    """
    Возвращает короткую человекочитаемую метку основной специальности врача.

    :param doc: карточка врача
    :param preferred_specialty: специальность из запроса, если она есть
    :return: короткая метка специальности для patient-facing ответа
    """

    preferred = _normalise_input(preferred_specialty)
    if preferred and _doctor_matches_primary_specialty(doc, preferred):
        return preferred_specialty.strip().capitalize()

    for unit_name in _collect_role_unit_names(doc, main_value=True) + _collect_role_unit_names(doc, main_value=False):
        found = _extract_specialties_from_text(unit_name)
        if found:
            return found[0].capitalize()

    display_spec = _pick_display_specialization(doc, preferred_specialty=preferred_specialty)
    for candidate in _split_spec_lines(display_spec):
        found = _extract_specialties_from_text(candidate)
        if found:
            return found[0].capitalize()

    return ""


def _compact_specialization(
    text: str,
    max_lines: int | None = None,
    max_chars: int | None = None,
) -> str:
    """
    Сжимает только дубли строк в специализации.
    Ограничения по строкам/символам отключены по умолчанию (полный текст),
    но могут быть включены параметрами max_lines/max_chars.
    """
    lines = [ln.strip() for ln in str(text or "").splitlines()]
    out: list[str] = []
    seen: set[str] = set()
    for ln in lines:
        if not ln:
            continue
        key = re.sub(r"\s+", " ", ln).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(ln)
        if isinstance(max_lines, int) and max_lines > 0 and len(out) >= max_lines:
            break

    compact = "\n".join(out).strip()
    if isinstance(max_chars, int) and max_chars > 0 and len(compact) > max_chars:
        compact = compact[:max_chars].rstrip() + "..."
    return compact


def _dedupe_doctors_by_fio(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for d in docs:
        fio = _normalise_input(str(d.get("fio") or ""))
        if not fio or fio in seen:
            continue
        seen.add(fio)
        out.append(d)
    return out


def _service_query_matches(service_q: str, service_name: str) -> bool:
    sq = _normalise_input(service_q)
    sn = _normalise_input(service_name)
    if not sq or not sn:
        return False
    if sq in sn:
        return True
    if "анализ" in sq or "лаборатор" in sq:
        analysis_tokens = (
            "анализ",
            "лаборатор",
            "биоматериал",
            "взятие",
            "кров",
            "моч",
            "мазок",
            "сыворот",
            "плазм",
        )
        return any(tok in sn for tok in analysis_tokens)
    if sq == "экг":
        return "экг" in sn or "электрокардиограм" in sn
    return False


# ---------------------------------------------------------------------------
# Mid-block helpers (service-kind detection)
# ---------------------------------------------------------------------------


def _is_consultation_service_query(value: str) -> bool:
    """
    Проверяет, что service_name относится к приему/консультации врача.

    :param value: строка услуги
    :return: True для консультационных услуг
    """

    # Lazy import to avoid circular imports with services_legacy on price regex.
    from .. import services_legacy as _legacy  # noqa: PLC0415

    norm = _normalise_input(str(value or ""))
    return bool(norm and _legacy._PRICE_CONSULT_HINT_RE.search(norm))


def _detect_service_kind(
    service_name: str,
    *,
    query_text: str = "",
    top_retail: dict[str, Any] | None = None,
    is_consult_query: bool = False,
) -> str:
    """
    Определяет тип услуги для PRICE-bundle:
    - `lab` для лабораторных анализов;
    - `doctor` для врачебных услуг/процедур/диагностики.

    :param service_name: итоговое название услуги
    :param query_text: исходный запрос пользователя
    :param top_retail: верхняя retail-строка (если есть)
    :param is_consult_query: заранее вычисленный признак консультации
    :return: `lab` | `doctor`
    """

    from .. import services_legacy as _legacy  # noqa: PLC0415

    norm_name = _normalise_input(str(service_name or ""))
    norm_query = _normalise_input(str(query_text or ""))

    if is_consult_query:
        return "doctor"

    if norm_name and _legacy._DOCTOR_SERVICE_HINT_RE.search(norm_name):
        return "doctor"

    if norm_name and _legacy._LAB_SERVICE_HINT_RE.search(norm_name):
        return "lab"

    if isinstance(top_retail, dict):
        homecode = _normalise_input(
            str(top_retail.get("serviceHomecode") or top_retail.get("homecode") or "")
        )
        deadline = _normalise_input(str(top_retail.get("deadline") or ""))
        if homecode.isdigit() and len(homecode) >= 4 and not _legacy._DOCTOR_SERVICE_HINT_RE.search(norm_name):
            return "lab"
        if deadline and _legacy._LAB_DEADLINE_HINT_RE.search(deadline) and not _legacy._DOCTOR_SERVICE_HINT_RE.search(norm_name):
            return "lab"

    if norm_query and _legacy._LAB_SERVICE_HINT_RE.search(norm_query) and not _legacy._DOCTOR_SERVICE_HINT_RE.search(norm_query):
        return "lab"

    return "doctor"


def _is_clean_consultation_row_name(value: str) -> bool:
    """
    Проверяет, что строка прайса похожа именно на услугу приема/консультации,
    а не на пакет/подготовку с вкраплением слова "прием".

    :param value: имя услуги из прайса
    :return: True для чистого консультационного тарифа
    """

    from .. import services_legacy as _legacy  # noqa: PLC0415

    norm = _normalise_input(str(value or ""))
    if not norm or not _legacy._PRICE_CONSULT_HINT_RE.search(norm):
        return False
    return _legacy._PRICE_CONSULT_EXCLUDE_RE.search(norm) is None


def _service_name_matches_specialty(service_name: str, specialty: str) -> bool:
    """
    Проверяет, что имя услуги относится к нужной специальности.

    :param service_name: строка услуги из каталога/контекста
    :param specialty: каноническая специальность
    :return: True, если в названии услуги есть термин специальности
    """

    return _is_direct_specialty_text_match(service_name, specialty)


def _service_name_allows_specialty(service_name: str, specialty: str) -> bool:
    """
    Мягкая проверка соответствия имени услуги специальности для PRICE-запросов.

    В отличие от strict-варианта (_service_name_matches_specialty) допускает
    гибриды: имя «травматолога-ортопеда» проходит запрос «травматолог», потому что
    атомарная специальность «травматолог» присутствует среди извлечённых. Нужно для
    price-consultation-фильтра, где составные специальности вида «травматолог-ортопед»
    являются канонической формой тарифа. Strict-проверка сохраняется для direct-doctor
    запросов, где важно не спутать «терапевт» с «гирудотерапевт».

    :param service_name: строка услуги из каталога
    :param specialty: каноническая специальность из запроса
    :return: True, если хотя бы одна извлечённая специальность эквивалентна целевой
    """

    target = _specialty_norm(specialty)
    if not target:
        return False
    found = _extract_specialties_from_text(service_name)
    if not found:
        return False
    return any(_specialty_equivalent(spec, target) for spec in found)


# ---------------------------------------------------------------------------
# Later-block helpers (catalog kind classification, price service names)
# ---------------------------------------------------------------------------


def _is_lab_like_service_name(value: str) -> bool:
    """
    Определяет, похожа ли строка прайса на лабораторный анализ.

    :param value: название услуги
    :return: True для lab-like строки
    """

    from .. import services_legacy as _legacy  # noqa: PLC0415

    norm = _normalise_input(str(value or ""))
    if not norm:
        return False
    if _legacy._PRICE_PROCEDURE_LIKE_RE.search(norm):
        return False
    return True


def _classify_catalog_service_kind(
    service_name: str,
    *,
    query_text: str,
    retail_rows: list[dict[str, Any]],
    has_exact_doctor_link: bool,
    is_consult_query: bool = False,
) -> str:
    """
    Классифицирует тип услуги по matched catalog rows, а не только по regex запроса.

    :param service_name: эффективное имя услуги
    :param query_text: исходный пользовательский запрос
    :param retail_rows: релевантные retail-строки
    :param has_exact_doctor_link: найден ли надежный exact-link в doctor_prices
    :param is_consult_query: является ли запрос консультационным
    :return: `lab`, `doctor_consult`, `procedure_with_doctor`, `diagnostic_no_doctor` или `ambiguous`
    """

    from .. import services_legacy as _legacy  # noqa: PLC0415

    if is_consult_query:
        return "doctor_consult"

    top_name = str(
        (retail_rows[0].get("serviceName") or retail_rows[0].get("name") or service_name)
        if retail_rows else service_name
    ).strip()
    top_norm = _normalise_input(top_name)
    service_norm = _normalise_input(service_name)
    query_norm = _normalise_input(query_text)
    top_rows = retail_rows[:3] if retail_rows else []
    lab_signal = bool(
        _legacy._LAB_SERVICE_HINT_RE.search(service_norm)
        or _legacy._LAB_SERVICE_HINT_RE.search(query_norm)
        or any(
            _legacy._LAB_SERVICE_HINT_RE.search(_normalise_input(str(row.get("serviceName") or row.get("name") or "")))
            or str(row.get("deadline") or "").strip()
            for row in top_rows
            if isinstance(row, dict)
        )
    )

    if _legacy._PRICE_DIAGNOSTIC_NO_DOCTOR_RE.search(top_norm) or _legacy._PRICE_DIAGNOSTIC_NO_DOCTOR_RE.search(service_norm):
        return "diagnostic_no_doctor"

    if has_exact_doctor_link:
        return "procedure_with_doctor"

    if top_rows and all(
        _is_lab_like_service_name(str(row.get("serviceName") or row.get("name") or ""))
        for row in top_rows
    ) and lab_signal:
        return "lab"

    if _is_lab_like_service_name(service_name) and not _legacy._PRICE_PROCEDURE_LIKE_RE.search(query_norm) and lab_signal:
        return "lab"

    if _legacy._PRICE_PROCEDURE_LIKE_RE.search(top_norm) or _legacy._PRICE_PROCEDURE_LIKE_RE.search(service_norm):
        return "procedure_with_doctor" if has_exact_doctor_link else "ambiguous"

    return "ambiguous"


def _has_reliable_doctor_service_link(
    matched_rows: list[tuple[int, int, int, int, dict[str, Any]]],
    query_norm: str,
) -> bool:
    """
    Проверяет, что doctor-price linkage достаточно надежный для показа врачей.

    Exact homecode остаётся самым сильным сигналом, но для старых кэшей иногда
    нет homecode на retail-строке. Тогда допускаем показ врачей только если
    строки doctor_prices почти буквально совпадают с целевой услугой.

    :param matched_rows: уже отфильтрованные matched doctor rows
    :param query_norm: нормализованное имя целевой услуги
    :return: True, если linkage можно считать надежным
    """

    from .. import services_legacy as _legacy  # noqa: PLC0415

    if not matched_rows or not query_norm:
        return False
    strong_hits = 0
    for _, matched, _, _, row in matched_rows[:3]:
        row_name = _normalise_input(str(row.get("serviceName") or row.get("name") or ""))
        if not row_name:
            continue
        if query_norm == row_name or query_norm in row_name or row_name in query_norm:
            strong_hits += 1
            continue
        if matched >= max(2, len(_legacy._price_query_tokens(query_norm)) - 1):
            strong_hits += 1
    return strong_hits >= 1


def _should_prefer_retail_query_candidate(query_candidate: str, service_name: str) -> bool:
    """
    Решает, когда для retail-поиска лучше взять текст из текущего запроса,
    а не уже резолвленную услугу.

    Это нужно для лабораторных случаев, где catalog-grounding может приземлить
    запрос в специальный вариант (`Cito`, капиллярная кровь), а пациент спросил
    про базовый анализ без уточняющих модификаторов.

    :param query_candidate: очищенная услуга из текущего запроса
    :param service_name: каноническая услуга после grounding
    :return: True, если для retail-ranking полезнее текущий текст запроса
    """

    from .. import services_legacy as _legacy  # noqa: PLC0415

    query_candidate_norm = _normalise_input(str(query_candidate or ""))
    service_name_norm = _normalise_input(str(service_name or ""))
    if not query_candidate_norm:
        return False
    if not service_name_norm:
        return True
    if query_candidate_norm == service_name_norm:
        return False
    if query_candidate_norm in service_name_norm and len(query_candidate_norm) < len(service_name_norm):
        return True

    candidate_tokens = [tok for tok in _legacy._price_query_tokens(query_candidate_norm) if tok not in _legacy._PRICE_GENERIC_SERVICE_TOKENS]
    service_tokens = [tok for tok in _legacy._price_query_tokens(service_name_norm) if tok not in _legacy._PRICE_GENERIC_SERVICE_TOKENS]
    if candidate_tokens and len(service_tokens) > len(candidate_tokens):
        candidate_covers_service_base = True
        for candidate_token in candidate_tokens:
            if not any(
                service_token == candidate_token
                or (
                    len(candidate_token) >= 4
                    and len(service_token) >= 4
                    and (
                        service_token.startswith(candidate_token[:4])
                        or candidate_token.startswith(service_token[:4])
                    )
                )
                for service_token in service_tokens
            ):
                candidate_covers_service_base = False
                break
        if candidate_covers_service_base:
            return True

    service_flags = _legacy._lab_price_variant_flags({"serviceName": service_name})
    if not service_flags:
        return False

    query_flags = _legacy._query_price_variant_flags(query_candidate)
    return not service_flags.issubset(query_flags)


def _meaningful_price_service_tokens(value: str) -> list[str]:
    """
    Возвращает информативные токены service_name без общих служебных слов.

    :param value: строка услуги
    :return: список нормализованных токенов
    """

    from .. import services_legacy as _legacy  # noqa: PLC0415

    return [
        tok
        for tok in _legacy._price_query_tokens(value)
        if tok not in _legacy._PRICE_GENERIC_SERVICE_TOKENS
    ]


def _is_price_service_noise_token(token: str) -> bool:
    """
    Проверяет, что токен не добавляет предметной специфики к услуге.

    :param token: нормализованный токен услуги
    :return: True для шумового токена
    """

    from .. import services_legacy as _legacy  # noqa: PLC0415

    norm = _legacy._normalise_price_token(token)
    if not norm:
        return True
    if norm.isdigit():
        return True
    return norm in _legacy._PRICE_QUERY_SERVICE_NOISE_TOKENS


def _select_effective_price_service_name(entity_service_name: str, query_service_name: str) -> str:
    """
    Выбирает итоговое имя услуги между извлеченной entity и candidate из query.

    Правило защищает от деградации, когда нижний слой повторно извлекает услугу
    из полного текста и получает более шумную строку с адресом, вторым интентом
    или служебными словами. При этом новый query-candidate все еще может
    победить, если он действительно задает другую или более точную услугу.

    :param entity_service_name: service_name, уже выделенный NLU/grounding слоем
    :param query_service_name: service_name, извлеченный из полного query
    :return: наиболее надежное имя услуги для дальнейшей обработки
    """

    entity = str(entity_service_name or "").strip()
    query = str(query_service_name or "").strip()
    if not entity:
        return query
    if not query:
        return entity

    entity_norm = _normalise_input(entity)
    query_norm = _normalise_input(query)
    if not query_norm or entity_norm == query_norm:
        return entity

    entity_tokens = _meaningful_price_service_tokens(entity)
    query_tokens = _meaningful_price_service_tokens(query)
    if not entity_tokens:
        return query
    if not query_tokens:
        return entity

    entity_set = set(entity_tokens)
    query_set = set(query_tokens)
    if not entity_set.intersection(query_set):
        return query
    if query_set.issubset(entity_set):
        return entity

    query_extra = [tok for tok in query_tokens if tok not in entity_set]
    if not query_extra:
        return entity

    if entity_set.issubset(query_set):
        if any(_is_price_service_noise_token(tok) for tok in query_extra):
            return entity
        entity_kind = _detect_service_kind(entity, query_text=entity)
        query_kind = _detect_service_kind(query, query_text=query)
        if entity_kind != query_kind:
            return entity
        return query

    return query


# ---------------------------------------------------------------------------
# Late-block helpers (doctor sort / specialty priority)
# ---------------------------------------------------------------------------


def _doctor_sort_key(doc: dict[str, Any]) -> tuple[int, str]:
    try:
        ord_value = int(doc.get("ord"))
    except Exception:
        ord_value = 10**9
    fio = _normalise_input(str(doc.get("fio") or ""))
    return ord_value, fio


def _specialty_priority_rank(doc: dict[str, Any], specialty: str) -> int:
    """
    Возвращает приоритет врача внутри специальности по бизнес-списку фамилий.

    :param doc: карточка врача
    :param specialty: специальность запроса
    :return: индекс приоритета (0..N-1), либо большой ранг если врач не в приоритете
    """
    spec_norm = _normalise_input(specialty)
    priorities = _SPECIALTY_PRIORITY_SURNAMES.get(spec_norm)
    if not priorities:
        return 10**6

    fio_norm = _normalise_input(str(doc.get("fio") or ""))
    if not fio_norm:
        return 10**6
    fio_tokens = [token for token in re.findall(r"[a-zа-я0-9]+", fio_norm) if token]
    if not fio_tokens:
        return 10**6
    surname = fio_tokens[0]

    for idx, wanted in enumerate(priorities):
        if surname.startswith(wanted):
            return idx
    return 10**6
