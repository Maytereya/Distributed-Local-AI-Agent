"""Жалоба на самочувствие → к каким врачам клиники обратиться (05.10, решение владельца).

Пациент пишет «у меня болит голова 3 дня». Бот не ставит диагноз и не назначает лечение,
но называет 1–3 специальности, к которым с такими жалобами обычно обращаются, и
предлагает записать. Раньше на это шёл шаблон «напишите возраст, симптомы и как давно»
— пациент их уже написал — или, если метку ставила LLM, фраза для непрофильных вопросов
«По техническим вопросам обратитесь к администратору клиники».

Специальности — роли приёма самарских врачей из МИС (`unit_links[].company_unit_name`),
не список в коде: в клинике нет специальности — бот её не назовёт. Выбирает LLM по
номеру из этого списка; при признаках неотложного состояния — «срочно» (шаблон скорой).

Сбой LLM, невнятный ответ или «не жалоба» → None: вызывающий отвечает прежним шаблоном
медвопроса (без диагноза, с оператором).

Kill-switch: [MESSENGER_ROUTER] llm_symptom_specialists = false (config.ini, дефолт ВКЛ).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from ..llm_runtime import generate_text
from ..prompt_registry import load_prompt_text
from ..runtime_config import config as _cfg
from ._regions import _has_explicit_non_samara_regions, _region_matches_samara_tokens

if TYPE_CHECKING:
    from .core import Services

logger = logging.getLogger(__name__)

MAX_SPECIALTIES = 3

_LLM_TIMEOUT_S = 12
_LLM_QUEUE_TIMEOUT_MS = 3000
_MAX_LEN = 600

_DOCTOR_PREFIX_RE = re.compile(r"^врач[\s\-–]+", re.I)
_PARENTHESISED_RE = re.compile(r"\([^)]*\)")
_NUMBER_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class SpecialistAdvice:
    """Совет по жалобе: срочно (скорая) либо специальности клиники."""

    urgent: bool = False
    specialties: tuple[str, ...] = ()


def specialty_display_name(unit_name: str) -> str:
    """«Врач-кардиолог» → «кардиолог», «Ревматолог (центр)» → «ревматолог»."""

    name = _PARENTHESISED_RE.sub(" ", str(unit_name or ""))
    name = _DOCTOR_PREFIX_RE.sub("", name.strip())
    return " ".join(name.split()).strip(" ,.-").lower()


def _is_samara_doctor(doctor: dict[str, Any], samara_tokens: set[str]) -> bool:
    regions = [str(x) for x in (doctor.get("regions") or []) if str(x).strip()]
    if samara_tokens:
        return any(_region_matches_samara_tokens(region, samara_tokens) for region in regions)
    return not _has_explicit_non_samara_regions(regions)


def samara_specialties(doctors: Sequence[dict[str, Any]], samara_tokens: set[str]) -> list[str]:
    """Роли приёма самарских врачей — без повторов, в порядке появления.

    :param doctors: карточки врачей из среза МИС
    :param samara_tokens: признаки самарских филиалов (`Services._samara_region_tokens`)
    :return: отображаемые названия специальностей
    """

    seen: dict[str, None] = {}
    for doctor in doctors:
        if not isinstance(doctor, dict) or not _is_samara_doctor(doctor, samara_tokens):
            continue
        for link in doctor.get("unit_links") or []:
            name = specialty_display_name(str((link or {}).get("company_unit_name") or ""))
            if name:
                seen.setdefault(name, None)
    return list(seen)


def _parse(answer: str, specialties: Sequence[str]) -> SpecialistAdvice | None:
    lines = str(answer or "").strip().splitlines()
    first = lines[0].strip().strip('"«»\'`.,:;!').upper() if lines else ""
    if first.startswith("СРОЧНО"):
        return SpecialistAdvice(urgent=True)
    if not first or first.startswith("НЕТ"):
        return None
    picked: list[str] = []
    for number in _NUMBER_RE.findall(first):
        index = int(number) - 1
        if 0 <= index < len(specialties) and specialties[index] not in picked:
            picked.append(specialties[index])
    return SpecialistAdvice(specialties=tuple(picked[:MAX_SPECIALTIES])) if picked else None


async def pick_specialists(text: str, specialties: Sequence[str]) -> SpecialistAdvice | None:
    """К каким из специальностей клиники обратиться с жалобой — решает LLM по номеру.

    :param text: реплика пациента
    :param specialties: специальности клиники (`samara_specialties`)
    :return: совет; None — LLM не ответила внятно или это не жалоба
    """

    if not bool(getattr(_cfg, "MR_LLM_SYMPTOM_SPECIALISTS", True)):
        return None
    raw = str(text or "").strip()
    if not raw or len(raw) > _MAX_LEN or not specialties:
        return None
    numbered = "\n".join(f"{i}. {name}" for i, name in enumerate(specialties, 1))
    try:
        prompt = load_prompt_text("symptom_specialists").replace("<<TEXT>>", raw).replace("<<SPECIALTIES>>", numbered)
        answer = await generate_text(prompt, timeout_s=_LLM_TIMEOUT_S, queue_timeout_ms=_LLM_QUEUE_TIMEOUT_MS, think=False)
    except Exception as exc:  # сбой LLM — вызывающий отвечает прежним шаблоном
        logger.warning("symptom_specialists_failed: %s", type(exc).__name__)
        return None
    advice = _parse(str(answer or ""), specialties)
    if advice is None:
        logger.info("symptom_specialists_none: %r", str(answer or "")[:40])
    return advice


async def advise(text: str, services: "Services | None") -> SpecialistAdvice | None:
    """Совет по жалобе для ответа на медвопрос; любой сбой — None (прежний шаблон).

    :param text: реплика пациента
    :param services: сервисный слой (срез врачей и самарские филиалы)
    :return: совет или None
    """

    if services is None:
        return None
    try:
        doctors = await services._ensure_doctors_cache_loaded()
        tokens = await services._samara_region_tokens()
        specialties = samara_specialties(doctors or [], tokens)
    except Exception as exc:
        logger.warning("symptom_specialists_catalog_failed: %s", type(exc).__name__)
        return None
    return await pick_specialists(text, specialties)
