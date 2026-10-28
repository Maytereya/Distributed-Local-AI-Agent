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

Выбор — на модуле `choose()` (ход (в), клиент 2; выключатель `choose_symptom` снят 28.10 — on на проде с 14.10): модель называет врача из перечня ролей МИС (`enum`
названий в JSON Schema, не больше двух — решение владельца 10.10), код сверяет; исходы «срочно»
(срочность можно только поднять) и «не жалоба». Политика сбоя прежняя: прежний шаблон медвопроса.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Sequence

from ..choice import Option, Status, Task, choose
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
# Роли МИС, к которым не приходят на приём с жалобой: диагностика, процедуры, медсестра, и
# служебные подразделения «<фамилия> -прием». Промпт и раньше просил их не выбирать, но при
# выборе «1–2 врачей» модель добирала второго из них («набираю вес» → анестезиолог-реаниматолог,
# «щитовидка» → ультразвуковой диагностики; стенд 10.10, BUG-2026-10-10-SYMPTOM-EXTRA-SPECIALTIES).
# Это то же правило промпта, исполненное кодом: модель их просто не видит.
_NOT_RECEPTION_RE = re.compile(
    r"диагностик|(?<![а-я])узи(?![а-я])|анестезиолог|медицинская сестра|физиотерап|эндоскопист|-\s*при[её]м\b"
)


# Задача модуля выбора: 1–2 врача первой очереди названиями из перечня; срочно или не жалоба.
SYMPTOM_TASK = Task(
    name="symptom_specialists",
    max_picks=2,
    by_text=True,
    outcomes=("urgent", "not_complaint"),
    num_predict=96,
    timeout_s=_LLM_TIMEOUT_S,
    queue_timeout_ms=_LLM_QUEUE_TIMEOUT_MS,
)
# Прежний промпт допускал до MAX_SPECIALTIES номеров — столько принимает и строгий разбор его ответа.


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


def is_reception_role(name: str) -> bool:
    """К этому врачу приходят на приём с жалобой (не диагностика, не процедура, не служебная роль)."""

    return bool(name) and not _NOT_RECEPTION_RE.search(str(name).lower())


def _is_samara_doctor(doctor: dict[str, Any], samara_tokens: set[str]) -> bool:
    regions = [str(x) for x in (doctor.get("regions") or []) if str(x).strip()]
    if samara_tokens:
        return any(_region_matches_samara_tokens(region, samara_tokens) for region in regions)
    return not _has_explicit_non_samara_regions(regions)


def in_region(tokens: set[str]) -> Callable[[dict[str, Any]], bool]:
    """Предикат «врач принимает в регионе бота» по признакам филиалов региона (`Services` — токены региона)."""

    return lambda doctor: _is_samara_doctor(doctor, tokens)


async def region_doctors(services: "Services") -> tuple[list[dict[str, Any]], set[str]]:
    """Срез врачей и признаки филиалов региона бота — одна точка для совета по жалобе и каталога ролей (ход (г))."""

    doctors = await services._ensure_doctors_cache_loaded()
    tokens = await services._samara_region_tokens()
    return list(doctors or []), tokens


def samara_specialties(doctors: Sequence[dict[str, Any]], samara_tokens: set[str]) -> list[str]:
    """Роли приёма самарских врачей — без повторов, в порядке появления; только те, к кому идут с жалобой.

    :param doctors: карточки врачей из среза МИС
    :param samara_tokens: признаки самарских филиалов (`Services._samara_region_tokens`)
    :return: отображаемые названия специальностей
    """

    seen: dict[str, None] = {}
    region = in_region(samara_tokens)
    for doctor in doctors:
        if not isinstance(doctor, dict) or not region(doctor):
            continue
        for link in doctor.get("unit_links") or []:
            name = specialty_display_name(str((link or {}).get("company_unit_name") or ""))
            if is_reception_role(name):
                seen.setdefault(name, None)
    return list(seen)


async def _pick_by_choice(text: str, specialties: Sequence[str]) -> SpecialistAdvice | None:
    options = [Option(id=name, text=name) for name in specialties]
    choice = await choose(SYMPTOM_TASK, text, options)
    if choice.status is Status.OUTCOME and choice.outcome == "urgent":
        return SpecialistAdvice(urgent=True)
    if choice.status is Status.PICKED:
        return SpecialistAdvice(specialties=choice.ids)
    return None  # не жалоба, ничего не подошло, сбой — прежний шаблон медвопроса


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
    return await _pick_by_choice(raw, specialties)


async def advise(text: str, services: "Services | None") -> SpecialistAdvice | None:
    """Совет по жалобе для ответа на медвопрос; любой сбой — None (прежний шаблон).

    :param text: реплика пациента
    :param services: сервисный слой (срез врачей и самарские филиалы)
    :return: совет или None
    """

    if services is None:
        return None
    try:
        doctors, tokens = await region_doctors(services)
        specialties = samara_specialties(doctors, tokens)
    except Exception as exc:
        logger.warning("symptom_specialists_catalog_failed: %s", type(exc).__name__)
        return None
    return await pick_specialists(text, specialties)
