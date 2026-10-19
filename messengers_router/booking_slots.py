"""Слоты записи по смыслу — LLM решает, правила дают кандидатов (принцип 2; решение владельца 18.10; облако 19.10, задача C).

Выключатели, по умолчанию off — как было:

- `[MESSENGER_ROUTER] booking_specialty_llm` — специальность записи из реплики, когда услуга из неё не подтвердилась.
  «На какое время запись к неврологу»: разбор хода брал «время» за услугу, LLM выбора строки прайса её отвергла (не
  услуга) или каталог её не нашёл, а специальности в решении хода не было — честный отказ гарда B′ с оффером оператора
  вместо вопроса о филиале (карточка test 18.10, g0792). Кандидат специальности — из правил (`policies.extract_specialty`, слова каталога ролей);
  к ней ли пациент хочет записаться, решает модуль выбора (`choose`, задача `booking_specialty`). Подтвердила — цель
  записи она; «нет», сбой — как было (отказ с оффером оператора).
"""

from __future__ import annotations

from typing import Any

from .choice import Option, Status, Task, choose
from .runtime_config import config as _cfg

BOOKING_SPECIALTY_TASK = Task(name="booking_specialty", max_picks=1, num_predict=32, timeout_s=5.0, queue_timeout_ms=1500)


def _on(attr: str) -> bool:
    return str(getattr(_cfg, attr, "off") or "").strip().lower() == "on"


def specialty_enabled() -> bool:
    """`booking_specialty_llm = on`."""

    return _on("MR_BOOKING_SPECIALTY_LLM")


def service_rejected_by_llm(flags: Any) -> bool:
    """Услугу этого хода сняла LLM (выбор строки прайса, `entity_grounder._apply_service_verdict`), а не правила: флаг
    снятия и внятный статус выбора `booking_service_<статус>` (недоступна — не решение)."""

    flags = set(flags or ())
    if "entity_dropped_unverified_service_name" not in flags:
        return False
    return any(f.startswith("booking_service_") and f not in ("booking_service_unavailable", "booking_service_strong") for f in flags)


async def confirms_booking_specialty(user_text: str, specialty: str) -> bool:
    """Пациент хочет записаться (узнать о записи) к врачу этой специальности. Сбой, мусор или «нет» — False."""

    specialty = str(specialty or "").strip()
    if not specialty:
        return False
    choice = await choose(BOOKING_SPECIALTY_TASK, user_text, [Option(id=specialty, text=specialty)])
    return choice.status is Status.PICKED and specialty in choice.ids
