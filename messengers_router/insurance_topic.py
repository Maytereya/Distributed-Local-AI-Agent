"""Вопрос об обслуживании по ДМС / ОМС — тема, которую решает LLM (класс К26, решение владельца 28.10, ответ клиники).

Было (выгрузки диалогов 08.10 и 26.10: 25 реплик пациентов о ДМС, ОМС, полисе, «бесплатно»): ни одно место хода не понимало
оплату страховкой как тему. Правила и сущности брали «ДМС» за название услуги или тему подбора: «Дмс» — «не понял», «Можно ли
сдать анализы по полису ДМС?» — «напишите названия анализов», «ваша клиника работает по ДМС?» — «у меня нет данных», «Как
записаться на процедуру по ДМС» — «к какому врачу записать?».

Под `[MESSENGER_ROUTER] insurance_topic = on`: реплику, в которой есть слово страхования (кандидат — слова из данных клиники,
`data/insurance_topic.yaml`; принцип 2: слова собирают кандидата, решает LLM), модуль выбора (`choose`, задача `insurance`)
относит к одному из вариантов:
- `dms` — как получить помощь по ДМС (работает ли клиника по ДМС, запись, анализы, УЗИ, процедуры, нужно ли гарантийное письмо, что
  взять с собой) —
  памятка клиники и оффер оператора: записать по ДМС может только сотрудник;
- `dms_doctor` — ведёт ли приём по ДМС конкретный врач: признака ДМС у врачей в данных МИС бота нет — перевод на оператора;
- `dms_case` — своё дело по ДМС (куда прислать гарантийное письмо, отдел ДМС, согласование со страховой, уже оформленная запись) —
  перевод на оператора;
- `oms` — ОМС, бесплатно — перевод на оператора (решение владельца 28.10: много нюансов, оператор передаёт оперменеджеру);
- `price` — цена по ДМС / ОМС — ход идёт путём цены (там решение владельца 28.10 о цене, С3 `position_qualifiers`);
- «ни одна» (слово не об оплате лечения), сбой LLM — ход идёт как было.
Тексты ответа — данные клиники, не кода. Что не закрыло ход, лимит решений хода не тратит (как ответ на висящий вопрос, Q5 18.10).
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import llm_runtime, turn_telemetry
from .choice import Option, Status, Task, choose
from .runtime_config import config as _cfg
from .russian_nlu import normalize_ru

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).resolve().parent / "data" / "insurance_topic.yaml"
TASK = Task(name="insurance", max_picks=1, num_predict=32, timeout_s=5.0, queue_timeout_ms=1500)
_OPTIONS = (
    Option(id="dms", text="Как получить помощь по ДМС: работает ли клиника по ДМС, как записаться на приём, сдать анализы, сделать УЗИ или процедуру по ДМС, нужно ли гарантийное письмо, что взять с собой"),
    Option(id="dms_doctor", text="Ведёт ли приём по ДМС конкретный врач, названный по фамилии или имени"),
    Option(id="dms_case", text="Своё дело по ДМС, которое делает или проверяет сотрудник клиники: куда прислать гарантийное письмо и что в нём указать, отдел ДМС, согласование со страховой компанией, документы для страховой, уже оформленная запись по ДМС"),
    Option(id="oms", text="Помощь по полису ОМС или бесплатно"),
    Option(id="price", text="Цена или стоимость услуги по ДМС или по ОМС"),
)
MEMO = "dms"  # памятка клиники и оффер оператора
HANDOFF = frozenset({"dms_doctor", "dms_case", "oms"})  # перевод на оператора
FLAG = "insurance_topic"


@dataclass(frozen=True)
class Reply:
    """Ответ темы: текст и перевод ли это (`handoff`) или оффер оператора вопросом."""

    kind: str
    text: str
    handoff: bool


def enabled() -> bool:
    """`[MESSENGER_ROUTER] insurance_topic = on`; мусор и отсутствие — off (как было)."""

    return str(getattr(_cfg, "MR_INSURANCE_TOPIC", "off") or "").strip().lower() == "on"


@functools.lru_cache(maxsize=1)
def _data() -> dict[str, Any]:
    """Тексты и слова-кандидаты клиники. Файла нет или он битый — пусто: модуль молчит, ход идёт как было."""

    try:
        data = yaml.safe_load(_DATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("insurance_topic_data_unreadable", exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}


def candidate(text: str) -> bool:
    """В реплике есть слово страхования из данных клиники — повод спросить LLM (кандидат, не вердикт)."""

    stems = tuple(normalize_ru(s) for s in (_data().get("candidate_stems") or ()) if str(s or "").strip())
    words = "".join(ch if ch.isalnum() else " " for ch in normalize_ru(text)).split()
    return bool(stems) and any(word.startswith(stems) for word in words)


def reply_for(kind: str) -> Reply | None:
    """Ответ варианта по данным клиники; варианта без ответа (`price`) или текста в данных нет — None (ход как было)."""

    data = _data()
    if kind == MEMO:
        memo, offer = str(data.get("dms_memo") or "").strip(), str(data.get("dms_offer") or "").strip()
        return Reply(kind, f"{memo}\n\n{offer}", handoff=False) if memo and offer else None
    if kind in HANDOFF:
        text = str((data.get("handoff") or {}).get(kind) or "").strip()
        return Reply(kind, text, handoff=True) if text else None
    return None


async def decide(user_text: str) -> Reply | None:
    """Ответ темы ДМС / ОМС по решению LLM; None — ход идёт как было: выключатель off, слова-кандидата нет, вариант «цена»,
    «ни одна», сбой LLM. Решение, не закрывшее ход, лимит решений хода не тратит."""

    if not enabled() or not candidate(user_text):
        return None
    spent_before = llm_runtime.turn_decisions_spent()
    picked = await choose(TASK, user_text, _OPTIONS)
    if picked.failed:
        kind = "failed"
    elif picked.status is Status.PICKED and picked.ids:
        kind = picked.ids[0]
    else:
        kind = "none"
    reply = reply_for(kind)
    refund = reply is None and llm_runtime.turn_decisions_spent() > spent_before and llm_runtime.refund_turn_decision()
    turn_telemetry.note(insurance_topic=kind, **({"insurance_topic_refund": 1} if refund else {}))
    return reply
