"""Тень LLM рядом с правилом темы (ход (б), решения владельца 20.10 (б) 1, 3; STEP-2026-10-21-B-TOPIC-SHADOW).

Главный принцип владельца: «у нас везде LLM, чтобы не было пространства для тихого бага». Под `legacy_v2` правило
классификатора, если сработало, решает тему само — LLM-классификатор не зовут (7–8 ходов из 10, план (б) §2), и
ошибку правила никто не видит. Тень: на ходе, где тему решило правило, тот же LLM-классификатор (тот же промпт, что
у хода без правила) решает тему параллельно ходу, вне его бюджета и предохранителя (`generate_structured(...,
shadow=True)`), и ход его не ждёт; решение пишется строкой после ответа (`topic_shadow`, журнал
`topic_shadow_<ГГГГММДД>.jsonl`, привязка к ходу по `sess` + `tid`, как тень `pending_reply`):

- `tu_topic_shadow` — метка LLM (или `unavailable` | `invalid` | `late` | `error`);
- `tu_topic_shadow_rule` — метка правила, `tu_topic_shadow_agree` — совпали ли.

Матрица «правило × LLM» по ветке правила (`nlu_rule`) и категории — `evalbench.topic_shadow`. Метка LLM-классификатора
и есть свойство вопроса (PRICE / PREPARE / APPOINTMENT …): расхождение «правило PRICE, LLM PREPARE» — ошибка свойства.

Выключатель `[MESSENGER_ROUTER] topic_shadow_llm = off | on` (по умолчанию off — ход не меняется ни на такт). Ответ хода
тень не меняет никогда: её решение никто не читает.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from . import classifier, llm_runtime, turn_telemetry
from .mess_types import PATIENT_LABEL_PRIORITY
from .runtime_config import config as _cfg

logger = logging.getLogger(__name__)

_SITE = "classifier_shadow"
_SIDE_KIND = "topic_shadow"
_CAP_S = 20.0
_PENDING: set[asyncio.Future[None]] = set()


def enabled() -> bool:
    return str(getattr(_cfg, "MR_TOPIC_SHADOW_LLM", "off") or "").strip().lower() == "on"


def launch(text: str, seeded: dict[str, Any], rule_label: str, *, queue_timeout_ms: int = 30000) -> None:
    """Ход, где тему решило правило: запустить тень (под `on`). Ход её не ждёт."""

    if not enabled() or not str(text or "").strip():
        return
    prompt = classifier._build_classify_prompt(text, seeded)
    task = asyncio.ensure_future(_decide(prompt, rule_label, turn_telemetry.current_ref(), queue_timeout_ms))
    _PENDING.add(task)
    task.add_done_callback(_PENDING.discard)


async def _decide(prompt: str, rule_label: str, ref: tuple[str, str] | None, queue_timeout_ms: int) -> None:
    token = turn_telemetry.side_begin(ref) if ref is not None else None
    cancelled = False
    try:
        raw, _usage = await asyncio.wait_for(
            llm_runtime.generate_structured(
                site=_SITE,
                system=None,
                user=prompt,
                schema="json",
                options={},
                inherit_options=True,
                timeout_s=classifier._CLASSIFY_TIMEOUT,
                queue_timeout_ms=queue_timeout_ms,
                shadow=True,
            ),
            _CAP_S,
        )
        verdict = _label(raw)
    except asyncio.TimeoutError:
        verdict = "late"
    except asyncio.CancelledError:  # остановка сервиса: строку всё равно пишем
        verdict, cancelled = "late", True
    except llm_runtime.LLM_UNAVAILABLE_ERRORS:
        verdict = "unavailable"
    except Exception:  # тень — новый слой: её сбой не ломает ход
        logger.warning("topic_shadow_failed", exc_info=True)
        verdict = "error"
    fields: dict[str, Any] = {"tu_topic_shadow": verdict, "tu_topic_shadow_rule": rule_label}
    if verdict in PATIENT_LABEL_PRIORITY:
        fields["tu_topic_shadow_agree"] = verdict == rule_label
    if token is not None:
        turn_telemetry.side_finish(token, _SIDE_KIND, **fields)
    else:
        turn_telemetry.note(**fields)
    if cancelled:
        raise asyncio.CancelledError


def _label(raw: str) -> str:
    """Метка из ответа модели — тем же разбором, что у хода без правила; вне контракта — `invalid`."""

    obj = classifier._extract_json(raw)
    if not isinstance(obj, dict) or obj.get("label") not in PATIENT_LABEL_PRIORITY:
        return "invalid"
    return str(obj["label"])


async def settle(timeout_s: float) -> None:
    """Дождаться теней не дольше `timeout_s` — для стенда и тестов, до чтения телеметрии хода. На проде не зовётся."""

    pending = [task for task in _PENDING if not task.done()]
    if pending:
        await asyncio.wait(pending, timeout=timeout_s)
