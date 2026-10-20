"""«Слово пациента → роль приёма» по каталогу МИС: точный слой + выбор LLM — в тени (ход (г), L-17, этап 2;
STEP-2026-10-21-G-ROLE-PICK-SHADOW).

Решения владельца 20.10 (г) 1, 2, 6: роль решает LLM по каталогу ролей из среза врачей МИС, наработки кода (точный слой
каталога, `extract_specialty`) — кандидаты; даже где точный слой нашёл роль, сначала проверить, насколько точно он
находит: LLM в тени рядом с ним, счёт расхождений. Детские и промахи резолвера — через LLM, а не правилами.

Под `[MESSENGER_ROUTER] roles_catalog_llm = shadow` на ходе о враче (метка врача или записи, специальность правил или
роль, названная словами МИС) — параллельно ходу, вне его бюджета (`choose(..., shadow=True)`), ход не ждёт:

1. точный слой каталога (`RoleCatalog.find`): `exact` / `partial` — кандидаты, первыми в списке вариантов;
2. `choose(role_pick)` по ролям приёма каталога: номер роли, исход `qualifier` (пациент уточнил то, чего нет в названии
   роли: «детский», «по ДМС»), `not_role` (это процедура или услуга, не приём врача) или «ни одна»;
3. строка `role_shadow` после ответа (как тень `pending_reply`): `role_layer` (exact | exact_many | partial | miss |
   no_catalog), `role_exact` — роль точного слоя, `role_rules` — роль, в которую ложится специальность правил,
   `role_llm` — роль LLM или исход, `role_llm_agree` — LLM согласна с точным слоем (там, где он нашёл одну роль),
   `role_rules_agree` — правила согласны с LLM. Названия ролей — данные МИС, реплики пациента в строке нет.

`off` (по умолчанию) — ход не меняется ни на такт; ответов тень не меняет никогда. Включение решения (`on`: «нет» по
данным + оффер, выбор роли для записи) — следующий этап, по карточке.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from . import turn_telemetry
from .choice import Option, Status, Task, choose
from .runtime_config import config as _cfg
from .services._role_catalog import RoleCatalog, build_role_catalog
from .services._symptom_specialists import in_region, region_doctors

if TYPE_CHECKING:
    from .services import Services

logger = logging.getLogger(__name__)

ROLE_TASK = Task(name="role_pick", max_picks=1, outcomes=("qualifier", "not_role"), num_predict=48, timeout_s=8.0, queue_timeout_ms=2000)
_DOCTOR_LABELS = frozenset({"DOCTOR_INFO", "DOCTOR_SCHEDULE", "APPOINTMENT"})
_SIDE_KIND = "role_shadow"
_CAP_S = 20.0
_PENDING: set[asyncio.Future[None]] = set()
_CATALOG: dict[str, Any] = {"key": None, "value": None}


def mode() -> str:
    raw = str(getattr(_cfg, "MR_ROLES_CATALOG_LLM", "off") or "").strip().lower()
    return raw if raw in {"off", "shadow"} else "off"


async def catalog(services: "Services") -> RoleCatalog:
    """Каталог ролей врачей региона из текущего среза — пересобирается, когда сменился срез (тот же объект — кэш)."""

    doctors, tokens = await region_doctors(services)
    key = (len(doctors), hash(tuple(id(d) for d in doctors[:3])), tuple(sorted(tokens)))
    if _CATALOG["key"] != key:
        _CATALOG["value"] = build_role_catalog(doctors, in_region=in_region(tokens))
        _CATALOG["key"] = key
    return _CATALOG["value"]


def launch(label: str, user_text: str, rules_specialty: str, services: "Services | None") -> None:
    """Ход о враче: запустить тень (под `shadow`). Ход её не ждёт."""

    if mode() != "shadow" or services is None or not str(user_text or "").strip():
        return
    if label not in _DOCTOR_LABELS and not str(rules_specialty or "").strip():
        return
    task = asyncio.ensure_future(_decide(label, user_text, str(rules_specialty or ""), services, turn_telemetry.current_ref()))
    _PENDING.add(task)
    task.add_done_callback(_PENDING.discard)


async def decide(label: str, user_text: str, rules_specialty: str, cat: RoleCatalog, *, shadow: bool = True) -> dict[str, Any] | None:
    """Решение «слово → роль» по каталогу: поля телеметрии; не ход о враче — None."""

    if not cat.roles:
        return {"role_layer": "no_catalog"} if label in _DOCTOR_LABELS else None
    match = cat.find(user_text)
    rules_match = cat.find(rules_specialty) if rules_specialty else None
    rules_role = rules_match.exact[0].name if rules_match is not None and len(rules_match.exact) == 1 else None
    if label not in _DOCTOR_LABELS and not (match.exact or match.partial or rules_specialty):
        return None
    if len(match.exact) == 1:
        layer = "exact"
    elif match.exact:
        layer = "exact_many"
    else:
        layer = "partial" if match.partial else "miss"
    reception = [role for role in cat.roles if role.reception]
    first = [role for role in (*match.exact, *match.partial) if role.reception]
    if rules_role:
        first += [role for role in reception if role.name == rules_role]
    ordered = list(dict.fromkeys([*first, *reception]))
    options = [Option(id=role.name, text=role.name) for role in ordered]
    fields: dict[str, Any] = {
        "role_layer": layer,
        "role_exact": match.exact[0].name if layer == "exact" else None,
        "role_rules": rules_role or (rules_specialty or None),
    }
    choice = await choose(ROLE_TASK, user_text, options, shadow=shadow)
    if choice.failed:
        llm = choice.status.value
    elif choice.status is Status.OUTCOME:
        llm = choice.outcome
    elif choice.status is Status.PICKED and choice.ids:
        llm = choice.ids[0]
    else:
        llm = "none"
    fields["role_llm"] = llm
    if not choice.failed:
        if fields["role_exact"]:
            fields["role_llm_agree"] = llm == fields["role_exact"]
        if rules_specialty:
            fields["role_rules_agree"] = rules_role is not None and llm == rules_role
    return fields


async def _decide(label: str, user_text: str, rules_specialty: str, services: "Services", ref: tuple[str, str] | None) -> None:
    token = turn_telemetry.side_begin(ref) if ref is not None else None
    cancelled = False
    fields: dict[str, Any] | None
    try:
        cat = await catalog(services)
        fields = await asyncio.wait_for(decide(label, user_text, rules_specialty, cat), _CAP_S)
    except asyncio.TimeoutError:
        fields = {"role_llm": "late"}
    except asyncio.CancelledError:  # остановка сервиса: строку всё равно пишем
        fields, cancelled = {"role_llm": "late"}, True
    except Exception:  # тень — новый слой: её сбой не ломает ход
        logger.warning("role_pick_shadow_failed", exc_info=True)
        fields = {"role_llm": "error"}
    if token is not None:
        turn_telemetry.side_finish(token, _SIDE_KIND, **(fields or {"role_layer": "not_doctor"}))
    elif fields:
        turn_telemetry.note(**fields)
    if cancelled:
        raise asyncio.CancelledError


async def settle(timeout_s: float) -> None:
    """Дождаться теней не дольше `timeout_s` — для стенда и тестов. На проде не зовётся."""

    pending = [task for task in _PENDING if not task.done()]
    if pending:
        await asyncio.wait(pending, timeout=timeout_s)
