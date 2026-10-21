"""«Слово пациента → роль приёма» по каталогу МИС: точный слой + выбор LLM — в тени (ход (г), L-17, этап 2;
STEP-2026-10-21-G-ROLE-PICK-SHADOW).

Решения владельца 20.10 (г) 1, 2, 6: роль решает LLM по каталогу ролей из среза врачей МИС, наработки кода (точный слой
каталога, `extract_specialty`) — кандидаты; даже где точный слой нашёл роль, сначала проверить, насколько точно он
находит: LLM в тени рядом с ним, счёт расхождений. Детские и промахи резолвера — через LLM, а не правилами.

Под `[MESSENGER_ROUTER] roles_catalog_llm = shadow` на ходе о враче (метка врача или записи, специальность правил или
роль, названная словами МИС, — при любой несрочной метке: «сколько стоит приём X», «есть ли у вас X» под OTHER) — параллельно ходу, вне его бюджета (`choose(..., shadow=True)`), ход не ждёт:

1. точный слой каталога (`RoleCatalog.find`): `exact` / `partial` — кандидаты, первыми в списке вариантов;
2. `choose(role_pick)` по ролям приёма каталога: номер роли, исход `qualifier` (пациент уточнил то, чего нет в названии
   роли: «детский», «по ДМС»), `not_role` (это процедура или услуга, не приём врача) или «ни одна»;
3. строка `role_shadow` после ответа (как тень `pending_reply`): `role_layer` (exact | exact_many | partial | miss |
   no_catalog), `role_exact` — роль точного слоя, `role_rules` — роль, в которую ложится специальность правил,
   `role_llm` — роль LLM или исход, `role_llm_agree` — LLM согласна с точным слоем (там, где он нашёл одну роль),
   `role_rules_agree` — правила согласны с LLM (если имя правил легло в роль каталога; иначе `role_rules_unmapped`).
   Названия ролей — данные МИС, реплики пациента в строке нет.

`off` (по умолчанию) — ход не меняется ни на такт; ответов тень не меняет никогда. `on` — решение действует (`apply`):
роль из каталога, честное «нет» по данным + оффер, уточнение — оператору (только метки врача и записи).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from . import llm_runtime, turn_telemetry
from .choice import Option, Status, Task, choose
from .runtime_config import config as _cfg
from .services._role_catalog import RoleCatalog, _hit, _stem, _tokens, build_role_catalog
from .services._symptom_specialists import in_region, region_doctors

if TYPE_CHECKING:
    from .services import Services

logger = logging.getLogger(__name__)

ROLE_TASK = Task(
    name="role_pick", max_picks=1, outcomes=("qualifier", "not_role", "absent"), num_predict=48, timeout_s=8.0, queue_timeout_ms=2000
)
_DOCTOR_LABELS = frozenset({"DOCTOR_INFO", "DOCTOR_SCHEDULE", "APPOINTMENT"})
_SAFETY_LABELS = frozenset({"URGENT", "COMPLAINT", "MEDICAL_ADVICE"})
_SIDE_KIND = "role_shadow"
_CAP_S = 20.0
_PENDING: set[asyncio.Future[None]] = set()
_CATALOG: dict[str, Any] = {"key": None, "value": None}


def mode() -> str:
    raw = str(getattr(_cfg, "MR_ROLES_CATALOG_LLM", "off") or "").strip().lower()
    return raw if raw in {"off", "shadow", "on"} else "off"


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
    if label in _SAFETY_LABELS:
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
        if rules_role is not None:
            fields["role_rules_agree"] = llm == rules_role
        elif rules_specialty:
            fields["role_rules_unmapped"] = True  # имя правил («ЛОР», «гинеколог») не ложится в роль каталога буквально
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
    if token is not None and fields is None:
        turn_telemetry.side_cancel(token)  # ход не о враче — строки нет
    elif token is not None:
        turn_telemetry.side_finish(token, _SIDE_KIND, **fields)
    elif fields:
        turn_telemetry.note(**fields)
    if cancelled:
        raise asyncio.CancelledError


# Режим `on` действует только на метках врача и записи: цену приёма («сколько стоит приём X») решает путь цены по строкам
# прайса — там роль только в тени.
_ACTING_LABELS = frozenset({"DOCTOR_INFO", "DOCTOR_SCHEDULE", "APPOINTMENT"})


async def apply(decision: Any, user_text: str, services: "Services | None") -> Any:
    """Режим `on` (ход (г), этап 3; STEP-2026-10-21-G-ROLE-PICK-ON): роль хода решает LLM по каталогу МИС, правила — кандидаты.

    - роль выбрана и правила назвали другую (или никакую) — специальность хода = роль каталога («окулист» → офтальмолог,
      «сердечно-сосудистый хирург» — не «хирург»); отказ стоп-листа снимается — роль в клинике есть;
    - `absent` — пациент назвал специалиста, которого нет у врачей региона: честное «такого специалиста нет» + оффер оператора
      (флаг `role_absent`, путь отказа каталога);
    - `qualifier` — «детский», «по ДМС»: подобрать врача поможет оператор (флаг `role_qualifier`);
    - `not_role`, «ни одна», сбой LLM, пустой каталог — как было (честное «нет» без данных невозможно — ход (д)).
    Решение — вызов хода (в его бюджете), поля `role_*` — в строку хода, `role_acted` — что поменялось.
    """

    if mode() != "on" or services is None or decision.label not in _ACTING_LABELS or not str(user_text or "").strip():
        return decision
    if "doctor_name_verified" in (decision.flags or ()) or any((decision.entities or {}).get(k) for k in _DOCTOR_KEYS):
        return decision  # назван врач (фамилия): роль решает его карточка, а не слово о специальности (dev g0124)
    try:
        cat = await catalog(services)
        fields = await decide(decision.label, user_text, str(decision.entities.get("specialty") or ""), cat, shadow=False)
    except Exception:  # новый слой: его сбой не ломает ход
        logger.warning("role_pick_on_failed", exc_info=True)
        return decision
    if not fields:
        return decision
    llm = str(fields.get("role_llm") or "")
    acted = ""
    flags = set(decision.flags or ())
    entities = dict(decision.entities or {})
    reception = {role.name for role in cat.roles if role.reception}
    if llm in reception:
        role = next(role for role in cat.roles if role.name == llm)
        rules_name = str(entities.get("specialty") or "")
        # Правила роли не дали, а точный слой нашёл ту же роль, что LLM, — дальше её найдёт и прежний путь по слову;
        # подстановка заранее сбивала поток записи (филиал из реплики терялся, dev g0023, g0613).
        same_as_exact = not rules_name and fields.get("role_exact") == llm
        if not same_as_exact and not _names_this_role(rules_name, role):
            entities["specialty"] = _atom_named(role, user_text)
            acted = "specialty"
        if flags & _REFUSAL_FLAGS:
            flags -= _REFUSAL_FLAGS
            acted = acted or "refusal_lifted"
    elif llm == "absent" and cat.roles:
        flags |= {"role_absent"}
        acted = "absent"
    elif llm == "qualifier":
        flags |= {"role_qualifier"}
        acted = "qualifier"
    turn_telemetry.note(**fields, role_acted=acted or "none")
    if not acted:
        # Решение роли ход не изменило — лимит решений хода оно не тратит (как висящий вопрос, решение владельца Q5 18.10):
        # иначе следующему выбору хода не хватало бюджета (test g0792: запись к неврологу → «не нашёл в онлайн-записи»).
        if "role_llm" in fields and llm_runtime.refund_turn_decision():
            turn_telemetry.note(role_refund=1)
        return decision
    return replace(decision, entities=entities, flags=flags)


_REFUSAL_FLAGS = frozenset({"unsupported_specialist", "unsupported_catalog"})
_DOCTOR_KEYS = ("doctor_name", "doctor_id", "last_name", "doctor_last_name")


def _names_this_role(name: str, role: Any) -> bool:
    """Специальность правил называет эту же роль: её название целиком или часть составного слова через дефис
    («аллерголог» — «аллерголог-иммунолог», «гинеколог» — «акушер-гинеколог»). Общее слово при уточнении прилагательным —
    другая роль («хирург» — не «сердечно-сосудистый хирург», решение владельца 08.10)."""

    words = _tokens(name)
    if not words:
        return False
    for atom in role.atoms:
        atom_words = _tokens(atom)
        if [_stem(w) for w in atom_words] == [_stem(w) for w in words]:
            return True
        if len(words) == 1 and len(atom.split()) == 1 and "-" in atom and any(_stem(part) == _stem(words[0]) for part in atom_words):
            return True
    return False


def _atom_named(role: Any, text: str) -> str:
    """Специальность роли, которую назвал пациент («нужен флеболог» у роли «сердечно-сосудистый хирург, флеболог» —
    флеболог); не назвал ни одну — первая."""

    words = _tokens(text)
    for atom in role.atoms:
        stems = [_stem(w) for w in _tokens(atom)]
        if stems and all(_hit(stem, words) for stem in stems):
            return atom
    return role.atoms[0] if role.atoms else role.name


async def settle(timeout_s: float) -> None:
    """Дождаться теней не дольше `timeout_s` — для стенда и тестов. На проде не зовётся."""

    pending = [task for task in _PENDING if not task.done()]
    if pending:
        await asyncio.wait(pending, timeout=timeout_s)
