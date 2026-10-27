"""Ответ адресами по данным филиалов, а не одним шаблоном списка (класс B метрики «рабочий бот», 25.10).

Замер `buttons_v1` (wb-metric, деплой 18): 87 провалов (9,8 п. п.) — на вопрос внутри «Адреса и часы работы» / «Где и до скольки
принимают» бот повторял общий список всех филиалов, а в 26 ответах «где сдать» в списке был филиал без забора анализов
(Ново-Садовая, 106 к.82: `analysis=false`). BUG-2026-10-25-PASS-BUTTON-FOLLOWUP-BRANCH-TEMPLATE.

* `address_branch_flags` — список филиалов в ответе о сдаче анализов (ЭКГ, УЗИ) без записи и на кнопке «Где и до скольки
  принимают» (лист раздела «Сдать анализы») и следующем ходе — только филиалы с этим флагом данных филиалов. Что именно без
  записи, называет ответ (`walkin`: LLM по правилу клиники), флаги — данные МИС (`/site/regions`).
* `address_followup_llm` (BUG-2026-10-25-ADDRESS-FOLLOWUP-TEMPLATE) — реплика сразу после адресного листа: LLM выбирает из
  закрытого списка филиалов, какие названы, и что спрошено (`FOLLOWUP_TASK`, промпт `choose_address_followup`, принцип 2); ответ
  печатает код из данных филиалов — адрес, телефон, график, что там делают (принцип 1). Когда можно прийти или принести
  биоматериал («кал в 15:00?», «ХГЧ вечером?») — пункты забора (`analysis`) с их графиком (BUG-2026-10-26-ADDRESS-FOLLOWUP-HOURS).
  Чего в данных нет (праздники, дети, конкретная услуга, кроме анализов / УЗИ / ЭКГ) — честно и оффер оператора (принцип 3).
  Общий вопрос, другая тема, сбой LLM — общий путь, как было; решение тогда лимит хода не тратит.
* `address_named_street` (BUG-2026-10-26-ADDRESS-FOLLOWUP-NAMED-BRANCH-HOURS) — там же: улица, на которой несколько филиалов
  («Победы»), называет их, и выбор LLM получает их карточки, а не вычеркивается как «слово, общее для многих адресов».

Метка адресной кнопки: нажатый лист адресов режима `pass` и возраст в ходах — на нажатии 0, на следующей реплике 1; другое
нажатие или второй ход после — метки нет. Метка хода — переменная хода; между ходами — `_address_button` в состоянии (пишет конец
хода нажатия), в инструменты не идёт (планировщик её снимает).
"""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import button_menu, turn_telemetry
from .choice import Option, Status, Task, choose
from .llm_mode_policy import fail_offer
from .mess_types import SessionState
from .runtime_config import config as _cfg

logger = logging.getLogger(__name__)

ADDRESS_BUTTON_KEY = "_address_button"
_FLAGS = ("analysis", "ecg", "usi")


def branch_flags_on() -> bool:
    """`[MESSENGER_ROUTER] address_branch_flags = on`."""

    return str(getattr(_cfg, "MR_ADDRESS_BRANCH_FLAGS", "off") or "").strip().lower() == "on"


def followup_on() -> bool:
    """`[MESSENGER_ROUTER] address_followup_llm = on`."""

    return str(getattr(_cfg, "MR_ADDRESS_FOLLOWUP_LLM", "off") or "").strip().lower() == "on"


def named_street_on() -> bool:
    """`[MESSENGER_ROUTER] address_named_street = on`."""

    return str(getattr(_cfg, "MR_ADDRESS_NAMED_STREET", "off") or "").strip().lower() == "on"


def tracks_address_button() -> bool:
    """Метку адресной кнопки ставим, только когда её кто-то читает (off — состояние как было)."""

    return branch_flags_on() or followup_on()


def _address_leaf(button_id: str) -> bool:
    action = button_menu.action_for(button_id) if button_id else None
    return action is not None and action.mode == "pass" and action.label == "ADDRESS"


_MARK: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("mr_address_button", default=None)


def track_address_button(state: SessionState, button_id: str) -> None:
    """Начало хода: нажатие адресного листа — метка этого хода (возраст 0), реплика после него — возраст 1; другое нажатие и
    вторая реплика — метки нет. Метка хода держится в переменной хода: состояние ход может сбросить (смена темы)."""

    _MARK.set(None)
    if not tracks_address_button():
        return
    entities = state.last_entities
    previous = entities.pop(ADDRESS_BUTTON_KEY, None)
    if button_id:
        if _address_leaf(button_id):
            _MARK.set({"id": button_id, "age": 0})
        return
    if isinstance(previous, dict) and int(previous.get("age") or 0) == 0 and previous.get("id"):
        _MARK.set({"id": str(previous["id"]), "age": 1})


def finish_turn(state: SessionState) -> None:
    """Конец хода: метку нажатия — в состояние, чтобы её увидела следующая реплика; прочее — снять."""

    mark = _MARK.get()
    if mark is not None and int(mark.get("age") or 0) == 0:
        state.last_entities[ADDRESS_BUTTON_KEY] = dict(mark)
    else:
        state.last_entities.pop(ADDRESS_BUTTON_KEY, None)


def followup_turn() -> bool:
    """Эта реплика — следующая после нажатия адресного листа."""

    mark = _MARK.get()
    return mark is not None and int(mark.get("age") or 0) == 1


def address_button() -> str:
    """Адресный лист, нажатый на этом ходе или на прошлом (эта реплика — следующая после нажатия); иначе пусто."""

    mark = _MARK.get()
    return str(mark.get("id") or "") if mark is not None else ""


def needed_flags(walkin_hint: str | None) -> frozenset[str]:
    """Флаги филиала, без которых филиал в этом ответе — неправда: что без записи назвал ответ («анализы», «ЭКГ», «УЗИ») и
    область адресного листа («Где и до скольки принимают» — `analysis`)."""

    from .services._regions import branch_flags_for

    wanted = set(branch_flags_for(walkin_hint or ""))
    action = button_menu.action_for(address_button())
    if action is not None and action.scope in _FLAGS:
        wanted.add(action.scope)
    return frozenset(wanted)


def filter_branches(branches: Iterable[Any], wanted: frozenset[str]) -> list[Any] | None:
    """Филиалы, у которых по данным есть все нужные флаги; флагов в данных нет или фильтр оставил пусто — None (как было)."""

    rows = list(branches)
    if not wanted or not rows:
        return None
    if not all(isinstance(r, dict) and all(f in r for f in wanted) for r in rows):
        return None
    kept = [r for r in rows if all(bool(r.get(f)) for f in wanted)]
    return kept if kept and len(kept) < len(rows) else None


def region_flags(region: dict[str, Any] | None) -> dict[str, bool]:
    """Флаги места оказания из строки `/site/regions` — в данные ответа адресами (только под выключателем)."""

    if not (branch_flags_on() or followup_on()) or not isinstance(region, dict):
        return {}
    return {flag: bool(region.get(flag)) for flag in _FLAGS if flag in region}


# --------------------------------------------------------------------------- реплика после адресного листа (`address_followup_llm`)

FOLLOWUP_TASK = Task(
    name="address_followup",
    max_picks=3,
    outcomes=("branch", "weekend", "hours", "no_data", "list", "other_topic"),
    num_predict=64,
    timeout_s=8,
    queue_timeout_ms=3000,
)
NO_DATA_TEXT = "Этого в данных о филиалах у меня нет — точно подскажет оператор. Перевести на оператора?"
# Сбой LLM на реплике после адресного листа (`llm_fail_offer` с узлом `address_followup`): не «нет в данных» — не разобрал.
FAILED_TEXT = "Не смог точно разобрать вопрос о филиалах — чтобы не ошибиться, подскажет оператор. Перевести на оператора?"
_CAN = {"analysis": "сдать анализы", "usi": "сделать УЗИ", "ecg": "сделать ЭКГ"}
HOURS_HEAD = "Анализы принимают без записи, в порядке живой очереди, в часы работы пунктов забора:"
# Часов забора по анализам в `/site/regions` нет, а у части анализов время сдачи задаёт подготовка в справочнике МИС («оптимальное
# время сдачи с 8 до 11 часов») — графиком филиала на них не отвечаем, отсылаем к подготовке.
HOURS_NOTE = "Если у анализа особые условия сдачи — натощак или в определённые часы, — они в подготовке к нему: спросите, как подготовиться к этому анализу, и я пришлю памятку."


@dataclass(frozen=True)
class Followup:
    """Решение о реплике после адресного листа: что спрошено и какие филиалы названы (филиалы ответа адресами)."""

    outcome: str
    branches: tuple[dict[str, Any], ...] = field(default_factory=tuple)


async def followup(text: str, branches: list[dict[str, Any]]) -> Followup | None:
    """Что спросил пациент сразу после адресного листа — выбор LLM (`FOLLOWUP_TASK`) среди филиалов ответа адресами
    (`address_info`: адрес, телефон, график, флаги); None — общий путь (общий вопрос, другая тема, сбой LLM), и решение тогда
    лимит хода не тратит."""

    from . import llm_runtime

    rows = [b for b in branches if isinstance(b, dict) and str(b.get("address") or "").strip()]
    spent = llm_runtime.turn_decisions_spent()
    try:
        choice = await choose(FOLLOWUP_TASK, text, [Option(id=str(i), text=str(b["address"])) for i, b in enumerate(rows, 1)])
    except Exception:  # новый слой: его сбой не ломает ход — общий путь
        logger.warning("address_followup_failed", exc_info=True)
        choice = None
    if (choice is None or choice.failed) and fail_offer("address_followup"):
        # Сбой LLM — не «общий вопрос»: общий путь печатал весь список филиалов / «без записи» во всех пунктах на вопрос с
        # условием (праздник, флюорография, конкретный филиал) — честный оффер (BUG-2026-10-27-LLM-QUEUE-GIVES-UP, шаг 4).
        turn_telemetry.note(address_followup="failed_offer", address_followup_picks=0)
        return Followup(outcome="failed")
    outcome = choice.outcome if choice is not None and choice.status is Status.OUTCOME else ""
    picks = tuple(rows[int(i) - 1] for i in (choice.ids if choice is not None else ()) if str(i).isdigit() and 1 <= int(i) <= len(rows))
    # Под `address_named_street` слово адреса не называет филиал, только когда филиалов с ним больше, чем карточек в ответе: улица
    # с тремя филиалами («Победы») называет все три (смоук деплоя 18: выбор LLM вычеркивался, и печатался список всех пунктов).
    common = _common_words([str(b["address"]) for b in rows], more_than=FOLLOWUP_TASK.max_picks if named_street_on() else 2)
    named = tuple(b for b in picks if named_in(text, str(b["address"]), common=common | frozenset(_address_words(str(b.get("city") or "")))))
    if len(named) < len(picks):
        turn_telemetry.note(address_followup_vetoed=len(picks) - len(named))  # филиал выбран, но в реплике не назван
    picks = named
    decided = outcome in ("no_data", "weekend", "hours") or (bool(picks) and outcome in ("branch", "list"))
    turn_telemetry.note(address_followup=outcome or "none", address_followup_picks=len(picks))
    if not decided:
        if llm_runtime.turn_decisions_spent() > spent and llm_runtime.refund_turn_decision():
            turn_telemetry.note(tu_address_followup_refund=1)
        return None
    return Followup(outcome=outcome, branches=picks)


def _address_words(address: str) -> list[str]:
    """Основы слов адреса (от четырёх букв, первые пять)."""

    from .russian_nlu import normalize_ru

    flat = normalize_ru(address or "")
    for mark in ",.-":
        flat = flat.replace(mark, " ")
    return [w[:5] for w in flat.split() if w.isalpha() and len(w) >= 4]


def _common_words(addresses: list[str], *, more_than: int = 2) -> frozenset[str]:
    """Основы, которые есть в адресах больше чем `more_than` филиалов («проспект», «шоссе»), — по ним филиал не назван (по
    данным, без списка); город филиала (`city` его данных) — тоже, его вызывающий добавляет к этим основам."""

    counts: dict[str, int] = {}
    for address in addresses:
        for stem in set(_address_words(address)):
            counts[stem] = counts.get(stem, 0) + 1
    return frozenset(stem for stem, n in counts.items() if n > more_than)


def named_in(text: str, address: str, *, common: frozenset[str] = frozenset()) -> bool:
    """Выбранный LLM филиал назван в реплике: основа слова его адреса, которое не общее для многих филиалов, есть в ней. Правило
    только сужает выбор модели (карточка 25.10: «рядом с метро», «рядом с моим адресом» давали карточки неназванных филиалов, а
    название города в реплике — карточку филиала, у которого город записан в адресе)."""

    from .russian_nlu import normalize_ru

    said = normalize_ru(text or "")
    return any(stem in said for stem in _address_words(address) if stem not in common)


def _card(branch: dict[str, Any]) -> list[str]:
    lines = [str(branch.get("address") or "")]
    if branch.get("phone"):
        lines.append(f"Телефон: {branch['phone']}")
    lines.append(f"График: {branch['work_time']}" if branch.get("work_time") else "График в данных не указан.")
    can = [verb for flag, verb in _CAN.items() if branch.get(flag)]
    cannot = [verb for flag, verb in _CAN.items() if flag in branch and not branch.get(flag)]
    if can:
        lines.append("Здесь можно " + ", ".join(can) + ".")
    if cannot:
        lines.append("Здесь нельзя " + ", ".join(cannot) + ".")
    return lines


def _open_on_weekend(work_time: str) -> bool:
    """График ответа адресами (`_extract_region_work_time`): «выходные …», «сб …», «вс …» — филиал работает в выходные."""

    head = str(work_time or "")
    return any(part in head for part in ("выходные ", "сб ", "вс "))


def followup_text(found: Followup, branches: list[dict[str, Any]]) -> tuple[str, bool]:
    """Ответ из данных филиалов — (текст, нужен ли оффер оператора). Код печатает адрес, телефон, график и флаги (принцип 1)."""

    if found.outcome == "failed":
        return FAILED_TEXT, True
    blocks: list[str] = []
    if found.branches:
        blocks = ["\n".join(_card(b)) for b in found.branches]
    elif found.outcome == "hours":
        rows = [b for b in branches if isinstance(b, dict) and b.get("address") and b.get("work_time")]
        labs = [b for b in rows if b.get("analysis")] if any("analysis" in b for b in rows) else rows
        if labs:
            blocks = ["\n".join([HOURS_HEAD] + [f"{i}. {b['address']} — {b['work_time']}" for i, b in enumerate(labs, 1)])]
    elif found.outcome == "weekend":
        scope = button_menu.action_for(address_button())
        rows = [b for b in branches if isinstance(b, dict) and b.get("address") and b.get("work_time")]
        if scope is not None and scope.scope in _FLAGS:
            rows = [b for b in rows if b.get(scope.scope)] or rows
        open_weekend = [b for b in rows if _open_on_weekend(str(b["work_time"]))]
        if open_weekend:
            lines = ["В выходные работают:"] + [f"{i}. {b['address']} — {b['work_time']}" for i, b in enumerate(open_weekend, 1)]
            blocks = ["\n".join(lines)]
    offer = found.outcome == "no_data" or not blocks
    if offer:
        blocks.append(NO_DATA_TEXT)
    elif found.outcome == "hours":
        blocks.append(HOURS_NOTE)
    return "\n\n".join(blocks), offer
