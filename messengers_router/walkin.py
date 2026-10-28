"""«Анализы и ЭКГ — без записи, в порядке живой очереди»: решает LLM по правилу клиники, детектор — кандидат.

Решения владельца 21.10 (принципы разметки 10, 14): знание «без записи» — не хардкод в адресах, а правило в промпте LLM с
примерами. До выключателя его применяли буквы: детектор `detect_nonbookable_walkin_intent` (слово «анализ» / «ЭКГ» плюс
глагол визита) переклеивал метку после NLU в ADDRESS (`policy_nonbookable_walkin`, `flow_price_followup_address`), а
фразу «Анализы выполняются без записи, в порядке живой очереди» в ответ адресами ставил `nonbookable_service_hint` по
тем же буквам — даже когда пациент спрашивал цену списка анализов (g0716: адреса вместо корзины) или процедуру по записи.

Под `walkin_by_llm = on` (`choose`, задача `walkin`, правило клиники и примеры — в промпте, две локации; до 26.10 — под
`service_answer_from_data`, связку владелец развязал: топ-5 разбора 95 цен, п. 2):
- ветка правил `rule_nonbookable_walkin` тему не решает — решает LLM-классификатор (как `topic_by_llm`; его промпт это
  правило знает);
- переклейка после NLU в адреса — только если LLM подтвердила «сдать анализы / сделать ЭКГ»; «по записи» или «о другом» —
  метка остаётся;
- фраза «… без записи, в порядке живой очереди» в ответе адресами — что именно без записи, называет LLM (анализы, ЭКГ),
  а не буквы; «нет» — фразы нет.
Сбой LLM — как было (детектор). Решение хода записывается флагом решения и не спрашивается дважды.
"""

from __future__ import annotations

from typing import Any

from . import llm_runtime, turn_telemetry
from .choice import Option, Status, Task, choose
from .runtime_config import config as _cfg

TASK = Task(name="walkin", max_picks=2, outcomes=("booking", "other"), num_predict=32, timeout_s=5.0, queue_timeout_ms=1500)
_OPTIONS = (
    Option(id="анализы", text="Сдать анализы: кровь, мочу, кал, мазок или другой биоматериал"),
    Option(id="ЭКГ", text="Сделать ЭКГ (электрокардиограмму)"),
)
FLAG = "walkin_llm"  # LLM решила: без записи — `walkin_llm:<что именно>`
FLAG_NO = "walkin_llm_no"  # LLM решила: не «без записи» (по записи или о другом)
_PREFIX = f"{FLAG}:"
_AND = " и "
# Что бот кладёт в `service_name` как класс «без записи»: варианты выбора и оба вместе. Тот же ответ при сбое LLM даёт детектор
# `policies.nonbookable_service_hint` («анализы», «ЭКГ», «анализы и ЭКГ»).
_CLASSES = frozenset({*(o.id.casefold() for o in _OPTIONS), _AND.join(o.id for o in _OPTIONS).casefold()})


def is_class(value: Any) -> bool:
    """Значение `service_name` — класс «без записи» («анализы», «ЭКГ», «анализы и ЭКГ»), который поставил сам бот, а не
    строка каталога. Одно определение — варианты задачи `walkin` (до 24.10 — отдельный список в заземлении,
    STEP-2026-10-24-S2-DOCS-PHRASE)."""

    return str(value or "").casefold() in _CLASSES


def enabled() -> bool:
    """`[MESSENGER_ROUTER] walkin_by_llm = on`; мусор и отсутствие — off (детектор по буквам, как было). Ответ по данным
    (`service_answer_from_data`) этот путь не включает: у каждого свой выключатель и своя карточка (решение владельца 26.10)."""

    return str(getattr(_cfg, "MR_WALKIN_BY_LLM", "off") or "").strip().lower() == "on"


def decided(flags: Any) -> str | None:
    """Решение LLM этого хода по флагам: что без записи («анализы», «ЭКГ», «анализы и ЭКГ»), "" — не без записи, None —
    не спрашивали (или сбой)."""

    flags = set(flags or ())
    if FLAG_NO in flags:
        return ""
    for flag in flags:
        if str(flag).startswith(_PREFIX):
            return str(flag)[len(_PREFIX) :]
    return None


async def service(user_text: str, *, site: str) -> str | None:
    """Что пациент хочет сдать или сделать без записи — решение LLM: «анализы» | «ЭКГ» | «анализы и ЭКГ»; "" — по записи
    или о другом; None — сбой (вызывающий делает как было)."""

    spent_before = llm_runtime.turn_decisions_spent()
    picked = await choose(TASK, user_text, _OPTIONS)
    if picked.failed:
        # Сбой — решает детектор, как было: лимит решений хода не тратится.
        spent = llm_runtime.turn_decisions_spent() > spent_before
        turn_telemetry.note(walkin_llm="failed", walkin_llm_site=site, walkin_llm_refund=int(spent and llm_runtime.refund_turn_decision()))
        return None
    if picked.status is Status.PICKED:
        ids = [o.id for o in _OPTIONS if o.id in picked.ids]
        answer = _AND.join(ids)
    else:
        answer = ""
    turn_telemetry.note(walkin_llm=answer or "no", walkin_llm_site=site)
    return answer


def flags_for(answer: str | None) -> set[str]:
    """Флаги решения для `RouteDecision`: решение хода видят следующие места (и ответ адресами)."""

    if answer is None:
        return set()
    return {f"{_PREFIX}{answer}"} if answer else {FLAG_NO}
