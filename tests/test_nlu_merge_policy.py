# tests/test_nlu_merge_policy.py
"""Unit tests for the LLM-first _merge() policy in nlu_pipeline."""
from messengers_router.mess_types import RouteDecision
from messengers_router.nlu_pipeline import _merge


def _rule(label, confidence=0.85, entities=None, flags=None):
    return RouteDecision(
        label=label,
        confidence=confidence,
        entities=entities or {},
        flags=set(flags or []),
        context_action="continue",
    )


def _llm(label, confidence=0.55, entities=None, flags=None, clarify_needed=False,
         clarify_slots=None, clarify_reason=""):
    return RouteDecision(
        label=label,
        confidence=confidence,
        entities=entities or {},
        flags=set(flags or []),
        context_action="continue",
        clarify_needed=clarify_needed,
        clarify_slots=clarify_slots or [],
        clarify_reason=clarify_reason,
    )


# --- Safety overrides ---

def test_rule_safety_wins_over_llm_other():
    rule = _rule("URGENT", confidence=0.90)
    llm = _llm("APPOINTMENT", confidence=0.80)
    decision, source = _merge(rule, llm)
    assert decision.label == "URGENT"
    assert source == "rule_safety"


def test_rule_safety_wins_over_llm_low_confidence():
    rule = _rule("COMPLAINT", confidence=0.90)
    llm = _llm("OTHER", confidence=0.30)
    decision, source = _merge(rule, llm)
    assert decision.label == "COMPLAINT"
    assert source == "rule_safety"


def test_llm_safety_wins_over_rule_other():
    rule = _rule("APPOINTMENT", confidence=0.85)
    llm = _llm("URGENT", confidence=0.70)
    decision, source = _merge(rule, llm)
    assert decision.label == "URGENT"
    assert source == "llm_safety"


# --- LLM intent always wins for non-safety ---

def test_llm_wins_even_at_low_confidence():
    """Core regression: low-confidence LLM should NOT be overridden by rule."""
    rule = _rule("DOCTOR_SCHEDULE", confidence=0.85)
    llm = _llm("APPOINTMENT", confidence=0.35)
    decision, source = _merge(rule, llm)
    assert decision.label == "APPOINTMENT"
    assert source == "llm_primary"


def test_llm_wins_in_hybrid_mode_at_low_confidence():
    rule = _rule("PRICE", confidence=0.85)
    llm = _llm("TEST_ASSIST", confidence=0.42)
    decision, source = _merge(rule, llm, llm_mode="hybrid")
    assert decision.label == "TEST_ASSIST"
    assert source == "llm_primary"


def test_llm_wins_in_rich_mode_at_low_confidence():
    rule = _rule("ADDRESS", confidence=0.85)
    llm = _llm("DOCTOR_INFO", confidence=0.38)
    decision, source = _merge(rule, llm, llm_mode="rich")
    assert decision.label == "DOCTOR_INFO"
    assert source == "llm_primary"


# --- Strong deterministic PRICE rescue ---

def test_rule_price_rescued_when_llm_drops_to_other():
    """Явный ценовой вопрос («Стоимость общего анализа крови»): rule даёт
    PRICE+rule_price, LLM ушёл в OTHER → итог PRICE (не утечка в оператора)."""
    rule = _rule("PRICE", confidence=0.72, flags=["rule_price"])
    llm = _llm("OTHER", confidence=0.30)
    decision, source = _merge(rule, llm)
    assert decision.label == "PRICE"
    assert source == "rule_price_rescue"
    assert decision.needs_handoff is False
    assert "rule_price_rescue" in decision.flags


def test_rule_price_not_rescued_without_rule_price_flag():
    """Без сильного rule-сигнала (нет флага rule_price) остаётся LLM-first."""
    rule = _rule("PRICE", confidence=0.72)
    llm = _llm("OTHER", confidence=0.30)
    decision, source = _merge(rule, llm)
    assert decision.label == "OTHER"
    assert source == "llm_primary"


def test_rule_price_not_rescued_when_llm_confident_non_other():
    """Спасаем только при llm.label == OTHER: уверенный APPOINTMENT с упоминанием
    цены не должен перехватываться в PRICE."""
    rule = _rule("PRICE", confidence=0.72, flags=["rule_price"])
    llm = _llm("APPOINTMENT", confidence=0.66)
    decision, source = _merge(rule, llm)
    assert decision.label == "APPOINTMENT"
    assert source == "llm_primary"


def test_rule_price_rescue_donates_llm_entities():
    """При спасении PRICE непустые LLM-сущности доносятся, rule-сущности базовые."""
    rule = _rule("PRICE", confidence=0.72, flags=["rule_price"], entities={"service_name": "общий анализ крови"})
    llm = _llm("OTHER", confidence=0.30, entities={"city": "самара"})
    decision, source = _merge(rule, llm)
    assert source == "rule_price_rescue"
    assert decision.entities["service_name"] == "общий анализ крови"
    assert decision.entities["city"] == "самара"


# --- Entity donation ---

def test_rule_entities_donated_when_llm_missing_them():
    """Rule entities fill slots the LLM didn't extract."""
    rule = _rule("APPOINTMENT", entities={"specialty": "кардиолог", "city": "самара"})
    llm = _llm("APPOINTMENT", entities={"doctor_name": "Иванова"})
    decision, source = _merge(rule, llm)
    assert decision.entities["doctor_name"] == "Иванова"
    assert decision.entities["specialty"] == "кардиолог"
    assert decision.entities["city"] == "самара"


def test_llm_entities_not_overridden_by_rule():
    """Rule must not stomp LLM entities that are already present."""
    rule = _rule("APPOINTMENT", entities={"specialty": "хирург"})
    llm = _llm("APPOINTMENT", entities={"specialty": "кардиолог"})
    decision, source = _merge(rule, llm)
    assert decision.entities["specialty"] == "кардиолог"


def test_rule_empty_entities_not_donated():
    """Empty-string values from rule should not pollute LLM decision."""
    rule = _rule("APPOINTMENT", entities={"doctor_name": "", "specialty": None})
    llm = _llm("DOCTOR_SCHEDULE", entities={"doctor_name": "Петрова"})
    decision, _ = _merge(rule, llm)
    assert decision.entities["doctor_name"] == "Петрова"
    assert "specialty" not in decision.entities


# --- Flags merged ---

def test_flags_are_union_of_both():
    rule = _rule("PRICE", flags=["rule_flag"])
    llm = _llm("PRICE", flags=["llm_flag"])
    decision, _ = _merge(rule, llm)
    assert "rule_flag" in decision.flags
    assert "llm_flag" in decision.flags


# --- Clarify fields preserved ---

def test_clarify_fields_preserved_from_llm():
    rule = _rule("DOCTOR_SCHEDULE")
    llm = _llm(
        "DOCTOR_SCHEDULE",
        clarify_needed=True,
        clarify_slots=["doctor_name"],
        clarify_reason="slot_request",
    )
    decision, _ = _merge(rule, llm)
    assert decision.clarify_needed is True
    assert decision.clarify_slots == ["doctor_name"]
    assert decision.clarify_reason == "slot_request"
