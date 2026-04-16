# tests/test_confidence_policy.py
from messengers_router.mess_types import ConfidencePolicy, CONFIDENCE


def test_confidence_singleton_is_frozen():
    """CONFIDENCE is a frozen dataclass — must not be mutable."""
    import dataclasses
    assert dataclasses.is_dataclass(CONFIDENCE)
    try:
        CONFIDENCE.rule_hardcode = 0.99  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except Exception:
        pass


def test_confidence_default_values():
    p = ConfidencePolicy()
    assert p.llm_promote_rich == 0.45
    assert p.llm_promote_hybrid == 0.55
    assert p.refine_min == 0.45
    assert p.rule_hardcode == 0.85
    assert p.high == 0.75
    assert p.moderate == 0.70
    assert p.price_floor == 0.55
    assert p.promoted_floor == 0.60
    assert p.llm_default == 0.20


def test_confidence_singleton_identity():
    """Module-level CONFIDENCE is a ConfidencePolicy instance."""
    assert isinstance(CONFIDENCE, ConfidencePolicy)


def test_confidence_thresholds_ordered():
    """Sanity: promote_hybrid > promote_rich; rule_hardcode > high > moderate."""
    p = CONFIDENCE
    assert p.llm_promote_hybrid > p.llm_promote_rich
    assert p.rule_hardcode > p.high > p.moderate


def test_confidence_custom_policy():
    """ConfidencePolicy supports custom tuning via constructor kwargs."""
    custom = ConfidencePolicy(rule_hardcode=0.90, llm_default=0.15)
    assert custom.rule_hardcode == 0.90
    assert custom.llm_default == 0.15
    # Other fields keep defaults
    assert custom.refine_min == 0.45
