import logging

import pytest
from messengers_router import resilience as R


@pytest.mark.parametrize("resp,exc,expected", [
    ({"ok": True, "data": "%PDF..."}, None, R.OK),
    ({"ok": True, "data": [1, 2]}, None, R.OK),
    ({"ok": True, "data": None}, None, R.NOT_FOUND),
    ({"ok": True, "data": []}, None, R.NOT_FOUND),
    ({"ok": False, "status_code": 404}, None, R.NOT_FOUND),
    ({"ok": False, "status_code": 500}, None, R.TECH_UNAVAILABLE),
    ({"ok": False, "status_code": 503}, None, R.TECH_UNAVAILABLE),
    ({"ok": False, "status_code": None}, None, R.TECH_UNAVAILABLE),
    ({"ok": False, "status_code": 400}, None, R.TECH_UNAVAILABLE),
    (None, None, R.TECH_UNAVAILABLE),
    ({"ok": True, "data": "x"}, ValueError("boom"), R.TECH_UNAVAILABLE),
])
def test_classify_api_response_class_invariant(resp, exc, expected):
    assert R.classify_api_response(resp, exc=exc) == expected


@pytest.mark.parametrize("resp,exc,expected", [
    ({"ok": False, "status_code": 404}, None, R.FM_HTTP_404),
    ({"ok": False, "status_code": 500}, None, R.FM_HTTP_5XX),
    ({"ok": False, "status_code": None}, None, R.FM_TIMEOUT),
    ({"ok": False, "status_code": 400}, None, R.FM_CONN_ERROR),
    (None, ValueError("x"), R.FM_EXCEPTION),
    ({"ok": True, "data": []}, None, R.FM_EMPTY_DATA),
])
def test_failure_mode_from_response(resp, exc, expected):
    assert R.failure_mode_from_response(resp, exc=exc) == expected


def test_log_degraded_emits_structured_line(caplog):
    with caplog.at_level(logging.WARNING):
        R.log_degraded(upstream="result_for_patient", failure_mode=R.FM_TIMEOUT,
                       latency_ms=1234, session_id="s1", fallback_used=False)
    msg = caplog.text
    assert "degraded_upstream" in msg
    assert "upstream=result_for_patient" in msg
    assert "failure_mode=timeout" in msg
    assert "fallback_used=False" in msg


def test_mark_degraded_sets_payload_fields():
    p = {"data": "x"}
    R.mark_degraded(p, upstream="regions", failure_mode=R.FM_TIMEOUT, fallback_used=True)
    assert p["degraded"] is True
    assert p["degraded_upstream"] == "regions"
    assert p["degraded_mode"] == "timeout"
    assert p["degraded_fallback_used"] is True


def test_tech_unavailable_text_substitutes_domain():
    t = R.tech_unavailable_text("результаты анализов")
    assert "результаты анализов" in t
    assert "техническ" in t.lower()
    assert "информацию" in R.tech_unavailable_text()
