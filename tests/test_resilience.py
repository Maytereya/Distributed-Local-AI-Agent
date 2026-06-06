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
