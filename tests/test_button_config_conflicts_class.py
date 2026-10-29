"""Н-7 (STEP-2026-10-30-BUTTON-KEYS-CONFLICT): несовместимые ключи кнопки видны громко, а не молча.

Инвариант класса «ловушка конфигурации»: сочетание ключей, при котором проверку, отданную одному слою, не делает никто,
даёт предупреждение при старте и строку в отпечатке рантайма (`debug.runtime.config_conflicts`); совместимые сочетания —
пусто. Сочетание: `button_answer_topic = frame` + `button_frame_check = judge` (шаг переклейки отдан судье) без
`button_reply_judge = on` (судья не видит кнопку и задачу с исходом `domain` не берёт).
"""

from __future__ import annotations

import itertools
import logging

import pytest

from messengers_router import button_flow, runtime_fingerprint
from messengers_router.runtime_config import config as cfg

_KEYS = ("MR_BUTTON_ANSWER_TOPIC", "MR_BUTTON_FRAME_CHECK", "MR_BUTTON_REPLY_JUDGE")
_VALUES = {
    "MR_BUTTON_ANSWER_TOPIC": ("off", "on", "frame"),
    "MR_BUTTON_FRAME_CHECK": ("off", "judge"),
    "MR_BUTTON_REPLY_JUDGE": ("off", "on"),
}
_ALL = list(itertools.product(*(_VALUES[k] for k in _KEYS)))


def _set(monkeypatch, combo):
    for key, value in zip(_KEYS, combo):
        monkeypatch.setattr(cfg, key, value, raising=False)


@pytest.mark.parametrize("combo", _ALL, ids=["-".join(c) for c in _ALL])
def test_conflict_exactly_when_frame_judge_has_no_button_context(monkeypatch, combo):
    _set(monkeypatch, combo)
    topic, check, judge = combo
    broken = topic == "frame" and check == "judge" and judge != "on"

    assert bool(button_flow.config_conflicts()) == broken


def test_conflict_is_loud_in_log_and_fingerprint(monkeypatch, caplog):
    _set(monkeypatch, ("frame", "judge", "off"))

    with caplog.at_level(logging.WARNING):
        conflicts = button_flow.warn_config_conflicts()
    assert conflicts and any("CONFIG CONFLICT" in r.getMessage() for r in caplog.records)
    assert runtime_fingerprint.runtime_fingerprint(fresh=True)["config_conflicts"] == conflicts


def test_deploy21_combination_is_clean(monkeypatch):
    """Конфиг деплоя 21 (`frame` + `judge` + `button_reply_judge = on`) — без предупреждений."""
    _set(monkeypatch, ("frame", "judge", "on"))

    assert button_flow.config_conflicts() == []
    assert runtime_fingerprint.runtime_fingerprint(fresh=True)["config_conflicts"] == []
