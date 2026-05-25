from datetime import datetime

import pytest

from messengers_router.policies import OPERATOR_HOURS_NOTE, operator_after_hours_note


# 2026-05-18 — понедельник (будни), 2026-05-23 — суббота, 2026-05-24 — воскресенье.
MON = (2026, 5, 18)
SAT = (2026, 5, 23)
SUN = (2026, 5, 24)


@pytest.mark.parametrize(
    "y_m_d, hh, mm",
    [
        (MON, 22, 5),   # реальный кейс: будни 22:05 — операторы уже не работают
        (MON, 20, 0),   # ровно закрытие будней
        (MON, 7, 59),   # до открытия
        (SAT, 19, 0),   # ровно закрытие выходных
        (SAT, 22, 5),   # выходные поздно
        (SUN, 7, 0),    # до открытия в выходной
    ],
)
def test_after_hours_returns_note(y_m_d, hh, mm):
    now = datetime(*y_m_d, hh, mm)
    assert operator_after_hours_note(now) == OPERATOR_HOURS_NOTE


@pytest.mark.parametrize(
    "y_m_d, hh, mm",
    [
        (MON, 8, 0),    # открытие будней
        (MON, 14, 0),   # середина будней
        (MON, 19, 59),  # до закрытия будней
        (SAT, 8, 0),    # открытие выходных
        (SAT, 18, 59),  # до закрытия выходных
        (SUN, 12, 30),  # середина выходного
    ],
)
def test_within_hours_returns_empty(y_m_d, hh, mm):
    now = datetime(*y_m_d, hh, mm)
    assert operator_after_hours_note(now) == ""
