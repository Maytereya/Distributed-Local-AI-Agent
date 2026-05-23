import pytest

from messengers_router.recovery_policy import explicit_operator_requested


@pytest.mark.parametrize(
    "text",
    [
        # реальный кейс из прод-диалога (раньше уходило в cancel-confirm записи)
        "Можно ли записаться через обычный чат? Не бот",
        "оператор",
        "позовите оператора",
        "соедините с оператором",
        "хочу к живому человеку",
        "переведите на человека",
        "позовите человека",
        "можно менеджера?",
        "не робот, а человек",
        "не бот",
    ],
)
def test_explicit_operator_requested_positive(text):
    assert explicit_operator_requested(text) is True


@pytest.mark.parametrize(
    "text",
    [
        # «с человека» = цена за человека — НЕ должно триггерить оператора
        "сколько стоит с человека",
        "цена анализа с человека",
        "записаться к врачу",
        "когда принимает врач",
        "2 человека на приём",
        "хочу записаться",
        "обычная консультация терапевта",
    ],
)
def test_explicit_operator_requested_negative(text):
    assert explicit_operator_requested(text) is False
