"""Кнопки Telegram-меню: что бот делает по нажатию (29.09).

Меню живёт у шлюза («Таблица экранов, версия 1»): кнопки-разделы и «Назад» он
обрабатывает сам, а сюда приходят кнопки-листья — идентификатором в поле
`button_id`. Нажатие — явный выбор темы пациентом, поэтому тема берётся отсюда,
а не угадывается NLU.

Подпись кнопки приходит в `text` (у API min_length=1), но словами пациента НЕ
является. Зонд 29.09 по проду: «Цена приёма врача», прочитанная как текст, дала
услугу «приём врача» и цену хирурга; «Подготовка к процедуре» — поиск правил
подготовки к «процедуре». Отсюда три режима:

* ``slot`` — тема из кнопки, сущности — только из СМЫСЛА кнопки (обычно ни
  одной), NLU не вызывается. Недостающее спрашивает штатный планировщик и
  запоминает контекст (pending), поэтому ответ пациента разбирается в теме кнопки.
* ``handoff`` — готовый текст и перевод на оператора.
* ``pass`` — подпись сама по себе полноценный запрос («Адреса и часы работы»,
  «Акции и скидки»), и текстовый путь отвечает верно (зонд 29.09): идёт как
  обычный текст.

Любое нажатие — явная смена темы (см. patient_routing_stream): висящий вопрос —
дата записи, «да/нет» оффера оператора — подпись кнопки не получает.
"""

from __future__ import annotations

from dataclasses import dataclass

from .policies import SICK_LEAVE_HANDOFF_TEXT, handoff_message


@dataclass(frozen=True)
class ButtonAction:
    label: str
    mode: str  # "slot" | "handoff" | "pass"
    entities: tuple[tuple[str, str], ...] = ()
    flags: frozenset[str] = frozenset()
    text: str = ""


def _slot(label: str, **entities: str) -> ButtonAction:
    return ButtonAction(label=label, mode="slot", entities=tuple(entities.items()))


def _handoff(label: str, text: str) -> ButtonAction:
    return ButtonAction(label=label, mode="handoff", text=text)


def _pass(label: str) -> ButtonAction:
    return ButtonAction(label=label, mode="pass")


BUTTONS: dict[str, ButtonAction] = {
    # --- Записаться к врачу
    "menu.appointment.book": _slot("APPOINTMENT"),
    "menu.appointment.schedule": _slot("DOCTOR_SCHEDULE"),
    # Решение владельца 25.09: существующую запись меняет только оператор, без вопроса.
    "menu.appointment.change": _handoff("APPOINTMENT", handoff_message("existing_appointment_change")),
    "menu.appointment.prepare": _slot("PREPARE"),
    # --- Сдать анализы
    "menu.tests.where": _pass("ADDRESS"),
    # Кнопка из раздела «Сдать анализы»: контекст анализов известен заранее. Штатный
    # walk-in ответ — «Анализы выполняются без записи, в порядке живой очереди» и
    # адреса. Флаг — ШТАТНЫЙ `policy_nonbookable_walkin`: по нему entity_grounder
    # оставляет «анализы» как класс услуги. Со своим флагом grounder отправлял
    # «анализы» в каталог — и бот предлагал «Анализ крови на аминокислоты» (29.09).
    "menu.tests.booking_needed": ButtonAction(
        label="ADDRESS",
        mode="slot",
        entities=(("service_name", "анализы"),),
        flags=frozenset({"policy_nonbookable_walkin"}),
    ),
    "menu.tests.prepare": _slot("PREPARE"),
    "menu.tests.suggest": _slot("TEST_ASSIST"),
    "menu.tests.price": _slot("PRICE"),
    # --- Результаты анализов
    "menu.results.lookup": _slot("TEST_RESULT"),
    "menu.results.missing": _slot("TEST_RESULT"),
    # Срок готовности анализа бот пока не знает: даже «когда будет готов анализ на
    # ферритин» текстом даёт цену (замер 29.09). Неверный ответ хуже уточнения —
    # до появления сроков в ответах честно переводим на оператора.
    "menu.results.eta": _handoff("TEST_ASSIST", "Срок готовности результата подскажет оператор — соединяю."),
    # --- Сколько стоит
    "menu.price.test": _slot("PRICE"),
    "menu.price.doctor": _slot("PRICE"),
    "menu.price.promo": _pass("NEWS"),
    # --- Адреса и часы работы (лист, решение владельца 28.09)
    "menu.address": _pass("ADDRESS"),
    # --- Справки и документы
    "menu.docs.tax": _pass("OTHER"),
    "menu.docs.contract": _handoff("OTHER", handoff_message("doc_request_handoff")),
    "menu.docs.sick_leave": _handoff("OTHER", SICK_LEAVE_HANDOFF_TEXT),
    # --- На каждом экране
    "menu.operator": _handoff("OTHER", handoff_message("manual_operator")),
}


def action_for(button_id: str) -> ButtonAction | None:
    """Действие для кнопки или None, если идентификатор бот не знает."""
    return BUTTONS.get((button_id or "").strip())
