"""Слоты записи по смыслу — LLM решает, правила дают кандидатов (принцип 2; решение владельца 18.10; облако 19.10, задача C).

Выключатели, по умолчанию off — как было:

- `[MESSENGER_ROUTER] booking_specialty_llm` — специальность записи из реплики, когда услуга из неё не подтвердилась.
  «На какое время запись к неврологу»: разбор хода брал «время» за услугу, LLM выбора строки прайса её отвергла (не
  услуга) или каталог её не нашёл, а специальности в решении хода не было — честный отказ гарда B′ с оффером оператора
  вместо вопроса о филиале (карточка test 18.10, g0792). Кандидат специальности — из правил (`policies.extract_specialty`, слова каталога ролей);
  к ней ли пациент хочет записаться, решает модуль выбора (`choose`, задача `booking_specialty`). Подтвердила — цель
  записи она; «нет», сбой — как было (отказ с оффером оператора).
- `[MESSENGER_ROUTER] slot_suppress_llm` — подавитель A′-1 (`router._suppress_grounder_rejected_slots`) держал отвергнутую
  в этом ходу услугу, если резолвер прайса находил по ней хоть что-то по буквам («Гастроскопию» → «Гастрин»): после
  честного отказа в состоянии оставалась фраза пациента (BUG-2026-10-18-BOOKING-SERVICE-BY-LETTERS, попутно (3);
  инвариант 2 BUG-2026-06-02-02). Под on: услугу, которую отвергла LLM этого хода, резолвер по буквам не воскрешает;
  снятие без суда LLM (правила; ответ на уточнение цены в P3) — как было.
- `[MESSENGER_ROUTER] prepare_vs_booking_llm` — правила нашли в реплике и запись, и подготовку: «как подготовиться к узи,
  нужно ли не есть» — «узи» + «нужно» правило записи на диагностику (`policies.detect_appointment_intent`) считает записью,
  и вопрос о подготовке получал отказ записи «эту услугу я не нашёл…» (BUG-2026-10-18-PREPARE-UZI-AS-BOOKING). Кандидаты —
  обе темы правил; главную выбирает модуль выбора (`choose`, задача `prepare_or_booking`): «подготовка» — ход идёт в
  памятку, запись остаётся вторым намерением, только если пациент написал о записи словом; «запись», «ни одна», сбой —
  как было.
- `[MESSENGER_ROUTER] slot_reply_llm` — ответ на вопрос-слот кнопки, который правила не разобрали. «Подготовка к процедуре»
  → «колоноскопия с наркозом»: извлекатель фразы услуги название не узнал, «наркоз» правила приняли за специальность —
  бот переспрашивал «К какому анализу…» (BUG-2026-10-08-PREPARE-ANESTHESIA-IN-QUERY); «Подобрать анализы» → «проверить
  щитовидку» (пример из подсказки самой кнопки) уходил в чужую тему (BUG-2026-10-08-SLOT-REPLY-LOST-TO-LLM). Ответ ли
  реплика на вопрос бота, решает модуль выбора (`choose`, задача `slot_reply`, вопрос бота — в данных хода): да — реплика
  целиком — значение слота (название процедуры, цель подбора), ход — в теме вопроса, а что это за памятка или анализы,
  решает инструмент темы по данным; «другое», сбой — как было. Вопросы с полем ожидания (`_expects`) разбирает одна функция
  ответа на висящий вопрос (`pending_answer`) — здесь их нет; срочное не трогается.
"""

from __future__ import annotations


from .choice import Option, Status, Task, choose
from .runtime_config import config as _cfg

BOOKING_SPECIALTY_TASK = Task(name="booking_specialty", max_picks=1, num_predict=32, timeout_s=5.0, queue_timeout_ms=1500)


def _on(attr: str) -> bool:
    return str(getattr(_cfg, attr, "off") or "").strip().lower() == "on"


def specialty_enabled() -> bool:
    """`booking_specialty_llm = on`."""

    return _on("MR_BOOKING_SPECIALTY_LLM")


def suppress_enabled() -> bool:
    """`slot_suppress_llm = on`."""

    return _on("MR_SLOT_SUPPRESS_LLM")


async def confirms_booking_specialty(user_text: str, specialty: str) -> bool:
    """Пациент хочет записаться (узнать о записи) к врачу этой специальности. Сбой, мусор или «нет» — False."""

    specialty = str(specialty or "").strip()
    if not specialty:
        return False
    choice = await choose(BOOKING_SPECIALTY_TASK, user_text, [Option(id=specialty, text=specialty)])
    return choice.status is Status.PICKED and specialty in choice.ids


PREPARE_OR_BOOKING_TASK = Task(name="prepare_or_booking", max_picks=1, num_predict=32, timeout_s=5.0, queue_timeout_ms=1500)
_PREPARE_OR_BOOKING_OPTIONS = (
    Option(id="PREPARE", text="Узнать, как подготовиться к процедуре, исследованию или анализу"),
    Option(id="APPOINTMENT", text="Записаться на приём, процедуру, исследование или анализ"),
)


def prepare_vs_booking_enabled() -> bool:
    """`prepare_vs_booking_llm = on`."""

    return _on("MR_PREPARE_VS_BOOKING_LLM")


async def prepare_wins_over_booking(user_text: str) -> bool:
    """Правила нашли и запись, и подготовку: главная тема реплики — подготовка (решение LLM). Сбой, «запись», «ни
    одна» — False (как было)."""

    choice = await choose(PREPARE_OR_BOOKING_TASK, user_text, _PREPARE_OR_BOOKING_OPTIONS)
    return choice.status is Status.PICKED and choice.ids == ("PREPARE",)


SLOT_REPLY_TASK = Task(name="slot_reply", max_picks=1, num_predict=32, timeout_s=5.0, queue_timeout_ms=1500)
# Тема кнопки → (слот, который ждёт её вопрос; вопрос бота — данные хода для LLM). Только слоты, где ответ — слова пациента
# целиком (название, цель): данные результата анализа (`TEST_RESULT`) — реквизиты, их разбирают правила формата.
SLOT_REPLY_SLOTS: dict[str, tuple[str, str]] = {
    "PREPARE": ("service_name", "К какой процедуре или анализу нужна подготовка? Напишите название."),
    "TEST_ASSIST": ("test_goal", "Для какой цели хотите подобрать анализы? Например: «проверить щитовидку», «витамины», «чекап»."),
}
_SLOT_REPLY_OPTIONS = (
    Option(id="answer", text="Ответ на вопрос бота"),
    Option(id="other", text="Другой вопрос или просьба"),
)
_SLOT_VALUE_MAX = 80


def slot_reply_enabled() -> bool:
    """`slot_reply_llm = on`."""

    return _on("MR_SLOT_REPLY_LLM")


def slot_value(user_text: str) -> str:
    """Реплика как значение слота: одна строка, без концевой пунктуации, разумной длины; длиннее — не ответ-слот."""

    value = " ".join(str(user_text or "").split()).strip(" .,!?;:")
    return value if 0 < len(value) <= _SLOT_VALUE_MAX else ""


async def reply_answers_slot(user_text: str, label: str) -> bool:
    """Реплика — ответ на вопрос-слот кнопки темы `label` (решение LLM). Сбой, мусор, «другое» — False."""

    slot = SLOT_REPLY_SLOTS.get(label)
    if slot is None or not slot_value(user_text):
        return False
    choice = await choose(SLOT_REPLY_TASK, user_text, _SLOT_REPLY_OPTIONS, asked=slot[1])
    return choice.status is Status.PICKED and choice.ids == ("answer",)
