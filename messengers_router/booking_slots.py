"""Слоты записи по смыслу — LLM решает, правила дают кандидатов (принцип 2; решение владельца 18.10; облако 19.10, задача C).

Выключатели, по умолчанию off — как было:

- `[MESSENGER_ROUTER] booking_specialty_llm` — специальность записи из реплики, когда услуга из неё не подтвердилась.
  «На какое время запись к неврологу»: разбор хода брал «время» за услугу, LLM выбора строки прайса её отвергла (не
  услуга) или каталог её не нашёл, а специальности в решении хода не было — честный отказ гарда B′ с оффером оператора
  вместо вопроса о филиале (карточка test 18.10, g0792). Кандидат специальности — из правил (`policies.extract_specialty`, слова каталога ролей);
  к ней ли пациент хочет записаться, решает модуль выбора (`choose`, задача `booking_specialty`). Подтвердила — цель
  записи она; «нет», сбой — как было (отказ с оффером оператора). С 20.10 (DEBT-2026-10-20-BOOKING-SPECIALTY-QUALIFIER)
  у задачи два исхода: подтверждение одного кандидата склоняло модель к «да» («к детскому аллергологу» → аллерголог,
  «узи кисти рук» → «приём к узи»), и сказать «есть уточнение» ей было некуда. `qualifier` — пациент уточняет то, чего
  нет в названии специальности (для кого врач, условие оплаты ДМС/ОМС/полис); `procedure` — пациент хочет на процедуру
  или исследование, а не на приём врача. Исход побеждает выбор: запись к кандидату не идёт — как под off.
- `[MESSENGER_ROUTER] slot_suppress_llm` — подавитель A′-1 (`router._suppress_grounder_rejected_slots`) держал отвергнутую
  в этом ходу услугу, если резолвер прайса находил по ней хоть что-то по буквам («Гастроскопию» → «Гастрин»): после
  честного отказа в состоянии оставалась фраза пациента (BUG-2026-10-18-BOOKING-SERVICE-BY-LETTERS, попутно (3);
  инвариант 2 BUG-2026-06-02-02). Под on: услугу, которую отвергла LLM этого хода, резолвер по буквам не воскрешает;
  снятие без суда LLM (правила; ответ на уточнение цены в P3) — как было.
- Подготовка или запись (выключатель `prepare_vs_booking_llm` снят 22.10 — неделя `on` на проде с деплоя 11) — правила нашли в реплике и запись, и подготовку: «как подготовиться к узи,
  нужно ли не есть» — «узи» + «нужно» правило записи на диагностику (`policies.detect_appointment_intent`) считает записью,
  и вопрос о подготовке получал отказ записи «эту услугу я не нашёл…» (BUG-2026-10-18-PREPARE-UZI-AS-BOOKING). Кандидаты —
  обе темы правил; главную выбирает модуль выбора (`choose`, задача `prepare_or_booking`): «подготовка» — ход идёт в
  памятку, запись остаётся вторым намерением, только если пациент написал о записи словом; «запись», «ни одна», сбой —
  как было.
- Ответ на вопрос-слот кнопки («Подготовка к процедуре» → «колоноскопия с наркозом», «Подобрать анализы» → «проверить
  щитовидку») — с 19.10 решает одна функция ответа на висящий вопрос (`pending_answer`, вопрос `slot_value`, под
  `pending_buttons`); прежний отдельный путь `slot_reply_llm` (задача `slot_reply`) снят — решение владельца 19.10, п. 5.
"""

from __future__ import annotations


from .choice import Choice, Option, Status, Task, choose
from .runtime_config import config as _cfg

BOOKING_SPECIALTY_TASK = Task(
    name="booking_specialty",
    max_picks=1,
    outcomes=("qualifier", "procedure"),
    num_predict=32,
    timeout_s=5.0,
    queue_timeout_ms=1500,
)


def _on(attr: str) -> bool:
    return str(getattr(_cfg, attr, "off") or "").strip().lower() == "on"


def specialty_enabled() -> bool:
    """`booking_specialty_llm = on`."""

    return _on("MR_BOOKING_SPECIALTY_LLM")


def suppress_enabled() -> bool:
    """`slot_suppress_llm = on`."""

    return _on("MR_SLOT_SUPPRESS_LLM")


async def booking_specialty_choice(user_text: str, specialty: str) -> Choice:
    """Решение LLM о записи к врачу этой специальности: PICKED — к ней; OUTCOME (`qualifier`, `procedure`), «нет», сбой,
    мусор — не к ней."""

    return await choose(BOOKING_SPECIALTY_TASK, user_text, [Option(id=specialty, text=specialty)])


def confirms(choice: Choice, specialty: str) -> bool:
    """Запись к специальности подтверждена: выбор без исхода. Исход побеждает выбор (`choose`) — уточнение пациента или
    процедура не дают записи к кандидату, даже если модель его назвала."""

    return choice.status is Status.PICKED and specialty in choice.ids


PREPARE_OR_BOOKING_TASK = Task(name="prepare_or_booking", max_picks=1, num_predict=32, timeout_s=5.0, queue_timeout_ms=1500)
_PREPARE_OR_BOOKING_OPTIONS = (
    Option(id="PREPARE", text="Узнать, как подготовиться к процедуре, исследованию или анализу"),
    Option(id="APPOINTMENT", text="Записаться на приём, процедуру, исследование или анализ"),
)


TEST_ASSIST_FACET_TASK = Task(name="test_assist_facet", max_picks=1, num_predict=32, timeout_s=5.0, queue_timeout_ms=1500)
_TEST_ASSIST_FACET_OPTIONS = (
    Option(id="price", text="Узнать стоимость анализа или услуги"),
    Option(id="availability", text="Узнать, есть ли такой анализ или услуга в клинике, делают ли его, что в него входит"),
    Option(id="where_when", text="Узнать, где или в какое время можно сдать анализ или пройти услугу, нужна ли запись"),
)


# Полный набор свойств (`p1_facet_none_offer = on`, BUG-2026-10-22-P1-FACET-NONE-AS-PRICE): «ни одно» прежней задачи сливало срок
# готовности, подготовку и голое название с «о другом» — оффер на «Сколько готовится ОАМ?» и «ТТГ» (карточка 22.10). Здесь
# они — свои варианты; «ни одно» — только вопрос о другом (возраст, значение показателя, свой материал).
TEST_ASSIST_FACET_FULL_TASK = Task(name="test_assist_facet_full", max_picks=1, num_predict=32, timeout_s=5.0, queue_timeout_ms=1500)
_TEST_ASSIST_FACET_FULL_OPTIONS = _TEST_ASSIST_FACET_OPTIONS + (
    Option(id="readiness", text="Узнать срок готовности результата"),
    Option(id="prepare", text="Узнать, как подготовиться к анализу или услуге"),
    Option(id="bare", text="Просто назвал анализ или услугу, без вопроса"),
)


async def test_assist_facet(user_text: str, *, full: bool = False) -> str:
    """Свойство вопроса об услуге, названной в реплике, — решение LLM (ход (б), решение владельца 20.10 (б) 3): `price` |
    `availability` | `where_when` (под `full` ещё `readiness` | `prepare` | `bare`); `none` — LLM ответила «ни одно»
    (прежняя задача: спрашивают о другом или смысл неясен; полная — только «о другом»); сбой, мусор — пустая строка
    (вызывающий делает как было)."""

    task, options = (TEST_ASSIST_FACET_FULL_TASK, _TEST_ASSIST_FACET_FULL_OPTIONS) if full else (TEST_ASSIST_FACET_TASK, _TEST_ASSIST_FACET_OPTIONS)
    choice = await choose(task, user_text, options)
    if choice.status is Status.NONE_FIT:
        return "none"
    return choice.ids[0] if choice.status is Status.PICKED and choice.ids else ""


async def prepare_wins_over_booking(user_text: str) -> bool:
    """Правила нашли и запись, и подготовку: главная тема реплики — подготовка (решение LLM). Сбой, «запись», «ни
    одна» — False (как было)."""

    choice = await choose(PREPARE_OR_BOOKING_TASK, user_text, _PREPARE_OR_BOOKING_OPTIONS)
    return choice.status is Status.PICKED and choice.ids == ("PREPARE",)
