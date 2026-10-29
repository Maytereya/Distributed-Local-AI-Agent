"""Контекст сессии = показанное — ход (б) L-14, шаг 3. Единственный путь с 17.10 (выключатель `[MESSENGER_ROUTER]
context_shown` снят, STEP-2026-10-17-ONE-PATH-CONTEXT-SHOWN; на проде `on` с 15.10).

Было: услугу в состояние (`service_name`) писало только заземление правил по буквам (`merge_entities` в
роутере). Услугу, которую ответ показал пациенту, но выбрали LLM (строка цены по термину: «ггтп» → «ГГТ
гамма-глютамилтрансфераза») или код из МИС (срок готовности — ранний возврат мимо слияния сущностей), состояние
не получало: следующее «сколько стоит?» не знало, о чём шла речь. Сводка диалога (`update_summary`, раз в 4 хода)
копировала `service_name` и называла его, когда состояние уже сменилось или было снято
(STEP-2026-10-14-L14-PREP-CONSUMERS, находка 6).

Стало: ответ с ценами пишет в состояние `_shown_services` — названия строк, которые увидел пациент
(как их печатает рендер), а `service_name` — показанную услугу, если она одна; срок из МИС — так же (правило
владельца 14.10: в контексте — услуга, которую ответ назвал пациенту одной строкой; догадка — никогда). У
корзины `_shown_services` — все её строки, `service_name` не пишется (услуг несколько). Реплика, в понимании
которой услуга не названа («а цена?»), не перебивает контекст фразой вырезателя (`prices._names_no_service`).
Сводка обновляется, как только называет не ту услугу. Читатели показанного прошлым ответом (`previous()`):
короткий вопрос о цене без названия услуги отвечает ровно показанными строками (`prices._rows_shown_before`),
вопрос о подготовке без своего предмета — о показанной услуге (`prepare._context_subject`), вопрос walk-in без
своей услуги («а где сдать?», «нужна ли запись?») — тоже о ней: класс «анализы» нужен только ответу об адресах,
контекст после ответа — показанное (`sync_address`, находка 3). Ход раскрытия семейства («все» на подсказку
«напишите: "все"») — не название услуги: роутер не ищет каталог по реплике (`expands_shown`), после раскрытия в
`service_name` — услуга контекста до хода, если она среди показанных (`_keep_service_on_expand`;
BUG-2026-10-14-SHOW-ALL-AS-CATALOG-SERVICE).
"""

from __future__ import annotations

import contextvars
from typing import Any, Iterable

from .mess_types import RouteDecision, SessionState

SHOWN_KEY = "_shown_services"
# Флаг решения walk-in: вопрос без своей услуги — о показанной прошлым ответом (ставит гард walk-in).
WALKIN_SHOWN_FLAG = "walkin_shown_service"
# Ход раскрытия семейства: решение классификатора на «показать все» после списка семейства с подсказкой
# «напишите: "все"» (`classifier._is_price_family_show_all_followup`), флаг роутера «каталог по реплике не
# искали» и пометка ответа цены, раскрывшего список из контекста
# (`_prices_helpers._price_family_payload_from_context`).
SHOW_ALL_FLAG = "rule_price_family_show_all"
SHOW_ALL_KEPT_FLAG = "show_all_context_kept"
# Номер строки показанного списка семейства («1», «2.»; `classifier._is_price_family_row_pick`): тоже не название услуги.
ROW_PICK_FLAG = "rule_price_family_row"
_EXPANDED_NOTE = "price_family_context"
_RENDERED_PRICE_ROWS = 5  # рендер цены печатает не больше пяти строк (`renderer.format_price_for_patient`)

# Что показал прошлый ответ — на весь ход (читатели: цена, подготовка, адрес). Инструменты состояния не видят,
# их сущности собирает планировщик; поэтому показанное кладём в ход, как понимание хода.
_PREVIOUS: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar("mr_shown_previous", default=())
# Услуга контекста до хода (`service_name`): ответ walk-in возвращает её, когда показано несколько строк.
_BEFORE: contextvars.ContextVar[str] = contextvars.ContextVar("mr_shown_service_before", default="")
# Услуга, которую этот ход записал в слот фразой разбора (`pending_in_frame`, `router._pending_slot_value`): вход инструмента хода,
# а не контекст — после ответа её заменяет показанное (`sync_prepare`) или снимает конец хода (`finish_turn`).
_PARSED: contextvars.ContextVar[str] = contextvars.ContextVar("mr_shown_parsed_input", default="")


def _row_name(row: Any) -> str:
    return str(row.get("serviceName") or row.get("name") or "").strip() if isinstance(row, dict) else ""


def shown_price_names(payload: dict[str, Any]) -> list[str]:
    """Названия строк с ценой, которые рендер цены покажет пациенту, — в его порядке и без повторов.

    Уточнение (`clarify_text`: ответ политики, вопрос) цен не показывает — пусто; семейство — видимые
    варианты (`visible_limit`, все — после «все»); строки цены — первые пять разных, как печатает рендер.
    """

    if str(payload.get("clarify_text") or "").strip():
        return []
    if str(payload.get("service_kind") or "").strip().lower() == "family_query":
        variants = payload.get("family_variants")
        if isinstance(variants, list) and variants:
            limit = max(1, int(payload.get("visible_limit") or 10))
            rows: Iterable[Any] = variants if payload.get("showing_all") else variants[:limit]
            return list(dict.fromkeys(name for name in map(_row_name, rows) if name))
    prices = payload.get("prices")
    names = [name for name in map(_row_name, prices if isinstance(prices, list) else []) if name]
    return list(dict.fromkeys(names))[:_RENDERED_PRICE_ROWS]


def shown_bundle_names(payload: dict[str, Any]) -> list[str]:
    """То же для ответа `service_bundle` (`renderer.format_service_bundle_for_patient`): уточнение — пусто;
    семейство и корзина — видимые варианты; несколько строк анализа или выбора LLM — первые пять разных; иначе —
    первая строка (рендер печатает одну цену)."""

    if str(payload.get("clarify_text") or "").strip() or str(payload.get("service_kind") or "").strip().lower() == "family_query":
        return shown_price_names(payload)
    rows = payload.get("retail_prices")
    rows = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    if not rows:
        return []
    if len(rows) > 1 and (str(payload.get("service_kind") or "").strip().lower() == "lab" or payload.get("llm_selected")):
        names = [name for name in map(_row_name, rows[:_RENDERED_PRICE_ROWS]) if name]
        return list(dict.fromkeys(names))
    name = _row_name(rows[0]) or str(payload.get("service_name") or "").strip()
    return [name] if name else []


def begin_turn(state: SessionState) -> None:
    """Новый ход: показанное прошлым ответом и услуга контекста до хода — из состояния."""

    shown = state.last_entities.get(SHOWN_KEY)
    _PREVIOUS.set(tuple(str(name) for name in shown if str(name or "").strip()) if isinstance(shown, list) else ())
    _BEFORE.set(str(state.last_entities.get("service_name") or "").strip())
    _PARSED.set("")


def previous() -> tuple[str, ...]:
    """Названия, которые показал пациенту прошлый ответ; пусто — не показывал."""

    return _PREVIOUS.get()


def remember(state: SessionState, names: list[str]) -> None:
    """Показанное — в состояние: `_shown_services` всегда, `service_name` — если ответ назвал одну услугу.

    Правило владельца 14.10 (уточнение решения 13.10): в контексте — только услуга, которую ответ назвал
    пациенту одной строкой (цена или срок); догадка, которой пациент не видел, — никогда. Ответ политики
    без цен (`clarify_text`) показанного не имеет — услуги нет.
    """

    state.last_entities[SHOWN_KEY] = list(names)
    if len(names) == 1:
        state.last_entities["service_name"] = names[0]
        state.last_entities.pop("test_name", None)


def parsed_input(phrases: str) -> None:
    """Слот-услуга этого хода — позиции разбора, слова пациента (`turn_understanding.service_slot`, `pending_in_frame`): их читает
    инструмент хода, а контекст следующего хода — то, что ответ показал (И-5.5; Н-1, второй ход: «ВПЧ» в слоте и памятка
    «Папилломавирус 16/18» на экране → «а сколько стоит?» отвечало ценой жидкостной цитологии, которой пациент не видел)."""

    _PARSED.set(_norm(phrases))


def sync_prepare(state: SessionState, payload: dict[str, Any]) -> None:
    """Ответ подготовки после фразы разбора в слоте (`parsed_input`): контекст — показанные памятки МИС (`prepare_shown`; названия
    карточек справочника — услуги клиники): одна — она (`remember`), несколько или ни одной (памятка базы знаний, общая памятка,
    «памятки нет») — без единственной услуги. Без фразы разбора в слоте — как было: подготовка контекст не пишет. Ответ — вопрос
    бота о том же предмете («с наркозом или без?», `prepare_variants`): разговор о предмете не закончен, фраза — вход висящего
    вопроса и остаётся (стенд 30.10, `yn_prep_offer_davai`: без неё «давай» на вопрос о наркозе теряло колоноскопию)."""

    if not _PARSED.get() or _norm(str(state.last_entities.get("service_name") or "")) != _PARSED.get():
        return
    _PARSED.set("")
    if isinstance(payload.get("prepare_variants"), dict):
        return
    names = [str(name) for name in payload.get("prepare_shown") or [] if str(name or "").strip()]
    remember(state, names)
    if len(names) != 1:
        state.last_entities.pop("service_name", None)


def finish_turn(state: SessionState) -> None:
    """Конец хода: фраза разбора, которую ответ не заменил показанным (переспрос, оффер, ответ не подготовки), — не контекст
    следующего хода: снимается, если в слоте всё ещё она. Услугу, которую ход записал сам (цена — `sync_price`), не трогает."""

    parsed = _PARSED.get()
    _PARSED.set("")
    if parsed and _norm(str(state.last_entities.get("service_name") or "")) == parsed:
        state.last_entities.pop("service_name", None)


def expands_shown(decision: RouteDecision) -> bool:
    """Ход раскрытия показанного семейства. Реплика («все», «Всё!», «ВСЕ АНАЛИЗЫ») — ответ на подсказку
    бота «напишите: "все"», а не название услуги: роутер не ищет по ней каталог (BUG-2026-10-14-SHOW-ALL-AS-CATALOG-
    SERVICE: «все» находило «УЗИ периферических лимфоузлов комплексно - все зоны (детское)», и подготовка следующим
    ходом отвечала про это УЗИ, которого пациент не видел)."""

    return SHOW_ALL_FLAG in decision.flags


def picks_shown_row(decision: RouteDecision) -> bool:
    """Ход выбора строки показанного списка по номеру («1»): реплика — номер строки, а не название услуги; каталог по ней
    роутер не ищет — строку отвечает инструмент цены из контекста семейства (BUG-2026-10-24-SHOW-ALL-HINT-ON-PROD)."""

    return ROW_PICK_FLAG in decision.flags


def _keep_service_on_expand(state: SessionState, payload: dict[str, Any], names: list[str]) -> None:
    """Ответ раскрыл список семейства из контекста (флаг раскрытия у ответа, а не у решения: так же и когда реплику
    понял не классификатор правил). Раскрытие услугу не выбирает — показал весь список, о котором уже шла речь:
    в `service_name` — услуга контекста до хода, если она среди показанных (принцип L-14: контекст = показанное);
    иначе услуги нет. Что подставили по самой реплике, в контекст не попадает — догадка, которой пациент не видел.
    Одна показанная строка — её запишет `remember` (раскрытие с одной строкой не бывает, но правило общее)."""

    if str(payload.get("note") or "") != _EXPANDED_NOTE or not payload.get("showing_all") or len(names) < 2:
        return
    keep = _BEFORE.get()
    if keep and _norm(keep) in {_norm(name) for name in names}:
        state.last_entities["service_name"] = keep
        state.last_entities.pop("test_name", None)
    else:
        state.last_entities.pop("service_name", None)


def sync_price(state: SessionState, payload: dict[str, Any], render_entities: dict[str, Any]) -> None:
    """Ответ цены (`build_price_response`). Цена у врача — услуга врача, а не контекст анализа: не трогаем."""

    if render_entities.get("doctor_id") or render_entities.get("doctor_name"):
        return
    names = shown_price_names(payload)
    remember(state, names)
    _keep_service_on_expand(state, payload, names)


def sync_bundle(state: SessionState, payload: dict[str, Any]) -> None:
    """Ответ `build_service_bundle_response` (одна услуга — «По услуге «X» нашёл следующее», корзина)."""

    names = shown_bundle_names(payload)
    remember(state, names)
    _keep_service_on_expand(state, payload, names)


def walkin_about_shown() -> frozenset[str]:
    """Флаг для решения walk-in, в реплике которого своей услуги нет (гард `_guard_walkin_named_service`):
    если прошлый ответ что-то показал — вопрос о показанном; иначе пусто, решение как было."""

    return frozenset({WALKIN_SHOWN_FLAG}) if previous() else frozenset()


def sync_address(state: SessionState, decision: RouteDecision, *, walkin: bool = False) -> None:
    """Ответ адреса (`build_address_response`) — после того, как текст построен.

    Находка 3 (STEP-…-L14-PREP-CONSUMERS): вопрос walk-in без своей услуги («а где сдать?», «нужна ли запись?»)
    заменял в контексте показанную услугу классом «анализы», и следующее «сколько стоит?» искало «анализы».
    Класс нужен ответу — инструмент адресов выбирает по нему пункты сдачи анализов, поэтому ход идёт как
    раньше, — а после ответа (флаг гарда) в контексте снова показанное: одна услуга — она (`service_name`),
    несколько — услуга контекста до хода, если она среди показанных (класс не подставляется, догадку
    заземления, которой пациент не видел, не возвращаем — тогда услуги нет); `_shown_services` не меняется.
    Трогаем только то, что записал сам walk-in (`service_name` решения).

    Обратный случай — своя услуга реплики («а где сделать экг?», «а на ттг нужна запись?»): ход записал в
    контекст услугу, которой среди показанных нет, — показанное прошлым ответом больше не предмет разговора,
    `_shown_services` снимается, и следующий короткий вопрос — о своей услуге (было: устаревшее
    показанное перебивало её — «сколько стоит?» после «а где сделать экг?» отвечало ценой ферритина). Класс
    walk-in (`walkin`: «анализы», «ЭКГ» правила, кнопка меню) — вид услуги, а не своя услуга: показанное не снимает.
    """

    shown = previous()
    if not shown:
        return
    held = str(state.last_entities.get("service_name") or "")
    if WALKIN_SHOWN_FLAG in decision.flags:
        if held != str(decision.entities.get("service_name") or ""):
            return
        keep = shown[0] if len(shown) == 1 else _BEFORE.get()
        if keep and _norm(keep) in {_norm(name) for name in shown}:
            state.last_entities["service_name"] = keep
        else:
            state.last_entities.pop("service_name", None)
        return
    if not held or _norm(held) == _norm(_BEFORE.get()) or _norm(held) in {_norm(name) for name in shown}:
        return  # контекст ход не менял или он — показанное
    # Классы walk-in. Импорт здесь: выбор LLM (`walkin` → `choice`) тянет рантайм LLM — модулю контекста он не нужен.
    from .walkin import is_class

    if walkin and is_class(held):
        return
    state.last_entities.pop(SHOWN_KEY, None)


def _norm(name: str) -> str:
    return " ".join(str(name or "").replace("ё", "е").casefold().split())
