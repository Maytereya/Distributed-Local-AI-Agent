import configparser
import os
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, "config.ini")

# Shared parser object (some modules read c.config directly).
config = configparser.ConfigParser()
config.read(config_path, encoding="utf-8")


def _norm_env(raw: str | None) -> str:
    txt = str(raw or "").strip().upper()
    return txt or "PRODUCTION"


def _as_bool(raw: object, default: bool = False) -> bool:
    if raw is None:
        return default
    txt = str(raw).strip().lower()
    if txt in {"1", "true", "yes", "on"}:
        return True
    if txt in {"0", "false", "no", "off"}:
        return False
    return default


def _initial_environment() -> str:
    if config.has_section("APP") and config.has_option("APP", "environment"):
        return _norm_env(config.get("APP", "environment"))
    if config.has_option("DEFAULT", "environment"):
        return _norm_env(config.get("DEFAULT", "environment"))
    return "PRODUCTION"


environment = _initial_environment()


def _section_candidates(section: str, env: str | None = None) -> list[str]:
    env_name = _norm_env(env or environment)
    base = str(section or "").strip()
    if not base:
        return []

    out = [f"{base}.{env_name}", base]

    # Backward compatibility with old [PRODUCTION]/[LOCAL]/... layout.
    if base in {"OLLAMA", "CHROMA", "MEILI", "ASR_VOSK", "ASR_WHISPER"}:
        out.append(env_name)

    # Deduplicate while preserving order.
    deduped: list[str] = []
    seen: set[str] = set()
    for sec in out:
        if sec not in seen:
            deduped.append(sec)
            seen.add(sec)
    return deduped


def _raw_value(
        section: str,
        key: str,
        default: str | None = None,
        *,
        env: str | None = None,
        legacy_key: str | None = None,
) -> str | None:
    opt = str(key or "").strip()
    if not opt:
        return default

    for sec in _section_candidates(section, env=env):
        if config.has_section(sec) and config.has_option(sec, opt):
            return config.get(sec, opt)

    legacy_opt = str(legacy_key or "").strip()
    if legacy_opt:
        env_name = _norm_env(env or environment)
        if config.has_section(env_name) and config.has_option(env_name, legacy_opt):
            return config.get(env_name, legacy_opt)
        if config.has_option("DEFAULT", legacy_opt):
            return config["DEFAULT"].get(legacy_opt)

    if config.has_option("DEFAULT", opt):
        return config["DEFAULT"].get(opt)

    return default


def _get_str(
        section: str,
        key: str,
        default: str = "",
        *,
        env: str | None = None,
        legacy_key: str | None = None,
) -> str:
    raw = _raw_value(section, key, default=default, env=env, legacy_key=legacy_key)
    return str(raw if raw is not None else default).strip()


def _get_int(
        section: str,
        key: str,
        default: int,
        *,
        env: str | None = None,
        legacy_key: str | None = None,
        min_value: int | None = None,
        max_value: int | None = None,
) -> int:
    raw = _raw_value(section, key, default=str(default), env=env, legacy_key=legacy_key)
    try:
        value = int(str(raw).strip())
    except Exception:
        value = int(default)

    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _get_bool(
        section: str,
        key: str,
        default: bool,
        *,
        env: str | None = None,
        legacy_key: str | None = None,
) -> bool:
    raw = _raw_value(section, key, default="true" if default else "false", env=env, legacy_key=legacy_key)
    return _as_bool(raw, default=default)


def _get_enum(
        section: str,
        key: str,
        default: str,
        allowed: set[str],
        *,
        env: str | None = None,
        legacy_key: str | None = None,
) -> str:
    raw = _get_str(section, key, default=default, env=env, legacy_key=legacy_key).lower()
    return raw if raw in allowed else default


def _get_int_setting(name: str, default: int, *, min_value: int = 1, max_value: int = 100) -> int:
    # Legacy helper kept for compatibility with earlier code style.
    return _get_int(
        "MESSENGER_ROUTER",
        name,
        default,
        legacy_key=name,
        min_value=min_value,
        max_value=max_value,
    )


@dataclass(frozen=True)
class AppConfig:
    environment: str
    app_data_dir: str
    agent_api_key: str


@dataclass(frozen=True)
class MessengerRouterConfig:
    use_doctor_prices_for_procedures: bool
    mr_doctors_top_n: int
    mr_schedule_fresh_ttl_seconds: int
    mr_schedule_stale_ttl_seconds: int
    mr_schedule_negative_ttl_seconds: int
    mr_schedule_cache_max_keys: int
    mr_schedule_cache_log_events: bool
    debug_raw_output: bool
    llm_procedure_normalization: bool
    llm_service_normalization: bool
    llm_price_service_select: bool
    llm_appointment_change_validation: bool
    llm_existing_appointment_validation: bool
    llm_empty_evidence_reply: bool
    llm_patient_name_validation: bool
    llm_service_slot_validation: bool
    llm_result_timing_validation: bool
    prepare_relevance_llm_enabled: bool
    llm_symptom_specialists: bool
    turn_telemetry: bool
    turn_understanding_price: str
    llm_decision_turn_calls: int
    llm_decision_turn_seconds: int
    llm_decision_breaker_failures: int
    llm_decision_breaker_open_s: int
    classifier_timeout_s: int
    llm_turn_deadline_s: int
    llm_fail_offer: str
    classifier_compact: str
    choose_cache_ttl_s: int
    parse_fail_variants: int
    topic_override_llm: str
    topic_shadow_llm: str
    p1_facet_none_offer: str
    facet_by_llm: str
    turn_spans_llm: str
    prepare_by_parse: str
    price_by_parse: str
    strong_family_letter: str
    position_qualifiers: str
    qualifier_pool: str
    qualifier_marks: str
    basket_fail_unpriced: str
    topic_by_llm: str
    booking_target_slot_llm: str
    button_answer_topic: str
    button_topic_hint: str
    button_reply_judge: str
    button_side_answer: str
    button_price_scope: str
    address_branch_flags: str
    address_followup_llm: str
    address_named_street: str
    turn_understanding_one: str
    turn_understanding_full: str
    service_answer_from_data: str
    test_assist_from_data: str
    test_assist_price: str
    walkin_by_llm: str
    button_topic_firm: str
    promo_by_llm: str
    button_leaf_frame: str
    button_frame_check: str
    pending_in_frame: str
    price_not_found_by_data: str
    frame_policy: str
    expects_by_understanding: str
    insurance_topic: str
    price_select_named: str
    bare_name_overview: str
    booking_options_price: str
    service_answer_no_similar: str
    address_service_llm: str
    doctors_by_service_data: str
    slot_suppress_llm: str
    t7_drop_k1: str
    t7_drop_k3: str
    price_judge_conditions: str
    basket_calc: str
    pending_buttons: str
    # D2.4 — предел переспросов (поле ожидания `_expects`, `messengers_router/expects.py`)
    reask_limit: str
    parse_fragment_windows: str
    parse_complete_windows: str
    strong_phrase_guard: str
    parse_agree_telemetry: str
    names_doctor_catalog: str
    names_doctor_noise: str
    names_doctor_followup: str
    names_result_surname: str
    names_patient_fio: str
    names_patient_rule: str
    names_patient_fill: str
    debug_final_input: bool
    url: str


@dataclass(frozen=True)
class RouterConfig:
    router_v2_enable: bool
    router_v2_shadow: bool
    default_city: str
    samara_only_operator_text: str


@dataclass(frozen=True)
class NLUConfig:
    nlu_shadow: bool


@dataclass(frozen=True)
class SberCloudConfig:
    giga_authorization_key: str


@dataclass(frozen=True)
class NaukaConfig:
    base_url: str
    base_url_no_site: str
    login: str
    password: str


@dataclass(frozen=True)
class OllamaConfig:
    url: str
    model_big: str
    model_small: str
    think: bool
    llm_max_concurrency: int


@dataclass(frozen=True)
class ChromaConfig:
    host: str
    port: int


@dataclass(frozen=True)
class WhisperConfig:
    ws_url: str
    http_api: str
    api_key: str


@dataclass(frozen=True)
class VoskConfig:
    ws_url: str


@dataclass(frozen=True)
class MeiliConfig:
    url: str
    master_key: str
    auth_name: str
    auth_pass: str


@dataclass(frozen=True)
class Settings:
    app: AppConfig
    messenger_router: MessengerRouterConfig
    router: RouterConfig
    nlu: NLUConfig
    sber_cloud: SberCloudConfig
    nauka: NaukaConfig
    ollama: OllamaConfig
    chroma: ChromaConfig
    whisper: WhisperConfig
    vosk: VoskConfig
    meili: MeiliConfig


settings = Settings(
    app=AppConfig(
        environment=environment,
        app_data_dir=_get_str("APP", "app_data_dir", default="/app_data", legacy_key="APP_DATA_DIR"),
        agent_api_key=_get_str("APP", "agent_api_key", default="", legacy_key="AGENT_API_KEY"),
    ),
    messenger_router=MessengerRouterConfig(
        use_doctor_prices_for_procedures=_get_bool(
            "MESSENGER_ROUTER",
            "use_doctor_prices_for_procedures",
            default=False,
            legacy_key="USE_DOCTOR_PRICES_FOR_PROCEDURES",
        ),
        mr_doctors_top_n=_get_int(
            "MESSENGER_ROUTER",
            "mr_doctors_top_n",
            default=4,
            legacy_key="MR_DOCTORS_TOP_N",
            min_value=1,
            max_value=20,
        ),
        mr_schedule_fresh_ttl_seconds=_get_int(
            "MESSENGER_ROUTER",
            "mr_schedule_fresh_ttl_seconds",
            default=30,
            legacy_key="MR_SCHEDULE_FRESH_TTL_SECONDS",
            min_value=1,
            max_value=300,
        ),
        mr_schedule_stale_ttl_seconds=_get_int(
            "MESSENGER_ROUTER",
            "mr_schedule_stale_ttl_seconds",
            default=600,
            legacy_key="MR_SCHEDULE_STALE_TTL_SECONDS",
            min_value=1,
            max_value=3600,
        ),
        mr_schedule_negative_ttl_seconds=_get_int(
            "MESSENGER_ROUTER",
            "mr_schedule_negative_ttl_seconds",
            default=15,
            legacy_key="MR_SCHEDULE_NEGATIVE_TTL_SECONDS",
            min_value=1,
            max_value=120,
        ),
        mr_schedule_cache_max_keys=_get_int(
            "MESSENGER_ROUTER",
            "mr_schedule_cache_max_keys",
            default=1000,
            legacy_key="MR_SCHEDULE_CACHE_MAX_KEYS",
            min_value=50,
            max_value=10000,
        ),
        mr_schedule_cache_log_events=_get_bool(
            "MESSENGER_ROUTER",
            "mr_schedule_cache_log_events",
            default=False,
            legacy_key="MR_SCHEDULE_CACHE_LOG_EVENTS",
        ),
        debug_raw_output=_get_bool(
            "MESSENGER_ROUTER",
            "debug_raw_output",
            default=False,
            legacy_key="DEBUG_RAW_OUTPUT",
        ),
        llm_procedure_normalization=_get_bool(
            "MESSENGER_ROUTER",
            "llm_procedure_normalization",
            default=False,
            legacy_key="LLM_PROCEDURE_NORMALIZATION",
        ),
        # П5 (вариант B): LLM-нормализация услуги на промахах лексики. Fail-open,
        # верифицируется каталогом → по умолчанию ВКЛ; kill-switch в config.ini:
        # [MESSENGER_ROUTER] llm_service_normalization = false
        llm_service_normalization=_get_bool(
            "MESSENGER_ROUTER",
            "llm_service_normalization",
            default=True,
            legacy_key="LLM_SERVICE_NORMALIZATION",
        ),
        # Выбор строки прайса LLM (01.10, класс «цена не той услуги»). Fail-open —
        # по умолчанию ВКЛ; kill-switch: [MESSENGER_ROUTER] llm_price_service_select = false
        llm_price_service_select=_get_bool(
            "MESSENGER_ROUTER",
            "llm_price_service_select",
            default=True,
            legacy_key="LLM_PRICE_SERVICE_SELECT",
        ),
        # «Отменить / перенести» без слова «запись» — спросить LLM, о записи ли речь
        # (05.10). Сбой LLM — как раньше, просьба о записи; по умолчанию ВКЛ;
        # kill-switch: [MESSENGER_ROUTER] llm_appointment_change_validation = false
        llm_appointment_change_validation=_get_bool(
            "MESSENGER_ROUTER",
            "llm_appointment_change_validation",
            default=True,
        ),
        # «Мои записи», «мне подтверждена запись?» — спросить LLM, о своей ли оформленной
        # записи речь (08.10). Сбой LLM — как раньше; по умолчанию ВКЛ; kill-switch:
        # [MESSENGER_ROUTER] llm_existing_appointment_validation = false
        llm_existing_appointment_validation=_get_bool(
            "MESSENGER_ROUTER",
            "llm_existing_appointment_validation",
            default=True,
        ),
        # Ход без данных: LLM выбирает вид реплики, текст печатает код (L-05, 08.10).
        # Выключить — вернуть свободный текст LLM; по умолчанию ВКЛ; kill-switch:
        # [MESSENGER_ROUTER] llm_empty_evidence_reply = false
        llm_empty_evidence_reply=_get_bool(
            "MESSENGER_ROUTER",
            "llm_empty_evidence_reply",
            default=True,
        ),
        # Выключатели LLM-проверок, которые журнал обещал, а config.py не объявлял — поэтому
        # они были всегда включены (ревью DLA, LLM-1; объявлены 08.10). По умолчанию ВКЛ.
        llm_patient_name_validation=_get_bool("MESSENGER_ROUTER", "llm_patient_name_validation", default=True),
        llm_service_slot_validation=_get_bool("MESSENGER_ROUTER", "llm_service_slot_validation", default=True),
        llm_result_timing_validation=_get_bool("MESSENGER_ROUTER", "llm_result_timing_validation", default=True),
        prepare_relevance_llm_enabled=_get_bool("MESSENGER_ROUTER", "prepare_relevance_llm_enabled", default=True),
        # Медвопрос: к каким врачам клиники обратиться с жалобой — выбирает LLM (05.10).
        # Сбой LLM — прежний шаблон медвопроса; по умолчанию ВКЛ; kill-switch:
        # [MESSENGER_ROUTER] llm_symptom_specialists = false
        llm_symptom_specialists=_get_bool(
            "MESSENGER_ROUTER",
            "llm_symptom_specialists",
            default=True,
        ),
        # Телеметрия хода — одна строка на ход, только числа и ключи (ход (а) рефакторинга,
        # решение владельца 08.10). По умолчанию ВКЛ; kill-switch:
        # [MESSENGER_ROUTER] turn_telemetry = false
        turn_telemetry=_get_bool("MESSENGER_ROUTER", "turn_telemetry", default=True),
        # Понимание хода на метке «цена» (ход (б), шаг 1, L-14): off — не строится; shadow — в
        # debug и телеметрию, поведение прежнее; on — список не сводится к одной услуге, вопрос
        # о сроке не становится ценой. По умолчанию on — как на проде с 14.10 (STEP-2026-10-15-DEFAULTS-B):
        # [MESSENGER_ROUTER] turn_understanding_price
        turn_understanding_price=_get_enum(
            "MESSENGER_ROUTER",
            "turn_understanding_price",
            default="on",
            allowed={"off", "shadow", "on"},
        ),
        # Бюджет LLM-решений на ход (ход (в), L-11): не больше N выборов и S секунд выбора на ход;
        # сверх бюджета — «LLM недоступна» без вызова, что ответить, решает вызывающий. Только для
        # структурных решений (модуль choose()), не для текста и классификатора. 0 — без ограничения.
        # 3 решения и 20 с с 19.10 (решение владельца: «рассчитываем именно на ответы LLM»): при 2 решениях на проде
        # 14–19.10 лимит отрезал 7 из 73 выборов строки цены — второй круг поиска после «ни одна» (разбор реплики,
        # выбор строки, второй выбор). Прежде — 2 и 10 (решение 13.10, Q4).
        llm_decision_turn_calls=_get_int("MESSENGER_ROUTER", "llm_decision_turn_calls", 3, min_value=0, max_value=20),
        # 25 с — прод с 26.10 (деплой 18, runbook §3: секунды LLM-решений хода не возвращаются; STEP-2026-10-15-DEFAULTS-B).
        llm_decision_turn_seconds=_get_int("MESSENGER_ROUTER", "llm_decision_turn_seconds", 25, min_value=0, max_value=120),
        # Предохранитель LLM-решений (L-11): N сбоев генерации подряд → S секунд без вызовов; таймаут
        # очереди не считается (иначе качество бота зависит от нагрузки соседей). 0 сбоев — выключен.
        llm_decision_breaker_failures=_get_int("MESSENGER_ROUTER", "llm_decision_breaker_failures", 3, min_value=0, max_value=100),
        llm_decision_breaker_open_s=_get_int("MESSENGER_ROUTER", "llm_decision_breaker_open_s", 30, min_value=1, max_value=600),
        # Предохранитель времени LLM-классификатора темы (BUG-2026-10-27-CLASSIFIER-NO-TIMEOUT): вызов целиком — очередь и
        # генерация — не дольше S секунд; не успел — тот же путь, что при сбое LLM (правило или «LLM недоступна»).
        # 0 (по умолчанию) — как было: генерация до 45 с плюс очередь до 30 с.
        classifier_timeout_s=_get_int("MESSENGER_ROUTER", "classifier_timeout_s", 0, min_value=0, max_value=45),
        # Дедлайн хода для LLM (решение владельца 27.10: «лучше подождать LLM, чем правила», BUG-2026-10-27-LLM-QUEUE-GIVES-UP):
        # вызов LLM ждёт слот до «остаток дедлайна − p95 задачи» вместо своих 1,5–3 с очереди, генерация — не дольше остатка;
        # слоты — начатым ходам первыми; не успел — исход `deadline`, ответ хода — честный оффер оператора, не правило.
        # Секунды от начала хода (шлюз режет на 60 с). 0 (по умолчанию) — как было. [MESSENGER_ROUTER] llm_turn_deadline_s
        llm_turn_deadline_s=_get_int("MESSENGER_ROUTER", "llm_turn_deadline_s", 0, min_value=0, max_value=58),
        # Узлы, где сбой LLM даёт честный оффер оператора, а не ответ правил «как было» (BUG-2026-10-27-LLM-QUEUE-GIVES-UP, шаг 4;
        # проверка ответов без LLM 27.10, таблица узлов), через запятую: `address_followup` — реплика после адресного листа (весь
        # список / «без записи» во всех пунктах); `topic` — ветка `topic_by_llm` (тема правила); `facet` — свойство вопроса П1
        # (переклейка в цену); `role_pick` — выбор роли врача («есть запись к X», список врачей по слову) — отдельно, на грани
        # правила ≤ 3 переводов на 1 чужую цену, решение владельца по карточке. Пусто (по умолчанию) — как было.
        # [MESSENGER_ROUTER] llm_fail_offer
        llm_fail_offer=_get_str("MESSENGER_ROUTER", "llm_fail_offer", default=""),
        # Короче ответ LLM-классификатора темы (BUG-2026-10-27-LLM-QUEUE-GIVES-UP, шаг 5): строка промпта
        # `classifier_compact_note` (две локации) — в `entities` только найденные поля. Половина ответов на проде печатала все
        # ~30 полей шаблона с null (≈ 227 токенов, ≈ 11,6 с против ≈ 70 токенов и 3,4 с). Следствие: правило дарит сущность,
        # которой LLM не назвала (`nlu_pipeline._merge`: ключ с null дарение блокировал) — как уже у коротких ответов.
        # off — как было. [MESSENGER_ROUTER] classifier_compact
        classifier_compact=_get_enum("MESSENGER_ROUTER", "classifier_compact", default="off", allowed={"off", "on"}),
        # Кэш внятных ответов модуля выбора (ход (в), L-12): секунд жизни; сбой и мусор не кэшируются.
        # Коротко — ошибка модели не становится ответом всем на часы; 0 — без кэша (стенд меряет без него).
        choose_cache_ttl_s=_get_int("MESSENGER_ROUTER", "choose_cache_ttl_s", 1800, min_value=0, max_value=86400),
        # Выбор строки прайса на модуле choose() (ход (в), клиент 1) — единственный путь с 17.10: ключ [MESSENGER_ROUTER]
        # choose_price (off — прежний вызов, legacy — прежний промпт через модуль) снят и больше не читается (в config.ini
        # прода безвреден; STEP-2026-10-17-ONE-PATH-CHOOSE-PRICE).
        # Симптом → врач — всегда выбор на модуле choose() (ход (в), клиент 2): ключ [MESSENGER_ROUTER] choose_symptom снят 28.10
        # (on на проде с 14.10) и больше не читается (в config.ini прода безвреден).
        # Срок готовности при слабом совпадении — всегда выбор LLM (ход (в)): ключ [MESSENGER_ROUTER] choose_readiness снят
        # 28.10 (on на проде с 14.10) и больше не читается (в config.ini прода безвреден).
        # Разбор ценовой реплики на фразы-услуги (L-15b, контракт C) — единственный режим с T7 шага 3б (16.10): ключ
        # [MESSENGER_ROUTER] turn_parse снят и больше не читается (в config.ini прода безвреден).
        # Политика сбоя под turn_parse = on (L-15b, шаг 6, T5): сколько вариантов кода с ценами показывать у
        # слабой позиции, когда LLM не выбрала строки (2–5; поправка владельца 13.10). По умолчанию 3 — как на
        # проде (STEP-2026-10-15-DEFAULTS-B). [MESSENGER_ROUTER] parse_fail_variants
        parse_fail_variants=_get_int("MESSENGER_ROUTER", "parse_fail_variants", default=3, min_value=2, max_value=5),
        # Фраза журнала синонимов — упоминание хода (L-14, шаг 2) — единственный путь с 17.10: ключ [MESSENGER_ROUTER]
        # journal_phrase снят и больше не читается (в config.ini прода безвреден; STEP-2026-10-17-ONE-PATH-JOURNAL-PHRASE).
        # Контекст сессии = показанное (L-14, шаг 3) — единственный путь с 17.10: ключ [MESSENGER_ROUTER] context_shown
        # снят и больше не читается (в config.ini прода безвреден; STEP-2026-10-17-ONE-PATH-CONTEXT-SHOWN).
        # Ответ на висящий вопрос бота — одна функция `pending_answer` — единственный путь с 24.10: ключи [MESSENGER_ROUTER]
        # pending_reply и pending_reply_types сняты и больше не читаются (в config.ini прода безвредны;
        # STEP-2026-10-22-SWITCH-REMOVAL).
        # Вопросы-слоты кнопок и остатки класса 5 — через одну функцию ответа на висящий вопрос (`pending_answer`; решение
        # владельца 19.10, п. 5): вопрос-слот («К какой процедуре нужна подготовка?», «Для какой цели подобрать анализы?»,
        # реквизиты результата) ставит `_expects` вида «свободный ответ», ответ решает функция; уточняющий вопрос на «Какой
        # филиал вам удобен?» получает ответ и тот же вопрос снова; «обе» на составную цену — цены обеих; второе «нет» на
        # «Похоже, вы имели в виду …?». off — как было. По умолчанию on — как на проде с 21.10 (деплой 12;
        # STEP-2026-10-15-DEFAULTS-B). [MESSENGER_ROUTER] pending_buttons
        pending_buttons=_get_enum("MESSENGER_ROUTER", "pending_buttons", default="on", allowed={"off", "on"}),
        # Предел переспросов (D2.4; пункт 5 принципов, инвариант 3bacf58): off — как было (обработчик висящего
        # вопроса переспрашивает без счёта); on — один вопрос звучит не больше двух раз подряд, третьего нет:
        # бот называет нужный ответ и предлагает оператора. [MESSENGER_ROUTER] reask_limit
        reask_limit=_get_enum("MESSENGER_ROUTER", "reask_limit", default="off", allowed={"off", "on"}),
        # Ключ [MESSENGER_ROUTER] llm_primary_rules_fallback снят 29.10 вместе с движком llm_primary (К2 карты мёртвого кода)
        # и больше не читается (в config.ini безвреден).
        # Гард П2 («уточнения нет в прайсе») у слабой фразы разбора судит выбор строки (класс A, решение владельца 15.10) —
        # единственный путь с 17.10: ключ [MESSENGER_ROUTER] parse_price_guard (rules — гард по словарю каталога) снят и
        # больше не читается (в config.ini прода безвреден; STEP-2026-10-17-ONE-PATH-PARSE-PRICE-GUARD).
        # Сильные окна внутри слабого фрагмента по разделителям — дополнительные кандидаты выбора LLM разбора
        # (15.10, смоук прода: «сколько стоят ферритин, …» → фрагмент «стоят ферритин», ферритин терялся): off — как
        # было; on — окно 1–3 слова, сильное по данным МИС («ферритин»), предлагается LLM; разрезом не служит.
        # [MESSENGER_ROUTER] parse_fragment_windows
        parse_fragment_windows=_get_enum("MESSENGER_ROUTER", "parse_fragment_windows", default="off", allowed={"off", "on"}),
        # Окно-кандидат разбора, оборванное перед словом реплики, с которым оно даёт сильное совпадение по данным МИС
        # (15.10, g0724: окно сегментации «взятие крови из» без «вены» — LLM разбора не видела названия целиком): off —
        # как было; on — окно, дополненное до сильного следующими словами (не дальше 2 и не через знак препинания),
        # предлагается LLM вариантом; разрезом не служит.
        # [MESSENGER_ROUTER] parse_complete_windows
        parse_complete_windows=_get_enum("MESSENGER_ROUTER", "parse_complete_windows", default="off", allowed={"off", "on"}),
        # Гард П2 у названия услуги из сущности NLU / контекста судит LLM (BUG-2026-10-15-P2-ENTITY-BYPASS) — единственный
        # путь с 17.10: ключ [MESSENGER_ROUTER] p2_llm_judge (off — цена по сущности без суда) снят и больше не читается
        # (в config.ini прода безвреден; STEP-2026-10-17-ONE-PATH-P2-LLM-JUDGE).
        # Гард П2 у СИЛЬНОЙ фразы разбора (16.10, BUG-2026-10-16-P2-STRONG-PHRASE-BYPASS: «приём терапевта по ОМС» → LLM
        # разбора выбрала «приём терапевта», её строки шли в ответ мимо гарда — платная цена): llm — фраза отдаёт свои
        # строки напрямую, только если вне неё в реплике нет слова, которого нет в прайсе; иначе «та ли услуга с этим
        # условием» решает выбор строки (`judge_qualifier`, правило 3 промпта отсекает «по ОМС»): строки — цена, «не та»
        # — «не нашёл», сбой LLM — ход как под turn_parse = off; off — как было с 15.10 (откат). Решение владельца 16.10 —
        # фикс следующим деплоем, поэтому по умолчанию llm. [MESSENGER_ROUTER] strong_phrase_guard
        strong_phrase_guard=_get_enum("MESSENGER_ROUTER", "strong_phrase_guard", default="llm", allowed={"off", "llm"}),
        # Телеметрия согласия разрезов `tu_parse_agree` под turn_parse = on (T7, шаг 4): on — как было (правила понимания
        # хода считаются на каждом ценовом ходе параллельно выбору фраз LLM, O3b, — согласию нужен их разрез); off —
        # согласия под on нет, правила считаются лениво: только когда разбор не годится для действия (`acting_for` —
        # None: empty, too_many, сбой LLM без разделителей) или решён одной слабой фразой (`single`). Ответы те же,
        # экономия — CPU хода. По умолчанию off с 16.10 (решение владельца,
        # T7 шаг 4). [MESSENGER_ROUTER] parse_agree_telemetry
        parse_agree_telemetry=_get_enum("MESSENGER_ROUTER", "parse_agree_telemetry", default="off", allowed={"off", "on"}),
        # Врач реплики для сверки с каталогом — по справочнику врачей МИС, без выбора слов списком (STEP-2026-10-24-NAMES-NO-STOPWORDS,
        # решения владельца 22.10): фамилия — слово реплики, точно совпавшее с фамилией врача в любой падежной форме; опечатка — только
        # в имени, которое назвала LLM, и исход — вопрос «Похоже, вы имели в виду…?». off — как было (_CATALOG_DOCTOR_STOPWORDS).
        # [MESSENGER_ROUTER] names_doctor_catalog
        # По умолчанию on — как на проде с 26.10 (деплой 18; STEP-2026-10-15-DEFAULTS-B).
        names_doctor_catalog=_get_enum("MESSENGER_ROUTER", "names_doctor_catalog", default="on", allowed={"off", "on"}),
        # doctor_name из ответа LLM из одного служебного слова («хочу», «запись») — не врач по справочнику МИС (resolve_doctor_name), а не по
        # списку (STEP-2026-10-24-NAMES-NO-STOPWORDS; свип 24.10: справочник ни одно из 14 слов не находит). off — как было (_DOCTOR_NOISE_TOKENS).
        # [MESSENGER_ROUTER] names_doctor_noise
        # По умолчанию on — как на проде с 26.10 (деплой 18; STEP-2026-10-15-DEFAULTS-B).
        names_doctor_noise=_get_enum("MESSENGER_ROUTER", "names_doctor_noise", default="on", allowed={"off", "on"}),
        # Короткий ответ в контексте врача («<фамилия>, да») — фамилия только по справочнику МИС (точное совпадение у правила), без списка
        # «да / хорошо / ладно» (STEP-2026-10-24-NAMES-NO-STOPWORDS; свип 24.10: список ни разу не решал исход). off — как было (_DOCTOR_FOLLOWUP_FILLERS).
        # [MESSENGER_ROUTER] names_doctor_followup
        # По умолчанию on — как на проде с 26.10 (деплой 18; STEP-2026-10-15-DEFAULTS-B).
        names_doctor_followup=_get_enum("MESSENGER_ROUTER", "names_doctor_followup", default="on", allowed={"off", "on"}),
        # Результаты анализов: одно слово на вопрос о реквизитах — фамилия, только если её назвала LLM-классификатор этого хода (сущность
        # surname / patient_name — те же слова реплики); без LLM — не фамилия, бот переспросит (решение владельца 22.10 (а);
        # STEP-2026-10-24-NAMES-NO-STOPWORDS). Порядок «фамилия, год, код, номер» — как было. off — как было (_QF_RESULT_SURNAME_STOPWORDS).
        # [MESSENGER_ROUTER] names_result_surname
        # По умолчанию on — как на проде с 26.10 (деплой 18; STEP-2026-10-15-DEFAULTS-B).
        names_result_surname=_get_enum("MESSENGER_ROUTER", "names_result_surname", default="on", allowed={"off", "on"}),
        # Реплика — ФИО пациента (12 мест: запись, вид недоступной цели, неподтверждённый врач → ФИО): форма 2–3 слова принимает и
        # «Общий анализ крови» — решает LLM-классификатор этого хода (patient_name — те же слова реплики); без LLM — не ФИО (решение
        # владельца 22.10 (а); STEP-2026-10-24-NAMES-NO-STOPWORDS). off — как было (_PATIENT_FIO_STOPWORDS).
        # [MESSENGER_ROUTER] names_patient_fio
        # По умолчанию on — как на проде с 26.10 (деплой 18; STEP-2026-10-15-DEFAULTS-B).
        names_patient_fio=_get_enum("MESSENGER_ROUTER", "names_patient_fio", default="on", allowed={"off", "on"}),
        # Правило шага ФИО записи (rule_appointment_patient_name) не решает по форме и списку: правило зовут до LLM, под on оно молчит, и
        # реплику (ФИО или нет, какая тема) решает LLM-классификатор хода (STEP-2026-10-24-NAMES-NO-STOPWORDS). off — как было
        # (_PATIENT_NAME_STOPWORDS). [MESSENGER_ROUTER] names_patient_rule
        # По умолчанию on — как на проде с 26.10 (деплой 18; STEP-2026-10-15-DEFAULTS-B).
        names_patient_rule=_get_enum("MESSENGER_ROUTER", "names_patient_rule", default="on", allowed={"off", "on"}),
        # Быстрое заполнение ФИО пациента (после «ФИО / меня зовут», голая форма, тройка слов) — только если кандидат — те же слова, что
        # patient_name LLM-классификатора этого хода; без LLM ФИО не заполняется (решение владельца 22.10 (а); STEP-2026-10-24-NAMES-NO-STOPWORDS;
        # свип 24.10: прежний путь клал «Общий Анализ Крови» в ФИО). off — как было (_QF_PATIENT_NAME_STOPWORDS).
        # [MESSENGER_ROUTER] names_patient_fill
        # По умолчанию on — как на проде с 26.10 (деплой 18; STEP-2026-10-15-DEFAULTS-B).
        names_patient_fill=_get_enum("MESSENGER_ROUTER", "names_patient_fill", default="on", allowed={"off", "on"}),
        # Вариант B L-13 — словоформы различающего слова всегда: ключ [MESSENGER_ROUTER] variant_b_wordforms снят 28.10 (on на
        # проде с 18.10) и больше не читается (в config.ini прода безвреден).
        # Ход (д), видимая деградация МИС — единственный путь: ключ [MESSENGER_ROUTER] mis_honest_degradation снят 28.10 (on на
        # проде с 18.10) и больше не читается (в config.ini прода безвреден).
        # Тема реестра перебивает метку OTHER только после подтверждения LLM по смыслу (решение владельца 18.10:
        # «чинить через LLM, уходим от совпадения букв к смыслу»; «Ново-Садовая» ловилась словом «новость»).
        # LLM недоступна — тема метку не перебивает. off — как было. [MESSENGER_ROUTER] topic_override_llm
        topic_override_llm=_get_enum("MESSENGER_ROUTER", "topic_override_llm", default="off", allowed={"off", "on"}),
        # Тень LLM рядом с правилом темы (ход (б), решения владельца 20.10 (б) 1, 3): где тему решило правило `legacy_v2`,
        # LLM-классификатор решает её параллельно ходу, вне его бюджета; строка `topic_shadow` после ответа — матрица
        # «правило × LLM». Ответ не меняется. off — как было. [MESSENGER_ROUTER] topic_shadow_llm
        topic_shadow_llm=_get_enum("MESSENGER_ROUTER", "topic_shadow_llm", default="off", allowed={"off", "on"}),
        # П1 на ходах `topic_by_llm` (BUG-2026-10-22-P1-FACET-NONE-AS-PRICE): LLM решила, что пациент спрашивает об услуге не цену,
        # не «есть ли» и не «где / когда» (подготовка, возраст, результат, неясно) — не цена, а честный оффер оператора (как OTHER
        # от LLM). Сбой LLM свойства — как было (П1 → цена). off — как было. [MESSENGER_ROUTER] p1_facet_none_offer
        # По умолчанию on — как на проде с 22.10 (деплой 15; STEP-2026-10-15-DEFAULTS-B).
        p1_facet_none_offer=_get_enum("MESSENGER_ROUTER", "p1_facet_none_offer", default="on", allowed={"off", "on"}),
        # Тип вопроса хода решает LLM (деплой 22, шаг 2, STEP-2026-10-30-FACET-BY-LLM): одно решение на ход (`turn_understanding.
        # turn_facet`, задача `turn_facet`) — цена / есть ли / где-когда / нужна ли запись / срок / подготовка / просто назвал / ни
        # одно; его читают срок (`_maybe_readiness_reply`), планировщик, П1, обзор названия и переклейки после цены и подготовки
        # вместо своих детекторов. Сбой LLM — как было (детекторы). off — как было. [MESSENGER_ROUTER] facet_by_llm
        facet_by_llm=_get_enum("MESSENGER_ROUTER", "facet_by_llm", default="off", allowed={"off", "on"}),
        # Разбор реплики на фразы-услуги без списков слов: LLM размечает номера слов (DESIGN-2026-10-23-NO-STOPWORDS, этап 1;
        # решения владельца 22.10 — 28 списков стоп-слов удалить, без LLM услугу не угадывать). off — прежний генератор окон.
        # [MESSENGER_ROUTER] turn_spans_llm
        # По умолчанию on — как на проде с 24.10 (деплой 17; STEP-2026-10-15-DEFAULTS-B).
        turn_spans_llm=_get_enum("MESSENGER_ROUTER", "turn_spans_llm", default="on", allowed={"off", "on"}),
        # Подготовка читает фразы разбора хода (DESIGN-2026-10-23-NO-STOPWORDS, этап 2; STEP-2026-10-24-PREPARE-BY-PARSE): предмет
        # памятки — услуги, которые разбор нашёл в реплике, а не реплика без слов пяти списков подготовки; ни одной услуги — без
        # поиска по словам; сбой разбора — «уточните». Разбор строится и на ходах подготовки. off — как было.
        # [MESSENGER_ROUTER] prepare_by_parse
        # По умолчанию on — как на проде с 24.10 (деплой 17; STEP-2026-10-15-DEFAULTS-B).
        prepare_by_parse=_get_enum("MESSENGER_ROUTER", "prepare_by_parse", default="on", allowed={"off", "on"}),
        # Цена читает фразу разбора хода (DESIGN-2026-10-25-PRICE-STAGE2, этап 2 «без стоп-слов»): сильное совпадение — фраза
        # как есть, без вырезателя; фраза поиска, подписи и журнала синонимов — из разбора, а не срез реплики по спискам;
        # «названа ли услуга» — решение понимания хода; без LLM — только дословное название или код, иначе «уточните» /
        # оффер, без списка правил с ценами. off — как было. [MESSENGER_ROUTER] price_by_parse
        # По умолчанию on — как на проде с 26.10 (деплой 18; STEP-2026-10-15-DEFAULTS-B).
        price_by_parse=_get_enum("MESSENGER_ROUTER", "price_by_parse", default="on", allowed={"off", "on"}),
        # Сильное совпадение не теряет букву семейства (BUG-2026-10-27-STRONG-DROPS-LETTER): «гепатит с» не сильна строке
        # «Гепатит В - HBsAg». Синоним МИС сильный, только если буквы семейства в нём те же, что у фразы пациента; строка,
        # названная другой буквой того же семейства, не сильная — её выбирает LLM. Семейства — из названий прайса.
        # off — как было. [MESSENGER_ROUTER] strong_family_letter
        strong_family_letter=_get_enum("MESSENGER_ROUTER", "strong_family_letter", default="off", allowed={"off", "on"}),
        # Уточнения позиции (DESIGN-2026-10-28-POSITION-QUALIFIERS, этап 2): разметка хода `turn_spans` отмечает у каждой услуги её
        # уточнения (оплата, условие, материал, метод, сопутствующая услуга) тем же вызовом LLM; условие оплаты (ОМС, ДМС, полис) у
        # позиции или на весь вопрос — без розничной цены; прочие уточнения вне фразы — суду условия корзины. Нужен
        # `turn_spans_llm = on`. off — как было. [MESSENGER_ROUTER] position_qualifiers
        position_qualifiers=_get_enum("MESSENGER_ROUTER", "position_qualifiers", default="off", allowed={"off", "on"}),
        # Уточнения позиции, этап 3 (DESIGN-2026-10-28-POSITION-QUALIFIERS): пул выбора строки по материалу позиции — строки, которые
        # по данным МИС (`biomatNames`) материал не принимают, не кандидаты; строки с ним — в пул («посев кала» не выбирает общий
        # посев). Нужен `position_qualifiers = on`. off — как было. [MESSENGER_ROUTER] qualifier_pool
        qualifier_pool=_get_enum("MESSENGER_ROUTER", "qualifier_pool", default="off", allowed={"off", "on"}),
        # Уточнения позиции, этап 4: выбор строки цены тем же вызовом отмечает по каждому уточнению пациента (метод, материал,
        # условие, сопутствующая услуга), есть ли оно в выбранных строках; нет — заголовок «Точно в таком виде в прайсе нет. Есть:»
        # пишет код («ВПЧ количественным методом», «антитела … суммарные»). Нужен `position_qualifiers = on`. off — как было.
        # [MESSENGER_ROUTER] qualifier_marks
        qualifier_marks=_get_enum("MESSENGER_ROUTER", "qualifier_marks", default="off", allowed={"off", "on"}),
        # Корзина при сбое выбора LLM (решение владельца 27.10, узел 7 проверки ответов без LLM; BUG-2026-10-27-LLM-QUEUE-GIVES-UP,
        # шаг 3): слабые позиции — варианты кода **названиями без цен** и оффер оператора, «Итого» нет; сильные (по данным) — с
        # ценой, как были. off — как было: два варианта и больше — с ценами и в итоге. [MESSENGER_ROUTER] basket_fail_unpriced
        basket_fail_unpriced=_get_enum("MESSENGER_ROUTER", "basket_fail_unpriced", default="off", allowed={"off", "on"}),
        # Ход (б), этап 4, вариант Б (решения владельца 20.10 (б) 1, 5): ветки правил `legacy_v2` через запятую
        # (`rule_test_assist`), у которых тему решает LLM-классификатор, а правило — кандидат (слияние: LLM решает метку,
        # правило дарит сущности; безопасность — как была; сбой LLM — решает правило). Пусто — как было.
        # [MESSENGER_ROUTER] topic_by_llm
        # По умолчанию — восемь веток, как на проде с 22.10 (деплой 15; STEP-2026-10-15-DEFAULTS-B).
        topic_by_llm=_get_str("MESSENGER_ROUTER", "topic_by_llm", default="rule_test_assist,rule_doctor_info_specialty,rule_schedule,rule_analysis_lookup_ambiguous,rule_website_help,rule_address_work_hours,rule_test_result,rule_prepare"),
        # Вопрос цели записи («К какому врачу или на какую услугу записать?» — кнопка «Записаться на приём» и тот же вопрос
        # текстом) — вопрос-слот одной функции ответа на висящий вопрос (`pending_buttons`): ответ, который LLM признала
        # ответом, идёт в запись, а не в вопрос «Отменить оформление?» правил смены темы
        # (BUG-2026-10-25-BOOK-BUTTON-REPLY-AS-TOPIC-SWITCH). off — как было. [MESSENGER_ROUTER] booking_target_slot_llm
        # По умолчанию on — как на проде с 26.10 (деплой 18; STEP-2026-10-15-DEFAULTS-B).
        booking_target_slot_llm=_get_enum("MESSENGER_ROUTER", "booking_target_slot_llm", default="on", allowed={"off", "on"}),
        # Ответ на вопрос кнопки-темы (`button_menu`, режим `slot`: «Какой анализ интересует?», «К какой процедуре нужна
        # подготовка?»): тему знает кнопка, LLM-классификатор её не переспрашивает (журнал прода 14–24.10: 59 % таких ответов
        # ждали его 3,7 с, до 12 с). Одна функция ответа на висящий вопрос решила «это ответ» — тема кнопки. `slot` — вопросы-
        # слоты (подготовка, подбор анализов, результаты), `on` — и кнопки цены (их ответ судит та же функция; карточка 25.10:
        # +3 чужие цены выбора строки — блок). `frame` — как `on`, и кнопка задаёт рамку хода: тему, которую судья признал
        # ответом, не перебивают и несрочные правила NLU (`rule_appointment`, `doc_request_*`, `unsupported_*`;
        # STEP-2026-10-29-BUTTON-FRAME). Нужен `pending_buttons = on`. off — как было. [MESSENGER_ROUTER] button_answer_topic
        button_answer_topic=_get_enum(
            "MESSENGER_ROUTER", "button_answer_topic", default="off", allowed={"off", "slot", "on", "frame"}
        ),
        # Чужая тема в ответ на вопрос кнопки («Цена анализа» → «а где вы находитесь?»): одна функция ответа решила «другая тема» —
        # LLM выбирает из закрытого списка листьев меню, бот подсказывает нужную кнопку («Это вопрос про … — основная информация в
        # кнопке «…»»), а не угадывает тему общим путём; в сообщении и ответ, и другой вопрос — ответ по теме кнопки и подсказка.
        # Нужен `pending_buttons = on`. off — как было. [MESSENGER_ROUTER] button_topic_hint
        button_topic_hint=_get_enum("MESSENGER_ROUTER", "button_topic_hint", default="off", allowed={"off", "on"}),
        # Судья ответа на вопрос кнопки видит нажатую кнопку и её тему: полный вопрос в теме кнопки («Сколько стоит общий анализ
        # крови?» на «Цене анализа») — ответ, а не «другая тема» (карточка 25.10: на «Цене анализа» 214 из 343 вопросов по теме
        # судья звал другой темой). Нужен `pending_buttons = on`. off — как было. [MESSENGER_ROUTER] button_reply_judge
        button_reply_judge=_get_enum("MESSENGER_ROUTER", "button_reply_judge", default="off", allowed={"off", "on"}),
        # Уточняющий вопрос о вопросе кнопки результатов («Где на бланке код?», «Как узнать сам результат?» после «Ввести
        # данные»): судья решил «вопрос о вопросе бота» — ответ, где код и номер и как посмотреть результат, и тот же вопрос
        # дальше, а не повтор инструкции и оффер оператора (класс К3 карты провалов деплоя 18). Нужен `pending_buttons = on`.
        # off — как было. [MESSENGER_ROUTER] button_side_answer
        button_side_answer=_get_enum("MESSENGER_ROUTER", "button_side_answer", default="off", allowed={"off", "on"}),
        # Область цены кнопки второго уровня (ТЗ 17.08, «сужение поиска по теме»): ответ на вопрос «Цена приёма врача» ищется
        # среди услуг без срока готовности (не анализов, данные прайса МИС), «Цена анализа» / «Сколько стоит анализ» — среди
        # анализов (со сроком готовности); в области нет кандидата на фразу — весь прайс, как без кнопки. `doctor` — только
        # «Цена приёма врача» (на «Цену анализа» пациенты ведут и УЗИ, и процедуры — набор buttons_v1: 84 из 264), `all` — обе.
        # off — как было. [MESSENGER_ROUTER] button_price_scope
        # По умолчанию doctor — как на проде с 26.10 (деплой 18; STEP-2026-10-15-DEFAULTS-B).
        button_price_scope=_get_enum("MESSENGER_ROUTER", "button_price_scope", default="doctor", allowed={"off", "doctor", "all"}),
        # Список филиалов в ответе о сдаче анализов (ЭКГ, УЗИ) без записи и на кнопке «Где и до скольки принимают» (раздел
        # «Сдать анализы») — только филиалы с этим флагом данных филиалов (`analysis` / `ecg` / `usi`): карточка buttons_v1 25.10 —
        # в 26 ответах «где сдать» был филиал без забора анализов. off — как было. [MESSENGER_ROUTER] address_branch_flags
        # По умолчанию on — как на проде с 26.10 (деплой 18; STEP-2026-10-15-DEFAULTS-B).
        address_branch_flags=_get_enum("MESSENGER_ROUTER", "address_branch_flags", default="on", allowed={"off", "on"}),
        # Реплика сразу после «Адреса и часы работы» / «Где и до скольки принимают»: LLM выбирает из списка филиалов, какие названы,
        # и что спрошено (график, день, анализы / УЗИ / ЭКГ, выходные, нет в данных); ответ печатает код из данных филиалов, чего нет —
        # честно и оффер оператора (buttons_v1: 87 провалов «общий список вместо ответа»). off — как было.
        # [MESSENGER_ROUTER] address_followup_llm
        # По умолчанию on — как на проде с 26.10 (деплой 18; STEP-2026-10-15-DEFAULTS-B).
        address_followup_llm=_get_enum("MESSENGER_ROUTER", "address_followup_llm", default="on", allowed={"off", "on"}),
        # Там же: улица, на которой несколько филиалов («Победы» — три), называет их — выбор LLM не вычёркивается правилом «слово
        # есть в адресах больше чем двух филиалов»; не называет только слово, общее для большего числа филиалов, чем карточек в
        # ответе (смоук деплоя 18: «а на Победы до скольки?» → часы всех пунктов забора). off — как было.
        # [MESSENGER_ROUTER] address_named_street
        address_named_street=_get_enum("MESSENGER_ROUTER", "address_named_street", default="off", allowed={"off", "on"}),
        # Ход (б), этап 2 (решения владельца 20.10 (б) 4, 21.10): «список ли» на пути правил решается один раз — пониманием
        # хода (с защитами K1, без K1 у явного перечня своих названий); корзина правил и подбор анализов читают его, а не
        # решают сами. off — как было. [MESSENGER_ROUTER] turn_understanding_one
        # По умолчанию on — как на проде с 22.10 (деплой 15; STEP-2026-10-15-DEFAULTS-B).
        turn_understanding_one=_get_enum("MESSENGER_ROUTER", "turn_understanding_one", default="on", allowed={"off", "on"}),
        # Ход (б), шаг 0 (консилиум 30.10, STEP-2026-10-30-B-STEP0-FULL): полное понимание хода на КАЖДОМ ходе — тема и кто
        # её решил, рамка кнопки, висящий вопрос и ответ на него, сущности — копиями уже принятых решений, в debug
        # (`understanding_full`) и телеметрию; новых вызовов LLM нет, ответы не меняются. off — как было (и сверка темы с меткой
        # решения, а не плана). [MESSENGER_ROUTER] turn_understanding_full
        turn_understanding_full=_get_enum("MESSENGER_ROUTER", "turn_understanding_full", default="shadow", allowed={"off", "shadow"}),
        # Ответ о клинике — только по её данным (решения владельца 21.10, «ни в коем случае не выдумывать»): «есть ли / делаете
        # ли / берёте ли X» решает LLM по прайсу и памяткам пациенту (`choose`, задача `service_answer`) там, где бот иначе
        # написал бы свободный текст или сдался бы без данных; текст печатает код, нет данных — честный оффер оператора.
        # off — как было. [MESSENGER_ROUTER] service_answer_from_data
        service_answer_from_data=_get_enum("MESSENGER_ROUTER", "service_answer_from_data", default="off", allowed={"off", "on"}),
        # Ход подбора анализов (TEST_ASSIST), дошедший до свободного текста LLM, отвечает по данным хода
        # (BUG-2026-10-21-WALKIN-LLM-FREE-TEXT): строки прайса выбирает LLM (`choose`, задача `test_assist_answer`), название
        # и срок печатает код, нет данных и сбой — честный оффер оператора. off — как было. [MESSENGER_ROUTER] test_assist_from_data
        # По умолчанию on — как на проде с 24.10 (деплой 17; STEP-2026-10-15-DEFAULTS-B).
        test_assist_from_data=_get_enum("MESSENGER_ROUTER", "test_assist_from_data", default="on", allowed={"off", "on"}),
        # Цена у строк, выбранных ответом подбора анализов по данным хода (решение владельца 26.10, топ-5 разбора 95 цен, п. 1):
        # сумму печатает код из той же строки прайса (принцип 1), частичный ответ — цены найденных строк и «остального в прайсе
        # нет — уточнит оператор». Давность прайса — как у пути цены (`services/_freshness`). Действует при
        # `test_assist_from_data = on`. off — как было (название и срок без цены). [MESSENGER_ROUTER] test_assist_price
        test_assist_price=_get_enum("MESSENGER_ROUTER", "test_assist_price", default="off", allowed={"off", "on"}),
        # «Анализы и ЭКГ — без записи» решает LLM по правилу клиники (STEP-2026-10-21-WALKIN-BY-LLM, `walkin.py`): ветка правил
        # `rule_nonbookable_walkin` тему не решает (её решает LLM-классификатор, как у `topic_by_llm`), переклейки после NLU в
        # адреса и фраза «… без записи» — по решению LLM. Свой выключатель — связку 21.10 с `service_answer_from_data` владелец
        # развязал 26.10 (топ-5 разбора 95 цен, п. 2). off — как было (детектор по буквам). [MESSENGER_ROUTER] walkin_by_llm
        walkin_by_llm=_get_enum("MESSENGER_ROUTER", "walkin_by_llm", default="off", allowed={"off", "on"}),
        # Тему ответа задала кнопка (`button_answer_topic`) — переклейки после NLU по буквам её не перебивают, как тему от LLM
        # (`topic_by_llm`): «без записи» в адреса и П1 по совпадению с каталогом в цену; «Подобрать анализы» решает LLM по данным
        # (STEP-2026-10-28-C6-BUTTON-TOPIC-FIRM, С6 шаг 2). off — как было. [MESSENGER_ROUTER] button_topic_firm
        button_topic_firm=_get_enum("MESSENGER_ROUTER", "button_topic_firm", default="off", allowed={"off", "on"}),
        # Какая акция отвечает на вопрос — выбор LLM среди действующих акций (`services/news.py`, `PROMO_TASK`), не счёт
        # совпадений слов (STEP-2026-10-29-PROMO-BY-MEANING). off — как было. [MESSENGER_ROUTER] promo_by_llm
        promo_by_llm=_get_enum("MESSENGER_ROUTER", "promo_by_llm", default="off", allowed={"off", "on"}),
        # Листы без своего вопроса, которые под `button_answer_topic = frame` держат рамку следующей реплики (id листьев через
        # запятую; пусто — ни один). Рамка без ответа по данным темы — шаблон: «Нужно ли записываться» 48 → 70 % (без записи ли
        # услуга решает каталог МИС), «Акции и скидки» 55 → 39 % (общий список акций), стенд 29.10. Включать по листу, по замеру
        # (STEP-2026-10-29-BUTTON-FRAME). [MESSENGER_ROUTER] button_leaf_frame
        button_leaf_frame=_get_str("MESSENGER_ROUTER", "button_leaf_frame", default=""),
        # Кто под `button_answer_topic = frame` проверяет «другую тему» судьи ответа на вопрос кнопки: `question` — отдельный
        # короткий вопрос рамки (`choose_button_frame`, вариант А), `judge` — сам судья с исходом «в теме кнопки» (`domain`,
        # промпт `choose_pending_free_reply_button_frame`, вариант Б). Выбор — по замеру (решение владельца 29.10).
        # `judge_leaf` — как `judge`, и лист без своего вопроса (`button_leaf_frame`) проверяет тот же судья по метке нажатия, а не
        # вопрос рамки (Р2 консилиума 30.10: одно определение «в теме кнопки», замер — STEP-2026-10-30-R2-LEAF-JUDGE).
        # [MESSENGER_ROUTER] button_frame_check
        button_frame_check=_get_enum(
            "MESSENGER_ROUTER", "button_frame_check", default="question", allowed={"question", "judge", "judge_leaf"}
        ),
        # Свой вопрос пациента в теме нажатой кнопки (исход судьи «domain», вариант Б) — отдельный исход `in_frame` ответа на висящий
        # вопрос, а не «ответ на вопрос-слот»: тема кнопки держится, запись не отменяется; в слот-услугу (`service_name`) —
        # позиции из разбора хода, а не реплика целиком; разбор пуст или упал — слот не пишется
        # (Н-1 карты «Понимания», STEP-2026-10-30-PENDING-IN-FRAME). off — как было. [MESSENGER_ROUTER] pending_in_frame
        pending_in_frame=_get_enum("MESSENGER_ROUTER", "pending_in_frame", default="off", allowed={"off", "on"}),
        # Путь цены сдался («Не нашёл актуальную стоимость … Уточните название», цен нет): сначала данные клиники
        # (`service_answer`, место `price_not_found`, без «похожего») — строки прайса и памяток, нет данных — честный оффер
        # (STEP-2026-10-29-PRICE-NOT-FOUND-BY-DATA). off — как было. [MESSENGER_ROUTER] price_not_found_by_data
        price_not_found_by_data=_get_enum("MESSENGER_ROUTER", "price_not_found_by_data", default="off", allowed={"off", "on"}),
        # Рамку кнопки проверяет одна точка — таблица политик темы (`topic_policy`, поле строки `in_frame`): переклейка после NLU
        # по буквам (`keeps`) не уводит из темы, которую держит кнопка, и не возвращает в неё реплику, которую LLM признала другой
        # темой (Д3); срочное, перевод на оператора, существующая и идущая запись (`overrides`) — как всегда
        # (STEP-2026-10-30-FRAME-POLICY, Н-2). off — как было. [MESSENGER_ROUTER] frame_policy
        frame_policy=_get_enum("MESSENGER_ROUTER", "frame_policy", default="off", allowed={"off", "on"}),
        # «Продолжает ли реплика висящий вопрос» решает одно решение хода — одна функция ответа (`pending_answer`, исход судьи);
        # правила предпроверки записи, prelock, переклеек записи и планировщика — только сужение структурного ответа и путь при
        # сбое LLM. Оффер оператора не ставится вопросом поверх вопроса бота: первый повтор — необязывающая строка, второй шаг
        # без продвижения — один вопрос-оффер с причиной (Д2 звено 2, вариант (А), решение владельца 29.10; шаг 3 деплоя 22,
        # STEP-2026-10-30-EXPECTS-BY-UNDERSTANDING). off — как было. [MESSENGER_ROUTER] expects_by_understanding
        expects_by_understanding=_get_enum("MESSENGER_ROUTER", "expects_by_understanding", default="off", allowed={"off", "on"}),
        # Вопрос об обслуживании по ДМС / ОМС как тема (класс К26, решение владельца 28.10, ответ клиники): о чём реплика со
        # словами страхования, решает LLM (`choose`, задача `insurance`; слова — кандидат); как получить помощь по ДМС — памятка
        # клиники и оффер оператора, ОМС, конкретный врач, гарантийное письмо и отдел ДМС — «ответит оператор» и оффер; цена — путь цены.
        # Тексты — данные клиники (`messengers_router/data/insurance_topic.yaml`). off — как было. [MESSENGER_ROUTER] insurance_topic
        insurance_topic=_get_enum("MESSENGER_ROUTER", "insurance_topic", default="off", allowed={"off", "on"}),
        # Выбор строки прайса на пути ответа о цене (топ-5 разбора 95 цен, п. 3, классы F + P1): промпты `*_general_named` с
        # правилами 6 «Cito — это срочно» и 7 «позиция для другого пациента, которого пациент не называл
        # (плод, ребёнок до года), — не та»; второй круг по редким словам терминов LLM — и
        # при пустом первом. Суд записи — прежние промпты. off — как было. [MESSENGER_ROUTER] price_select_named
        price_select_named=_get_enum("MESSENGER_ROUTER", "price_select_named", default="off", allowed={"off", "on"}),
        # Голое название услуги → максимум по данным (STEP-2026-10-21-BARE-NAME-OVERVIEW) — своим выключателем, без
        # `service_answer_from_data` (решение владельца 26.10, топ-5 разбора 95 цен, п. 4): и на темах APPOINTMENT / ADDRESS /
        # DOCTOR_INFO, голое ли название — решает LLM; на записи и врачах — без строк приёма врача (цену приёма к специальности
        # не называем, пока врач не выбран — 25.09). off — как было. [MESSENGER_ROUTER] bare_name_overview
        bare_name_overview=_get_enum("MESSENGER_ROUTER", "bare_name_overview", default="off", allowed={"off", "on"}),
        # Варианты записи на услугу («на какую услугу вас записать: — …») — с ценой строки прайса процедуры (п. 4); строки
        # приёма врача — без цены (25.09). off — названия без цен, как было. [MESSENGER_ROUTER] booking_options_price
        booking_options_price=_get_enum("MESSENGER_ROUTER", "booking_options_price", default="off", allowed={"off", "on"}),
        # Ответ о клинике по данным (`service_answer_from_data`) без исхода «похожее» (BUG-2026-10-21-SVC-SIMILAR-NOISE; топ-5
        # разбора 95 цен, п. 5): близкое, но другое — «нет данных» и честный оффер, как у подбора анализов по данным; промпт
        # `choose_service_answer_exact`. Включение — после окна (е), решение владельца. off — как было. [MESSENGER_ROUTER] service_answer_no_similar
        service_answer_no_similar=_get_enum("MESSENGER_ROUTER", "service_answer_no_similar", default="off", allowed={"off", "on"}),
        # Инструмент адресов при пустой сущности услуги сам не решает её по буквам реплики: правила дают кандидатов (строки
        # прайса, по которым ищутся места оказания), выбирает LLM (`choose`, задача `address_service`); «ни одна» — общий
        # список филиалов, сбой LLM — как было. Услуга, которую LLM разбора хода уже не подтвердила, заново не ищется
        # («где сделать магнитотерапию» → адрес «Mg+ (магний)», BUG-2026-10-18-BOOKING-SERVICE-BY-LETTERS, остаток).
        # off — как было. [MESSENGER_ROUTER] address_service_llm
        address_service_llm=_get_enum("MESSENGER_ROUTER", "address_service_llm", default="off", allowed={"off", "on"}),
        # Ход (г), L-06: «кто делает услугу» — врачи по коду услуги в срезе `doctor_prices` (МИС), а не по правилу
        # «процедура → специальность» в коде. shadow — ответ не меняется, в ход пишется согласие нынешней выдачи врачей с
        # данными (`svc_doc_*`); on — врачи услуги из данных (роль правил только сужает, если среди них есть); off — как было.
        # [MESSENGER_ROUTER] doctors_by_service_data
        # По умолчанию shadow — как на проде с 22.10 (деплой 15; STEP-2026-10-15-DEFAULTS-B).
        doctors_by_service_data=_get_enum("MESSENGER_ROUTER", "doctors_by_service_data", default="shadow", allowed={"off", "shadow", "on"}),
        # Подавитель A′-1 (`router._suppress_grounder_rejected_slots`): услугу, которую в этом ходу отвергла LLM, резолвер
        # прайса по буквам в состоянии не воскрешает (инвариант 2 BUG-2026-06-02-02); off — как было.
        # [MESSENGER_ROUTER] slot_suppress_llm
        slot_suppress_llm=_get_enum("MESSENGER_ROUTER", "slot_suppress_llm", default="off", allowed={"off", "on"}),
        # T7 K1 (облако 19.10, задача B): защиты `price_list_services` — «одно название прайса покрывает реплику» и
        # «сплошной список — только распознанный целиком». Под разбором хода (`turn_parse`) они решают только на пути
        # правил (`turn_parse.acting_for` = None). on — защит нет: «список или нет» на этом пути решает то же, что
        # строит корзина (`_multi_price_items`); off — как было. [MESSENGER_ROUTER] t7_drop_k1
        t7_drop_k1=_get_enum("MESSENGER_ROUTER", "t7_drop_k1", default="off", allowed={"off", "on"}),
        # T7 K3: гард третьего стыка корзины (`_multi_price_items`: условие П2 в остатке реплики гасит корзину). Под
        # разбором его заменяет флаг `unsatisfiable` у позиций разбора (`phrase_price_items`). on — гард снят; off —
        # как было. [MESSENGER_ROUTER] t7_drop_k3
        t7_drop_k3=_get_enum("MESSENGER_ROUTER", "t7_drop_k3", default="off", allowed={"off", "on"}),
        # Выбор строки прайса и суд условия П2 — промпт с уточнённым правилом 3: вопрос о том, как сдать, подготовиться,
        # записаться, сколько готовится, «условия сбора» — не условие услуги (BUG-2026-10-17-P2-JUDGE-COLLECTION-CONDITIONS);
        # промпт `choose_price_service_conditions`. off — прежний промпт `choose_price_service`. [MESSENGER_ROUTER] price_judge_conditions
        price_judge_conditions=_get_enum("MESSENGER_ROUTER", "price_judge_conditions", default="off", allowed={"off", "on"}),
        # Корзина-калькулятор, базовый объём ТЗ 17.08 §6 (ответы заказчика 16.09): под позициями корзины — итог (одна цена у
        # каждой позиции — сумма; у позиции развилка вариантов — «от … до …» без выдуманной суммы) и «Взятие биоматериала
        # оплачивается отдельно» с перечнем взятия по данным (название строки прайса и материалы МИС `biomatNames`, цены — из
        # прайса). off — как было. По умолчанию on — как на проде с 26.10 (деплой 18; STEP-2026-10-15-DEFAULTS-B).
        # [MESSENGER_ROUTER] basket_calc
        basket_calc=_get_enum("MESSENGER_ROUTER", "basket_calc", default="on", allowed={"off", "on"}),
        debug_final_input=_get_bool(
            "MESSENGER_ROUTER",
            "debug_final_input",
            default=False,
            legacy_key="DEBUG_FINAL_INPUT",
        ),
        url=_get_str(
            "MESSENGER_ROUTER",
            "url", default="http://172.16.0.28/api/messenger-generate",
            legacy_key="MESSENGER_API_URL"),
    ),
    router=RouterConfig(
        router_v2_enable=_get_bool("ROUTER", "router_v2_enable", default=True, legacy_key="MR_ROUTER_V2_ENABLE"),
        router_v2_shadow=_get_bool("ROUTER", "router_v2_shadow", default=False, legacy_key="MR_ROUTER_V2_SHADOW"),
        default_city=_get_str("ROUTER", "default_city", default="Самара"),
        samara_only_operator_text=_get_str(
            "ROUTER",
            "samara_only_operator_text",
            default="Сейчас могу помочь только по Самаре. Соединяю с оператором.",
        ),
    ),
    nlu=NLUConfig(
        # Ключ [NLU] nlu_engine (MR_NLU_ENGINE) снят 29.10 вместе с движком llm_primary (К2 карты мёртвого кода): движок один —
        # legacy_v2; ключ больше не читается (в config.ini безвреден).
        nlu_shadow=_get_bool("NLU", "nlu_shadow", default=False, legacy_key="MR_NLU_SHADOW"),
    ),
    sber_cloud=SberCloudConfig(
        giga_authorization_key=_get_str("SBER_CLOUD", "giga_authorization_key", default="",
                                        legacy_key="giga_authorization_key"),
    ),
    nauka=NaukaConfig(
        base_url=_get_str("NAUKA", "base_url", default="", legacy_key="base_url"),
        base_url_no_site=_get_str("NAUKA", "base_url_no_site", default="", legacy_key="base_url_no_site"),
        login=_get_str("NAUKA", "login", default="", legacy_key="nayka_login"),
        password=_get_str("NAUKA", "password", default="", legacy_key="nayka_pass"),
    ),
    ollama=OllamaConfig(
        url=_get_str("OLLAMA", "url", default="http://localhost:11434", legacy_key="ollama_url"),
        model_big=_get_str("OLLAMA", "ll_model_big", default="", legacy_key="ll_model_big"),
        model_small=_get_str("OLLAMA", "ll_model_small", default="", legacy_key="ll_model_small"),
        think=_get_bool("OLLAMA", "think", default=False, legacy_key="think"),
        # Сколько вызовов LLM роутера идут одновременно (семафор `messengers_router/llm_runtime.py`). По умолчанию
        # 2 — как на проде, = OLLAMA_NUM_PARALLEL сервера (STEP-2026-10-15-DEFAULTS-B). [OLLAMA] llm_max_concurrency
        llm_max_concurrency=_get_int(
            "OLLAMA",
            "llm_max_concurrency",
            default=2,
            legacy_key="MR_LLM_MAX_CONCURRENCY",
            min_value=1,
            max_value=100,
        ),
    ),
    chroma=ChromaConfig(
        host=_get_str("CHROMA", "host", default="localhost", legacy_key="chroma_host"),
        port=_get_int("CHROMA", "port", default=8000, legacy_key="chroma_port", min_value=1, max_value=65535),
    ),
    whisper=WhisperConfig(
        ws_url=_get_str("ASR_WHISPER", "ws_url", default="", legacy_key="WHISPER_URL"),
        http_api=_get_str("ASR_WHISPER", "http_api", default="", legacy_key="WHISPER_HTTP_API"),
        api_key=_get_str("ASR_WHISPER", "api_key", default="", legacy_key="WHISPER_API_KEY"),
    ),
    vosk=VoskConfig(
        ws_url=_get_str("ASR_VOSK", "ws_url", default="", legacy_key="VOSK_URL"),
    ),
    meili=MeiliConfig(
        url=_get_str("MEILI", "url", default="", legacy_key="MEILI_URL"),
        master_key=_get_str("MEILI", "master_key", default="", legacy_key="MASTER_KEY"),
        auth_name=_get_str("MEILI", "auth_name", default="", legacy_key="AUTH_NAME"),
        auth_pass=_get_str("MEILI", "auth_pass", default="", legacy_key="AUTH_PASS"),
    ),
)

# Legacy aliases (keep old import style intact across the project).
environment = settings.app.environment

ollama_url = settings.ollama.url
chroma_host = settings.chroma.host
chroma_port = settings.chroma.port
MEILI_URL = settings.meili.url
VOSK_URL = settings.vosk.ws_url
WHISPER_URL = settings.whisper.ws_url
WHISPER_HTTP_API = settings.whisper.http_api

MASTER_KEY = settings.meili.master_key
AUTH_NAME = settings.meili.auth_name
AUTH_PASS = settings.meili.auth_pass
APP_DATA_DIR = settings.app.app_data_dir
AGENT_API_KEY = settings.app.agent_api_key
WHISPER_API_KEY = settings.whisper.api_key

nayka_base_url = settings.nauka.base_url
nayka_base_url_no_site = settings.nauka.base_url_no_site
nayka_login = settings.nauka.login
nayka_pass = settings.nauka.password

think = settings.ollama.think
giga_authorization = settings.sber_cloud.giga_authorization_key
ll_model_big = settings.ollama.model_big
ll_model_small = settings.ollama.model_small

USE_DOCTOR_PRICES_FOR_PROCEDURES = settings.messenger_router.use_doctor_prices_for_procedures
MR_DOCTORS_TOP_N = settings.messenger_router.mr_doctors_top_n
MR_SCHEDULE_FRESH_TTL_SECONDS = settings.messenger_router.mr_schedule_fresh_ttl_seconds
MR_SCHEDULE_STALE_TTL_SECONDS = settings.messenger_router.mr_schedule_stale_ttl_seconds
MR_SCHEDULE_NEGATIVE_TTL_SECONDS = settings.messenger_router.mr_schedule_negative_ttl_seconds
MR_SCHEDULE_CACHE_MAX_KEYS = settings.messenger_router.mr_schedule_cache_max_keys
MR_SCHEDULE_CACHE_LOG_EVENTS = settings.messenger_router.mr_schedule_cache_log_events
DEBUG_RAW_OUTPUT = settings.messenger_router.debug_raw_output
LLM_PROCEDURE_NORMALIZATION = settings.messenger_router.llm_procedure_normalization
MR_LLM_SERVICE_NORMALIZATION = settings.messenger_router.llm_service_normalization
MR_LLM_PRICE_SERVICE_SELECT = settings.messenger_router.llm_price_service_select
MR_LLM_APPOINTMENT_CHANGE_VALIDATION = settings.messenger_router.llm_appointment_change_validation
MR_LLM_EXISTING_APPOINTMENT_VALIDATION = settings.messenger_router.llm_existing_appointment_validation
MR_LLM_EMPTY_EVIDENCE_REPLY = settings.messenger_router.llm_empty_evidence_reply
MR_LLM_PATIENT_NAME_VALIDATION = settings.messenger_router.llm_patient_name_validation
MR_LLM_SERVICE_SLOT_VALIDATION = settings.messenger_router.llm_service_slot_validation
MR_LLM_RESULT_TIMING_VALIDATION = settings.messenger_router.llm_result_timing_validation
MR_PREPARE_RELEVANCE_LLM_ENABLED = settings.messenger_router.prepare_relevance_llm_enabled
MR_LLM_SYMPTOM_SPECIALISTS = settings.messenger_router.llm_symptom_specialists
MR_TURN_TELEMETRY = settings.messenger_router.turn_telemetry
MR_TURN_UNDERSTANDING_PRICE = settings.messenger_router.turn_understanding_price
MR_LLM_DECISION_TURN_CALLS = settings.messenger_router.llm_decision_turn_calls
MR_LLM_DECISION_TURN_SECONDS = settings.messenger_router.llm_decision_turn_seconds
MR_LLM_DECISION_BREAKER_FAILURES = settings.messenger_router.llm_decision_breaker_failures
MR_LLM_DECISION_BREAKER_OPEN_S = settings.messenger_router.llm_decision_breaker_open_s
MR_CLASSIFIER_TIMEOUT_S = settings.messenger_router.classifier_timeout_s
MR_LLM_TURN_DEADLINE_S = settings.messenger_router.llm_turn_deadline_s
MR_LLM_FAIL_OFFER = settings.messenger_router.llm_fail_offer
MR_CLASSIFIER_COMPACT = settings.messenger_router.classifier_compact
MR_CHOOSE_CACHE_TTL_S = settings.messenger_router.choose_cache_ttl_s
MR_PARSE_FAIL_VARIANTS = settings.messenger_router.parse_fail_variants
MR_TOPIC_OVERRIDE_LLM = settings.messenger_router.topic_override_llm
MR_TOPIC_SHADOW_LLM = settings.messenger_router.topic_shadow_llm
MR_P1_FACET_NONE_OFFER = settings.messenger_router.p1_facet_none_offer
MR_FACET_BY_LLM = settings.messenger_router.facet_by_llm
MR_TURN_SPANS_LLM = settings.messenger_router.turn_spans_llm
MR_PREPARE_BY_PARSE = settings.messenger_router.prepare_by_parse
MR_PRICE_BY_PARSE = settings.messenger_router.price_by_parse
MR_STRONG_FAMILY_LETTER = settings.messenger_router.strong_family_letter
MR_POSITION_QUALIFIERS = settings.messenger_router.position_qualifiers
MR_QUALIFIER_POOL = settings.messenger_router.qualifier_pool
MR_QUALIFIER_MARKS = settings.messenger_router.qualifier_marks
MR_BASKET_FAIL_UNPRICED = settings.messenger_router.basket_fail_unpriced
MR_TOPIC_BY_LLM = settings.messenger_router.topic_by_llm
MR_BOOKING_TARGET_SLOT_LLM = settings.messenger_router.booking_target_slot_llm
MR_BUTTON_ANSWER_TOPIC = settings.messenger_router.button_answer_topic
MR_BUTTON_TOPIC_HINT = settings.messenger_router.button_topic_hint
MR_BUTTON_REPLY_JUDGE = settings.messenger_router.button_reply_judge
MR_BUTTON_SIDE_ANSWER = settings.messenger_router.button_side_answer
MR_BUTTON_PRICE_SCOPE = settings.messenger_router.button_price_scope
MR_ADDRESS_BRANCH_FLAGS = settings.messenger_router.address_branch_flags
MR_ADDRESS_FOLLOWUP_LLM = settings.messenger_router.address_followup_llm
MR_ADDRESS_NAMED_STREET = settings.messenger_router.address_named_street
MR_TURN_UNDERSTANDING_ONE = settings.messenger_router.turn_understanding_one
MR_TURN_UNDERSTANDING_FULL = settings.messenger_router.turn_understanding_full
MR_SERVICE_ANSWER_FROM_DATA = settings.messenger_router.service_answer_from_data
MR_TEST_ASSIST_FROM_DATA = settings.messenger_router.test_assist_from_data
MR_TEST_ASSIST_PRICE = settings.messenger_router.test_assist_price
MR_WALKIN_BY_LLM = settings.messenger_router.walkin_by_llm
MR_BUTTON_TOPIC_FIRM = settings.messenger_router.button_topic_firm
MR_PROMO_BY_LLM = settings.messenger_router.promo_by_llm
MR_BUTTON_LEAF_FRAME = settings.messenger_router.button_leaf_frame
MR_BUTTON_FRAME_CHECK = settings.messenger_router.button_frame_check
MR_PENDING_IN_FRAME = settings.messenger_router.pending_in_frame
MR_PRICE_NOT_FOUND_BY_DATA = settings.messenger_router.price_not_found_by_data
MR_FRAME_POLICY = settings.messenger_router.frame_policy
MR_EXPECTS_BY_UNDERSTANDING = settings.messenger_router.expects_by_understanding
MR_INSURANCE_TOPIC = settings.messenger_router.insurance_topic
MR_PRICE_SELECT_NAMED = settings.messenger_router.price_select_named
MR_BARE_NAME_OVERVIEW = settings.messenger_router.bare_name_overview
MR_BOOKING_OPTIONS_PRICE = settings.messenger_router.booking_options_price
MR_SERVICE_ANSWER_NO_SIMILAR = settings.messenger_router.service_answer_no_similar
MR_ADDRESS_SERVICE_LLM = settings.messenger_router.address_service_llm
MR_DOCTORS_BY_SERVICE_DATA = settings.messenger_router.doctors_by_service_data
MR_SLOT_SUPPRESS_LLM = settings.messenger_router.slot_suppress_llm
MR_T7_DROP_K1 = settings.messenger_router.t7_drop_k1
MR_T7_DROP_K3 = settings.messenger_router.t7_drop_k3
MR_PRICE_JUDGE_CONDITIONS = settings.messenger_router.price_judge_conditions
MR_BASKET_CALC = settings.messenger_router.basket_calc
MR_PENDING_BUTTONS = settings.messenger_router.pending_buttons
MR_REASK_LIMIT = settings.messenger_router.reask_limit
MR_PARSE_FRAGMENT_WINDOWS = settings.messenger_router.parse_fragment_windows
MR_PARSE_COMPLETE_WINDOWS = settings.messenger_router.parse_complete_windows
MR_STRONG_PHRASE_GUARD = settings.messenger_router.strong_phrase_guard
MR_PARSE_AGREE_TELEMETRY = settings.messenger_router.parse_agree_telemetry
MR_NAMES_DOCTOR_CATALOG = settings.messenger_router.names_doctor_catalog
MR_NAMES_DOCTOR_NOISE = settings.messenger_router.names_doctor_noise
MR_NAMES_DOCTOR_FOLLOWUP = settings.messenger_router.names_doctor_followup
MR_NAMES_RESULT_SURNAME = settings.messenger_router.names_result_surname
MR_NAMES_PATIENT_FIO = settings.messenger_router.names_patient_fio
MR_NAMES_PATIENT_RULE = settings.messenger_router.names_patient_rule
MR_NAMES_PATIENT_FILL = settings.messenger_router.names_patient_fill
DEBUG_FINAL_INPUT = settings.messenger_router.debug_final_input
MESSENGER_API_URL = settings.messenger_router.url

MR_ROUTER_V2_ENABLE = settings.router.router_v2_enable
MR_ROUTER_V2_SHADOW = settings.router.router_v2_shadow
MR_NLU_SHADOW = settings.nlu.nlu_shadow
MR_LLM_MAX_CONCURRENCY = settings.ollama.llm_max_concurrency
