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
    choose_cache_ttl_s: int
    choose_symptom: str
    choose_readiness: str
    parse_fail_variants: int
    variant_b_wordforms: str
    mis_honest_degradation: str
    topic_override_llm: str
    pending_reply: str
    pending_reply_types: str
    # D2.4 — предел переспросов (поле ожидания `_expects`, `messengers_router/expects.py`)
    reask_limit: str
    parse_fragment_windows: str
    parse_complete_windows: str
    strong_phrase_guard: str
    parse_agree_telemetry: str
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
    nlu_engine: str
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
        llm_decision_turn_seconds=_get_int("MESSENGER_ROUTER", "llm_decision_turn_seconds", 20, min_value=0, max_value=120),
        # Предохранитель LLM-решений (L-11): N сбоев генерации подряд → S секунд без вызовов; таймаут
        # очереди не считается (иначе качество бота зависит от нагрузки соседей). 0 сбоев — выключен.
        llm_decision_breaker_failures=_get_int("MESSENGER_ROUTER", "llm_decision_breaker_failures", 3, min_value=0, max_value=100),
        llm_decision_breaker_open_s=_get_int("MESSENGER_ROUTER", "llm_decision_breaker_open_s", 30, min_value=1, max_value=600),
        # Кэш внятных ответов модуля выбора (ход (в), L-12): секунд жизни; сбой и мусор не кэшируются.
        # Коротко — ошибка модели не становится ответом всем на часы; 0 — без кэша (стенд меряет без него).
        choose_cache_ttl_s=_get_int("MESSENGER_ROUTER", "choose_cache_ttl_s", 1800, min_value=0, max_value=86400),
        # Выбор строки прайса на модуле choose() (ход (в), клиент 1) — единственный путь с 17.10: ключ [MESSENGER_ROUTER]
        # choose_price (off — прежний вызов, legacy — прежний промпт через модуль) снят и больше не читается (в config.ini
        # прода безвреден; STEP-2026-10-17-ONE-PATH-CHOOSE-PRICE).
        # Симптом → врач на модуле choose() (ход (в), клиент 2): off — прежний вызов и разбор; legacy —
        # прежний промпт через модуль; on — врач названием из перечня ролей МИС (не больше двух), исходы
        # «срочно» и «не жалоба». Политика сбоя прежняя. По умолчанию on — как на проде с 14.10
        # (STEP-2026-10-15-DEFAULTS-B). [MESSENGER_ROUTER] choose_symptom
        choose_symptom=_get_enum("MESSENGER_ROUTER", "choose_symptom", default="on", allowed={"off", "legacy", "on"}),
        # Срок готовности при слабом совпадении — услугу выбирает LLM, срок печатает код из МИС (ход (в)).
        # off | on, по умолчанию on — как на проде с 14.10 (STEP-2026-10-15-DEFAULTS-B); действует под пониманием
        # хода on. [MESSENGER_ROUTER] choose_readiness
        choose_readiness=_get_enum("MESSENGER_ROUTER", "choose_readiness", default="on", allowed={"off", "on"}),
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
        # Ответ на висящий вопрос бота силами LLM (D2.3, `messengers_router/pending_reply.py`): off — как было;
        # shadow — тень: решение LLM вне бюджета хода и предохранителя, только телеметрия (tu_pending_*), ответ
        # прежний; on — пока как shadow (исполнение — следующий шаг, после живого замера и встречного свипа).
        # [MESSENGER_ROUTER] pending_reply
        pending_reply=_get_enum("MESSENGER_ROUTER", "pending_reply", default="off", allowed={"off", "shadow", "on"}),
        # Под pending_reply = on — типы вопросов (`_expects.type`, через запятую), на которые решение LLM исполняется в
        # ходе (асимметрия безопасности по типу, решение владельца 15.10); прочие типы — тень. Пусто — как shadow.
        # Сейчас исполняется только operator_offer: LLM может поднять ответ до «да» (перевод), не снять.
        # [MESSENGER_ROUTER] pending_reply_types
        pending_reply_types=_get_str("MESSENGER_ROUTER", "pending_reply_types", default=""),
        # Предел переспросов (D2.4; пункт 5 принципов, инвариант 3bacf58): off — как было (обработчик висящего
        # вопроса переспрашивает без счёта); on — один вопрос звучит не больше двух раз подряд, третьего нет:
        # бот называет нужный ответ и предлагает оператора. [MESSENGER_ROUTER] reask_limit
        reask_limit=_get_enum("MESSENGER_ROUTER", "reask_limit", default="off", allowed={"off", "on"}),
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
        # Вариант B L-13 (названия без цен при недоступной LLM): строка со словоформой различающего слова пациента
        # («сифилиса» к «сифилис») — тоже кандидат, если покрывает больше слов пациента; порядок — по их числу. off — как
        # было (только слово целиком); on — словоформы, если слово вместе с ними различает (облако 16.10; общее слово —
        # BUG-2026-10-17-VARIANT-B-COMMON-WORD). По умолчанию on с 17.10: офлайн-замер dev — 0 хуже (решение владельца 16.10:
        # включать только при 0 хуже). [MESSENGER_ROUTER] variant_b_wordforms
        variant_b_wordforms=_get_enum("MESSENGER_ROUTER", "variant_b_wordforms", default="on", allowed={"off", "on"}),
        # Ход (д), видимая деградация (решения владельца 08.10 и 18.10): сбой МИС → честный перевод на оператора вместо
        # правдоподобной неправды («расписание не найдено», «акций нет», адреса без фильтра врачей); устаревший срез —
        # приписка «по данным на ДД.ММ» или оператор. off — как было (правило 13.10: видимое поведение под
        # выключателем, умолчание «как было»). [MESSENGER_ROUTER] mis_honest_degradation
        mis_honest_degradation=_get_enum("MESSENGER_ROUTER", "mis_honest_degradation", default="off", allowed={"off", "on"}),
        # Тема реестра перебивает метку OTHER только после подтверждения LLM по смыслу (решение владельца 18.10:
        # «чинить через LLM, уходим от совпадения букв к смыслу»; «Ново-Садовая» ловилась словом «новость»).
        # LLM недоступна — тема метку не перебивает. off — как было. [MESSENGER_ROUTER] topic_override_llm
        topic_override_llm=_get_enum("MESSENGER_ROUTER", "topic_override_llm", default="off", allowed={"off", "on"}),
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
        nlu_engine=_get_enum(
            "NLU",
            "nlu_engine",
            default="legacy_v2",
            allowed={"legacy_v2", "llm_primary"},
            legacy_key="MR_NLU_ENGINE",
        ),
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
MR_CHOOSE_CACHE_TTL_S = settings.messenger_router.choose_cache_ttl_s
MR_CHOOSE_SYMPTOM = settings.messenger_router.choose_symptom
MR_CHOOSE_READINESS = settings.messenger_router.choose_readiness
MR_PARSE_FAIL_VARIANTS = settings.messenger_router.parse_fail_variants
MR_VARIANT_B_WORDFORMS = settings.messenger_router.variant_b_wordforms
MR_MIS_HONEST_DEGRADATION = settings.messenger_router.mis_honest_degradation
MR_TOPIC_OVERRIDE_LLM = settings.messenger_router.topic_override_llm
MR_PENDING_REPLY = settings.messenger_router.pending_reply
MR_PENDING_REPLY_TYPES = settings.messenger_router.pending_reply_types
MR_REASK_LIMIT = settings.messenger_router.reask_limit
MR_PARSE_FRAGMENT_WINDOWS = settings.messenger_router.parse_fragment_windows
MR_PARSE_COMPLETE_WINDOWS = settings.messenger_router.parse_complete_windows
MR_STRONG_PHRASE_GUARD = settings.messenger_router.strong_phrase_guard
MR_PARSE_AGREE_TELEMETRY = settings.messenger_router.parse_agree_telemetry
DEBUG_FINAL_INPUT = settings.messenger_router.debug_final_input
MESSENGER_API_URL = settings.messenger_router.url

MR_ROUTER_V2_ENABLE = settings.router.router_v2_enable
MR_ROUTER_V2_SHADOW = settings.router.router_v2_shadow
MR_NLU_ENGINE = settings.nlu.nlu_engine
MR_NLU_SHADOW = settings.nlu.nlu_shadow
MR_LLM_MAX_CONCURRENCY = settings.ollama.llm_max_concurrency
