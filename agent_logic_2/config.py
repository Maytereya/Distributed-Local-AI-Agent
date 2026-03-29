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
        debug_final_input=_get_bool(
            "MESSENGER_ROUTER",
            "debug_final_input",
            default=False,
            legacy_key="DEBUG_FINAL_INPUT",
        ),
        url=_get_str(
            "MESSENGER_ROUTER",
            "url", default="http://172.16.0.16/api/messenger-generate",
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
        llm_max_concurrency=_get_int(
            "OLLAMA",
            "llm_max_concurrency",
            default=5,
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
DEBUG_FINAL_INPUT = settings.messenger_router.debug_final_input
MESSENGER_API_URL = settings.messenger_router.url

MR_ROUTER_V2_ENABLE = settings.router.router_v2_enable
MR_ROUTER_V2_SHADOW = settings.router.router_v2_shadow
MR_NLU_ENGINE = settings.nlu.nlu_engine
MR_NLU_SHADOW = settings.nlu.nlu_shadow
MR_LLM_MAX_CONCURRENCY = settings.ollama.llm_max_concurrency
