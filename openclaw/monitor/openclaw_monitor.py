#!/usr/bin/env python3
from __future__ import annotations

import json
import hmac
import os
import re
import shlex
import socket
import signal
import subprocess
import threading
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, time as dt_time, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
from probe_contract import decode_probe_output
from reporting_time import covered_samples, period_window
from report_outbox import ReportOutbox
from reporting_projection import corporate_card, failure_card, validate_snapshot
from host_metrics_contract import validate_host_metrics


OK = "ok"
WARN = "warn"
CRIT = "crit"
SEVERITY = {OK: 0, WARN: 1, CRIT: 2}
STATUS_LABELS = {
    OK: "ШТАТНО",
    WARN: "ВНИМАНИЕ",
    CRIT: "КРИТИЧНО",
}
STATUS_ICONS = {
    OK: "✓",
    WARN: "!",
    CRIT: "✕",
}
SERVICE_PURPOSES = {
    "OpenClaw gateway": "Telegram-интерфейс OpenClaw и агент, который отвечает на запросы.",
    "sing-box Telegram gateway": "локальный proxy-канал к Telegram для контейнеров сервера.",
    "Ollama server": "сервер локальных языковых моделей.",
    "Whisper server": "сервер распознавания речи.",
    "Support messenger Telegram bot": "Telegram-адаптер корпоративного чатбота.",
    "Support messenger web": "web-панель операторов и API агрегатора сообщений.",
}
LOG_TITLE_PREFIXES = ("Logs: ", "Логи: ")

DEFAULT_CONFIG: dict[str, Any] = {
    "listen_host": "0.0.0.0",
    "listen_port": 18080,
    "timezone": "Europe/Samara",
    "check_interval_seconds": 300,
    "menu_interval_seconds": 300,
    "report_time": "09:00",
    "alert_on_status_change": True,
    "openclaw_config_path": "/openclaw/openclaw.json",
    "data_dir": "/data",
    "remediation": {
        "enabled": False,
        "openclaw_telegram_restart": {
            "enabled": True,
            "container": "openclaw-openclaw-gateway-1",
            "failure_threshold": 1,
            "min_interval_seconds": 900,
            "startup_wait_seconds": 12,
        },
    },
    "telegram": {
        "proxy": "http://singbox:1080",
        "chat_id": "",
        "report_targets": [],
        "commands": [
            {"command": "server_status", "description": "Проверить AI-сервер"},
            {"command": "chatbot_status", "description": "Краткий отчет чатбота"},
            {"command": "telegram_pulse", "description": "Пульс связи Telegram"},
            {"command": "failures", "description": "Разбор проблемных обращений — личный чат"},
            {"command": "failure", "description": "Карточка обращения по номеру — личный чат"},
            {"command": "security_status", "description": "Состояние проверки безопасности — личный чат"},
            {"command": "status", "description": "Состояние OpenClaw"},
            {"command": "diagnostics", "description": "Диагностика OpenClaw"},
            {"command": "whoami", "description": "Ваш Telegram ID"},
            {"command": "commands", "description": "Краткая справка"},
            {"command": "help", "description": "Помощь"},
        ],
    },
    "containers": [
        {"key": "openclaw", "name": "OpenClaw gateway", "container": "openclaw-openclaw-gateway-1"},
        {"key": "singbox", "name": "sing-box Telegram gateway", "container": "singbox_telegram_gateway"},
        {"key": "ollama", "name": "Ollama server", "container": "ollama"},
        {"key": "whisper", "name": "Whisper server", "container": "whisper-gpu"},
        {
            "key": "aggregator_bot",
            "name": "Support messenger Telegram bot",
            "container": "support-messenger-aggregator-telegram_bot-1",
        },
        {
            "key": "aggregator_web",
            "name": "Support messenger web",
            "container": "support-messenger-aggregator-web-1",
        },
    ],
    "network_checks": {
        "singbox_host": "singbox",
        "singbox_port": 1080,
        "telegram_proxies": ["http://singbox:1080", "socks5h://singbox:1080"],
    },
    "telegram_pulse": {
        "enabled": True,
        "interval_seconds": 60,
        "timeout_seconds": 10,
        "history_days": 95,
        "url": "https://api.telegram.org",
        "daily_line_enabled": True,
        "channels": [
            {"id": "http_singbox", "name": "HTTP-вход sing-box", "proxy": "http://singbox:1080"},
            {"id": "socks_singbox", "name": "SOCKS-вход sing-box", "proxy": "socks5h://singbox:1080"},
        ],
    },
    "api_checks": {
        "ollama_tags_url": "http://ollama:11434/api/tags",
        "whisper_health_url": "http://whisper-gpu:8000/healthz",
        "aggregator_login_url": "http://support-messenger-aggregator-web-1:8000/login/",
        "aggregator_host_header": "127.0.0.1",
    },
    "postgres_stats": {
        "enabled": True,
        "container": "support-messenger-aggregator-postgres-1",
    },
    "logs": {
        "since": "24h",
        "tail": "500",
        "max_lines_per_container": 5,
    },
}

ERROR_RE = re.compile(
    r"\b(error|failed|exception|traceback|panic|fatal|oom|out of memory|cuda|unhealthy|timeout|fallback|falling back|no compatible gpu|no gpu|cpu fallback)\b",
    re.IGNORECASE,
)
CRITICAL_LOG_RE = re.compile(
    r"\b(panic|fatal|oom|out of memory|segmentation fault|cuda error|unhealthy|no compatible gpu|cpu fallback)\b",
    re.IGNORECASE,
)
SECRET_PATTERNS = [
    re.compile(r"bot\d+:[A-Za-z0-9_-]+"),
    re.compile(r"(?i)(token|password|secret|api[_-]?key)=([^&\s]+)"),
    re.compile(r"(?i)(token|password|secret|api[_-]?key)(['\"]?\s*[:=]\s*['\"])[^'\"\s]+"),
]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config() -> dict[str, Any]:
    config = DEFAULT_CONFIG
    config_path = os.environ.get("MONITOR_CONFIG", "/etc/openclaw-monitor/config.json")
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            config = deep_merge(config, json.load(fh))

    telegram = dict(config.get("telegram", {}))
    if os.environ.get("TELEGRAM_CHAT_ID"):
        telegram["chat_id"] = os.environ["TELEGRAM_CHAT_ID"]
    if os.environ.get("TELEGRAM_PROXY"):
        telegram["proxy"] = os.environ["TELEGRAM_PROXY"]
    config["telegram"] = telegram

    for env_key, config_key in [
        ("OPENCLAW_CONFIG_PATH", "openclaw_config_path"),
        ("REPORT_TIME", "report_time"),
        ("REPORT_TIMEZONE", "timezone"),
        ("CHECK_INTERVAL_SECONDS", "check_interval_seconds"),
        ("MENU_INTERVAL_SECONDS", "menu_interval_seconds"),
    ]:
        if os.environ.get(env_key):
            value: Any = os.environ[env_key]
            if config_key.endswith("_seconds"):
                value = int(value)
            config[config_key] = value
    return config


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def truncate(value: str, limit: int = 500) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def sanitize(value: str) -> str:
    sanitized = value
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub(lambda m: f"{m.group(1)}=<redacted>" if m.lastindex else "<redacted>", sanitized)
    return truncate(sanitized.replace("\r", " ").replace("\n", " "))


def split_message(text: str, limit: int = 3600) -> list[str]:
    # Telegram counts UTF-16 code units, including two units for most emoji.
    def size(value):
        return len(value.encode("utf-16-le")) // 2

    def take(value):
        used = 0
        for i, char in enumerate(value):
            used += size(char)
            if used > limit:
                return value[:i], value[i:]
        return value, ""

    chunks: list[str] = []
    current = ""
    for line in text.splitlines():
        candidate = line if not current else current + "\n" + line
        if size(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        while size(line) > limit:
            chunk, line = take(line)
            chunks.append(chunk)
        current = line
    if current:
        chunks.append(current)
    return chunks or [""]


def run_cmd(
    args: list[str],
    *,
    timeout: int = 20,
    input_text: str | None = None,
) -> tuple[int, str, str]:
    if args and args[0] == "docker" and os.environ.get("DOCKER_BROKER_SOCKET"):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(max(timeout, 70))
                connection.connect(os.environ["DOCKER_BROKER_SOCKET"])
                request = json.dumps({"args": args, "input_text": input_text}) + "\n"
                connection.sendall(request.encode("utf-8"))
                with connection.makefile("r", encoding="utf-8") as stream:
                    response = json.loads(stream.readline(500_001))
                return int(response["code"]), str(response["stdout"]), str(response["stderr"])
        except (OSError, ValueError, KeyError):
            return 1, "", "docker_broker_unavailable"
    try:
        proc = subprocess.run(
            args,
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return 124, stdout, stderr or f"timeout after {timeout}s"
    except OSError as exc:
        return 127, "", str(exc)


def check(
    status: str,
    key: str,
    title: str,
    summary: str,
    details: list[str] | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item = {
        "key": key,
        "title": title,
        "status": status,
        "summary": summary,
        "details": details or [],
    }
    if data is not None:
        item["data"] = data
    return item


def worst_status(items: list[dict[str, Any]]) -> str:
    status = OK
    for item in items:
        if SEVERITY.get(item.get("status"), 0) > SEVERITY[status]:
            status = item["status"]
    return status


def read_openclaw_token(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        config = json.load(fh)
    telegram = config.get("channels", {}).get("telegram", {})
    token = (
        telegram.get("botToken")
        or telegram.get("token")
        or telegram.get("bot_token")
        or telegram.get("bot", {}).get("token")
    )
    if not token:
        raise ValueError("Telegram bot token was not found in OpenClaw config")
    return str(token)


class TelegramClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._token: str | None = None

    def token(self) -> str:
        if not self._token:
            token_file = os.environ.get("TELEGRAM_MONITOR_BOT_TOKEN_FILE")
            self._token = Path(token_file).read_text(encoding="utf-8").strip() if token_file else read_openclaw_token(self.config["openclaw_config_path"])
            if not self._token:
                raise RuntimeError("telegram_credential_unavailable")
        return self._token

    def call(
        self,
        method: str,
        payload: dict[str, Any] | None = None,
        timeout: int = 15,
        retries: int = 3,
    ) -> dict[str, Any]:
        url = f"https://api.telegram.org/bot{self.token()}/{method}"
        data = json.dumps(payload or {}).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        proxy = self.config.get("telegram", {}).get("proxy")
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"https": proxy, "http": proxy}) if proxy else urllib.request.ProxyHandler({})
        )
        last_exc: Exception | None = None
        for attempt in range(1, max(1, retries) + 1):
            try:
                with opener.open(request, timeout=timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code < 500 and exc.code != 429:
                    raise
                last_exc = exc
            except (TimeoutError, OSError, urllib.error.URLError) as exc:
                last_exc = exc
            if attempt < max(1, retries):
                time.sleep(min(2 * attempt, 5))
        if last_exc:
            raise last_exc
        raise RuntimeError("Telegram API call failed")

    def send_message(self, text: str, chat_id: str | None = None) -> tuple[bool, str]:
        chat_id = chat_id or self.config.get("telegram", {}).get("chat_id")
        if not chat_id:
            return False, "TELEGRAM_CHAT_ID is not configured"
        try:
            for index, chunk in enumerate(split_message(text)):
                response = self.send_chunk(chunk, chat_id)
                if response["state"] != "sent":
                    return False, response["reason"]
                if index:
                    time.sleep(0.2)
            return True, "sent"
        except Exception:
            return False, "transport_unconfirmed"

    def send_chunk(self, text: str, chat_id: str, created_at: str | None = None) -> dict:
        if created_at and (datetime.now(timezone.utc) - datetime.fromisoformat(created_at)).total_seconds() > 60:
            text = f"Отчёт доставлен с задержкой\nСформирован: {created_at}\nДоставляется: {now_iso()}\nДанные относятся к исходному периоду.\n\n{text}"
        try:
            response = self.call("sendMessage", {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}, retries=1)
            message_id = (response.get("result") or {}).get("message_id")
            if response.get("ok") is True and isinstance(message_id, int):
                return {"state": "sent", "message_id": message_id, "reason": "accepted_by_telegram"}
            code = response.get("error_code")
            return {"state": "pending" if code == 429 else "failed", "reason": "telegram_rejected", "retry_after": (response.get("parameters") or {}).get("retry_after", 60)}
        except urllib.error.HTTPError as exc:
            retry_after = 60
            if exc.code == 429:
                try:
                    retry_after = json.loads(exc.read(16_000)).get("parameters", {}).get("retry_after", 60)
                except (ValueError, OSError):
                    pass
            return {"state": "pending" if exc.code == 429 else "failed" if 400 <= exc.code < 500 else "uncertain", "reason": "telegram_http_" + str(exc.code), "retry_after": retry_after}
        except Exception:
            return {"state": "uncertain", "reason": "transport_unconfirmed"}

    def set_commands(self) -> tuple[bool, str]:
        commands = self.config.get("telegram", {}).get("commands") or []
        if not commands:
            return True, "no commands configured"
        scopes: list[dict[str, Any] | None] = [None, {"type": "all_private_chats"}, {"type": "all_group_chats"}]
        seen_scopes = {json.dumps(scope or {"type": "default"}, sort_keys=True) for scope in scopes}
        for target in self.config.get("telegram", {}).get("report_targets") or []:
            chat_id = str(target.get("chat_id") or "").strip()
            if not chat_id.startswith("-"):
                continue
            scope = {"type": "chat", "chat_id": chat_id}
            key = json.dumps(scope, sort_keys=True)
            if key not in seen_scopes:
                scopes.append(scope)
                seen_scopes.add(key)
        language_codes: list[str | None] = [None, "ru", "en"]
        try:
            for scope in scopes:
                for language_code in language_codes:
                    payload: dict[str, Any] = {"commands": commands}
                    if language_code:
                        payload["language_code"] = language_code
                    if scope:
                        payload["scope"] = scope
                    response = self.call("setMyCommands", payload)
                    if not response.get("ok"):
                        return False, "telegram_menu_rejected"
            return True, "commands updated"
        except Exception:
            return False, "telegram_menu_unavailable"


class Monitor:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.telegram = TelegramClient(config)
        self.data_dir = Path(config.get("data_dir", "/data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.data_dir / "state.json"
        self.history_path = self.data_dir / "checks.jsonl"
        self.pulse_path = self.data_dir / "telegram_pulse.jsonl"
        self.pending_path = self.data_dir / "pending_reports.jsonl"
        self.lock = threading.Lock()
        self.state_lock = threading.RLock()
        self.pulse_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.started_at = time.monotonic()
        self.heartbeats = {}
        self.outbox = ReportOutbox(self.data_dir / "report_outbox.sqlite3")
        self.last_result: dict[str, Any] | None = None
        self.state = self.load_state()

    def load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def save_state(self) -> None:
        with self.state_lock:
            tmp = self.state_path.with_name("state." + uuid.uuid4().hex + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.state, fh, ensure_ascii=False, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp, 0o600)
            tmp.replace(self.state_path)

    def record_state(self, key, value):
        with self.state_lock:
            self.state[key] = value
            self.save_state()

    def health_status(self):
        now = time.monotonic()
        required = {"scheduler": 2 * int(self.config.get("check_interval_seconds", 300)) + 120}
        if self.pulse_enabled():
            required["pulse"] = 3 * self.pulse_interval() + 30
        failures = [key for key, timeout in required.items() if now - self.heartbeats.get(key, self.started_at) > timeout]
        return {"status": "unhealthy" if failures else "ok", "stale_workers": failures, "outbox": self.outbox.summary(), "timestamp": now_iso()}

    def append_history(self, result: dict[str, Any]) -> None:
        compact = {
            "timestamp": result["timestamp"],
            "status": result["status"],
            "duration_ms": result["duration_ms"],
            "checks": [
                {"key": item["key"], "status": item["status"], "summary": item["summary"]}
                for item in result["checks"]
            ],
        }
        with open(self.history_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(compact, ensure_ascii=False) + "\n")

    def pulse_config(self) -> dict[str, Any]:
        return self.config.get("telegram_pulse", {}) or {}

    def pulse_enabled(self) -> bool:
        return bool(self.pulse_config().get("enabled", True))

    def pulse_interval(self) -> int:
        return max(15, int(self.pulse_config().get("interval_seconds", 60)))

    def pulse_channels(self) -> list[dict[str, str]]:
        configured = self.pulse_config().get("channels") or []
        channels: list[dict[str, str]] = []
        for index, channel in enumerate(configured):
            if not isinstance(channel, dict):
                continue
            proxy = str(channel.get("proxy") or "").strip()
            if not proxy:
                continue
            channel_id = str(channel.get("id") or f"channel_{index + 1}").strip()
            channels.append(
                {
                    "id": re.sub(r"[^A-Za-z0-9_.-]+", "_", channel_id)[:80] or f"channel_{index + 1}",
                    "name": str(channel.get("name") or proxy),
                    "proxy": proxy,
                }
            )
        if channels:
            return channels

        fallback_proxies = self.config.get("network_checks", {}).get("telegram_proxies") or []
        for index, proxy in enumerate(fallback_proxies):
            proxy_text = str(proxy).strip()
            if not proxy_text:
                continue
            channels.append(
                {
                    "id": f"proxy_{index + 1}",
                    "name": proxy_text,
                    "proxy": proxy_text,
                }
            )
        return channels

    def probe_telegram_pulse_channel(self, channel: dict[str, str]) -> dict[str, Any]:
        pulse_config = self.pulse_config()
        url = str(pulse_config.get("url") or "https://api.telegram.org")
        timeout = max(3, int(pulse_config.get("timeout_seconds", 10)))
        started = time.monotonic()
        code, stdout, stderr = run_cmd(
            [
                "curl",
                "-sS",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}\t%{time_total}",
                "-x",
                channel["proxy"],
                "--connect-timeout",
                str(timeout),
                "--max-time",
                str(timeout),
                url,
            ],
            timeout=timeout + 5,
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        parts = stdout.strip().split("\t")
        http_status = parts[0] if parts else ""
        if len(parts) > 1:
            try:
                elapsed_ms = int(float(parts[1]) * 1000)
            except ValueError:
                pass
        ok_statuses = {"200", "301", "302", "404"}
        ok = code == 0 and http_status in ok_statuses
        item: dict[str, Any] = {
            "id": channel["id"],
            "name": channel["name"],
            "proxy": channel["proxy"],
            "ok": ok,
            "http_status": http_status,
            "elapsed_ms": elapsed_ms,
        }
        if not ok:
            item["error"] = sanitize(stderr or stdout or f"curl exit {code}")
            item["exit_code"] = code
        return item

    def run_telegram_pulse_once(self) -> dict[str, Any] | None:
        if not self.pulse_enabled():
            return None
        channels = self.pulse_channels()
        if not channels:
            return None

        results = [self.probe_telegram_pulse_channel(channel) for channel in channels]
        ok_count = sum(1 for item in results if item.get("ok"))
        if ok_count == len(results):
            status = OK
            state = "ok"
        elif ok_count:
            status = WARN
            state = "partial"
        else:
            status = CRIT
            state = "down"
        sample = {
            "timestamp": now_iso(),
            "status": status,
            "state": state,
            "ok_count": ok_count,
            "channel_count": len(results),
            "channels": results,
        }
        self.append_telegram_pulse(sample)
        self.record_state("last_telegram_pulse", {
            "at": sample["timestamp"],
            "status": status,
            "state": state,
            "ok_count": ok_count,
            "channel_count": len(results),
        })
        self.trim_telegram_pulse_history_if_due()
        return sample

    def append_telegram_pulse(self, sample: dict[str, Any]) -> None:
        with self.pulse_lock:
            with open(self.pulse_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def parse_timestamp(self, value: str) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except (TypeError, ValueError):
            return None

    def load_telegram_pulse_samples(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[dict[str, Any]]:
        if not self.pulse_path.exists():
            return []
        samples: list[dict[str, Any]] = []
        with self.pulse_lock:
            with open(self.pulse_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = self.parse_timestamp(str(sample.get("timestamp") or ""))
                    if not ts:
                        continue
                    if start and ts < start.astimezone(timezone.utc):
                        continue
                    if end and ts >= end.astimezone(timezone.utc):
                        continue
                    sample["_dt"] = ts
                    samples.append(sample)
        return samples

    def trim_telegram_pulse_history_if_due(self) -> None:
        tz = ZoneInfo(str(self.config.get("timezone", "Europe/Samara")))
        today = datetime.now(tz).date().isoformat()
        if (self.state.get("telegram_pulse") or {}).get("last_trim_date") == today:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(7, int(self.pulse_config().get("history_days", 95))))
        with self.pulse_lock:
            samples = self.load_telegram_pulse_samples(start=cutoff)
            tmp = self.pulse_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                for sample in samples:
                    sample.pop("_dt", None)
                    fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
            tmp.replace(self.pulse_path)
        self.record_state("telegram_pulse", {"last_trim_date": today})

    def aggregate_telegram_pulse(self, start: datetime, end: datetime) -> dict[str, Any]:
        coverage = covered_samples(self.load_telegram_pulse_samples(start=start, end=end), start, end, self.pulse_interval())
        samples = coverage.pop("samples")
        interval = self.pulse_interval()
        channels = {}
        counts = {"ok": 0, "partial": 0, "down": 0}
        latencies = []
        for sample in samples:
            counts[sample["state"]] += 1
            for channel in sample["channels"]:
                key = str(channel.get("id") or "unknown")
                stats = channels.setdefault(key, {"id": key, "name": channel.get("name") or key, "samples": 0, "ok": 0, "fail": 0, "latencies": []})
                stats["samples"] += 1
                stats["ok" if channel["ok"] else "fail"] += 1
                delay = channel.get("elapsed_ms")
                if channel["ok"] and isinstance(delay, (int, float)) and delay >= 0:
                    stats["latencies"].append(delay)
                    latencies.append(delay)
        for stats in channels.values():
            delays = stats.pop("latencies")
            stats["ok_percent"] = round(stats["ok"] * 100 / stats["samples"], 3)
            stats["avg_latency_ms"] = int(sum(delays) / len(delays)) if delays else None
        total = len(samples)
        longest = coverage["max_down_streak_samples"] * interval
        return {
            **coverage, "start": start.isoformat(), "end": end.isoformat(), "samples": total,
            "interval_seconds": interval, "all_ok_samples": counts["ok"],
            "partial_samples": counts["partial"], "full_down_samples": counts["down"],
            "partial_seconds": counts["partial"] * interval, "full_down_seconds": counts["down"] * interval,
            "max_down_streak_seconds": longest,
            "disruption_character": self.telegram_pulse_disruption_character(counts["down"], counts["partial"], longest) if total else "нет наблюдений",
            "impact_seconds": (counts["down"] + counts["partial"] * 0.5) * interval,
            "full_down_percent": round(counts["down"] * 100 / total, 3) if total else None,
            "partial_percent": round(counts["partial"] * 100 / total, 3) if total else None,
            "availability_percent": round((total - counts["down"]) * 100 / total, 3) if total else None,
            "avg_latency_ms": int(sum(latencies) / len(latencies)) if latencies else None,
            "channels": sorted(channels.values(), key=lambda item: item["id"]),
        }

    def pulse_periods(self, now_local=None) -> dict[str, tuple[datetime, datetime, datetime, datetime]]:
        tz = ZoneInfo(str(self.config.get("timezone", "Europe/Samara")))
        now_local = now_local or datetime.now(tz)
        result = {}
        for period in ("day", "week", "month"):
            start, end = period_window(period, now_local)
            if period == "day":
                previous_start, previous_end = start - timedelta(days=1), start
            elif period == "week":
                previous_start, previous_end = start - timedelta(days=7), start
            else:
                local_start = start.astimezone(tz)
                previous_start = (local_start - timedelta(days=1)).replace(day=1).astimezone(timezone.utc)
                previous_end = start
            result[period] = (start, end, previous_start, previous_end)
        return result

    def compare_pulse_periods(self, current: dict[str, Any], previous: dict[str, Any]) -> str:
        if (current.get("coverage_percent") or 0) < 95 or (previous.get("coverage_percent") or 0) < 95:
            return "недостаточно полного наблюдения для сравнения"
        if current.get("samples", 0) < 3:
            return "данных пока мало"
        if previous.get("samples", 0) < 3:
            return "нет достаточной базы для сравнения"

        current_observed = max(1, int(current.get("observed_seconds") or 0))
        previous_observed = max(1, int(previous.get("observed_seconds") or 0))
        current_rate = float(current.get("impact_seconds") or 0) / current_observed
        previous_rate = float(previous.get("impact_seconds") or 0) / previous_observed
        delta_rate = current_rate - previous_rate
        delta_seconds = (current_rate - previous_rate) * current_observed
        abs_delta_minutes = abs(delta_seconds) / 60
        abs_delta_points = abs(delta_rate) * 100

        if abs_delta_minutes < 5 and abs_delta_points < 0.5:
            return "без ощутимых изменений"

        if delta_rate > 0:
            if current_rate >= 0.5 or (current.get("full_down_percent") or 0) >= 25:
                return "критическое нарастание сбоев"
            if abs_delta_minutes >= 60 or abs_delta_points >= 10:
                return "значительное нарастание сбоев"
            if abs_delta_minutes >= 15 or abs_delta_points >= 2:
                return "существенное нарастание сбоев"
            return "незначительное нарастание сбоев"

        if abs_delta_minutes >= 15 or abs_delta_points >= 2:
            return "существенное снижение сбоев"
        return "несущественное снижение сбоев"

    def human_duration(self, seconds: float | int | None) -> str:
        if not seconds:
            return "0 мин"
        total = int(seconds)
        if total < 60:
            return f"{total} сек"
        minutes = total // 60
        hours = minutes // 60
        minutes = minutes % 60
        if hours:
            return f"{hours} ч {minutes} мин"
        return f"{minutes} мин"

    def human_percent(self, value: Any) -> str:
        if value is None:
            return "n/a"
        try:
            percent = float(value)
        except (TypeError, ValueError):
            return str(value)
        text = f"{percent:.1f}"
        if text.endswith(".0"):
            text = text[:-2]
        return f"{text}%"

    def telegram_pulse_disruption_character(
        self,
        full_down_samples: int,
        partial_samples: int,
        max_down_streak_seconds: int,
    ) -> str:
        if full_down_samples <= 0:
            if partial_samples > 0:
                return "частичные сбои без потери связи"
            return "сбоев не было"

        if max_down_streak_seconds <= 2 * 60:
            return "разрозненные короткие сбои"
        if max_down_streak_seconds <= 5 * 60:
            return "серии коротких сбоев"
        if max_down_streak_seconds <= 15 * 60:
            return "сбои средней длительности"
        if max_down_streak_seconds <= 60 * 60:
            return "продолжительные сбои"
        return "полный обрыв связи"

    def is_sunday(self) -> bool:
        tz = ZoneInfo(str(self.config.get("timezone", "Europe/Samara")))
        return datetime.now(tz).weekday() == 6

    def is_last_day_of_month(self) -> bool:
        tz = ZoneInfo(str(self.config.get("timezone", "Europe/Samara")))
        tomorrow = datetime.now(tz).date() + timedelta(days=1)
        return tomorrow.day == 1

    def telegram_pulse_snapshot(self, now_local=None) -> dict[str, Any]:
        periods = self.pulse_periods(now_local)
        day = self.aggregate_telegram_pulse(*periods["day"][:2])
        previous_day = self.aggregate_telegram_pulse(*periods["day"][2:])
        week = self.aggregate_telegram_pulse(*periods["week"][:2])
        previous_week = self.aggregate_telegram_pulse(*periods["week"][2:])
        month = self.aggregate_telegram_pulse(*periods["month"][:2])
        previous_month = self.aggregate_telegram_pulse(*periods["month"][2:])
        current = dict(self.state.get("last_telegram_pulse") or {})
        last = self.parse_timestamp(current.get("at"))
        if not last or not 0 <= (datetime.now(timezone.utc) - last).total_seconds() <= 3 * self.pulse_interval():
            current.update(status=WARN, state="unknown", fresh=False)
        else:
            current["fresh"] = True
        return {"timestamp": now_iso(), "current": current, "today": day, "yesterday_same_time": previous_day,
                "week_to_date": week, "previous_week_same_time": previous_week,
                "month_to_date": month, "previous_month_same_time": previous_month,
                "day_trend": self.compare_pulse_periods(day, previous_day),
                "week_trend": self.compare_pulse_periods(week, previous_week),
                "month_trend": self.compare_pulse_periods(month, previous_month)}

    def format_telegram_pulse_line(self, include_calendar: bool = True) -> str:
        if not self.pulse_enabled():
            return "Связь с Telegram: наблюдение выключено."
        snapshot = self.telegram_pulse_snapshot()
        day = snapshot["today"]
        fresh = "нет свежих данных" if not snapshot["current"].get("fresh") else self.status_mark(snapshot["current"]["status"])
        return (f"Telegram API: сейчас {fresh}; связь сохранилась в {self.human_percent(day['availability_percent'])} проверок; "
                f"наблюдений {day['samples']} из {day['expected_samples']}, покрытие {self.human_percent(day['coverage_percent'])}; "
                f"оба входа недоступны: {day['full_down_samples']}; один: {day['partial_samples']}. Доставка сообщений отдельно не проверяется.")

    def format_telegram_pulse_report(self, now_local=None,period="day") -> str:
        if not self.pulse_enabled():
            return "Связь с Telegram\nНаблюдение выключено."
        snapshot = self.telegram_pulse_snapshot(now_local)
        current = snapshot["current"]
        day = snapshot["today"]
        tz = ZoneInfo(str(self.config.get("timezone", "Europe/Samara")))
        if period != "day":
            anchor=now_local or datetime.now(tz)
            first=min((s["_dt"] for s in self.load_telegram_pulse_samples()),default=anchor) if period=="all" else anchor
            start,end=period_window(period,anchor,first)
            day=self.aggregate_telegram_pulse(start,min(end,datetime.now(timezone.utc)))
        stamp = lambda value: datetime.fromisoformat(value).astimezone(tz).strftime("%d.%m.%Y %H:%M:%S")
        state = "нет свежих данных" if not current.get("fresh") else self.status_mark(current["status"])
        lines = ["Связь с Telegram", f"Период: {stamp(day['start'])} — {stamp(day['end'])}, Самара",
                 f"Сейчас: {state}", f"Последняя проверка: {stamp(current['at']) if current.get('at') else 'отсутствует'}", "",
                 "──── За период ────", f"Выполнено проверок: {day['samples']}",
                 f"Покрытие наблюдениями: {self.human_percent(day['coverage_percent'])}",
                 f"Пропущено минутных наблюдений: {day['missing_samples']}",
                 f"Связь сохранилась: {self.human_percent(day['availability_percent'])} выполненных проверок",
                 f"Оба входа недоступны: {day['full_down_samples']} проверок",
                 f"Один вход недоступен: {day['partial_samples']} проверок",
                 f"Самый длинный наблюдаемый сбой: около {self.human_duration(day['max_down_streak_seconds'])}",
                 f"Средняя задержка: {day['avg_latency_ms'] if day['avg_latency_ms'] is not None else 'нет данных'} мс", "",
                 "──── Проверяемые входы ────"]
        latest_samples = self.load_telegram_pulse_samples()
        latest = {item.get('id'): item for item in latest_samples[-1].get('channels', [])} if latest_samples else {}
        for channel in day['channels']:
            last = latest.get(channel['id'])
            state = 'нет свежих данных' if not current.get('fresh') or not last else 'доступен' if last['ok'] else 'недоступен'
            lines.append(f"{channel['name']}: {state}, ошибок {channel['fail']} из {channel['samples']}")
        lines.extend(["", "──── Динамика ────", f"За сутки: {snapshot['day_trend']}",
                      f"За неделю: {snapshot['week_trend']}", f"За месяц: {snapshot['month_trend']}", "",
                      "Проверяется доступность Telegram API. Доставка сообщений этим тестом не подтверждается.",
                      "HTTP и SOCKS — два входа одного sing-box. Длительность сбоев оценена по наблюдаемым минутам; пропуски не считаются успехом."])
        return "\n".join(lines)

    def telegram_pulse_loop(self) -> None:
        while not self.stop_event.is_set():
            self.heartbeats["pulse"] = time.monotonic()
            if self.pulse_enabled():
                try:
                    self.run_telegram_pulse_once()
                except Exception:
                    self.record_state("last_telegram_pulse_error", {"at": now_iso(), "error": "pulse_collection_failed"})
            self.stop_event.wait(self.pulse_interval())

    def queue_report(self, text: str, reason: str, target: dict[str, str] | None = None, report_id=None, start=None, end=None) -> None:
        target = target or self.primary_report_target() or {}
        chat_id = target.get("chat_id") or self.config.get("telegram", {}).get("chat_id")
        if not chat_id:
            self.record_state("delivery_error", {"at": now_iso(), "reason": "target_unavailable"})
            return
        self.outbox.enqueue(report_id or "event:" + uuid.uuid4().hex, str(chat_id), split_message(text), start, end)

    def flush_pending(self) -> None:
        summary = self.outbox.flush(self.telegram.send_chunk)
        self.record_state("outbox", summary)

    def inspect_container(self, container: str) -> dict[str, Any] | None:
        code, stdout, _stderr = run_cmd(["docker", "inspect", container], timeout=12)
        if code != 0:
            return None
        try:
            data = json.loads(stdout)
            return data[0] if data else None
        except json.JSONDecodeError:
            return None

    def check_containers(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for service in self.config.get("containers", []):
            title = service["name"]
            container = service["container"]
            info = self.inspect_container(container)
            if not info:
                items.append(
                    check(
                        CRIT,
                        f"container:{service['key']}",
                        title,
                        f"{container}: container not found",
                        data={"container": container, "running": False, "found": False},
                    )
                )
                continue

            state = info.get("State", {})
            running = bool(state.get("Running"))
            health = (state.get("Health") or {}).get("Status")
            restart_count = info.get("RestartCount", 0)
            data = {
                "container": container,
                "found": True,
                "running": running,
                "health": health,
                "restart_count": restart_count,
            }
            details = [f"container={container}", f"restart_count={restart_count}"]
            if health:
                details.append(f"health={health}")

            if not running:
                status = CRIT
                summary = "container is not running"
            elif health == "unhealthy":
                status = CRIT
                summary = "container health is unhealthy"
            elif health == "starting":
                status = WARN
                summary = "container health is still starting"
            else:
                status = OK
                summary = "running" + (f", health={health}" if health else "")
            items.append(check(status, f"container:{service['key']}", title, summary, details, data=data))
        return items

    def check_tcp(self, host: str, port: int) -> dict[str, Any]:
        started = time.monotonic()
        try:
            with socket.create_connection((host, port), timeout=5):
                elapsed = int((time.monotonic() - started) * 1000)
                return check(
                    OK,
                    "network:singbox_tcp",
                    "sing-box DNS/TCP",
                    f"{host}:{port} reachable in {elapsed} ms",
                    data={"host": host, "port": port, "elapsed_ms": elapsed},
                )
        except OSError as exc:
            return check(
                CRIT,
                "network:singbox_tcp",
                "sing-box DNS/TCP",
                f"{host}:{port} failed: {sanitize(str(exc))}",
                data={"host": host, "port": port, "error": sanitize(str(exc))},
            )

    def check_telegram_proxy(self, proxy: str) -> dict[str, Any]:
        title = f"Telegram API через {proxy}"
        failures: list[str] = []
        for attempt in range(1, 4):
            code, stdout, stderr = run_cmd(
                [
                    "curl",
                    "-sS",
                    "-o",
                    "/dev/null",
                    "-w",
                    "%{http_code}",
                    "-x",
                    proxy,
                    "--max-time",
                    "15",
                    "https://api.telegram.org",
                ],
                timeout=20,
            )
            if code == 0 and stdout.strip() in {"200", "301", "302", "404"}:
                suffix = "" if attempt == 1 else f" after {attempt} attempts"
                return check(
                    OK,
                    f"network:telegram:{proxy}",
                    title,
                    f"HTTP {stdout.strip()}{suffix}",
                    failures[-2:],
                    data={"proxy": proxy, "http_status": stdout.strip(), "attempts": attempt},
                )
            failures.append(f"attempt {attempt}: {sanitize(stderr or stdout)}")
            time.sleep(1)
        detail = "; ".join(failures[-3:])
        return check(
            CRIT,
            f"network:telegram:{proxy}",
            title,
            f"failed: {detail}",
            data={"proxy": proxy, "error": detail},
        )

    def check_api_json(self, key: str, title: str, url: str, parser: str | None = None) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(url, timeout=8) as response:
                body = response.read(500_000)
                status = response.status
            if status >= 400:
                return check(CRIT, key, title, f"HTTP {status}", data={"http_status": status})
            summary = f"HTTP {status}"
            details: list[str] = []
            data: dict[str, Any] = {"http_status": status}
            if parser == "ollama":
                payload = json.loads(body.decode("utf-8"))
                models = payload.get("models", [])
                summary = f"HTTP {status}, models={len(models)}"
                data.update(
                    {
                        "models_count": len(models),
                        "models": [str(model.get("name", "")) for model in models[:5] if model.get("name")],
                    }
                )
                if models:
                    details.append("models=" + ", ".join(data["models"]))
            elif parser == "whisper":
                payload = json.loads(body.decode("utf-8"))
                data.update(
                    {
                        "status": payload.get("status"),
                        "model": payload.get("model"),
                        "device": payload.get("device"),
                    }
                )
                summary = (
                    f"HTTP {status}, status={data.get('status')}, "
                    f"model={data.get('model')}, device={data.get('device')}"
                )
            return check(OK, key, title, summary, details, data=data)
        except urllib.error.HTTPError as exc:
            return check(CRIT if exc.code >= 500 else WARN, key, title, f"HTTP {exc.code}", data={"http_status": exc.code})
        except Exception as exc:  # noqa: BLE001 - probe failure should become monitor data.
            return check(CRIT, key, title, f"failed: {sanitize(str(exc))}", data={"error": sanitize(str(exc))})

    def check_ollama_gpu(self) -> dict[str, Any]:
        container = next(
            (
                service.get("container")
                for service in self.config.get("containers", [])
                if service.get("key") == "ollama"
            ),
            "ollama",
        )
        code, stdout, stderr = run_cmd(
            [
                "docker",
                "exec",
                str(container),
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=12,
        )
        if code != 0:
            return check(
                CRIT,
                "api:ollama_gpu",
                "Ollama GPU",
                f"GPU недоступна в контейнере Ollama: {sanitize(stderr or stdout)}",
                data={"container": container, "error": sanitize(stderr or stdout)},
            )

        gpus: list[dict[str, Any]] = []
        for line in stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                gpus.append(
                    {
                        "name": parts[0],
                        "utilization": int(float(parts[1])),
                        "memory_used_mb": int(float(parts[2])),
                        "memory_total_mb": int(float(parts[3])),
                        "temperature_c": int(float(parts[4])),
                    }
                )
            except ValueError:
                continue

        if not gpus:
            return check(
                CRIT,
                "api:ollama_gpu",
                "Ollama GPU",
                "nvidia-smi не вернул список GPU внутри контейнера Ollama",
                data={"container": container, "raw": sanitize(stdout)},
            )

        total_used = sum(gpu["memory_used_mb"] for gpu in gpus)
        total_memory = sum(gpu["memory_total_mb"] for gpu in gpus)
        max_util = max(gpu["utilization"] for gpu in gpus)
        max_temp = max(gpu["temperature_c"] for gpu in gpus)
        names = sorted({str(gpu["name"]) for gpu in gpus})
        summary = (
            f"GPU доступна: {len(gpus)} шт., "
            f"VRAM {total_used / 1024:.1f}/{total_memory / 1024:.1f} GB, "
            f"max util {max_util}%, max temp {max_temp}C"
        )
        return check(
            OK,
            "api:ollama_gpu",
            "Ollama GPU",
            summary,
            details=[", ".join(names)],
            data={
                "container": container,
                "gpu_count": len(gpus),
                "names": names,
                "memory_used_mb": total_used,
                "memory_total_mb": total_memory,
                "max_utilization": max_util,
                "max_temperature_c": max_temp,
            },
        )

    def check_aggregator_login(self, url: str) -> dict[str, Any]:
        try:
            host_header = self.config.get("api_checks", {}).get("aggregator_host_header")
            request = urllib.request.Request(url, headers={"Host": host_header} if host_header else {})
            with urllib.request.urlopen(request, timeout=8) as response:
                status = response.status
                content_type = response.headers.get("Content-Type", "")
            if status == 200:
                return check(
                    OK,
                    "api:aggregator_web",
                    "Aggregator web-панель",
                    f"login page HTTP 200, {content_type}",
                    data={"http_status": status, "content_type": content_type},
                )
            return check(
                WARN,
                "api:aggregator_web",
                "Aggregator web-панель",
                f"unexpected HTTP {status}",
                data={"http_status": status, "content_type": content_type},
            )
        except Exception as exc:  # noqa: BLE001
            return check(CRIT, "api:aggregator_web", "Aggregator web-панель", f"failed: {sanitize(str(exc))}", data={"error": sanitize(str(exc))})

    def check_openclaw_probe(self) -> dict[str, Any]:
        code, stdout, stderr = run_cmd(
            [
                "docker",
                "exec",
                "openclaw-openclaw-gateway-1",
                "node",
                "dist/index.js",
                "channels",
                "status",
                "--probe",
                "--json",
            ],
            timeout=60,
        )
        if code != 0:
            return check(
                CRIT,
                "openclaw:telegram_probe",
                "OpenClaw Telegram probe",
                sanitize(stderr or stdout),
                data={"error": sanitize(stderr or stdout)},
            )
        try:
            data = decode_probe_output(stdout)
            telegram = self.find_telegram_status(data)
            if not telegram:
                return check(WARN, "openclaw:telegram_probe", "OpenClaw Telegram probe", "telegram status not found")
            running = telegram.get("running")
            if not isinstance(running, bool):
                running = None
            probe = telegram.get("probe") or {}
            probe_ok = probe.get("ok")
            last_error = telegram.get("lastError") or probe.get("error")
            item_data = {
                "running": running,
                "probe_ok": probe_ok,
                "last_error": sanitize(str(last_error)) if last_error else "",
            }
            if running and probe_ok is True and not last_error:
                return check(
                    OK,
                    "openclaw:telegram_probe",
                    "OpenClaw Telegram probe",
                    "running=true, probe.ok=true",
                    data=item_data,
                )
            status = CRIT if running is False or probe_ok is False else WARN
            return check(
                status,
                "openclaw:telegram_probe",
                "OpenClaw Telegram probe",
                f"running={running}, probe.ok={probe_ok}, last_error={sanitize(str(last_error or 'none'))}",
                data=item_data,
            )
        except (json.JSONDecodeError, ValueError):
            return check(WARN, "openclaw:telegram_probe", "OpenClaw Telegram probe", "could not parse JSON output")

    def maybe_restart_openclaw_telegram(self, probe_item: dict[str, Any]) -> dict[str, Any] | None:
        remediation_config = self.config.get("remediation", {})
        action_config = remediation_config.get("openclaw_telegram_restart", {})
        state = dict(self.state.get("remediation") or {})
        data = probe_item.get("data") or {}

        if data.get("running") is not False:
            if state.get("openclaw_telegram_failures"):
                state["openclaw_telegram_failures"] = 0
                self.record_state("remediation", state)
            return None
        if not remediation_config.get("enabled", False) or not action_config.get("enabled", True):
            return None

        failures = int(state.get("openclaw_telegram_failures") or 0) + 1
        state["openclaw_telegram_failures"] = failures
        threshold = max(1, int(action_config.get("failure_threshold", 1)))
        if failures < threshold:
            self.record_state("remediation", state)
            return check(
                WARN,
                "remediation:openclaw_telegram_restart",
                "Автовосстановление OpenClaw Telegram",
                f"Telegram polling остановлен; наблюдение {failures}/{threshold} перед рестартом",
                data={"failures": failures, "threshold": threshold, "action": "wait"},
            )

        now = time.time()
        min_interval = max(60, int(action_config.get("min_interval_seconds", 900)))
        last_restart = float(state.get("last_openclaw_telegram_restart_ts") or 0)
        if last_restart and now - last_restart < min_interval:
            remaining = int(min_interval - (now - last_restart))
            self.record_state("remediation", state)
            return check(
                WARN,
                "remediation:openclaw_telegram_restart",
                "Автовосстановление OpenClaw Telegram",
                f"Telegram polling остановлен, но cooldown рестарта еще {remaining} сек.",
                data={
                    "failures": failures,
                    "min_interval_seconds": min_interval,
                    "remaining_seconds": remaining,
                    "action": "cooldown",
                },
            )

        container = str(action_config.get("container") or "openclaw-openclaw-gateway-1")
        code, stdout, stderr = run_cmd(["docker", "restart", container], timeout=60)
        state["last_openclaw_telegram_restart_ts"] = now
        state["last_openclaw_telegram_restart_at"] = now_iso()
        state["openclaw_telegram_failures"] = 0 if code == 0 else failures
        self.record_state("remediation", state)
        if code != 0:
            return check(
                CRIT,
                "remediation:openclaw_telegram_restart",
                "Автовосстановление OpenClaw Telegram",
                f"не удалось перезапустить {container}: {sanitize(stderr or stdout)}",
                data={"container": container, "action": "restart_failed", "exit_code": code},
            )
        return check(
            WARN,
            "remediation:openclaw_telegram_restart",
            "Автовосстановление OpenClaw Telegram",
            f"перезапущен {container}: Telegram polling был остановлен",
            details=[f"до рестарта: {probe_item.get('summary', '')}"],
            data={"container": container, "action": "restarted"},
        )

    def find_telegram_status(self, value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            channels = value.get("channels")
            if isinstance(channels, dict) and isinstance(channels.get("telegram"), dict):
                return channels["telegram"]
            if isinstance(value.get("telegram"), dict):
                return value["telegram"]
            if (value.get("id") == "telegram" or value.get("type") == "telegram") and (
                "running" in value or "probe" in value
            ):
                return value
            for nested in value.values():
                found = self.find_telegram_status(nested)
                if found:
                    return found
        elif isinstance(value, list):
            for nested in value:
                found = self.find_telegram_status(nested)
                if found:
                    return found
        return None

    def check_postgres_stats(self) -> dict[str, Any]:
        stats_config = self.config.get("postgres_stats", {})
        if not stats_config.get("enabled", True):
            return check(OK, "aggregator:stats", "Статистика агрегатора", "выключено")
        container = stats_config.get("container", "support-messenger-aggregator-postgres-1")
        tz = str(self.config.get("timezone", "Europe/Samara")).replace("'", "''")
        sql = f"""
WITH bounds AS (
  SELECT date_trunc('day', timezone('{tz}', now())) AS local_start
)
SELECT key, value::text
FROM (
  SELECT 'conversations_today' AS key, count(*) AS value
    FROM core_conversation c, bounds
    WHERE timezone('{tz}', c.started_at) >= bounds.local_start
  UNION ALL
  SELECT 'messages_today', count(*)
    FROM core_message m, bounds
    WHERE timezone('{tz}', m.created_at) >= bounds.local_start
  UNION ALL
  SELECT 'user_messages_today', count(*)
    FROM core_message m, bounds
    WHERE m.sender = 'user' AND timezone('{tz}', m.created_at) >= bounds.local_start
  UNION ALL
  SELECT 'bot_messages_today', count(*)
    FROM core_message m, bounds
    WHERE m.sender = 'bot' AND timezone('{tz}', m.created_at) >= bounds.local_start
  UNION ALL
  SELECT 'operator_messages_today', count(*)
    FROM core_message m, bounds
    WHERE m.sender = 'operator' AND timezone('{tz}', m.created_at) >= bounds.local_start
  UNION ALL
  SELECT 'new_clients_today', count(*)
    FROM core_channelidentity ci, bounds
    WHERE timezone('{tz}', ci.created_at) >= bounds.local_start
  UNION ALL
  SELECT 'clients_with_new_conversations_today', count(DISTINCT c.channel_identity_id)
    FROM core_conversation c, bounds
    WHERE timezone('{tz}', c.started_at) >= bounds.local_start
  UNION ALL
  SELECT 'unique_clients_today', count(DISTINCT ci.external_id)
    FROM core_message m
    JOIN core_conversation c ON c.id = m.conversation_id
    JOIN core_channelidentity ci ON ci.id = c.channel_identity_id,
    bounds
    WHERE m.sender = 'user' AND timezone('{tz}', m.created_at) >= bounds.local_start
  UNION ALL
  SELECT 'waiting_now', count(*) FROM core_conversation WHERE status = 'waiting_for_operator'
  UNION ALL
  SELECT 'operator_now', count(*) FROM core_conversation WHERE status = 'operator'
  UNION ALL
  SELECT 'bot_now', count(*) FROM core_conversation WHERE status = 'bot'
  UNION ALL
  SELECT 'closed_today', count(*)
    FROM core_conversation c, bounds
    WHERE c.status = 'closed'
      AND c.closed_at IS NOT NULL
      AND timezone('{tz}', c.closed_at) >= bounds.local_start
) s
ORDER BY key;
"""
        code, stdout, stderr = run_cmd(
            [
                "docker",
                "exec",
                "-i",
                container,
                "sh",
                "-lc",
                'psql -X -qAt -F "\t" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f -',
            ],
            timeout=20,
            input_text=sql,
        )
        if code != 0:
            return check(CRIT, "aggregator:stats", "Статистика агрегатора", sanitize(stderr or stdout))
        stats: dict[str, str] = {}
        for line in stdout.splitlines():
            if "\t" in line:
                key, value = line.split("\t", 1)
                stats[key] = value
        summary = (
            f"новые диалоги сегодня: {stats.get('conversations_today', 'n/a')}, "
            f"новые клиенты сегодня: {stats.get('new_clients_today', 'n/a')}, "
            f"клиенты, писавшие сегодня: {stats.get('unique_clients_today', 'n/a')}, "
            f"сообщения сегодня: {stats.get('messages_today', 'n/a')}, "
            f"ожидают оператора: {stats.get('waiting_now', 'n/a')}"
        )
        details = [
            f"сообщения клиентов сегодня: {stats.get('user_messages_today', 'n/a')}",
            f"сообщения бота сегодня: {stats.get('bot_messages_today', 'n/a')}",
            f"сообщения операторов сегодня: {stats.get('operator_messages_today', 'n/a')}",
            f"сейчас в режиме бота: {stats.get('bot_now', 'n/a')}",
            f"сейчас у оператора: {stats.get('operator_now', 'n/a')}",
            f"закрыто сегодня: {stats.get('closed_today', 'n/a')}",
        ]
        status = WARN if stats.get("waiting_now") not in {None, "0"} else OK
        return check(status, "aggregator:stats", "Статистика агрегатора", summary, details, data=stats)

    def check_logs(self) -> list[dict[str, Any]]:
        logs_config = self.config.get("logs", {})
        since = str(logs_config.get("since", "24h"))
        tail = str(logs_config.get("tail", "500"))
        items: list[dict[str, Any]] = []
        for service in self.config.get("containers", []):
            container = service["container"]
            key = f"logs:{service['key']}"
            code, stdout, stderr = run_cmd(
                ["docker", "logs", "--since", since, "--tail", tail, container],
                timeout=20,
            )
            title = f"Логи: {service['name']}"
            if code != 0:
                items.append(
                    check(
                        WARN,
                        key,
                        title,
                        "Журналы недоступны",
                        data={"container": container, "since": since, "error": "logs_unavailable"},
                    )
                )
                continue
            matches = 0
            critical = False
            for line in (stdout + "\n" + stderr).splitlines():
                if ERROR_RE.search(line):
                    matches += 1
                    critical = critical or bool(CRITICAL_LOG_RE.search(line))
            if not matches:
                items.append(
                    check(
                        OK,
                        key,
                        title,
                        f"no error-like lines in last {since}",
                        data={"container": container, "since": since, "matches": 0, "critical": False},
                    )
                )
            else:
                status = CRIT if critical else WARN
                items.append(
                    check(
                        status,
                        key,
                        title,
                        f"{matches} error-like lines in last {since}",
                        data={"container": container, "since": since, "matches": matches, "critical": critical},
                    )
                )
        return items

    def run_checks(self) -> dict[str, Any]:
        with self.lock:
            started = time.monotonic()
            checks: list[dict[str, Any]] = []
            checks.extend(self.check_containers())
            checks.append(self.check_host_metrics())

            network = self.config.get("network_checks", {})
            checks.append(self.check_tcp(network.get("singbox_host", "singbox"), int(network.get("singbox_port", 1080))))
            for proxy in network.get("telegram_proxies", []):
                checks.append(self.check_telegram_proxy(proxy))

            api = self.config.get("api_checks", {})
            checks.append(self.check_api_json("api:ollama", "Ollama API", api["ollama_tags_url"], parser="ollama"))
            checks.append(self.check_ollama_gpu())
            checks.append(self.check_api_json("api:whisper", "Whisper API", api["whisper_health_url"], parser="whisper"))
            checks.append(self.check_aggregator_login(api["aggregator_login_url"]))
            openclaw_probe = self.check_openclaw_probe()
            remediation_item = self.maybe_restart_openclaw_telegram(openclaw_probe)
            if remediation_item and (remediation_item.get("data") or {}).get("action") == "restarted":
                time.sleep(max(0, int(self.config.get("remediation", {}).get("openclaw_telegram_restart", {}).get("startup_wait_seconds", 12))))
                refreshed_probe = self.check_openclaw_probe()
                refreshed_probe["details"] = [
                    f"до автовосстановления: {openclaw_probe.get('summary', '')}",
                    *refreshed_probe.get("details", []),
                ]
                openclaw_probe = refreshed_probe
            checks.append(openclaw_probe)
            if remediation_item:
                checks.append(remediation_item)
            checks.append(self.check_postgres_stats())
            if self.config.get("appeal_reporting",{}).get("enabled",False):
                checks.append(self.check_appeal_collector())
            checks.extend(self.check_logs())

            result = {
                "timestamp": now_iso(),
                "status": worst_status(checks),
                "duration_ms": int((time.monotonic() - started) * 1000),
                "checks": checks,
            }
            self.last_result = result
            self.append_history(result)
            return result

    def check_host_metrics(self):
        item={"key":"host:resources","title":"Ресурсы сервера","status":WARN,"summary":"Нет свежих измерений CPU, памяти и дисков.","details":[],"data":{}}
        try:
            source=Path("/host-metrics/host.json")
            if source.stat().st_size>64000: raise ValueError
            data=validate_host_metrics(json.loads(source.read_text()))
            age=(datetime.now(timezone.utc)-datetime.fromisoformat(data["checked_at"])).total_seconds()
            if not 0<=age<=150: return item
            measured=all(data[k] is not None for k in ("cpu_percent","cpu_temperature","memory_total","memory_available","filesystem_total","filesystem_free"))
            measured=measured and bool(data["nvmes"]) and all(d["passed"] is not None and d["temperature"] is not None for d in data["nvmes"])
            critical=any(d["passed"] is False or (d["critical_warning"] or 0)>0 for d in data["nvmes"])
            item.update(data=data,status=CRIT if critical else OK if measured else WARN,
                summary="SMART сообщает неисправность диска." if critical else "Измерения свежие; SMART штатный." if measured else "Часть аппаратных измерений отсутствует.")
        except (OSError,ValueError,TypeError,KeyError): pass
        return item

    def format_host_metrics(self,item):
        data=item.get("data") or {}
        if not data: return [self.status_mark(WARN)+" Нет свежих измерений ресурсов сервера."]
        show=lambda v:"нет данных" if v is None else f"{v:g}"
        gib=lambda v:"нет данных" if v is None else f"{v/1024**3:.1f} GiB"
        lines=[self.status_mark(item["status"])+" "+item["summary"],"Измерено: "+data["checked_at"],
            f"CPU: {show(data['cpu_percent'])}%; логических ядер {show(data['cpu_count'])}; максимальная температура датчиков CPU: {show(data['cpu_temperature'])} °C.",
            f"RAM: доступно {gib(data['memory_available'])} из {gib(data['memory_total'])}.",
            f"Системный раздел: свободно {gib(data['filesystem_free'])} из {gib(data['filesystem_total'])}."]
        for disk in data["nvmes"]:
            health="штатно" if disk["passed"] is True else "ошибка" if disk["passed"] is False else "нет данных"
            lines.append(f"NVMe {disk['id']}: SMART {health}; температура {show(disk['temperature'])} °C; износ {show(disk['percentage_used'])}%; ошибок носителя {show(disk['media_errors'])}.")
        lines.append("Температурные пороги и автоматические аппаратные реакции ещё не утверждены.")
        return lines

    def check_appeal_collector(self):
        code,out,_=run_cmd(["docker","exec","reporting-worker","python","manage.py","reporting_snapshot","--period","day"],timeout=30)
        try:
            if code: raise ValueError
            data=validate_snapshot(json.loads(out))
            fresh=data["enabled"] and data["status"]=="fresh"
            return {"key":"reporting:collector","title":"Учёт обращений","status":OK if fresh else WARN,
                    "summary":"Сборщик работает; есть свежая отметка." if fresh else "Нет свежего подтверждения работы сборщика.","data":{},"details":[]}
        except (ValueError,TypeError,KeyError):
            return {"key":"reporting:collector","title":"Учёт обращений","status":WARN,
                    "summary":"Учёт обращений недоступен.","data":{},"details":[]}

    def status_mark(self, status: str) -> str:
        return f"{STATUS_ICONS.get(status, '?')} {STATUS_LABELS.get(status, str(status).upper())}"

    def ru_summary(self, item: dict[str, Any]) -> str:
        summary = str(item.get("summary", "")).strip()
        replacements = [
            ("container not found", "контейнер не найден"),
            ("container is not running", "контейнер не запущен"),
            ("container health is unhealthy", "Docker healthcheck сообщает ошибку"),
            ("container health is still starting", "Docker healthcheck еще стартует"),
            ("running, health=healthy", "контейнер запущен, healthcheck здоров"),
            ("running", "контейнер запущен"),
            ("telegram status not found", "статус Telegram в OpenClaw не найден"),
            ("could not parse JSON output", "не удалось разобрать JSON-ответ"),
            ("login page HTTP 200", "страница входа доступна, HTTP 200"),
            ("unexpected HTTP", "неожиданный HTTP"),
            ("failed:", "ошибка:"),
            ("disabled", "выключено"),
        ]
        for source, target in replacements:
            summary = summary.replace(source, target)
        return summary

    def container_state_text(self, item: dict[str, Any]) -> str:
        data = item.get("data") or {}
        if data.get("found") is False:
            return f"контейнер {data.get('container', item['title'])} не найден."
        if data:
            container = data.get("container", item["title"])
            restart_count = data.get("restart_count")
            if not data.get("running"):
                suffix = f"; перезапусков: {restart_count}" if restart_count is not None else ""
                return f"контейнер {container} не запущен{suffix}."
            parts = ["контейнер запущен"]
            health = data.get("health")
            if health == "healthy":
                parts.append("healthcheck здоров")
            elif health == "starting":
                parts.append("healthcheck еще стартует")
            elif health == "unhealthy":
                parts.append("healthcheck сообщает ошибку")
            elif health:
                parts.append(f"healthcheck: {health}")
            if restart_count is not None:
                parts.append(f"перезапусков: {restart_count}")
            return ", ".join(parts) + "."
        return self.ru_summary(item)

    def human_result(self, item: dict[str, Any]) -> str:
        key = str(item.get("key", ""))
        data = item.get("data") or {}
        if key.startswith("container:"):
            return self.container_state_text(item)
        if key == "network:singbox_tcp":
            if item["status"] == OK:
                return (
                    f"имя {data.get('host', 'singbox')} резолвится внутри Docker, "
                    f"порт {data.get('port', 1080)} доступен за {data.get('elapsed_ms', '?')} мс."
                )
            return f"нет TCP-доступа к {data.get('host', 'singbox')}:{data.get('port', 1080)}: {data.get('error') or self.ru_summary(item)}."
        if key.startswith("network:telegram:"):
            proxy = data.get("proxy") or item["title"].replace("Telegram API через ", "")
            if item["status"] == OK:
                attempts = int(data.get("attempts") or 1)
                suffix = "" if attempts == 1 else f" после {attempts} попыток"
                return f"Telegram API отвечает через {proxy}: HTTP {data.get('http_status', '?')}{suffix}."
            return f"Telegram API недоступен через {proxy}: {data.get('error') or self.ru_summary(item)}."
        if key == "openclaw:telegram_probe":
            if item["status"] == OK:
                return "polling Telegram запущен, проверка OpenClaw Telegram probe успешна."
            last_error = data.get("last_error") or self.ru_summary(item)
            return (
                f"polling={data.get('running', '?')}, "
                f"probe={data.get('probe_ok', '?')}; ошибка: {last_error}."
            )
        if key == "api:ollama":
            if item["status"] == OK:
                return f"Ollama API отвечает HTTP {data.get('http_status', '?')}; список моделей доступен."
            return f"Ollama API не отвечает штатно: {data.get('error') or self.ru_summary(item)}."
        if key == "api:ollama_gpu":
            if item["status"] == OK:
                used_gb = float(data.get("memory_used_mb") or 0) / 1024
                total_gb = float(data.get("memory_total_mb") or 0) / 1024
                names = ", ".join(data.get("names") or [])
                return (
                    f"GPU видна из контейнера Ollama: {data.get('gpu_count', '?')} шт. "
                    f"({names or 'модель GPU не указана'}), VRAM {used_gb:.1f}/{total_gb:.1f} GB, "
                    f"пиковая загрузка сейчас {data.get('max_utilization', '?')}%."
                )
            return f"GPU для Ollama недоступна или не читается: {data.get('error') or self.ru_summary(item)}."
        if key == "api:whisper":
            if item["status"] == OK:
                return (
                    f"Whisper API отвечает HTTP {data.get('http_status', '?')}; "
                    f"модель: {data.get('model') or 'не указана'}, устройство: {data.get('device') or 'не указано'}."
                )
            return f"Whisper API не отвечает штатно: {data.get('error') or self.ru_summary(item)}."
        if key == "api:aggregator_web":
            if item["status"] == OK:
                return f"страница входа web-панели доступна, HTTP {data.get('http_status', '?')}."
            return f"web-панель агрегатора отвечает нештатно: {data.get('error') or self.ru_summary(item)}."
        if key == "aggregator:stats":
            return self.ru_summary(item)
        if key.startswith("logs:"):
            if item["status"] == OK:
                return f"подозрительных ошибок за {data.get('since', '24h')} не найдено."
            count = data.get("matches")
            return f"найдено подозрительных строк: {count if count is not None else '?'}."
        return self.ru_summary(item)

    def format_container_block(self, item: dict[str, Any]) -> list[str]:
        return [
            f"{self.status_mark(item['status'])} {item['title']}",
            f"  Роль: {SERVICE_PURPOSES.get(item['title'], 'сервис AI-сервера.')}",
            f"  Сейчас: {self.human_result(item)}",
        ]

    def format_check_block(self, item: dict[str, Any], purpose: str) -> list[str]:
        return [
            f"{self.status_mark(item['status'])} {item['title']}",
            f"  Что проверяем: {purpose}",
            f"  Результат: {self.human_result(item)}",
        ]

    def short_check_result(self, item: dict[str, Any] | None, ok_text: str, bad_text: str) -> str:
        if not item:
            return f"{self.status_mark(WARN)} {bad_text}: нет данных проверки."
        if item.get("status") == OK:
            return f"{self.status_mark(OK)} {ok_text}"
        return f"{self.status_mark(item.get('status', WARN))} {bad_text}: {self.human_result(item)}"

    def format_telegram_block(self, result: dict[str, Any], by_key: dict[str, dict[str, Any]]) -> list[str]:
        local_proxy = by_key.get("network:singbox_tcp")
        openclaw_probe = by_key.get("openclaw:telegram_probe")
        telegram_proxy_checks = [
            item for item in result.get("checks", []) if item.get("key", "").startswith("network:telegram:")
        ]
        http_proxy = next(
            (item for item in telegram_proxy_checks if "http://singbox:1080" in item.get("key", "")),
            None,
        )
        socks_proxy = next(
            (item for item in telegram_proxy_checks if "socks5h://singbox:1080" in item.get("key", "")),
            None,
        )
        telegram_status = self.aggregate_status([local_proxy, *telegram_proxy_checks, openclaw_probe])

        if openclaw_probe and openclaw_probe.get("status") == OK:
            current = "Telegram API доступен, OpenClaw polling запущен; доставка ответов отдельно не проверена."
        elif self.aggregate_status([local_proxy, *telegram_proxy_checks]) == OK:
            current = "сеть до Telegram есть, но OpenClaw polling работает нештатно."
        else:
            current = "есть проблема с маршрутом до Telegram; бот может отвечать нестабильно."

        return [
            f"{self.status_mark(telegram_status)} Telegram для OpenClaw",
            f"  Сейчас: {current}",
            self.short_check_result(local_proxy, "sing-box внутри Docker доступен.", "sing-box внутри Docker недоступен"),
            self.short_check_result(
                http_proxy,
                "HTTP-вход sing-box для приложений работает.",
                "HTTP-вход sing-box для приложений не проходит Telegram API",
            ),
            self.short_check_result(
                socks_proxy,
                "SOCKS-вход sing-box для приложений работает.",
                "SOCKS-вход sing-box для приложений не проходит Telegram API",
            ),
            self.short_check_result(
                openclaw_probe,
                "OpenClaw polling запущен.",
                "OpenClaw polling не запущен",
            ),
            (
                "  Пояснение: HTTP и SOCKS здесь не два независимых внешних канала, "
                "а два способа подключиться к одному sing-box на порту 1080."
            ),
            (
                "  Маршрут Telegram сейчас идет через SOCKS failover. Если один SOCKS upstream отвалится, "
                "sing-box может переключиться на второй; если отвалятся оба, Telegram станет недоступен. "
                "VLESS/Reality используется для других внешних доменов и сам зависит от SOCKS-underlay."
            ),
            "  HTTP 302 от api.telegram.org в этих проверках считается нормой: это значит, что proxy и TLS-доступ до Telegram работают.",
        ]

    def format_stats_block(self, item: dict[str, Any]) -> list[str]:
        stats = item.get("data") or {}
        lines = [
            f"{self.status_mark(item['status'])} Статистика агрегатора",
            "  Что это: клиентская активность в корпоративном чатботе за текущий день.",
        ]
        if not stats:
            lines.append(f"  Результат: {self.human_result(item)}")
            return lines

        waiting = stats.get("waiting_now", "n/a")
        if item["status"] == OK:
            lines.append("  Сейчас: очереди к оператору нет.")
        else:
            lines.append(f"  Сейчас: обращений в очереди к оператору: {waiting}.")

        labels = [
            ("conversations_today", "Новые диалоги"),
            ("new_clients_today", "Новые клиенты"),
            ("clients_with_new_conversations_today", "Клиенты с новыми диалогами"),
            ("unique_clients_today", "Клиенты, писавшие сегодня"),
            ("messages_today", "Сообщений за сегодня, всего"),
            ("user_messages_today", "От клиентов"),
            ("bot_messages_today", "От бота"),
            ("operator_messages_today", "От операторов"),
            ("waiting_now", "Открытые диалоги сейчас, включая старые: ждут оператора"),
            ("operator_now", "Открытые диалоги сейчас, включая старые: у оператора"),
            ("bot_now", "Открытые диалоги сейчас, включая старые: в режиме бота"),
            ("closed_today", "Закрыто сегодня"),
        ]
        for key, label in labels:
            lines.append(f"  {label}: {stats.get(key, 'n/a')}")
        return lines

    def checks_by_key(self, result: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {item["key"]: item for item in result.get("checks", [])}

    def aggregate_status(self, checks: list[dict[str, Any] | None]) -> str:
        return worst_status([item for item in checks if item]) if checks and all(item is not None for item in checks) else WARN

    def compact_status_line(self, title: str, status: str, details: str = "") -> str:
        suffix = f" — {details}" if details else ""
        return f"{self.status_mark(status)} {title}{suffix}"

    def format_aggregator_brief_report(
        self,
        result: dict[str, Any],
        title: str = "Отчет корпоративного чатбота",
        period: str = "day", start=None, end=None,
    ) -> str:
        # The old counters measured CRM dialogs; the new source measures appeals.
        if self.config.get("appeal_reporting",{}).get("enabled",False):
            return self.format_appeal_report(result,period,start,end)
        local_tz = ZoneInfo(str(self.config.get("timezone", "Europe/Samara")))
        timestamp = datetime.fromisoformat(result["timestamp"]).astimezone(local_tz)
        by_key = self.checks_by_key(result)
        telegram_items = [
            by_key.get("network:singbox_tcp"),
            *[item for item in result.get("checks", []) if item.get("key", "").startswith("network:telegram:")],
        ]
        ai_items = [by_key.get("container:ollama"), by_key.get("api:ollama"), by_key.get("api:ollama_gpu")]
        chatbot_infra_items = [
            by_key.get("container:aggregator_bot"),
            by_key.get("container:aggregator_web"),
            by_key.get("api:aggregator_web"),
        ]
        telegram_status = self.aggregate_status(telegram_items)
        ai_status = self.aggregate_status(ai_items)
        stats_item = by_key.get("aggregator:stats") or {}
        stats_status = stats_item.get("status", WARN)
        stats_health_status = CRIT if stats_status == CRIT else OK
        chatbot_status = self.aggregate_status([*chatbot_infra_items, {"status": stats_health_status}])
        overall_status = self.aggregate_status(
            [
                {"status": telegram_status},
                {"status": ai_status},
                {"status": chatbot_status},
            ]
        )

        ai_problem = next((item for item in ai_items if item and item.get("status") != OK), None)
        stats = stats_item.get("data") or {}

        lines = [
            title,
            f"Время: {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            f"Итог: {self.status_mark(overall_status)}",
            "",
            self.compact_status_line("Доступность Telegram API", telegram_status),
            self.compact_status_line("Административный OpenClaw", (by_key.get("openclaw:telegram_probe") or {}).get("status", WARN)),
            self.compact_status_line(
                "Сервер ИИ (Ollama)",
                ai_status,
                self.human_result(ai_problem) if ai_problem else "",
            ),
            self.compact_status_line("Корпоративный чатбот", chatbot_status),
            "",
            "──── Корпоративный чатбот ────",
        ]

        if not stats:
            lines.append(f"{self.status_mark(stats_item.get('status', WARN))} Статистика недоступна")
            if stats_item:
                lines.append(f"Причина: {self.human_result(stats_item)}")
            if self.pulse_config().get("daily_line_enabled", True):
                lines.extend(["", self.format_telegram_pulse_line()])
            return "\n".join(lines)

        waiting = stats.get("waiting_now", "n/a")
        operator_now = stats.get("operator_now", "n/a")
        bot_now = stats.get("bot_now", "n/a")
        lines.extend(
            [
                f"Новые диалоги сегодня: {stats.get('conversations_today', 'n/a')}",
                f"Новые клиенты сегодня: {stats.get('new_clients_today', 'n/a')}",
                f"Клиенты с новыми диалогами сегодня: {stats.get('clients_with_new_conversations_today', 'n/a')}",
                f"Клиенты, писавшие сегодня: {stats.get('unique_clients_today', 'n/a')}",
                (
                    "Сообщений за сегодня: "
                    f"всего {stats.get('messages_today', 'n/a')}, "
                    f"от клиентов {stats.get('user_messages_today', 'n/a')}, "
                    f"от бота {stats.get('bot_messages_today', 'n/a')}, "
                    f"от операторов {stats.get('operator_messages_today', 'n/a')}"
                ),
                (
                    "Открытые диалоги сейчас, включая старые: "
                    f"ждут оператора {waiting}, у оператора {operator_now}, в режиме бота {bot_now}"
                ),
                f"Закрыто сегодня: {stats.get('closed_today', 'n/a')}",
            ]
        )
        if self.pulse_config().get("daily_line_enabled", True):
            lines.extend(["", self.format_telegram_pulse_line()])
        return "\n".join(lines)

    def format_appeal_report(self,result,period="day",start=None,end=None):
        args=["docker","exec","reporting-worker","python","manage.py","reporting_snapshot","--period",period]
        if start is not None and end is not None: args += ["--start",str(start),"--end",str(end)]
        code,out,_=run_cmd(args,timeout=65)
        if code: return "Корпоративный чатбот\n! Учёт обращений временно недоступен."
        try:
            text=corporate_card(json.loads(out))
        except (ValueError,TypeError,KeyError):
            return "Корпоративный чатбот\n! Не удалось проверить полноту данных отчёта."
        by_key=self.checks_by_key(result)
        text += "\n\n──── Сейчас ────\n"+self.compact_status_line("Административный OpenClaw",(by_key.get("openclaw:telegram_probe") or {}).get("status",WARN))
        text += "\n"+self.compact_status_line("Корпоративный чатбот",self.aggregate_status([by_key.get("container:aggregator_bot"),by_key.get("container:aggregator_web"),by_key.get("api:aggregator_web")]))
        text += "\n"+self.compact_status_line("Сервер ИИ (Ollama)",self.aggregate_status([by_key.get("container:ollama"),by_key.get("api:ollama"),by_key.get("api:ollama_gpu")]))
        text += "\n"+self.compact_status_line("Учёт обращений",(by_key.get("reporting:collector") or {}).get("status",WARN))
        text += "\n\n"+self.format_telegram_pulse_line()
        return text

    def format_failure_report(self,period="day",case_id=None):
        args=["docker","exec","reporting-worker","python","manage.py","reporting_failures"]
        args += ["--id",str(case_id)] if case_id is not None else ["--period",period]
        code,out,_=run_cmd(args,timeout=65)
        if code: return "Локальный разбор временно недоступен."
        try: return failure_card(json.loads(out),detail=case_id is not None)
        except (ValueError,TypeError,KeyError): return "Не удалось проверить результат локального разбора."

    def format_security_report(self):
        # Audit data contains only build/version, counts and fixed status codes.
        try:
            review=json.loads(Path("/etc/openclaw-monitor/security_review.json").read_text())
            stamp=datetime.fromisoformat(review["checked_at"])
            critical=review["critical"]
            if set(review)!={"checked_at","critical","warnings","version","admitted"} or stamp.tzinfo is None: raise ValueError
            if type(critical) is not int or not 0<=critical<=100: raise ValueError
            if type(review["warnings"]) is not int or not 0<=review["warnings"]<=100: raise ValueError
            if type(review["admitted"]) is not bool or not re.fullmatch(r"2026\.\d{1,2}\.\d{1,2}",review["version"]): raise ValueError
            age=(datetime.now(timezone.utc)-stamp).total_seconds()
            if age<0: raise ValueError
            return ("Проверка безопасности\nПроверено: "+stamp.isoformat()+"\nOpenClaw: "+review["version"]+"\n"
                +f"Критичных замечаний встроенного аудита: {critical}.\n"
                +f"Предупреждений: {review['warnings']}.\n"
                +("Запись старше суток; новой проверки с тех пор не проводилось.\n" if age>86400 else "")
                +("Новый учёт допущен по проверенному объёму.\n" if review["admitted"] else "Проверка продолжается; допуск ещё не выдан.\n")
                +"API защищены ключами; персональные тексты исключены из отчётов.\n"
                +"Инструменты изолированы; исходящие направления ограничены.\n"
                +"ИИ-анализатор проходит проверку качества; автоматическая рассылка анализа выключена.\n"
                +"Область и ограничения проверки зафиксированы в журнале безопасности.")
        except (OSError,ValueError,KeyError,TypeError): return "Проверка безопасности\nНет актуальной записи о завершённом аудите."

    def format_logs_block(self, log_items: list[dict[str, Any]]) -> list[str]:
        problem_logs = [item for item in log_items if item["status"] != OK]
        since = str(self.config.get("logs", {}).get("since", "24h"))
        if not problem_logs:
            return [
                f"{self.status_mark(OK)} Логи",
                f"  Проверено: за последние {since}; подозрительных ошибок не найдено.",
            ]

        lines = ["Логи с подозрительными строками:"]
        for item in problem_logs[:6]:
            service = item["title"]
            for prefix in LOG_TITLE_PREFIXES:
                if service.startswith(prefix):
                    service = service[len(prefix) :]
                    break
            data = item.get("data") or {}
            lines.append(f"{self.status_mark(item['status'])} {service}")
            lines.append(f"  Результат: {self.human_result(item)}")
            for detail in item.get("details", [])[:2]:
                lines.append(f"  Последнее: {detail}")
        return lines

    def format_report(self, result: dict[str, Any], title: str = "Отчет AI-сервера") -> str:
        local_tz = ZoneInfo(str(self.config.get("timezone", "Europe/Samara")))
        timestamp = datetime.fromisoformat(result["timestamp"]).astimezone(local_tz)
        status = result["status"]
        lines = [
            title,
            f"Сформирован: {datetime.now(local_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}",
            f"Данные обновлены: {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            "Свежесть: " + ("данные устарели" if (datetime.now(timezone.utc) - timestamp).total_seconds() > 2 * int(self.config.get("check_interval_seconds", 300)) + 120 else "в пределах интервала проверки"),
            f"Итог: {self.status_mark(status)}",
            f"Доставка отчётов, частей: {self.outbox.summary()}",
            "",
        ]
        important = [item for item in result["checks"] if item["status"] != OK]
        if not important:
            lines.append("Все ключевые проверки штатные.")
        else:
            lines.append("Что требует внимания:")
            for item in important[:10]:
                lines.append(f"{self.status_mark(item['status'])} {item['title']}: {self.human_result(item)}")

        by_key = {item["key"]: item for item in result["checks"]}
        lines.extend(["","──── CPU, память и диски ────",*self.format_host_metrics(by_key.get("host:resources") or {})])
        container_items = [item for item in result["checks"] if item["key"].startswith("container:")]
        lines.append("")
        lines.append("──── Контейнеры ────")
        for item in container_items:
            lines.extend(self.format_container_block(item))

        lines.append("")
        lines.append("──── Связь с Telegram ────")
        lines.extend(self.format_telegram_block(result, by_key))

        lines.append("")
        lines.append("──── API и модели ────")
        api_purposes = {
            "api:ollama": "Ollama отвечает локально и может отдать список доступных моделей.",
            "api:ollama_gpu": "контейнер Ollama видит NVIDIA GPU. Если эта проверка падает, инференс может резко замедлиться или уйти на CPU.",
            "api:whisper": "Whisper server отвечает на healthcheck и готов к распознаванию речи.",
            "api:aggregator_web": "web-панель корпоративного агрегатора открывает страницу входа.",
        }
        for key in ["api:ollama", "api:ollama_gpu", "api:whisper", "api:aggregator_web"]:
            if key in by_key:
                lines.extend(self.format_check_block(by_key[key], api_purposes[key]))

        if "aggregator:stats" in by_key:
            lines.append("")
            lines.append("──── Корпоративный чатбот ────")
            lines.extend(self.format_stats_block(by_key["aggregator:stats"]))

        log_items = [item for item in result["checks"] if item["key"].startswith("logs:")]
        if log_items:
            lines.append("")
            lines.append("──── Логи ────")
            lines.extend(self.format_logs_block(log_items))
        if self.pulse_config().get("daily_line_enabled", True):
            lines.append("")
            lines.append(self.format_telegram_pulse_line())
        return "\n".join(lines)

    def format_report_for_target(
        self,
        result: dict[str, Any],
        target: dict[str, str] | None = None,
        title: str = "Ежедневный отчет AI-сервера",
    ) -> str:
        report_format = (target or {}).get("format", "full")
        if report_format in {"aggregator_brief", "corporate_brief"}:
            return self.format_aggregator_brief_report(result,period=(target or {}).get("period","day"),
                start=(target or {}).get("start"),end=(target or {}).get("end"))
        if report_format in {"telegram_pulse", "telegram-pulse", "pulse"}:
            return self.format_telegram_pulse_report()
        return self.format_report(result, title=title)

    def report_targets(self) -> list[dict[str, str]]:
        telegram_config = self.config.get("telegram", {})
        configured = telegram_config.get("report_targets") or []
        targets: list[dict[str, str]] = []
        default_chat_id = str(telegram_config.get("chat_id") or "").strip()
        if default_chat_id and not configured:
            targets.append(
                {
                    "id": "default",
                    "name": "Основной чат",
                    "chat_id": default_chat_id,
                    "report_time": str(self.config.get("report_time", "09:00")),
                    "format": "full",
                }
            )
        for index, target in enumerate(configured):
            chat_id = str(target.get("chat_id") or "").strip()
            if chat_id in {"$TELEGRAM_CHAT_ID", "default"}:
                chat_id = default_chat_id
            if not chat_id:
                continue
            target_id = str(target.get("id") or f"target_{index + 1}")
            if any(existing["id"] == target_id for existing in targets):
                continue
            targets.append(
                {
                    "id": target_id,
                    "name": str(target.get("name") or target_id),
                    "chat_id": chat_id,
                    "report_time": str(target.get("report_time") or self.config.get("report_time", "09:00")),
                    "format": str(target.get("format") or "full"),
                    "frequency": str(target.get("frequency") or "daily"),
                    "period": str(target.get("period") or "day"),
                    "starts_at": target.get("starts_at"),
                }
            )
        return targets

    def primary_report_target(self) -> dict[str, str] | None:
        targets = self.report_targets()
        return targets[0] if targets else None

    def parse_report_time(self, value: str) -> dt_time:
        try:
            hour, minute = [int(part) for part in str(value).split(":", 1)]
            return dt_time(hour, minute)
        except (TypeError, ValueError):
            return dt_time(9, 0)

    def scheduled_window(self, target, current=None):
        tz = ZoneInfo(str(self.config.get("timezone", "Europe/Samara")))
        current = current or datetime.now(tz)
        frequency = target.get("frequency", "daily")
        if frequency not in {"daily", "weekly", "monthly"}:
            raise ValueError("invalid_frequency")
        if frequency == "weekly" and current.weekday() != 0:
            return None
        if frequency == "monthly" and current.day != 1:
            return None
        at = self.parse_report_time(target.get("report_time", "19:00"))
        anchor = current.replace(hour=at.hour, minute=at.minute, second=0, microsecond=0)
        if target.get("starts_at"):
            first=datetime.fromisoformat(target["starts_at"])
            if first.tzinfo is None or anchor<first: return None
        if current < anchor:
            return None
        start, end = period_window(target.get("period", "day"), anchor)
        return anchor, start, end

    def report_time_due(self, target, current=None) -> bool:
        window = self.scheduled_window(target, current)
        if not window:
            return False
        today = window[0].date().isoformat()
        return (self.state.get("last_report_dates") or {}).get(target.get("id", "default")) != today

    def mark_report_sent(self, target, ok, reason):
        # Kept as a compatibility method; 'queued' is explicitly distinct from delivered.
        target_id = target.get("id", "default")
        tz = ZoneInfo(str(self.config.get("timezone", "Europe/Samara")))
        with self.state_lock:
            self.state.setdefault("last_report_dates", {})[target_id] = datetime.now(tz).date().isoformat()
            self.state.setdefault("last_reports", {})[target_id] = {"at": now_iso(), "ok": ok, "reason": reason}
            self.save_state()

    def scheduler_step(self, current=None):
        now = time.monotonic()
        self.heartbeats["scheduler"] = now
        if now - getattr(self, "_last_menu_at", -1e9) >= int(self.config.get("menu_interval_seconds", 300)):
            ok, reason = self.telegram.set_commands()
            self.record_state("last_menu_update", {"at": now_iso(), "ok": ok, "reason": reason})
            self._last_menu_at = now
        if now - getattr(self, "_last_check_at", -1e9) >= int(self.config.get("check_interval_seconds", 300)):
            result = self.run_checks()
            previous = self.state.get("last_status")
            self.record_state("last_status", result["status"])
            self.record_state("last_check", {"at": result["timestamp"], "status": result["status"]})
            self._last_check_at = time.monotonic()
            if self.config.get("alert_on_status_change", True) and previous and previous != result["status"]:
                affected = [item for item in result["checks"] if item["status"] != OK and not item["key"].startswith("logs:")]
                text = ("⚠️ Нарушение работы" if affected else "✅ Работа восстановлена") + "\nВремя: " + result["timestamp"]
                if affected:
                    text += "\nКомпоненты: " + ", ".join(item["title"] for item in affected)
                # Detailed diagnostic output is available only through the private command.
                target = next((t for t in self.report_targets() if t.get("format") in {"telegram_pulse", "pulse"}), self.primary_report_target())
                self.queue_report(text, "state_transition", target)
        for target in self.report_targets():
            if self.report_time_due(target, current):
                anchor, start, end = self.scheduled_window(target, current)
                result = self.last_result or self.run_checks()
                if target.get("format") in {"telegram_pulse", "telegram-pulse", "pulse"}:
                    text = self.format_telegram_pulse_report(anchor)
                else:
                    text = self.format_report_for_target(result, {**target,"start":start.isoformat(),"end":end.isoformat()})
                report_id = "scheduled:" + target["id"] + ":" + anchor.date().isoformat()
                self.queue_report(text, "scheduled", target, report_id, start.isoformat(), end.isoformat())
                self.mark_report_sent(target, False, "queued")
        self.flush_pending()

    def scheduler_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.scheduler_step()
            except Exception:
                self.record_state("scheduler_error", {"at": now_iso(), "reason": "scheduler_iteration_failed"})
            self.stop_event.wait(5)


class Handler(BaseHTTPRequestHandler):
    monitor: Monitor

    def log_message(self, fmt: str, *args: Any) -> None:
        # Paths/query strings and exception text are untrusted input.
        print("monitor_http_request", flush=True)

    def authenticated(self) -> bool:
        secret_path = os.environ.get("MONITOR_API_TOKEN_FILE", "/run/secrets/monitor_api_token")
        try:
            expected = Path(secret_path).read_text(encoding="utf-8").strip()
        except OSError:
            expected = ""
        if not expected:
            self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "authentication unavailable"})
            return False
        provided = self.headers.get("Authorization", "")
        if not hmac.compare_digest(provided.encode("utf-8"), ("Bearer " + expected).encode("utf-8")):
            self.send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return False
        return True

    def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_text(self, status: HTTPStatus, payload: str) -> None:
        data = payload.encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        if not self.authenticated():
            return
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        period=(query.get("period") or ["day"])[0]
        if path in {"/report","/failures"}:
            try:
                if len(period)>24: raise ValueError
                period_window(period,datetime.now(ZoneInfo("Europe/Samara")),datetime(2000,1,1,tzinfo=timezone.utc))
            except ValueError:
                self.send_text(HTTPStatus.BAD_REQUEST,"Неизвестный период. Доступны day, today, week, month, all и YYYY-MM-DD..YYYY-MM-DD.")
                return
        if path == "/security_status":
            self.send_text(HTTPStatus.OK,self.monitor.format_security_report())
            return
        if path in {"/failures","/failure"}:
            case_id=(query.get("id") or [""])[0] if path=="/failure" else None
            if path=="/failure" and not re.fullmatch(r"[1-9][0-9]{0,17}",case_id):
                self.send_text(HTTPStatus.BAD_REQUEST,"Укажите номер: /failure 123.")
                return
            self.send_text(HTTPStatus.OK,self.monitor.format_failure_report(period,case_id))
            return
        if path == "/healthz":
            health = self.monitor.health_status()
            self.send_json(HTTPStatus.OK if health["status"] == "ok" else HTTPStatus.SERVICE_UNAVAILABLE, health)
            return
        if path == "/check":
            result = self.monitor.run_checks()
            self.send_json(HTTPStatus.OK, result)
            return
        if path == "/report":
            result = self.monitor.last_result or self.monitor.run_checks()
            report_format = (query.get("format") or ["full"])[0]
            if report_format in {"aggregator_brief", "corporate_brief"}:
                self.send_text(HTTPStatus.OK, self.monitor.format_aggregator_brief_report(result,period=period))
            elif report_format in {"telegram_pulse", "telegram-pulse", "pulse"}:
                self.send_text(HTTPStatus.OK, self.monitor.format_telegram_pulse_report())
            else:
                self.send_text(HTTPStatus.OK, self.monitor.format_report(result))
            return
        if path == "/telegram_pulse":
            try: text=self.monitor.format_telegram_pulse_report(period=period)
            except ValueError:
                self.send_text(HTTPStatus.BAD_REQUEST,"Неизвестный период.")
                return
            self.send_text(HTTPStatus.OK,text)
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        if not self.authenticated():
            return
        path = urllib.parse.urlparse(self.path).path
        if path == "/telegram/test":
            result = self.monitor.run_checks()
            text = self.monitor.format_report(result, title="Тестовый отчет AI-сервера")
            target = self.monitor.primary_report_target()
            ok, reason = self.monitor.telegram.send_message(text, chat_id=(target or {}).get("chat_id"))
            self.send_json(HTTPStatus.OK if ok else HTTPStatus.BAD_GATEWAY, {"ok": ok, "reason": reason})
            return
        if path == "/telegram/menu":
            ok, reason = self.monitor.telegram.set_commands()
            self.send_json(HTTPStatus.OK if ok else HTTPStatus.BAD_GATEWAY, {"ok": ok, "reason": reason})
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})


def main() -> None:
    config = load_config()
    monitor = Monitor(config)
    Handler.monitor = monitor

    scheduler = threading.Thread(target=monitor.scheduler_loop, daemon=True)
    scheduler.start()
    pulse = threading.Thread(target=monitor.telegram_pulse_loop, daemon=True)
    pulse.start()

    host = str(config.get("listen_host", "0.0.0.0"))
    port = int(config.get("listen_port", 18080))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"openclaw-monitor listening on {host}:{port}; authenticated API", flush=True)
    def stop(_signal, _frame):
        monitor.stop_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever()
    finally:
        monitor.stop_event.set()
        server.server_close()
        scheduler.join(timeout=30)
        pulse.join(timeout=30)


if __name__ == "__main__":
    main()
