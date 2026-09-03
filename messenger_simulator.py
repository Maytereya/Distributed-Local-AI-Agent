#START: python messenger_simulator.py "http://172.16.0.28/api/messenger-generate" s_test
# Модуль тестирования роутера для общения с пациентами через месенджеры.

import json
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, List, Optional, Tuple

import httpx


@dataclass
class BotResult:
    text: str
    attachments: List[dict]
    handoff: bool
    error: Optional[str] = None


@dataclass
class BotChunk:
    text: str = ""
    attachments: List[dict] = field(default_factory=list)
    handoff: bool = False
    error: Optional[str] = None
    raw: Optional[dict[str, Any]] = None


@dataclass
class RuntimeOptions:
    llm_mode: str = "hybrid"
    self_check: bool = False
    self_check_max_retries: int = 1
    queue_timeout_ms: int = 30000


def sanitize_text(s: str) -> str:
    if not s:
        return s
    # удаляем невалидные unicode surrogate
    return s.encode("utf-8", "ignore").decode("utf-8")


def _build_payload(session_id: str, text: str, opts: RuntimeOptions) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "text": sanitize_text(text),
        "llm_mode": opts.llm_mode,
        "self_check": bool(opts.self_check),
        "self_check_max_retries": int(opts.self_check_max_retries),
        "queue_timeout_ms": int(opts.queue_timeout_ms),
    }


def _build_headers(host_header: Optional[str]) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if host_header:
        headers["Host"] = host_header
    return headers


def _extract_text_delta(text_value: str, last_text: str) -> Tuple[str, str]:
    if not text_value:
        return "", last_text
    # Поддерживаем оба формата:
    # 1) delta-чанки: "текущий кусок"
    # 2) partial-чанки: "весь накопленный ответ на текущий момент"
    if last_text and text_value.startswith(last_text):
        delta = text_value[len(last_text):]
        return delta, text_value
    return text_value, text_value


def _parse_stream_line(raw: str, last_text: str) -> Tuple[BotChunk, str]:
    raw = (raw or "").strip()
    if not raw:
        return BotChunk(), last_text

    try:
        obj = json.loads(raw)
    except Exception:
        # Иногда в stream может прийти plain-text строка.
        delta, new_last_text = _extract_text_delta(raw, last_text)
        return BotChunk(text=delta), new_last_text

    if not isinstance(obj, dict):
        return BotChunk(), last_text

    chunk = BotChunk(raw=obj)

    if obj.get("handoff") is True:
        chunk.handoff = True

    err = obj.get("error")
    if isinstance(err, str) and err.strip():
        chunk.error = err

    att = obj.get("attachments")
    if isinstance(att, list) and att:
        chunk.attachments = [a for a in att if isinstance(a, dict)]

    text_value = obj.get("text")
    if isinstance(text_value, str) and text_value:
        delta, last_text = _extract_text_delta(text_value, last_text)
        chunk.text = delta

    return chunk, last_text


async def stream_message(
    url: str,
    session_id: str,
    text: str,
    host_header: Optional[str] = None,
    runtime_options: Optional[RuntimeOptions] = None,
) -> AsyncGenerator[BotChunk, None]:
    """
    Потоковый клиент NDJSON для /api/messenger-generate.
    Отдает чанки по мере прихода строк от сервера.
    """
    opts = runtime_options or RuntimeOptions()
    payload = _build_payload(session_id=session_id, text=text, opts=opts)
    headers = _build_headers(host_header=host_header)

    last_text = ""
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                body_text = body.decode("utf-8", errors="replace") if body else ""
                yield BotChunk(error=f"HTTP {resp.status_code}: {body_text}")
                return

            async for line in resp.aiter_lines():
                chunk, last_text = _parse_stream_line(line, last_text=last_text)
                if chunk.text or chunk.attachments or chunk.handoff or chunk.error:
                    yield chunk


def parse_jsonl_stream(lines: List[str]) -> BotResult:
    """
    текущий формат JSONL:
      {"text": "...", "attachments": [...], "handoff": bool}
    Стримит по токенам, поэтому мы:
      - склеиваем text
      - собираем attachments
      - handoff=true если хоть раз встретился (но обычно это финальная пустая строка)
    """
    parts: List[str] = []
    attachments: List[dict] = []
    handoff = False
    error: Optional[str] = None

    # На всякий случай поддержим режим "partial" (если когда-то возвратится накопление).
    last_text = ""

    for raw in lines:
        chunk, last_text = _parse_stream_line(raw, last_text=last_text)
        if chunk.text:
            parts.append(chunk.text)
        if chunk.attachments:
            attachments.extend(chunk.attachments)
        if chunk.handoff:
            handoff = True
        if chunk.error:
            error = chunk.error

    return BotResult(text="".join(parts).strip(), attachments=attachments, handoff=handoff, error=error)


def send_message(
    url: str,
    session_id: str,
    text: str,
    host_header: Optional[str] = None,
    runtime_options: Optional[RuntimeOptions] = None,
) -> BotResult:
    opts = runtime_options or RuntimeOptions()
    payload = _build_payload(session_id=session_id, text=text, opts=opts)
    headers = _build_headers(host_header=host_header)

    raw_lines: List[str] = []

    # timeout=None: чтобы не рвать долгий стрим
    with httpx.Client(timeout=None) as client:
        with client.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code >= 400:
                body = resp.read()
                text = body.decode("utf-8", errors="replace") if body else ""
                return BotResult(text="", attachments=[], handoff=False, error=f"HTTP {resp.status_code}: {text}")

            for line in resp.iter_lines():
                if line is None:
                    continue
                s = line.strip()
                if not s:
                    continue
                raw_lines.append(s)

    return parse_jsonl_stream(raw_lines)


def repl():
    if len(sys.argv) < 2:
        print("Usage: python messenger_simulator.py <url> [session_id] [host_header]")
        print('Example: python messenger_simulator.py "http://localhost:8000/v1/agent/stream" local_test')
        print('Example: python messenger_simulator.py "http://172.16.0.28/api/messenger-generate" s_test')
        print('Example: python messenger_simulator.py "http://172.16.0.28/api/messenger-generate" s_test ontheflyai.ru')
        sys.exit(1)

    url = sys.argv[1]
    session_id = sys.argv[2] if len(sys.argv) >= 3 else f"sim_{uuid.uuid4().hex[:8]}"
    host_header = sys.argv[3] if len(sys.argv) >= 4 else None

    print("=== Messenger Simulator (Telegram/WhatsApp style) ===")
    print(f"URL: {url}")
    print(f"session_id: {session_id}")
    if host_header:
        print(f"Host header: {host_header}")
    opts = RuntimeOptions()

    print("Type messages.")
    print("Commands: /new, /session <id>, /mode <strict|hybrid|rich>, /selfcheck <on|off>, /retries <0..2>, /qtimeout <ms>, /opts, /quit\n")
    print(f"[opts] llm_mode={opts.llm_mode}, self_check={opts.self_check}, retries={opts.self_check_max_retries}, queue_timeout_ms={opts.queue_timeout_ms}\n")

    while True:
        try:
            user_text = input("YOU: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not user_text:
            continue

        if user_text == "/quit":
            print("Bye.")
            return

        if user_text == "/opts":
            print(
                f"[opts] llm_mode={opts.llm_mode}, "
                f"self_check={opts.self_check}, "
                f"retries={opts.self_check_max_retries}, "
                f"queue_timeout_ms={opts.queue_timeout_ms}"
            )
            continue

        if user_text == "/new":
            session_id = f"sim_{uuid.uuid4().hex[:8]}"
            print(f"[system] new session_id: {session_id}")
            continue

        if user_text.startswith("/session "):
            session_id = user_text.split(" ", 1)[1].strip() or session_id
            print(f"[system] session_id set to: {session_id}")
            continue

        if user_text.startswith("/mode "):
            mode = user_text.split(" ", 1)[1].strip().lower()
            if mode not in {"strict", "hybrid", "rich"}:
                print("[system] invalid mode. use: strict|hybrid|rich")
                continue
            opts.llm_mode = mode
            print(f"[system] llm_mode set to: {opts.llm_mode}")
            continue

        if user_text.startswith("/selfcheck "):
            raw = user_text.split(" ", 1)[1].strip().lower()
            if raw in {"on", "1", "true", "yes"}:
                opts.self_check = True
            elif raw in {"off", "0", "false", "no"}:
                opts.self_check = False
            else:
                print("[system] invalid value. use: on|off")
                continue
            print(f"[system] self_check set to: {opts.self_check}")
            continue

        if user_text.startswith("/retries "):
            raw = user_text.split(" ", 1)[1].strip()
            try:
                val = int(raw)
            except Exception:
                print("[system] invalid retries. use integer 0..2")
                continue
            if val < 0 or val > 2:
                print("[system] retries must be 0..2")
                continue
            opts.self_check_max_retries = val
            print(f"[system] self_check_max_retries set to: {opts.self_check_max_retries}")
            continue

        if user_text.startswith("/qtimeout "):
            raw = user_text.split(" ", 1)[1].strip()
            try:
                val = int(raw)
            except Exception:
                print("[system] invalid qtimeout. use integer milliseconds")
                continue
            if val < 1000:
                print("[system] qtimeout should be >= 1000 ms")
                continue
            opts.queue_timeout_ms = val
            print(f"[system] queue_timeout_ms set to: {opts.queue_timeout_ms}")
            continue

        try:
            result = send_message(
                url,
                session_id,
                user_text,
                host_header=host_header,
                runtime_options=opts,
            )
        except httpx.HTTPStatusError as e:
            # покажем ответ сервера для диагностики
            resp = e.response
            print(f"[HTTP {resp.status_code}] {resp.text}")
            continue
        except Exception as e:
            print(f"[error] {e}")
            continue

        print("\nBOT:")
        if result.error:
            print(f"[error] {result.error}")

        if result.text:
            print(result.text)
        else:
            print("(no text)")

        if result.attachments:
            print("\nATTACHMENTS:")
            for a in result.attachments:
                print("-", a)

        if result.handoff:
            print("\n[SYSTEM] handoff=true -> transfer to operator")

        print("")

async def main():
    """
    Локальная диагностическая проверка сервисов (опционально).
    """
    from messengers_router.services import Services

    s = Services()
    print(await s.doctors_info("уролог Дразнин", {"specialty": "уролог", "last_name": "Дразнин"}))
    print(await s.doctors_schedule_week("покажи расписание Дразнина", {"last_name": "Дразнин"}))

# Для ручной отладки:
# import asyncio
# asyncio.run(main())


if __name__ == "__main__":
    repl()
