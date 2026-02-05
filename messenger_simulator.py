#START: python messenger_simulator.py "http://172.16.0.16/api/messenger-generate" s_test

import json
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class BotResult:
    text: str
    attachments: List[dict]
    handoff: bool
    error: Optional[str] = None


def parse_jsonl_stream(lines: List[str]) -> BotResult:
    """
    Твой текущий формат JSONL:
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

    # На всякий случай поддержим режим "partial" (если когда-то вернёшь накопление).
    last_text = ""

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue

        try:
            obj = json.loads(raw)
        except Exception:
            # Если вдруг прилетел голый текст
            parts.append(raw)
            continue

        if isinstance(obj, dict):
            if obj.get("handoff") is True:
                handoff = True

            if "error" in obj and isinstance(obj["error"], str):
                error = obj["error"]

            # attachments
            att = obj.get("attachments")
            if isinstance(att, list) and att:
                for a in att:
                    if isinstance(a, dict):
                        attachments.append(a)

            # text
            t = obj.get("text", "")
            if isinstance(t, str) and t:
                # Если это delta — просто добавляем:
                # parts.append(t)

                # Если это partial (накопление) — добавляем только разницу:
                if t.startswith(last_text):
                    delta = t[len(last_text):]
                    if delta:
                        parts.append(delta)
                    last_text = t
                else:
                    parts.append(t)
                    last_text = t

    return BotResult(text="".join(parts).strip(), attachments=attachments, handoff=handoff, error=error)


def send_message(url: str, session_id: str, text: str, host_header: Optional[str] = None) -> BotResult:
    payload = {"session_id": session_id, "text": text}

    headers = {"Content-Type": "application/json"}
    if host_header:
        headers["Host"] = host_header

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
        print('Example: python messenger_simulator.py "http://172.16.0.16/api/messenger-generate" s_test')
        print('Example: python messenger_simulator.py "http://172.16.0.16/api/messenger-generate" s_test ontheflyai.ru')
        sys.exit(1)

    url = sys.argv[1]
    session_id = sys.argv[2] if len(sys.argv) >= 3 else f"sim_{uuid.uuid4().hex[:8]}"
    host_header = sys.argv[3] if len(sys.argv) >= 4 else None

    print("=== Messenger Simulator (Telegram/WhatsApp style) ===")
    print(f"URL: {url}")
    print(f"session_id: {session_id}")
    if host_header:
        print(f"Host header: {host_header}")
    print("Type messages. Commands: /new, /session <id>, /quit\n")

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

        if user_text == "/new":
            session_id = f"sim_{uuid.uuid4().hex[:8]}"
            print(f"[system] new session_id: {session_id}")
            continue

        if user_text.startswith("/session "):
            session_id = user_text.split(" ", 1)[1].strip() or session_id
            print(f"[system] session_id set to: {session_id}")
            continue

        try:
            result = send_message(url, session_id, user_text, host_header=host_header)
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


if __name__ == "__main__":
    repl()