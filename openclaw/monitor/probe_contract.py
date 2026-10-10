"""Parse OpenClaw CLI diagnostics without returning arbitrary diagnostic text."""
import json


def decode_probe_output(text: str) -> dict:
    decoder = json.JSONDecoder()
    for line_number, line in enumerate(text.splitlines(keepends=True)):
        if not line.lstrip().startswith("{"):
            continue
        tail = "".join(text.splitlines(keepends=True)[line_number:]).lstrip()
        try:
            value, _ = decoder.raw_decode(tail)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and any(key in value for key in ("channels", "channelAccounts", "telegram")):
            return value
    raise ValueError("probe_json_unavailable")


def telegram_probe_only(payload: dict) -> dict:
    telegram = payload.get("channels", {}).get("telegram") or payload.get("telegram")
    if not isinstance(telegram, dict):
        return {}
    probe = telegram.get("probe") or {}
    running = telegram.get("running")
    ok = probe.get("ok")
    return {"telegram": {
        "running": running if isinstance(running, bool) else None,
        "probe": {"ok": ok if isinstance(ok, bool) else None},
        "lastError": "probe_error" if telegram.get("lastError") or probe.get("error") else "",
    }}
