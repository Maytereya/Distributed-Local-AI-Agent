"""Проверка живости бота: доходит ли запрос до LLM и приходит ли осмысленный ответ.

Зачем именно так. 15.09.2026 прод сутки отвечал «Соединяю с оператором» почти на
всё: ollama не могла загрузить модель. HTTP при этом был жив, `/api/tags`
работал, контейнеры были `Up` — ни один обычный healthcheck этого не видел.

Мерить «нет успешных ответов за N минут» нельзя: ночью и в выходные обращений
может не быть вовсе, и монитор будет будить зря. Поэтому проверка СИНТЕТИЧЕСКАЯ —
шлём собственный запрос и смотрим на ответ. Тишина перестаёт быть сигналом.

Запуск (cron, раз в 10-15 минут):
    ./venv/bin/python scripts/check_bot_health.py --url http://ХОСТ/api/messenger-generate-once

Коды возврата: 0 — здоров, 1 — болен (текст причины в stderr), 2 — не достучались.
Пригоден для cron с отправкой письма при ненулевом коде.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# Зонд намеренно требует РАБОТЫ LLM: короткие ценовые запросы обслуживает
# rule-path и каталог, они прошли бы и 15.09, когда бот был фактически мёртв.
_PROBE_TEXT = "Добрый день, подскажите, а что вы можете?"

# Тексты, которыми бот сообщает о собственном отказе. Появление любого — сигнал.
_FAILURE_MARKERS = (
    "не удалось получить данные автоматически",
    "соединяю с оператором",
)


def probe(url: str, timeout: int) -> tuple[bool, str]:
    """Шлёт один запрос и оценивает ответ.

    :param url: эндпоинт messenger-generate-once
    :param timeout: таймаут ответа в секундах
    :return: (здоров, описание)
    """

    payload = {
        "session_id": f"healthcheck_{int(time.time())}",
        "text": _PROBE_TEXT,
        "debug": True,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    elapsed = time.perf_counter() - started

    debug = (data.get("state_update") or {}).get("debug") or {}
    # Самый точный сигнал: эндпоинт положил сюда текст исключения (endpoint.py).
    route_error = str(debug.get("route_error") or debug.get("endpoint_error") or "")
    if route_error:
        return False, f"ошибка маршрутизации за {elapsed:.0f}с: {route_error[:200]}"

    text = str(data.get("text") or "")
    if not text.strip():
        return False, f"пустой ответ за {elapsed:.0f}с"

    lowered = text.lower()
    for marker in _FAILURE_MARKERS:
        if marker in lowered:
            return False, f"бот сдался оператору за {elapsed:.0f}с: {text[:120]}"

    label = str((debug.get("decision") or {}).get("label") or "")
    return True, f"ок за {elapsed:.0f}с, метка {label or '—'}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="эндпоинт messenger-generate-once")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2,
                        help="повторы перед вердиктом: единичный таймаут не повод будить")
    args = parser.parse_args()

    last = ""
    for attempt in range(1, args.retries + 1):
        try:
            healthy, detail = probe(args.url, args.timeout)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = f"не достучались: {type(exc).__name__}: {exc}"
            if attempt < args.retries:
                time.sleep(5)
                continue
            print(f"НЕДОСТУПЕН: {last}", file=sys.stderr)
            return 2
        if healthy:
            print(f"ЗДОРОВ: {detail}")
            return 0
        last = detail
        if attempt < args.retries:
            time.sleep(5)

    print(f"БОЛЕН: {last}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
