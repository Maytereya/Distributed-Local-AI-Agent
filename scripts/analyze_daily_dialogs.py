"""Ежедневный триаж диалогов бота: эвристический поиск подозрительных сессий.

Зачем: весь цикл «прислал диалог → нашёл промах правила → фикс + eval-кейс»
делался руками. Стадия 1-2 автоматизации (без LLM, ничего не покидает сервер):
парсим лог бот-воркера агрегатора за день → ранжируем диалоги по «красным
флагам» (оператор-фолбэк, повтор-луп, дефлекты) → markdown-дайджест. Глубокий
разбор топ-N (стадия 3) остаётся за Claude — по короткому отобранному списку.

Источник (доступен без чужой БД):
    docker logs --since 24h support-messenger-aggregator-telegram_bot-1 > day.log
    ./venv/bin/python scripts/analyze_daily_dialogs.py day.log

Формат лога (наблюдён 10.07):
    ... channels_app.services: Incoming message conv=#619 from Losinui: <text>
    ... channels_app.services: Sending to telegram:<chat> conv=#524: <text>
    ... channels_app.adapters.telegram: Telegram edit msg_id=.. in chat=<chat>: <text>
    ... channels_app.management...run_telegram: Received from X (chat=<chat>): <text>

Реплей/PII/LLM здесь НЕ делаются — только детерминированный триаж.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field

# --- сигнатуры строк лога ------------------------------------------------------
# Ответы бота идут строкой «Telegram edit ... in chat=<chat>» (БЕЗ conv), а
# conv известен из «Incoming ... conv=#N from USER». Мост chat↔conv строим по
# имени пользователя из пары «Received from USER (chat=<chat>)» + «Incoming ...
# from USER» (обе строки на каждый входящий, наблюдено 10.07).
_RE_INCOMING = re.compile(r"Incoming message conv=#(\d+) from (.+?):\s*(.*)$")
_RE_SENDING = re.compile(r"Sending to telegram:(\d+) conv=#(\d+):\s*(.*)$")
_RE_RECEIVED = re.compile(r"Received from (.+?)\s*\(chat=(\d+)\):\s*(.*)$")
_RE_EDIT = re.compile(r"Telegram (?:edit msg_id=\d+ in|send to) chat=(\d+):\s*(.*)$")

# «бот не обслужил» — агрегатор отдал фолбэк оператора
_OPERATOR_FALLBACK_RE = re.compile(r"ожидайте,?\s*оператор|соединяю с оператором|переключаю на оператора", re.I)
# бот не понял запрос (clarify-дефлект / low-confidence)
_DEFLECT_RE = re.compile(r"не совсем понял|нет информации для ответа|уточните,? что именно", re.I)
# security/непрофильный дефлект (сам по себе не баг, но частота = сигнал)
_GENERIC_DEFLECT_RE = re.compile(r"виртуальный помощник клиники", re.I)
# служебный шум — не считаем ответом бота (плейсхолдер стрима с хвостом «...»)
_NOISE_RE = re.compile(r"^(ответ в процессе|печатает)[.…\s]*$", re.I)

_TELEGRAM_LIMIT = 4096


@dataclass
class Dialog:
    conv: str
    user_msgs: list[str] = field(default_factory=list)
    bot_msgs: list[str] = field(default_factory=list)
    operator_fallback: int = 0

    def flags(self) -> list[tuple[str, str]]:
        """Красные флаги диалога: (код, человекочитаемая причина)."""
        out: list[tuple[str, str]] = []
        if self.operator_fallback:
            out.append(("operator_fallback", f"оператор-фолбэк ×{self.operator_fallback} (бот не обслужил)"))
        # повтор-луп: один и тот же ответ бота ≥2 раз (кейс УЗИ-мошонки)
        seen: dict[str, int] = defaultdict(int)
        for m in self.bot_msgs:
            key = re.sub(r"\s+", " ", m.strip().lower())[:200]
            if key:
                seen[key] += 1
        repeats = max(seen.values(), default=0)
        if repeats >= 2:
            out.append(("repeat_loop", f"идентичный ответ бота ×{repeats}"))
        deflects = sum(1 for m in self.bot_msgs if _DEFLECT_RE.search(m))
        if deflects >= 2:
            out.append(("clarify_loop", f"дефлект «не понял» ×{deflects}"))
        elif deflects == 1:
            out.append(("deflect", "дефлект «не понял» ×1"))
        oversize = sum(1 for m in self.bot_msgs if len(m) > _TELEGRAM_LIMIT)
        if oversize:
            out.append(("oversize", f"ответ >{_TELEGRAM_LIMIT} симв. ×{oversize} (телеграм режет)"))
        generic = sum(1 for m in self.bot_msgs if _GENERIC_DEFLECT_RE.search(m))
        if generic >= 2:
            out.append(("generic_deflect", f"«виртуальный помощник» ×{generic}"))
        return out

    def score(self) -> int:
        """Вес подозрительности — для ранжирования (operator > loop > deflect)."""
        weights = {
            "operator_fallback": 5,
            "repeat_loop": 4,
            "oversize": 3,
            "clarify_loop": 3,
            "generic_deflect": 2,
            "deflect": 1,
        }
        return sum(weights.get(code, 1) for code, _ in self.flags())


def parse_log(lines: list[str]) -> dict[str, Dialog]:
    """Собирает диалоги по conv=# из строк лога воркера.

    chat→conv мостится по conv-тегированным строкам; ответы бота (Telegram
    edit/send, где conv нет) привязываются к диалогу через chat.
    """
    dialogs: dict[str, Dialog] = {}
    chat_to_conv: dict[str, str] = {}
    user_to_chat: dict[str, str] = {}

    def dlg(conv: str) -> Dialog:
        return dialogs.setdefault(conv, Dialog(conv=conv))

    for raw in lines:
        line = raw.rstrip("\n")

        m = _RE_RECEIVED.search(line)
        if m:
            # запоминаем chat пользователя — следующая Incoming даст его conv
            user, chat = m.group(1).strip(), m.group(2)
            user_to_chat[user] = chat
            continue

        m = _RE_INCOMING.search(line)
        if m:
            conv, user, text = m.group(1), m.group(2).strip(), m.group(3).strip()
            chat = user_to_chat.get(user)
            if chat:
                chat_to_conv[chat] = conv
            if text:
                dlg(conv).user_msgs.append(text)
            continue

        m = _RE_SENDING.search(line)
        if m:
            chat, conv, text = m.group(1), m.group(2), m.group(3).strip()
            chat_to_conv[chat] = conv
            d = dlg(conv)
            if _OPERATOR_FALLBACK_RE.search(text):
                d.operator_fallback += 1
            elif text and not _NOISE_RE.match(text):
                d.bot_msgs.append(text)
            continue

        m = _RE_EDIT.search(line)
        if m:
            chat, text = m.group(1), m.group(2).strip()
            conv = chat_to_conv.get(chat)
            if conv and text and not _NOISE_RE.match(text):
                if _OPERATOR_FALLBACK_RE.search(text):
                    dlg(conv).operator_fallback += 1
                else:
                    dlg(conv).bot_msgs.append(text)
            continue

    return dialogs


def build_digest(dialogs: dict[str, Dialog], top_n: int = 15) -> str:
    flagged = [(d.score(), d) for d in dialogs.values() if d.flags()]
    flagged.sort(key=lambda x: (-x[0], x[1].conv))

    total = len(dialogs)
    n_flagged = len(flagged)
    counts: dict[str, int] = defaultdict(int)
    for _, d in flagged:
        for code, _reason in d.flags():
            counts[code] += 1

    lines = [
        "# Дайджест диалогов бота",
        "",
        f"- Всего диалогов: **{total}**",
        f"- С красными флагами: **{n_flagged}** ({(n_flagged / total * 100 if total else 0):.0f}%)",
        "- По типам: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])) or "—"),
        "",
        f"## Топ-{min(top_n, n_flagged)} подозрительных (для разбора)",
        "",
    ]
    if not flagged:
        lines.append("_Красных флагов нет — день чистый._")
    for score, d in flagged[:top_n]:
        reasons = "; ".join(reason for _code, reason in d.flags())
        first_user = (d.user_msgs[0] if d.user_msgs else "").strip()[:80]
        lines.append(f"- **conv #{d.conv}** (вес {score}, ходов {len(d.user_msgs)}): {reasons}")
        if first_user:
            lines.append(f"  - первый запрос: «{first_user}»")
    lines.append("")
    lines.append("> Триаж детерминированный (без LLM). Глубокий разбор топа — реплей "
                 "conv через debug-эндпоинт + Claude; в прод-логику ничего автоматически не идёт.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Эвристический триаж дневных диалогов бота.")
    ap.add_argument("logfile", nargs="?", default="-", help="лог воркера (или - для stdin)")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    if args.logfile == "-":
        lines = sys.stdin.readlines()
    else:
        with open(args.logfile, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

    dialogs = parse_log(lines)
    print(build_digest(dialogs, top_n=args.top))


if __name__ == "__main__":
    main()
