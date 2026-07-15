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
    ... channels_app.services: Incoming message conv=#619 from user_alpha: <text>
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


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


@dataclass
class Dialog:
    conv: str
    events: list[tuple[str, str]] = field(default_factory=list)  # ('user'|'bot', text) в порядке лога
    operator_fallback: int = 0

    def add_user(self, text: str) -> None:
        self.events.append(("user", text))

    def add_bot(self, text: str) -> None:
        self.events.append(("bot", text))

    @property
    def user_msgs(self) -> list[str]:
        return [t for r, t in self.events if r == "user"]

    @property
    def bot_msgs(self) -> list[str]:
        return [t for r, t in self.events if r == "bot"]

    def _loop_repeats(self) -> tuple[int, str]:
        """Настоящий луп: ОДИН ответ бота на РАЗНЫЕ реплики пользователя.

        Кейс УЗИ-мошонки: «рекомендуй»/«кто топ» (разное) → один список.
        Повтор одного и того же вопроса пациентом (дважды «Флюорография» →
        тот же ответ) НЕ считается лупом — это ожидаемо.
        Возвращает (число повторов, сниппет ответа).
        """
        pairs: dict[str, list[str]] = defaultdict(list)  # ответ → список реплик-до
        last_user: str | None = None
        for role, text in self.events:
            if role == "user":
                last_user = _norm(text)
            elif role == "bot" and last_user is not None:
                pairs[_norm(text)[:200]].append(last_user)
        best_n, best_key = 0, ""
        for ans_key, users in pairs.items():
            if not ans_key:
                continue
            distinct_users = len(set(users))
            if len(users) >= 2 and distinct_users >= 2 and len(users) > best_n:
                best_n, best_key = len(users), ans_key
        return best_n, best_key[:70]

    def flags(self) -> list[tuple[str, str]]:
        """Красные флаги диалога: (код, человекочитаемая причина)."""
        out: list[tuple[str, str]] = []
        if self.operator_fallback:
            out.append(("operator_fallback", f"оператор-фолбэк ×{self.operator_fallback} (бот не обслужил)"))
        repeats, snippet = self._loop_repeats()
        if repeats >= 2:
            out.append(("repeat_loop", f"один ответ на {repeats} разных реплик: «{snippet}…»"))
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
                dlg(conv).add_user(text)
            continue

        m = _RE_SENDING.search(line)
        if m:
            chat, conv, text = m.group(1), m.group(2), m.group(3).strip()
            chat_to_conv[chat] = conv
            d = dlg(conv)
            if _OPERATOR_FALLBACK_RE.search(text):
                d.operator_fallback += 1
            elif text and not _NOISE_RE.match(text):
                d.add_bot(text)
            continue

        m = _RE_EDIT.search(line)
        if m:
            chat, text = m.group(1), m.group(2).strip()
            conv = chat_to_conv.get(chat)
            if conv and text and not _NOISE_RE.match(text):
                if _OPERATOR_FALLBACK_RE.search(text):
                    dlg(conv).operator_fallback += 1
                else:
                    dlg(conv).add_bot(text)
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


def dump_conversations(dialogs: dict[str, Dialog], convs: list[str]) -> str:
    """Выгрузка ходов конкретных диалогов — для курируемой передачи на разбор.

    Позволяет вытащить ТОЛЬКО отобранные подозрительные conv (а не весь дневной
    PII) → реплей через debug-эндпоинт / разбор Claude.
    """
    out: list[str] = []
    for conv in convs:
        d = dialogs.get(conv)
        if d is None:
            out.append(f"=== conv #{conv}: не найден в логе ===\n")
            continue
        reasons = "; ".join(r for _c, r in d.flags()) or "нет флагов"
        out.append(f"=== conv #{conv} (флаги: {reasons}) ===")
        for role, text in d.events:
            tag = "пациент" if role == "user" else "бот    "
            out.append(f"  [{tag}] {text}")
        if d.operator_fallback:
            out.append(f"  [оператор-фолбэк ×{d.operator_fallback}]")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Эвристический триаж дневных диалогов бота.")
    ap.add_argument("logfile", nargs="?", default="-", help="лог воркера (или - для stdin)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--conv", default="", help="выгрузить ходы конкретных диалогов (через запятую), без дайджеста")
    args = ap.parse_args()

    if args.logfile == "-":
        lines = sys.stdin.readlines()
    else:
        with open(args.logfile, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

    dialogs = parse_log(lines)
    if args.conv.strip():
        convs = [c.strip().lstrip("#") for c in args.conv.split(",") if c.strip()]
        print(dump_conversations(dialogs, convs))
        return
    print(build_digest(dialogs, top_n=args.top))


if __name__ == "__main__":
    main()
