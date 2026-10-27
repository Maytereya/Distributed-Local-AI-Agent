"""Подмена клиента LLM бота: режимы стенда и учёт вызовов по месту вызова.

Все вызовы LLM пациентского бота идут через один объект `llm_runtime._OLLAMA_CLIENT`
(и короткие решения, и поток свободного ответа). Стенд ставит на его место
`HarnessClient`: семафор, тайм-ауты и обработка ошибок в местах вызова остаются
настоящими, меняется только то, что «ответила модель».

Режимы:
- `real` — настоящая LLM; каждый ответ записывается в кэш (`record`);
- `cached` — ответ из кэша, при промахе — настоящая LLM с записью;
- `replay` — только кэш; промах — отказ соединения и счётчик промахов;
- `adversary` — уверенно-неверный, но разбираемый ответ для каждого места;
- `down` — отказ соединения (как упавший ollama); `down_timeout` — тайм-аут: сразу или, с `hang_s`, после «зависания» на
  столько секунд (время хода при зависшей LLM: вызов обрывает тайм-аут места вызова, исход — `cancelled`);
- `replay` с `hang_s` — «медленная, но отвечающая» LLM: ответ кэша через `hang_s` секунд, если место вызова дождётся (иначе его
  тайм-аут — `cancelled`). Вместе с `--llm-slots` (слотов меньше, чем ходов) — очередь к LLM под нагрузкой (стенд разбора 27.10,
  BUG-2026-10-27-LLM-QUEUE-GIVES-UP).

Место вызова (site) определяется отпечатком шаблона промпта: прод читает шаблоны через
`prompt_registry.load_prompt_text`, стенд берёт их оттуда же в том дереве кода, которое
гоняет. Нераспознанный промпт — громкий счётчик `unknown`, а не тихий ответ по умолчанию.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODES = ("real", "cached", "replay", "adversary", "down", "down_timeout")
_LIVE_MODES = {"real", "cached"}

WARM_UP_PROMPT = "Ответь одним словом: да"
_PLACEHOLDER_RE = re.compile(r"<<[A-Z0-9_]+>>")
_MIN_SEGMENT = 25
_SITE_THRESHOLD = 0.6


# --------------------------------------------------------------------------- места вызова


@dataclass(frozen=True)
class SiteIndex:
    """Отпечатки шаблонов: для каждого ключа — литеральные куски длиной от 25 символов."""

    segments: dict[str, tuple[str, ...]]

    @classmethod
    def from_templates(cls, templates: dict[str, str]) -> "SiteIndex":
        segments: dict[str, tuple[str, ...]] = {}
        for key, text in templates.items():
            parts: list[str] = []
            for line in text.splitlines():
                for chunk in _PLACEHOLDER_RE.split(line):
                    chunk = chunk.strip()
                    if len(chunk) >= _MIN_SEGMENT:
                        parts.append(chunk)
            segments[key] = tuple(dict.fromkeys(parts))
        return cls(segments=segments)

    @classmethod
    def from_code_root(cls, code_root: Path) -> "SiteIndex":
        """Шаблоны так, как их видит бот в этом дереве кода (копия на хосте побеждает пакетную)."""

        from messengers_router import prompt_registry

        keys = sorted(p.stem for p in (code_root / "messengers_router" / "prompts").glob("*.txt"))
        return cls.from_templates({key: prompt_registry.load_prompt_text(key) for key in keys})

    def scores(self, prompt: str) -> dict[str, float]:
        out = {}
        for key, segs in self.segments.items():
            total = sum(len(s) for s in segs)
            if total:
                out[key] = sum(len(s) for s in segs if s in prompt) / total
        return out

    def site_of(self, prompt: str) -> str:
        if prompt.strip() == WARM_UP_PROMPT:
            return "warm_up"
        scores = self.scores(prompt)
        if not scores:
            return "unknown"
        key, best = max(scores.items(), key=lambda kv: kv[1])
        return key if best >= _SITE_THRESHOLD else "unknown"

    def ambiguous_pairs(self) -> list[tuple[str, str, float]]:
        """Пары шаблонов, которые отпечаток путает: шаблон A набирает порог на шаблоне B."""

        bad = []
        for key, segs in self.segments.items():
            sample = "\n".join(segs)
            for other, score in self.scores(sample).items():
                if other != key and score >= _SITE_THRESHOLD and score >= self.scores(sample)[key]:
                    bad.append((key, other, score))
        return bad


# --------------------------------------------------------------------------- adversary

_ADVERSARY_LABELS = ("OTHER", "ADDRESS", "NEWS", "DOCTOR_INFO", "TEST_RESULT", "APPOINTMENT", "PRICE", "PREPARE")
_PRICE_KINDS = ("lab", "doctor_consult", "diagnostic_no_doctor", "family_query", "procedure_with_doctor")
_NUMBERED_RE = re.compile(r"^\s*(\d+)\.\s", re.M)
ADVERSARY_RENDER_TEXT = (
    "Стоимость этой услуги — 1 990 руб. Записаться можно в любом нашем филиале, "
    "ближайшее время — завтра с 9:00."
)


def _pick(options: tuple[str, ...], prompt: str) -> str:
    digest = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
    return options[digest % len(options)]


def _last_number(block: str) -> int | None:
    nums = [int(n) for n in _NUMBERED_RE.findall(block)]
    return nums[-1] if nums else None


def _adversary_choice(prompt: str) -> str:
    """Модуль выбора (`choose_*`): последний вариант списка — номер или название; исход — никогда.

    Варианты стоят в конце данных хода (`<<OPTIONS>>` — последняя подстановка промпта задачи):
    берётся последняя строка, если это вариант «N. …» или «- …». Особый исход (срочно) adversary
    не выбирает — срочность он только понижает.
    """

    lines = prompt.strip().splitlines()
    last = lines[-1] if lines else ""
    numbered = _NUMBERED_RE.match(last)
    picks: list[Any] = [int(numbered.group(1))] if numbered else ([last[2:].strip()] if last.startswith("- ") else [])
    return json.dumps({"picks": picks, "outcome": "none", "reason": "none", "search_terms": []}, ensure_ascii=False)


def adversary_answer(site: str, prompt: str) -> str | None:
    """Худший правдоподобный ответ места: формат верный, решение — неверное.

    Срочность adversary всегда ПОНИЖАЕТ (метка не URGENT, «НЕТ» у специалистов): так
    стенд проверяет монотонную безопасность — LLM не может снять срочность.

    :return: текст ответа модели или None для неизвестного места
    """

    if site.startswith("choose_"):
        return _adversary_choice(prompt)
    if site in ("classifier_patient", "classifier_refine_patient"):
        payload = {"label": _pick(_ADVERSARY_LABELS, prompt), "confidence": 0.95, "context_action": "new_topic", "entities": {}}
        return json.dumps(payload, ensure_ascii=False)
    if site == "price_service_select":  # прежний промпт выбора строки — снят 17.10; место нужно прогонам кода до 17.10
        last = _last_number(prompt.split("Ответь только JSON")[0])
        if last is None:
            return json.dumps({"match": [], "search_terms": ["Общий анализ крови"]}, ensure_ascii=False)
        return json.dumps({"match": [last], "search_terms": []})
    if site == "prepare_document_select":
        last = _last_number(prompt.split("Ответь только JSON")[0])
        return json.dumps({"match": last})
    if site == "prepare_same_memo":
        return json.dumps({"same": True})
    if site == "turn_spans":  # разметка слов реплики (DESIGN-2026-10-23-NO-STOPWORDS): вся реплика одной услугой — список слит, вопрос внутри
        numbered = prompt.split("Слова сообщения с номерами:")[-1]
        count = len(re.findall(r"(?:^|\s)(\d+):", numbered))
        return json.dumps({"phrases": [list(range(1, count + 1))] if count else []})
    if site == "price_kind_ambiguous":
        return json.dumps({"kind": _pick(_PRICE_KINDS, prompt), "reason": "так"}, ensure_ascii=False)
    fixed = {
        "service_normalizer": "Общий анализ крови",
        "patient_name_validator": "OTHER",
        "service_slot_validator": "OTHER",
        "result_timing_validator": "TIMING",
        "appointment_change_validator": "ДРУГОЕ",
        "existing_appointment_validator": "НЕТ",
        "empty_evidence_kind": "BYE",
        "symptom_specialists": "НЕТ",
        "renderer_patient": ADVERSARY_RENDER_TEXT,
        "renderer_patient_rich": ADVERSARY_RENDER_TEXT,
        "renderer_critic_patient_alignment": json.dumps(
            {"aligned_with_user_intent": False, "fact_consistency": False, "score": 0}
        ),
        "warm_up": "да",
    }
    return fixed.get(site)


# --------------------------------------------------------------------------- кэш ответов


class ResponseCache:
    """Записанные ответы настоящей LLM: ключ — модель, формат, поток и полный промпт.

    Хранится вне репо (в нём промпты с репликами пациентов). Повторный прогон «после»
    с теми же промптами не грузит LLM и не шумит температурой.

    В записи — отпечаток параметров генерации (`options`): ответ, записанный при других параметрах
    (температура 0,2 вместо 0), при повторе не подходит. Записи до 10.10 параметров не хранят и
    подходят всегда. При дублях побеждает ПЕРВАЯ подходящая запись — её видел прогон, против которого
    сравнивают (дубли дают параллельные части и повторные живые прогоны).
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._data: dict[str, list[tuple[str | None, str]]] = {}
        if path is not None and path.exists():
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    self._data.setdefault(row["key"], []).append((row.get("options"), row["response"]))

    @staticmethod
    def key(model: str, prompt: str, fmt: Any, stream: bool, system: str | None = None) -> str:
        """Схема ответа (dict) — в каноническом виде, инструкция роли system — часть ключа.

        Ключ записи без system и со строковым форматом прежний: записанные ответы в силе.
        """

        fmt_key = json.dumps(fmt, sort_keys=True, ensure_ascii=False) if isinstance(fmt, dict) else fmt
        parts: list[Any] = [model, fmt_key, bool(stream), prompt]
        if system:
            parts.append(system)
        raw = json.dumps(parts, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str, options: str = "") -> str | None:
        for recorded, response in self._data.get(key, ()):
            if recorded is None or recorded == options:
                return response
        return None

    def put(self, key: str, response: str, site: str, options: str = "") -> None:
        self._data.setdefault(key, []).append((options, response))
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            row = {"key": key, "site": site, "options": options, "response": response}
            line = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            # Одна запись O_APPEND — строки параллельных частей прогона (--shard) не перемешиваются.
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)


# --------------------------------------------------------------------------- клиент


class LiveBudgetExhausted(RuntimeError):
    """Исчерпан бюджет вызовов настоящей LLM на прогон."""


@dataclass
class CallRecord:
    site: str
    mode: str
    outcome: str  # ok | adversary | cache_hit | down | timeout | replay_miss | unknown_site | error:<тип>
    ms: int
    stream: bool
    fmt: str | None
    prompt_chars: int
    eval_count: int | None = None


@dataclass
class HarnessClient:
    """Объект на месте `llm_runtime._OLLAMA_CLIENT` с тем же методом `generate`."""

    mode: str
    sites: SiteIndex
    real: Any = None
    cache: ResponseCache | None = None
    live_pause_s: float = 0.0
    max_live_calls: int | None = None
    hang_s: float = 0.0
    calls: list[CallRecord] = field(default_factory=list)
    live_calls: int = 0
    default_options: dict[str, Any] | None = None
    _last_live: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"режим LLM: {self.mode!r}, ожидался один из {MODES}")
        if self.mode in _LIVE_MODES and self.real is None:
            raise ValueError("режиму с настоящей LLM нужен настоящий клиент")

    def _options_fingerprint(self, options: Any) -> str:
        """Отпечаток параметров генерации (температура, длина ответа); умолчание — пустая строка."""

        if options is None:
            return ""
        dumped = options.model_dump(exclude_none=True) if hasattr(options, "model_dump") else dict(options)
        if self.default_options is not None and dumped == self.default_options:
            return ""
        return json.dumps(dumped, sort_keys=True, ensure_ascii=False)

    def take_turn_calls(self) -> list[CallRecord]:
        """Вызовы с прошлого запроса — их пишет прогон в строку хода."""

        out, self.calls = self.calls, []
        return out

    async def generate(
        self,
        model: str = "",
        prompt: str | None = None,
        *,
        stream: bool = False,
        format: Any = None,  # noqa: A002 — имя параметра как у ollama
        system: str | None = None,
        **kwargs: Any,
    ):
        prompt = prompt or ""
        # Модель видит инструкцию (роль system) и данные хода вместе — по ним место вызова и adversary.
        seen = f"{system}\n\n{prompt}" if system else prompt
        site = self.sites.site_of(seen)
        started = time.monotonic()
        fmt = format if isinstance(format, str) else ("schema" if format else None)

        def record(outcome: str, eval_count: int | None = None) -> None:
            self.calls.append(
                CallRecord(site, self.mode, outcome, int((time.monotonic() - started) * 1000), stream, fmt, len(seen), eval_count)
            )

        if site == "unknown":
            record("unknown_site")
            raise ConnectionError("стенд: нераспознанный промпт")
        if self.mode == "down":
            record("down")
            raise ConnectionError("стенд: LLM недоступна")
        if self.hang_s and self.mode in ("down_timeout", "replay"):
            try:
                await asyncio.sleep(self.hang_s)
            except asyncio.CancelledError:
                record("cancelled")
                raise
        if self.mode == "down_timeout":
            record("timeout")
            raise asyncio.TimeoutError()
        if self.mode == "adversary":
            text = adversary_answer(site, seen)
            if text is None:
                record("unknown_site")
                raise ConnectionError(f"стенд: нет ответа adversary для {site}")
            record("adversary")
            return _response(model, text, stream)

        key = ResponseCache.key(model, prompt, format, stream, system=system)
        opts = self._options_fingerprint(kwargs.get("options"))
        cached = self.cache.get(key, opts) if self.cache is not None and self.mode in ("cached", "replay") else None
        if cached is not None:
            record("cache_hit")
            return _response(model, cached, stream)
        if self.mode == "replay":
            record("replay_miss")
            raise ConnectionError("стенд: ответа нет в записи")

        if self.max_live_calls is not None and self.live_calls >= self.max_live_calls:
            record("budget")
            raise LiveBudgetExhausted(f"исчерпан бюджет живых вызовов: {self.max_live_calls}")
        wait = self.live_pause_s - (time.monotonic() - self._last_live)
        if wait > 0:
            await asyncio.sleep(wait)
        self.live_calls += 1
        if system is not None:
            kwargs["system"] = system
        try:
            res = await self.real.generate(model=model, prompt=prompt, stream=stream, format=format, **kwargs)
            if stream:
                text, eval_count = "", None
                async for chunk in res:
                    text += getattr(chunk, "response", "") or ""
                    eval_count = getattr(chunk, "eval_count", None) or eval_count
            else:
                text, eval_count = getattr(res, "response", "") or "", getattr(res, "eval_count", None)
        except asyncio.TimeoutError:
            record("timeout")
            raise
        except asyncio.CancelledError:
            # Вызов оборвал тайм-аут места вызова (asyncio.wait_for) — это тоже исход, не пропуск.
            record("cancelled")
            raise
        except Exception as exc:
            record(f"error:{type(exc).__name__}")
            raise
        finally:
            self._last_live = time.monotonic()
        if self.cache is not None:
            self.cache.put(key, text, site, opts)
        record("ok", eval_count)
        return _response(model, text, stream)


def _response(model: str, text: str, stream: bool):
    """Ответ настоящего типа ollama: подделка-dict давала зелёный тест и TypeError вживую (01.10)."""

    from ollama import GenerateResponse

    final = GenerateResponse(model=model, response=text, done=True, eval_count=len(text.split()))
    if not stream:
        return final

    async def chunks():
        yield final

    return chunks()
