"""Оракул разметки `turn_spans` для тестов пути прода (`turn_spans_llm = on`, деплой 17; STEP-2026-10-15-DEFAULTS-B, переворот 27.10).

Под `turn_spans_llm = on` разбор реплики — LLM размечает номера слов (промпт `turn_spans`, контракт `{"phrases": [[первое,
последнее], ...]}`). Подделка LLM теста, которая знает только свои места (классификатор, выбор строки, старый промпт выбора фраз
`choose_turn_phrases`), на разметку отвечает не по контракту — разбор `invalid`, и тест видит путь сбоя вместо пути прода. Оракул
отвечает на разметку ровно заданными фразами, остальное отдаёт подделке теста — настоящего типа (`GenerateResponse` на границе
`llm_runtime._OLLAMA_CLIENT`), дальше — настоящий разбор.

Образец — `tests/test_price_stage2_turns._Oracle` (там же — выбор строки по началу названия). Фраза ищется в словах реплики после
той же нормализации, что у каталога (`_normalise_catalog_text`): первое вхождение; фразы, которой в реплике нет, — пропуск, как у
модели, которая её не отметила.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from ollama import GenerateResponse

from messengers_router.services._common import _normalise_catalog_text
from messengers_router.services._prices_helpers import _PRICE_TOKEN_RE

SPANS_MARK = "Слова сообщения с номерами:"
INSURANCE_MARK = "Слова пациента об оплате"  # вопрос об оплате по метке разметки (`choose_payment_insurance`)


def is_spans(prompt: str) -> bool:
    """Запрос — разметка слов `turn_spans`."""

    return SPANS_MARK in (prompt or "")


def insurance_response(prompt: str, confirm: bool) -> GenerateResponse:
    """Ответ на вопрос об оплате: все отмеченные слова — оплата по ОМС / ДМС / страховке (`confirm`) или ни одно."""

    numbers = [int(head) for head, dot, _rest in (line.partition(". ") for line in prompt.split(INSURANCE_MARK, 1)[1].splitlines()) if dot and head.isdigit()]
    return GenerateResponse(model="m", response=json.dumps({"picks": numbers if confirm else []}), done=True)


def _words(prompt: str) -> list[str]:
    numbered = prompt.split(SPANS_MARK, 1)[1].strip().splitlines()[0].split()
    return [_normalise_catalog_text(item.partition(":")[2]) for item in numbered]


def _find(words: list[str], text: str) -> list[int] | None:
    """Первое вхождение слов `text` в слова разметки — [первое, последнее] с 1; нет — None. Слова — как у разметки
    (`_turn_spans.words`): дефис и «+» — разделители («врача-терапевта» — два слова)."""

    want = [_normalise_catalog_text(token) for token in _PRICE_TOKEN_RE.findall(str(text or ""))]
    for k in range(len(words) - len(want) + 1):
        if want and words[k : k + len(want)] == want:
            return [k + 1, k + len(want)]
    return None


def _groups(prompt: str, phrases: Iterable[str]) -> list[tuple[str, list[int]]]:
    words, groups = _words(prompt), []
    for phrase in phrases:
        found = _find(words, phrase)
        if found and not any(found[0] <= last and first <= found[1] for _phrase, (first, last) in groups):
            groups.append((phrase, found))
    return groups


def spans_groups(prompt: str, phrases: Iterable[str]) -> list[list[int]]:
    """Номера [первое, последнее] каждой фразы в словах запроса разметки (по порядку фраз; нет в реплике — пропуск).

    Группы не пересекаются (правило 7 промпта): фраза, задевающая уже отмеченную, — пропуск. Поэтому один перечень фраз годится
    на весь диалог — длинная фраза раньше короткой («колоноскопии под наркозом», затем «колоноскопии»).
    """

    return [group for _phrase, group in _groups(prompt, phrases)]


def spans_response(prompt: str, phrases: Iterable[str], qualifiers: Iterable[tuple[str | None, str, str]] | None = None) -> GenerateResponse:
    """Ответ модели на разметку — JSON-строкой, как у живой.

    `qualifiers` (уточнения позиции, `position_qualifiers = on`): `(фраза или None, слова уточнения, вид)` — уточнение фразы из
    `phrases` (её номер группы) или ни к одной (`None` → `of = 0`); уточнения, слов которых нет в реплике или чья фраза не
    отмечена, — пропуск. None — ответ без поля (промпт без уточнений)."""

    groups = _groups(prompt, phrases)
    answer: dict[str, Any] = {"phrases": [group for _phrase, group in groups]}
    if qualifiers is not None:
        words, numbers = _words(prompt), {phrase: k for k, (phrase, _group) in enumerate(groups, 1)}
        answer["qualifiers"] = [
            {"of": numbers[of] if of is not None else 0, "words": found, "kind": kind}
            for of, text, kind in qualifiers
            if (found := _find(words, text)) and (of is None or of in numbers)
        ]
    return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)


class SpansOracle:
    """Разметка — заданные фразы (пусто — «ни одной услуги», None — разметка недоступна); остальное — `inner` или «недоступна».

    `phrases` можно сменить между ходами (`oracle.phrases = (...)`): у каждого хода своя реплика. `asked` — места, куда пришёл
    ход (`spans` / `spans:down` / `inner`), для проверки, что разметку вообще спросили. `qualifiers` — уточнения позиции в ответе
    (см. `spans_response`; None — ответ без поля). `insurance` — ответ на вопрос об оплате по метке «insurance»: True — все
    отмеченные слова про ОМС / ДМС / страховку, False — ни одно, None — вопрос недоступен (прежнее правило «суд + метка»).
    """

    def __init__(
        self,
        phrases: Iterable[str] | None = (),
        inner: Any = None,
        qualifiers: Iterable[tuple[str | None, str, str]] | None = None,
        insurance: bool | None = None,
    ) -> None:
        self.phrases = tuple(phrases) if phrases is not None else None
        self.inner = inner
        self.qualifiers = tuple(qualifiers) if qualifiers is not None else None
        self.insurance = insurance
        self.asked: list[str] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **kwargs):
        if is_spans(prompt):
            if self.phrases is None:
                self.asked.append("spans:down")
                raise ConnectionError("тест: разметка недоступна")
            self.asked.append("spans")
            return spans_response(prompt, self.phrases, self.qualifiers)
        if INSURANCE_MARK in (prompt or ""):
            self.asked.append("insurance" if self.insurance is not None else "insurance:down")
            if self.insurance is None:
                raise ConnectionError("тест: вопрос об оплате недоступен")
            return insurance_response(prompt, self.insurance)
        if self.inner is None:
            raise ConnectionError("тест: LLM недоступна для этого места")
        self.asked.append("inner")
        return await self.inner.generate(prompt=prompt, system=system, **kwargs)


def keep_llm_breaker_closed(monkeypatch) -> None:
    """Предохранитель LLM-решений (`llm_decision_breaker_failures`: 3 сбоя подряд — пауза) в многоходовом тесте не размыкается.

    Подделка, которая отвечает только на часть мест (разметка — да, классификатор и прочее — «недоступна»), размыкает его уже
    в первом ходе, и разметку следующего хода код не спрашивает — тест видит сбой разметки вместо пути прода. На проде так не
    бывает: LLM либо отвечает, либо лежит целиком (тогда разомкнутый предохранитель — честный путь «LLM недоступна»).
    """

    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_LLM_DECISION_BREAKER_FAILURES", 0, raising=False)


def prepare_turn(services: Any, reply: str, entities: dict[str, Any] | None = None, *, phrases: Iterable[str] | None, monkeypatch) -> dict[str, Any]:
    """Ход подготовки, как на проде (`prepare_by_parse = on`, деплой 17): разбор реплики (`observe_prepare_turn`) — оракул с
    фразами `phrases` (None — разметка недоступна), затем `services.test_prepare` в том же ходе.

    Разбор хода живёт в переменной контекста хода: модульный тест, который зовёт `test_prepare` напрямую, получал «разбора нет»
    — на проде так не бывает (роутер строит разбор на ходу подготовки). Прочие места LLM — прежний клиент гейта или теста.
    """

    import asyncio

    from messengers_router import llm_runtime, turn_understanding

    oracle = SpansOracle(phrases, inner=llm_runtime._OLLAMA_CLIENT)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", oracle)

    async def go() -> dict[str, Any]:
        turn_understanding.begin_turn()
        await turn_understanding.observe_prepare_turn("PREPARE", user_text=reply)
        return await services.test_prepare(reply, dict(entities or {}))

    return asyncio.run(go())
