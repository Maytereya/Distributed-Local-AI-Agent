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

SPANS_MARK = "Слова сообщения с номерами:"


def is_spans(prompt: str) -> bool:
    """Запрос — разметка слов `turn_spans`."""

    return SPANS_MARK in (prompt or "")


def spans_groups(prompt: str, phrases: Iterable[str]) -> list[list[int]]:
    """Номера [первое, последнее] каждой фразы в словах запроса разметки (по порядку фраз; нет в реплике — пропуск)."""

    numbered = prompt.split(SPANS_MARK, 1)[1].strip().splitlines()[0].split()
    words = [_normalise_catalog_text(item.partition(":")[2]) for item in numbered]
    groups: list[list[int]] = []
    for phrase in phrases:
        want = _normalise_catalog_text(phrase).split()
        for k in range(len(words) - len(want) + 1):
            if want and words[k : k + len(want)] == want:
                groups.append([k + 1, k + len(want)])
                break
    return groups


def spans_response(prompt: str, phrases: Iterable[str]) -> GenerateResponse:
    """Ответ модели на разметку — JSON-строкой, как у живой."""

    return GenerateResponse(model="m", response=json.dumps({"phrases": spans_groups(prompt, phrases)}), done=True)


class SpansOracle:
    """Разметка — заданные фразы (пусто — «ни одной услуги», None — разметка недоступна); остальное — `inner` или «недоступна».

    `phrases` можно сменить между ходами (`oracle.phrases = (...)`): у каждого хода своя реплика. `asked` — места, куда пришёл
    ход (`spans` / `spans:down` / `inner`), для проверки, что разметку вообще спросили.
    """

    def __init__(self, phrases: Iterable[str] | None = (), inner: Any = None) -> None:
        self.phrases = tuple(phrases) if phrases is not None else None
        self.inner = inner
        self.asked: list[str] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **kwargs):
        if is_spans(prompt):
            if self.phrases is None:
                self.asked.append("spans:down")
                raise ConnectionError("тест: разметка недоступна")
            self.asked.append("spans")
            return spans_response(prompt, self.phrases)
        if self.inner is None:
            raise ConnectionError("тест: LLM недоступна для этого места")
        self.asked.append("inner")
        return await self.inner.generate(prompt=prompt, system=system, **kwargs)
