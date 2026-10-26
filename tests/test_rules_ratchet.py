"""Храповик правил: регулярок и списков слов в ядре не становится больше (контрметрика Гудхарта).

Тема рефакторинга — единая система и LLM first, а не нагромождение правил на правила (владелец,
11.10). Стенд меряет качество ответов, и метрику легко поднять подгонкой: ещё одной регуляркой,
словарём или исключением под фразу, найденную на том же эталоне. Этот тест держит обратную сторону:
правил в ядре (`messengers_router`, без тестов) не прибавляется.

Что считается (по совету сессии этапа 1 — классами, чтобы правило не «переехало» из одной формы в
другую):
- `re.compile(` — именованные регулярки;
- регулярка-литерал прямо в вызове `re.search / match / fullmatch / findall / finditer / sub / subn /
  split` — та же регулярка, спрятанная внутрь функции;
- список слов на уровне модуля — `set / frozenset / list / tuple / dict` из пяти и больше строк.

Как работать: новое правило — только вместе со снятием другого в том же коммите (+1 только с −1).
Число упало — опустите потолок здесь же. Потолок не поднимают, чтобы позеленеть: если без нового
правила нельзя, это решение владельца, и оно записывается в журнал вместе с ходом, который правило
снимет. Главный сигнал качества — разметка реального трафика раз в 2–4 недели; стенд — ограждение.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_CORE = Path(__file__).resolve().parents[1] / "messengers_router"

# Потолки на 11.10.2026 (после хода (б) шаг 1). Эвристики шага 1 — долг с ходом снятия, журнал
# STEP-2026-10-10-B1-TURN-UNDERSTANDING.
_CEILINGS = {
    "re.compile": 258,  # 259 → 258: снят путь off `grounder_generic_by_llm` — регулярка общих слов заземления (STEP-2026-10-31-SWITCH-REMOVAL-CONSUMERS)
    "регулярка-литерал в вызове re.*": 158,  # 160 → 158: мёртвые литералы `_extract_schedule_specialty` — их слова раньше ловит `extract_specialty` (STEP-2026-10-27-DEAD-RULES)
    "список слов на уровне модуля (≥5 строк)": 100,  # 101 → 100: мёртвый `_CYR_TO_LAT` (`fold_cyrillic_homoglyphs` без вызовов) (STEP-2026-10-27-DEAD-RULES)
}

_INLINE_RE = re.compile(r"re\.(?:search|match|fullmatch|findall|finditer|sub|subn|split)\(\s*r?[\"']")


def _sources() -> list[tuple[Path, str]]:
    return [(p, p.read_text(encoding="utf-8")) for p in sorted(_CORE.rglob("*.py")) if "tests" not in p.parts]


def _word_lists(source: str) -> int:
    count = 0
    for node in ast.parse(source).body:
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
        if isinstance(value, ast.Call) and getattr(value.func, "id", None) in {"frozenset", "set", "tuple", "list"} and value.args:
            value = value.args[0]
        if isinstance(value, (ast.Set, ast.List, ast.Tuple)):
            elements = value.elts
        elif isinstance(value, ast.Dict):
            elements = value.keys
        else:
            continue
        if sum(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in elements) >= 5:
            count += 1
    return count


def _counts() -> dict[str, int]:
    sources = _sources()
    return {
        "re.compile": sum(src.count("re.compile(") for _, src in sources),
        "регулярка-литерал в вызове re.*": sum(len(_INLINE_RE.findall(src)) for _, src in sources),
        "список слов на уровне модуля (≥5 строк)": sum(_word_lists(src) for _, src in sources),
    }


def test_rules_in_core_do_not_grow():
    counts = _counts()
    grown = {name: f"{counts[name]} > потолка {ceiling}" for name, ceiling in _CEILINGS.items() if counts[name] > ceiling}
    assert grown == {}, f"в ядре прибавилось правил (снимите другое в том же коммите): {grown}"
