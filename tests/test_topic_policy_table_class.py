"""Таблица политик темы (ход (б), этап 3, шаг 1; решение владельца 20.10 (б) 7; STEP-2026-10-21-B-POLICY-TABLE).

Инвариант класса: метку решения после NLU меняет только строка таблицы `topic_policy.TABLE` — через
`router._relabel(decision, <строка>, ...)`, и каждая сработавшая строка, сменившая метку, пишет себя в телеметрию хода
(`tu_relabel_src`). Архитектурно: `_copy_decision(..., label=...)` в ядре — красный; строка вне таблицы, строка таблицы без
места в коде, место не в той функции (`site`) — красный. Поведенчески: у хода, где `tu_label_nlu ≠ tu_label_final`,
есть `tu_relabel_src`, и его последняя строка ведёт к итоговой метке.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from messengers_router import topic_policy, turn_telemetry
from tests.test_llm_primary_rules_fallback_class import _turn
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_CORE = Path(__file__).resolve().parents[1] / "messengers_router"
# Классификатор отвечает подбором анализов: под `topic_by_llm` прода (с 22.10) тему веток подбора и поиска анализа решает он, и
# переклейки «названная услуга нашлась в каталоге — цена» — строки таблицы (с OTHER корпус переклеек не давал: OTHER — итог).
_LLM_ANSWER = json.dumps({"label": "TEST_ASSIST", "confidence": 0.9, "context_action": "new_topic", "entities": {}})
_REPLIES = (
    "есть ли у вас анализ на ферритин",
    "анализ на ферритин и глюкозу есть?",
    "сдаете ли вы анализ на витамин д",
    "сколько стоит ферритин",
    "мне нужно к лору",
    "хочу записаться к кардиологу",
    "спасибо большое",
)


def _calls(name: str) -> list[tuple[Path, str, ast.Call]]:
    found: list[tuple[Path, str, ast.Call]] = []
    for path in sorted(_CORE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        stack: list[tuple[ast.AST, str]] = [(tree, "<module>")]
        while stack:
            node, fn = stack.pop()
            for child in ast.iter_child_nodes(node):
                inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
                if isinstance(child, ast.Call) and getattr(child.func, "id", None) == name:
                    found.append((path, inner, child))
                stack.append((child, inner))
    return found


def test_label_changes_only_through_policy_table():
    bare = [
        f"{path.name}:{call.lineno} ({fn})"
        for path, fn, call in _calls("_copy_decision")
        if any(kw.arg == "label" for kw in call.keywords)
    ]
    assert not bare, f"метка меняется мимо таблицы политик (нужен _relabel со строкой topic_policy.TABLE): {bare}"


def test_every_relabel_names_a_table_row_at_its_site():
    used: set[str] = set()
    for path, fn, call in _calls("_relabel"):
        assert len(call.args) >= 2 and isinstance(call.args[1], ast.Constant), f"{path.name}:{call.lineno}: строка — константой"
        policy = call.args[1].value
        assert policy in topic_policy.TABLE, f"{path.name}:{call.lineno}: строки {policy!r} нет в таблице"
        row = topic_policy.TABLE[policy]
        assert row.site == fn, f"{policy}: в таблице место {row.site}, в коде — {fn}"
        to = next((kw.value for kw in call.keywords if kw.arg == "label"), None)
        if row.to is not None and isinstance(to, ast.Constant):
            assert to.value == row.to, f"{policy}: в таблице → {row.to}, в коде → {to.value}"
        used.add(policy)
    assert set(topic_policy.TABLE) == used, f"строки таблицы без места в коде: {sorted(set(topic_policy.TABLE) - used)}"


def test_row_ids_are_unique_and_described():
    ids = [row.id for row in topic_policy._ROWS]
    assert len(ids) == len(set(ids)), ids
    assert all(row.why and row.layer and row.site for row in topic_policy._ROWS)


def test_relabel_writes_itself_only_when_label_changes():
    token = turn_telemetry.begin("test", session_id="s")
    topic_policy.note("flow_pending_answer", "APPOINTMENT", "APPOINTMENT")
    topic_policy.note("test_assist_catalog_hit_is_price", "TEST_ASSIST", "PRICE")
    row = turn_telemetry.finish(token, handoff=False, chars=0)
    assert row is not None and row["tu_relabel_src"] == ["test_assist_catalog_hit_is_price:TEST_ASSIST>PRICE"], row


def test_changed_label_is_explained_by_table(mini_prod_world, monkeypatch):  # noqa: F811
    changed = 0
    for reply in _REPLIES:
        tel = _turn(mini_prod_world, monkeypatch, reply, engine="legacy_v2", fallback="off", answer=_LLM_ANSWER)["telemetry"]
        if "tu_label_nlu" not in tel or tel["tu_label_nlu"] == tel.get("tu_label_final"):
            continue
        changed += 1
        trail = tel.get("tu_relabel_src") or []
        assert trail, (reply, tel)
        assert trail[-1].endswith(f">{tel['tu_label_final']}"), (reply, trail)
        assert all(step.split(":", 1)[0] in topic_policy.TABLE for step in trail), trail
    assert changed >= 2, "корпус вырожден: ни одна переклейка не сработала"
