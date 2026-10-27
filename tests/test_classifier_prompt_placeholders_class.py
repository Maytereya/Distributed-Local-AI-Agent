"""Заглушки промптов классификатора и подстановки кода совпадают (STEP-2026-10-27-DEAD-RULES).

`_build_classify_prompt` с 24.03 считал хинты правил и подставлял `<<RULE_HINTS>>`, а заглушки в промпте не было ни в одной из
двух локаций — хинты считались и выбрасывались. Инвариант класса: код подставляет только заглушки, которые есть в шаблоне
(подстановка без заглушки — мёртвый код или потерянный вход модели), в обеих локациях шаблона заглушки одни и те же, и в
собранном промпте не остаётся ни одной.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from messengers_router import classifier
from messengers_router.mess_types import RouteDecision

_ROOT = Path(__file__).resolve().parents[1]
_PLACEHOLDER_RE = re.compile(r"<<[A-Z_]+>>")
_REPLACED_RE = re.compile(r"replace\(\s*\"(<<[A-Z_]+>>)\"")
_BASE = RouteDecision(label="DOCTOR_SCHEDULE", confidence=0.74, entities={}, flags=set(), needs_handoff=False, context_action="continue")
_BUILDERS = {
    "classifier_patient": (classifier._build_classify_prompt, lambda: classifier._build_classify_prompt("расписание кардиолога", {})),
    "classifier_refine_patient": (
        classifier._build_refine_prompt,
        lambda: classifier._build_refine_prompt("расписание кардиолога", {}, _BASE),
    ),
}


def _templates(key: str) -> list[str]:
    return [
        (_ROOT / "messengers_router" / "prompts" / f"{key}.txt").read_text(encoding="utf-8"),
        (_ROOT / "app_data" / "prompts" / f"mr_{key}.txt").read_text(encoding="utf-8"),
    ]


@pytest.mark.parametrize("key", sorted(_BUILDERS))
def test_code_replaces_only_placeholders_present_in_both_templates(key):
    builder, _ = _BUILDERS[key]
    replaced = set(_REPLACED_RE.findall(inspect.getsource(builder)))
    bundle, host = (set(_PLACEHOLDER_RE.findall(t)) for t in _templates(key))

    assert bundle == host, (key, bundle ^ host)
    assert replaced <= bundle, (key, replaced - bundle)
    assert bundle <= replaced, (key, bundle - replaced)


@pytest.mark.parametrize("key", sorted(_BUILDERS))
def test_built_prompt_has_no_placeholders_left(key):
    _, build = _BUILDERS[key]
    assert not _PLACEHOLDER_RE.findall(build()), key
