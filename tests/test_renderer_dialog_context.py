"""Диалоговый контекст в LLM-рендере: follow-up больше не отвечается вслепую.

До 2026-07-10 промпт рендера получал ТОЛЬКО текущую реплику + Данные: вопрос
«а это подходит студентам?» после карточки акции LLM видел без карточки —
отвечал общими словами или предлагал оператора. Multi-turn вытягивали
рукописные followup-правила по одному кейсу (промо, price-family…).

Инварианты (класс):
1. Хвост state.history попадает в промпт (оба режима: обычный и rich),
   с ролями «Пациент:/Ассистент:» и в хронологическом порядке.
2. Бюджет жёсткий: не более 6 ходов, длинные сообщения обрезаются,
   суммарный блок ограничен — история не раздувает промпт бесконечно.
3. Текущая реплика в контекст НЕ дублируется (append_turn идёт после
   пайплайна — проверяется тем, что берётся ровно state.history).
4. Плейсхолдер <<DIALOG_CONTEXT>> не протекает в готовый промпт.
"""

from __future__ import annotations

import asyncio

from messengers_router.mess_types import Evidence, RouteDecision, SessionState
from messengers_router.orchestrator import OrchestratorContext, render
from messengers_router.renderer import (
    _DIALOG_CTX_BUDGET,
    _DIALOG_CTX_EMPTY,
    _DIALOG_CTX_MAX_TURNS,
    _final_prompt,
    _final_prompt_rich,
    _format_dialog_context,
)


def run(coro):
    return asyncio.run(coro)


def _turn(role, text):
    return {"role": role, "text": text}


def test_format_orders_and_labels_roles():
    hist = [
        _turn("user", "какие акции есть?"),
        _turn("assistant", "Сейчас в клинике действуют акции: 1. ЧЕКАП + ВИТАМИН D"),
    ]
    block = _format_dialog_context(hist)
    lines = block.split("\n")
    assert lines[0].startswith("Пациент: какие акции")
    assert lines[1].startswith("Ассистент: Сейчас в клинике")


def test_format_empty_history_placeholder():
    assert _format_dialog_context([]) == _DIALOG_CTX_EMPTY
    assert _format_dialog_context(None) == _DIALOG_CTX_EMPTY


def test_format_caps_turns_and_length():
    hist = [_turn("user", f"сообщение номер {i} " + "х" * 400) for i in range(20)]
    block = _format_dialog_context(hist)
    lines = block.split("\n")
    assert len(lines) <= _DIALOG_CTX_MAX_TURNS
    assert len(block) <= _DIALOG_CTX_BUDGET + 350  # бюджет + одна строка сверху
    assert "…" in lines[-1], "длинное сообщение обрезано"
    # свежайшее сообщение обязано выжить при любом бюджете
    assert "номер 19" in block


def test_format_multiline_bot_answer_flattened():
    hist = [_turn("assistant", "Акция «ЧЕКАП»\nусловия:\nстрока")]
    block = _format_dialog_context(hist)
    assert "\n" not in block.replace("", ""), block  # одна строка на ход
    assert "Акция «ЧЕКАП» условия: строка" in block


def _decision():
    return RouteDecision(label="OTHER", confidence=0.5, entities={}, flags=set())


def test_prompt_includes_context_and_no_placeholder_leak():
    hist = [_turn("assistant", "Акция «ЧЕКАП + ВИТАМИН D» действует до 31.08.2026")]
    for build in (
        lambda: _final_prompt("а студентам подходит?", _decision(), Evidence(), history=hist),
        lambda: _final_prompt_rich("а студентам подходит?", _decision(), Evidence(), history=hist),
    ):
        prompt = build()
        assert "ЧЕКАП + ВИТАМИН D" in prompt
        assert "<<DIALOG_CONTEXT>>" not in prompt
        assert "Ассистент: Акция" in prompt


def test_prompt_without_history_uses_placeholder_text():
    prompt = _final_prompt("привет", _decision(), Evidence())
    assert _DIALOG_CTX_EMPTY in prompt
    assert "<<DIALOG_CONTEXT>>" not in prompt


def test_orchestrator_passes_state_history_to_renderer(monkeypatch):
    captured = {}

    async def fake_render_stream(user_text, decision, evidence, runtime_options=None, *, history=None):
        _ = user_text, decision, evidence, runtime_options
        captured["history"] = history
        yield "ответ"

    monkeypatch.setattr("messengers_router.renderer.render_stream", fake_render_stream)

    state = SessionState(session_id="dlg-ctx")
    state.history = [_turn("user", "какие акции есть?"), _turn("assistant", "1. ЧЕКАП")]
    ctx = OrchestratorContext(text="а вторая какая?", state=state)
    ctx.decision = _decision()
    ctx.evidence = Evidence()

    out = run(render(ctx))

    assert out.response is not None and out.response.text == "ответ"
    assert captured["history"] == state.history
