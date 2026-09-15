"""Кэш имени модели: переключение рычагом в Gradio должен видеть и бот.

Инцидент 15.09.2026. Модель переключили в Gradio, `main_model.txt` перезаписался,
но прод продолжил работать на старой модели, пока контейнер бота не перезапустили.
Причина — `LLMName.current_llm` читается ОДИН РАЗ за жизнь процесса:

    if cls.current_llm is not None:
        return

Gradio и бот живут в разных контейнерах, файл у них общий, а кэш — нет. Снаружи
выглядит так, будто модель сменилась (файл-то переписан), а прод использует
старую. Полдня диагностики ушло, чтобы дойти до этого.

Инвариант: имя модели перечитывается, когда файл изменился на диске, — как это
уже сделано для дневного среза МИС (кэш по mtime).
"""

from __future__ import annotations

import time
from pathlib import Path

from agent_logic_2 import ollama_settings as os_mod


def _point_to(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "main_model.txt"
    monkeypatch.setattr(os_mod, "main_model_path", path)
    monkeypatch.setattr(os_mod, "settings_dir", tmp_path)
    monkeypatch.setattr(os_mod.LLMName, "current_llm", None)
    return path


def test_model_name_follows_the_file_after_external_change(tmp_path: Path, monkeypatch):
    """Файл переписали снаружи (рычагом в другом процессе) — имя обязано обновиться."""
    path = _point_to(tmp_path, monkeypatch)
    path.write_text("mistral-small3.2:24b-instruct-2506-fp16", encoding="utf-8")
    assert os_mod.LLMName.get() == "mistral-small3.2:24b-instruct-2506-fp16"

    time.sleep(0.01)
    path.write_text("qwen3.5:35b-a3b-bf16", encoding="utf-8")

    assert os_mod.LLMName.get() == "qwen3.5:35b-a3b-bf16", (
        "имя модели закэшировано намертво — это и есть дефект 15.09"
    )


def test_model_name_survives_missing_file(tmp_path: Path, monkeypatch):
    """Файла нет — падаем на config.ini, как и раньше (fail-open)."""
    _point_to(tmp_path, monkeypatch)
    assert os_mod.LLMName.get() == os_mod.c.ll_model_small


def test_set_updates_both_file_and_cache(tmp_path: Path, monkeypatch):
    """Рычаг в своём процессе обязан обновить и файл, и кэш — без перечитывания."""
    path = _point_to(tmp_path, monkeypatch)
    path.write_text("gpt-oss:20b", encoding="utf-8")
    assert os_mod.LLMName.get() == "gpt-oss:20b"

    os_mod.LLMName.set("mistral-small3.2:24b-instruct-2506-q4_K_M")

    assert path.read_text(encoding="utf-8").strip() == "mistral-small3.2:24b-instruct-2506-q4_K_M"
    assert os_mod.LLMName.get() == "mistral-small3.2:24b-instruct-2506-q4_K_M"
