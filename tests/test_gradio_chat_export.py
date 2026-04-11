import json
from pathlib import Path

from agent_logic_2.gradio_ui.tabs import assistant_tab as assistant_tab_mod


def test_build_chat_export_payload_preserves_messages_and_mode():
    payload = assistant_tab_mod._build_chat_export_payload(
        chat_history=[
            {"role": "user", "content": "Привет"},
            {"role": "assistant", "content": "Здравствуйте"},
        ],
        mode="Free-talk-Ai",
        session_id="gr_ft_test",
    )

    assert payload["mode"] == "Free-talk-Ai"
    assert payload["session_id"] == "gr_ft_test"
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][1]["content"] == "Здравствуйте"
    assert "user: Привет" in payload["transcript"]


def test_export_chat_dialog_writes_json_file(tmp_path, monkeypatch):
    monkeypatch.setattr(assistant_tab_mod, "_EXPORT_DIR", tmp_path)

    path = assistant_tab_mod.export_chat_dialog(
        [
            {"role": "user", "content": "Первый вопрос"},
            {"role": "assistant", "content": "Первый ответ"},
        ],
        "Messengers-Ai",
        "gr_mr_test",
    )

    file_path = Path(path)
    assert file_path.exists()
    data = json.loads(file_path.read_text(encoding="utf-8"))
    assert data["mode"] == "Messengers-Ai"
    assert data["session_id"] == "gr_mr_test"
    assert len(data["messages"]) == 2
