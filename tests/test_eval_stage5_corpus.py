import json
import sys

from messengers_router.scripts import eval_stage5_corpus as stage5


def test_stage5_diagnostic_mode_prints_timings_and_slowest_cases(monkeypatch, tmp_path, capsys):
    golden = tmp_path / "stage5_golden.jsonl"
    golden.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "case_id": "G001",
                        "text": "Сколько стоит анализ?",
                        "expected_label": "PRICE",
                        "expected_handoff": False,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "case_id": "G002",
                        "text": "Где вы находитесь?",
                        "expected_label": "ADDRESS",
                        "expected_handoff": False,
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    responses = iter(
        [
            {
                "handoff": False,
                "state_update": {
                    "debug": {
                        "decision": {
                            "label": "PRICE",
                            "confidence": 0.91,
                            "flags": ["rule_price"],
                        },
                        "last_entities": {"service_name": "Анализ", "city": "Самара"},
                    }
                },
            },
            {
                "handoff": False,
                "state_update": {
                    "debug": {
                        "decision": {
                            "label": "ADDRESS",
                            "confidence": 0.88,
                            "flags": ["topic_registry:address"],
                        },
                        "last_entities": {"branch_name": "Ленина, 5"},
                    }
                },
            },
        ]
    )
    perf_values = iter([1.0, 1.4, 10.0, 11.2, 20.0, 24.6])

    monkeypatch.setattr(stage5, "preflight_endpoint", lambda url: (True, "ok"))
    monkeypatch.setattr(stage5, "post_json", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(stage5.time, "perf_counter", lambda: next(perf_values))
    monkeypatch.setattr(stage5.time, "time", lambda: 1776369999)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_stage5_corpus.py",
            "--golden",
            str(golden),
            "--url",
            "http://example.test/api/messenger-generate-once",
            "--diagnostic",
            "--slow-case-threshold-sec",
            "2.0",
        ],
    )

    result = stage5.main()

    captured = capsys.readouterr().out
    assert result == 0
    assert "[diag] PREFLIGHT ok elapsed=0.4s note=ok" in captured
    assert "[diag] START G001" in captured
    assert "[diag] END G001 elapsed=1.2s status=ok" in captured
    assert "[diag] START G002" in captured
    assert "[diag] END G002 elapsed=4.6s status=ok" in captured
    assert "Slowest cases:" in captured
    assert "- G002: 4.6s status=ok label=ADDRESS" in captured


def test_stage5_diagnostic_mode_prints_timeout_error(monkeypatch, tmp_path, capsys):
    golden = tmp_path / "stage5_golden.jsonl"
    golden.write_text(
        json.dumps(
            {
                "case_id": "G777",
                "text": "Тест timeout",
                "expected_label": "PRICE",
                "expected_handoff": False,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    perf_values = iter([1.0, 1.5, 100.0, 175.0])

    monkeypatch.setattr(stage5, "preflight_endpoint", lambda url: (True, "ok"))
    monkeypatch.setattr(stage5, "post_json", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("timed out")))
    monkeypatch.setattr(stage5.time, "perf_counter", lambda: next(perf_values))
    monkeypatch.setattr(stage5.time, "time", lambda: 1776370001)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_stage5_corpus.py",
            "--golden",
            str(golden),
            "--url",
            "http://example.test/api/messenger-generate-once",
            "--diagnostic",
        ],
    )

    result = stage5.main()

    captured = capsys.readouterr().out
    assert result == 2
    assert "[diag] PREFLIGHT ok elapsed=0.5s note=ok" in captured
    assert "[diag] START G777" in captured
    assert "[diag] ERROR G777 elapsed=75.0s error=TimeoutError: timed out" in captured
    assert "transport_errors: 1" in captured
