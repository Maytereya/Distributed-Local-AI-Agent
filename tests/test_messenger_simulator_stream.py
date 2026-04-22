from messenger_simulator import parse_jsonl_stream


def test_parse_jsonl_stream_supports_delta_chunks():
    result = parse_jsonl_stream(
        [
            '{"text":"Привет", "attachments":[], "handoff":false}',
            '{"text":", мир", "attachments":[], "handoff":false}',
        ]
    )

    assert result.text == "Привет, мир"
    assert result.handoff is False
    assert result.attachments == []
    assert result.error is None


def test_parse_jsonl_stream_supports_partial_chunks_and_meta():
    result = parse_jsonl_stream(
        [
            '{"text":"Прив", "attachments":[], "handoff":false}',
            '{"text":"Привет", "attachments":[{"type":"url","name":"Документ","url":"https://example.com"}], "handoff":false}',
            '{"text":"", "attachments":[], "handoff":true}',
        ]
    )

    assert result.text == "Привет"
    assert result.handoff is True
    assert result.attachments == [{"type": "url", "name": "Документ", "url": "https://example.com"}]
