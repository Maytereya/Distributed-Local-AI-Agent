"""Эвристический триаж дневных диалогов: детект красных флагов без LLM.

Проверяем на синтетическом логе в РЕАЛЬНОМ формате воркера (наблюдён 10.07):
операторский фолбэк, повтор-луп (кейс УЗИ-мошонки), дефлект-петля, чистый диалог.
"""

from __future__ import annotations

from scripts.analyze_daily_dialogs import build_digest, parse_log

# Синтетический лог: формат строк 1:1 с прод-воркером. Каждый входящий — пара
# «Received from USER (chat=..)» + «Incoming ... conv=#N from USER» (мост chat↔conv).
_LOG = """\
2026-07-10 16:00:16,311 [INFO] channels_app.management.commands.run_telegram: Received from Losinui (chat=111): Стоимость колоноскопии
2026-07-10 16:00:16,315 [INFO] channels_app.services: Incoming message conv=#619 from Losinui: Стоимость колоноскопии
2026-07-10 16:00:20,034 [INFO] channels_app.adapters.telegram: Telegram edit msg_id=5941 in chat=111: По услуге колоноскопия нашёл следующее: 1) цена 3500
2026-07-10 16:15:08,765 [INFO] channels_app.management.commands.run_telegram: Received from Anastasiiichik (chat=222): Дементьев 1995 стб 493
2026-07-10 16:15:08,769 [INFO] channels_app.services: Incoming message conv=#524 from Anastasiiichik: Дементьев 1995 стб 493
2026-07-10 16:15:08,815 [INFO] channels_app.services: Sending to telegram:222 conv=#524: Ожидайте, оператор скоро ответит.
2026-07-10 22:21:09,000 [INFO] channels_app.management.commands.run_telegram: Received from UziGuy (chat=333): УЗИ органов мошонки
2026-07-10 22:21:10,000 [INFO] channels_app.services: Incoming message conv=#700 from UziGuy: УЗИ органов мошонки
2026-07-10 22:21:11,000 [INFO] channels_app.adapters.telegram: Telegram edit msg_id=6001 in chat=333: 1. Казакова 2. Ларионова 3. Портянникова 4. Свиридова
2026-07-10 22:21:29,000 [INFO] channels_app.management.commands.run_telegram: Received from UziGuy (chat=333): Кого рекомендуешь
2026-07-10 22:21:30,000 [INFO] channels_app.services: Incoming message conv=#700 from UziGuy: Кого рекомендуешь
2026-07-10 22:21:31,000 [INFO] channels_app.adapters.telegram: Telegram edit msg_id=6002 in chat=333: 1. Казакова 2. Ларионова 3. Портянникова 4. Свиридова
2026-07-10 22:21:59,000 [INFO] channels_app.management.commands.run_telegram: Received from UziGuy (chat=333): Кто из них топ
2026-07-10 22:22:00,000 [INFO] channels_app.services: Incoming message conv=#700 from UziGuy: Кто из них топ
2026-07-10 22:22:01,000 [INFO] channels_app.adapters.telegram: Telegram edit msg_id=6003 in chat=333: 1. Казакова 2. Ларионова 3. Портянникова 4. Свиридова
2026-07-10 12:00:00,000 [INFO] channels_app.management.commands.run_telegram: Received from Confused (chat=444): расскажи анекдот
2026-07-10 12:00:00,500 [INFO] channels_app.services: Incoming message conv=#800 from Confused: расскажи анекдот
2026-07-10 12:00:02,000 [INFO] channels_app.adapters.telegram: Telegram send to chat=444: Ответ в процессе...
2026-07-10 12:00:03,000 [INFO] channels_app.adapters.telegram: Telegram edit msg_id=7001 in chat=444: Не совсем понял ваш запрос
2026-07-10 12:00:19,000 [INFO] channels_app.management.commands.run_telegram: Received from Confused (chat=444): ну давай
2026-07-10 12:00:20,000 [INFO] channels_app.services: Incoming message conv=#800 from Confused: ну давай
2026-07-10 12:00:21,000 [INFO] channels_app.adapters.telegram: Telegram edit msg_id=7002 in chat=444: Нет информации для ответа на ваш запрос
"""


def _dialogs():
    return parse_log(_LOG.splitlines())


def test_parses_conversations_and_turns():
    d = _dialogs()
    assert set(d.keys()) == {"619", "524", "700", "800"}
    assert d["700"].user_msgs == ["УЗИ органов мошонки", "Кого рекомендуешь", "Кто из них топ"]
    assert len(d["700"].bot_msgs) == 3


def test_operator_fallback_flagged_not_counted_as_answer():
    d = _dialogs()["524"]
    assert d.operator_fallback == 1
    assert d.bot_msgs == []  # «Ожидайте оператор» — не ответ бота
    codes = [c for c, _ in d.flags()]
    assert "operator_fallback" in codes


def test_repeat_loop_detected():
    d = _dialogs()["700"]
    codes = [c for c, _ in d.flags()]
    assert "repeat_loop" in codes, "один список на три РАЗНЫЕ реплики — луп (УЗИ-мошонка)"


def test_same_question_repeated_is_not_a_loop():
    """Прод-обкатка 15.07: пациент дважды пишет «Флюорография» и получает тот же
    ответ — это НЕ баг (тот же вопрос → тот же ответ). Луп = один ответ на
    РАЗНЫЕ реплики, а не любой повтор."""
    log = (
        "2026-07-15 10:00:00,000 [INFO] ...run_telegram: Received from Fluo (chat=900): Флюорография\n"
        "2026-07-15 10:00:00,100 [INFO] channels_app.services: Incoming message conv=#900 from Fluo: Флюорография\n"
        "2026-07-15 10:00:01,000 [INFO] channels_app.adapters.telegram: Telegram edit msg_id=1 in chat=900: Флюорография ведётся через регистратуру филиала на Ленина 5\n"
        "2026-07-15 10:05:00,000 [INFO] ...run_telegram: Received from Fluo (chat=900): Флюорография\n"
        "2026-07-15 10:05:00,100 [INFO] channels_app.services: Incoming message conv=#900 from Fluo: Флюорография\n"
        "2026-07-15 10:05:01,000 [INFO] channels_app.adapters.telegram: Telegram edit msg_id=2 in chat=900: Флюорография ведётся через регистратуру филиала на Ленина 5\n"
    )
    d = parse_log(log.splitlines())["900"]
    codes = [c for c, _ in d.flags()]
    assert "repeat_loop" not in codes, "тот же вопрос → тот же ответ не должен быть лупом"


def test_clarify_loop_detected():
    d = _dialogs()["800"]
    codes = [c for c, _ in d.flags()]
    assert "clarify_loop" in codes  # два разных дефлекта подряд


def test_clean_dialog_has_no_flags():
    d = _dialogs()["619"]
    assert d.flags() == []
    assert d.score() == 0


def test_noise_placeholder_ignored():
    # «Ответ в процессе...» не должен попадать в bot_msgs
    d = _dialogs()["800"]
    assert all("ответ в процессе" not in m.lower() for m in d.bot_msgs)


def test_ranking_puts_operator_and_loop_on_top():
    d = _dialogs()
    flagged = sorted((x for x in d.values() if x.flags()), key=lambda x: -x.score())
    top_convs = [x.conv for x in flagged[:2]]
    # operator_fallback (вес 5) и repeat_loop (вес 4) — самые тяжёлые
    assert "524" in top_convs or "700" in top_convs


def test_digest_renders_summary_and_top():
    digest = build_digest(_dialogs())
    assert "Всего диалогов: **4**" in digest
    assert "С красными флагами: **3**" in digest
    assert "conv #700" in digest
    assert "conv #619" not in digest  # чистый — не в топе
