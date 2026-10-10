import sys
import tempfile
import unittest
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parents[1] / "openclaw/monitor"))
from report_outbox import ReportOutbox
from reporting_time import covered_samples, period_window
from openclaw_monitor import Monitor, TelegramClient, split_message


class CoverageTests(unittest.TestCase):
    def test_gap_breaks_outage_and_duplicates_cannot_inflate_coverage(self):
        start = datetime(2026, 10, 9, tzinfo=timezone.utc)
        sample = lambda minute: {"_dt": start + timedelta(minutes=minute), "channels": [{"id": "http", "ok": False}]}
        result = covered_samples([sample(0), sample(0), sample(1), sample(4)], start, start + timedelta(minutes=6), 60)
        self.assertEqual(result["coverage_percent"], 50)
        self.assertEqual(result["missing_samples"], 3)
        self.assertEqual(result["max_down_streak_samples"], 2)

    def test_missing_or_unknown_observation_does_not_become_an_outage(self):
        start = datetime(2026, 10, 9, tzinfo=timezone.utc)
        result = covered_samples([{"_dt": start, "channels": [{"ok": None}]}], start, start + timedelta(minutes=1), 60)
        self.assertEqual(result["samples"], [])
        self.assertEqual(result["coverage_percent"], 0)
        self.assertEqual(result["max_down_streak_samples"], 0)

    def test_full_calendar_periods_and_leap_month(self):
        now = datetime(2028, 3, 6, 19, tzinfo=ZoneInfo("Europe/Samara"))
        start, end = period_window("week", now)
        self.assertEqual((end - start).days, 7)
        self.assertEqual(start.astimezone(now.tzinfo).isoformat(), "2028-02-28T00:00:00+04:00")
        start, end = period_window("month", now)
        self.assertEqual((end - start).days, 29)
        for invalid in ("latest", "2026-13-01", "2026-10-09..2026-10-08", "2026-10-01..2026-10-02..2026-10-03"):
            with self.assertRaises(ValueError):
                period_window(invalid, now)


class OutboxTests(unittest.TestCase):
    def test_receipt_survives_restart_and_duplicate_report_preserves_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outbox.sqlite3"
            box = ReportOutbox(path)
            box.enqueue("day:a", "synthetic-chat", ["part1", "part2"], "start", "end")
            calls = []
            sender = lambda text, chat, created: calls.append(text) or {"state": "sent", "message_id": 12}
            box.flush(sender)
            box.close()
            box = ReportOutbox(path)
            self.assertFalse(box.enqueue("day:a", "synthetic-chat", ["new values"]))
            box.flush(sender)
            self.assertEqual(calls, ["part1", "part2"])
            self.assertEqual(box.summary(), {"sent": 2})
            box.close()

    def test_ambiguous_acceptance_blocks_automatic_retry_and_remaining_parts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outbox.sqlite3"
            box = ReportOutbox(path)
            box.enqueue("one", "synthetic", ["first", "second"])
            calls = []
            sender = lambda text, chat, created: calls.append(text) or {"state": "uncertain", "reason": "transport_unconfirmed"}
            box.flush(sender)
            box.close()
            box = ReportOutbox(path)
            box.flush(sender)
            self.assertEqual(calls, ["first"])
            self.assertEqual(box.summary(), {"pending": 1, "uncertain": 1})
            box.close()

    def test_process_interruption_during_send_stays_uncertain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outbox.sqlite3"
            box = ReportOutbox(path)
            box.enqueue("one", "synthetic", ["first"])
            with box.db:
                box.db.execute("UPDATE chunks SET state='sending'")
            box.close()
            box = ReportOutbox(path)
            self.assertEqual(box.summary(), {"uncertain": 1})
            box.close()


class MonitorReliabilityTests(unittest.TestCase):
    def test_stale_worker_and_pulse_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = Monitor({"data_dir": directory, "telegram_pulse": {"enabled": True}})
            with patch("openclaw_monitor.time.monotonic", return_value=monitor.started_at + 1000):
                health = monitor.health_status()
                self.assertEqual(health["status"], "unhealthy")
                self.assertEqual(set(health["stale_workers"]), {"pulse", "scheduler"})
            monitor.state["last_telegram_pulse"] = {"at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(), "status": "ok"}
            self.assertFalse(monitor.telegram_pulse_snapshot()["current"]["fresh"])
            self.assertIn("нет свежих данных", monitor.format_telegram_pulse_report())
            monitor.outbox.close()

    def test_scheduled_period_is_anchored_to_1900_even_if_delivery_is_late(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = Monitor({"data_dir": directory, "timezone": "Europe/Samara"})
            now = datetime(2026, 10, 12, 21, tzinfo=ZoneInfo("Europe/Samara"))
            target = {"report_time": "19:00", "period": "day"}
            anchor, start, end = monitor.scheduled_window(target, now)
            self.assertEqual(anchor.hour, 19)
            self.assertEqual(end.astimezone(now.tzinfo).hour, 19)
            self.assertEqual(end - start, timedelta(days=1))
            target["starts_at"]=now.isoformat()
            self.assertIsNone(monitor.scheduled_window(target,now))
            target.pop("starts_at")
            target.update(frequency="weekly", period="week")
            self.assertIsNone(monitor.scheduled_window(target, now + timedelta(days=1)))
            monitor.outbox.close()

    def test_utf16_limit_and_transport_error_cannot_leak_payload(self):
        text = "🧭" * 4000
        chunks = split_message(text)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(chunk.encode("utf-16-le")) // 2 <= 3600 for chunk in chunks))
        client = TelegramClient({})
        with patch.object(client, "call", side_effect=urllib.error.URLError("PATIENT_CANARY")) as call:
            result = client.send_chunk("synthetic", "synthetic")
            self.assertEqual(result, {"state": "uncertain", "reason": "transport_unconfirmed"})
            self.assertEqual(call.call_args.kwargs["retries"], 1)


if __name__ == "__main__":
    unittest.main()
