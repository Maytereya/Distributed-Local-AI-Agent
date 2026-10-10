"""Durable chunk receipts. An ambiguous send is never retried automatically.

Telegram has no sendMessage idempotency key: a crash after acceptance but before
the receipt commits cannot provide exactly-once delivery. Such chunks stay
uncertain and are visible to the administrator instead of silently duplicating.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from datetime import datetime, timezone


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ReportOutbox:
    def __init__(self, path):
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY, chat_id TEXT NOT NULL, created_at TEXT NOT NULL,
                period_start TEXT, period_end TEXT, content_hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                report_id TEXT NOT NULL REFERENCES reports(id), ordinal INTEGER NOT NULL,
                text TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
                message_id INTEGER, delivered_at TEXT, reason TEXT, available_at REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (report_id, ordinal)
            );
        """)
        with self.db:
            self.db.execute("UPDATE chunks SET state='uncertain', reason='process_interrupted' WHERE state='sending'")

    def enqueue(self, report_id, chat_id, chunks, start=None, end=None):
        digest = hashlib.sha256("\n".join(chunks).encode()).hexdigest()
        with self.lock, self.db:
            existing = self.db.execute("SELECT chat_id FROM reports WHERE id=?", (report_id,)).fetchone()
            if existing:
                if existing["chat_id"] != chat_id:
                    raise ValueError("report_target_changed")
                return False  # Preserve the first snapshot and its original period.
            self.db.execute("INSERT INTO reports VALUES (?,?,?,?,?,?)", (report_id, chat_id, utcnow(), start, end, digest))
            self.db.executemany("INSERT INTO chunks (report_id,ordinal,text) VALUES (?,?,?)", [(report_id, i, text) for i, text in enumerate(chunks)])
            return True

    def flush(self, send_chunk):
        # Only one sender holds the lock. Persist 'sending' before calling Telegram.
        with self.lock:
            rows = self.db.execute("""SELECT c.*,r.chat_id,r.created_at FROM chunks c
                JOIN reports r ON r.id=c.report_id WHERE c.state='pending'
                AND c.available_at<=?
                AND NOT EXISTS (SELECT 1 FROM chunks earlier WHERE earlier.report_id=c.report_id
                  AND earlier.ordinal<c.ordinal AND earlier.state!='sent')
                ORDER BY r.created_at,c.report_id,c.ordinal""", (time.time(),)).fetchall()
            for row in rows:
                with self.db:
                    self.db.execute("UPDATE chunks SET state='sending' WHERE report_id=? AND ordinal=?", (row["report_id"], row["ordinal"]))
                try:
                    receipt = send_chunk(row["text"], row["chat_id"], row["created_at"])
                except Exception:
                    receipt = {"state": "uncertain", "reason": "transport_unconfirmed"}
                state = receipt.get("state")
                if state not in {"sent", "pending", "failed", "uncertain"}:
                    state = "uncertain"
                with self.db:
                    self.db.execute("""UPDATE chunks SET state=?,message_id=?,delivered_at=?,reason=?,available_at=?
                        WHERE report_id=? AND ordinal=?""", (state, receipt.get("message_id") if state == "sent" else None,
                        utcnow() if state == "sent" else None, receipt.get("reason", "transport_unconfirmed"),
                        time.time() + max(1, int(receipt.get("retry_after") or 60)) if state == "pending" else 0,
                        row["report_id"], row["ordinal"]))
                if state == "pending":
                    break  # Rate limit or confirmed rejection; wait for next iteration.
            return self.summary_unlocked()

    def summary_unlocked(self):
        return {row["state"]: row["n"] for row in self.db.execute("SELECT state,count(*) n FROM chunks GROUP BY state")}

    def summary(self):
        with self.lock:
            return self.summary_unlocked()

    def close(self):
        self.db.close()
