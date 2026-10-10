"""Content-free logging for processes that handle patient messages.

Redaction by regex cannot reliably recognize names or arbitrary clinical text.
The record factory therefore keeps only the code location and exception class.
It removes interpolated arguments and tracebacks before *any* logging handler,
including third-party handlers, can format or persist the record.
"""

from __future__ import annotations

import logging
import re
import sys
import io

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,99}$")
_diagnostic_stream = None


class _DiscardText(io.TextIOBase):
    """Legacy print() calls must not bypass content-free logging."""

    encoding = "utf-8"
    errors = "replace"
    _patient_safe = True

    def write(self, value):
        return len(value)

    def flush(self):
        pass

    def isatty(self):
        return False


def _safe_identifier(value: object) -> str:
    return value if isinstance(value, str) and _IDENTIFIER.fullmatch(value) else "event"


def scrub_record(record: logging.LogRecord) -> logging.LogRecord:
    event = _safe_identifier(record.funcName)
    # Even exception messages/tracebacks can contain response bodies and tokens.
    error = ""
    if record.exc_info and record.exc_info[0]:
        error = " error_type=" + _safe_identifier(record.exc_info[0].__name__)
    record.msg = "event=" + event + error
    record.args = ()
    record.exc_info = None
    record.exc_text = None
    record.stack_info = None
    return record


def install_privacy_logging(*, suppress_console: bool = False) -> None:
    global _diagnostic_stream
    if suppress_console and not getattr(sys.stderr, "_patient_safe", False):
        _diagnostic_stream = sys.stderr
        sys.stdout = _DiscardText()
        sys.stderr = _DiscardText()
    factory = logging.getLogRecordFactory()
    if getattr(factory, "_patient_safe", False):
        return

    def safe_factory(*args, **kwargs):
        record = scrub_record(factory(*args, **kwargs))
        if _diagnostic_stream is not None and record.levelno >= logging.WARNING:
            # A dedicated channel survives suppression of legacy print/traceback.
            _diagnostic_stream.write(record.levelname + " " + record.msg + "\n")
            _diagnostic_stream.flush()
        return record

    safe_factory._patient_safe = True
    logging.setLogRecordFactory(safe_factory)
