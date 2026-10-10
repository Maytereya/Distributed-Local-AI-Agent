import io
import logging
import os
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from api_security import messenger_debug_allowed, verify_messenger_api_key, verify_secret
from privacy_logging import install_privacy_logging
import privacy_logging


class MessengerAuthenticationTests(unittest.TestCase):
    def test_authentication_precedes_patient_endpoint(self):
        app = FastAPI()
        calls = []

        @app.post("/api/messenger-generate", dependencies=[Depends(verify_messenger_api_key)])
        def endpoint():
            calls.append(True)
            return {"ok": True}

        with tempfile.TemporaryDirectory() as directory:
            credential = Path(directory) / "key"
            credential.write_text("synthetic-test-key\n")
            with patch.dict(os.environ, {"MESSENGER_API_KEY_FILE": str(credential)}):
                client = TestClient(app)
                for key in (None, "wrong", "Пациент CANARY"):
                    headers = {b"X-API-Key": key.encode("utf-8")} if key is not None else {}
                    response = client.post("/api/messenger-generate", headers=headers)
                    self.assertEqual(response.status_code, 401)
                    self.assertNotIn("CANARY", response.text)
                self.assertEqual(calls, [])
                self.assertEqual(client.post("/api/messenger-generate", headers={"X-API-Key": "synthetic-test-key"}).status_code, 200)
                credential.unlink()
                self.assertEqual(client.post("/api/messenger-generate", headers={"X-API-Key": "synthetic-test-key"}).status_code, 503)
                self.assertEqual(len(calls), 1)

    def test_empty_secret_and_diagnostics_fail_closed(self):
        with self.assertRaises(HTTPException) as error:
            verify_secret("synthetic", "")
        self.assertEqual(error.exception.status_code, 503)
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(messenger_debug_allowed(True))
        with patch.dict(os.environ, {"MESSENGER_ALLOW_DEBUG": "1"}):
            self.assertTrue(messenger_debug_allowed(True))
            self.assertFalse(messenger_debug_allowed(False))


class PrivacyLoggingTests(unittest.TestCase):
    def test_legacy_prints_and_stderr_are_suppressed(self):
        old_factory = logging.getLogRecordFactory()
        old_sink = privacy_logging._diagnostic_stream
        output = io.StringIO()
        marker = "PATIENT_CANARY_LEGACY_PRINT"
        try:
            with patch.object(sys, "stdout", output), patch.object(sys, "stderr", output):
                install_privacy_logging(suppress_console=True)
                print(marker)
                sys.stderr.write(marker)
                logging.getLogger("synthetic").warning("Response %s", marker)
            self.assertNotIn(marker, output.getvalue())
            self.assertIn("WARNING event=", output.getvalue())
        finally:
            privacy_logging._diagnostic_stream = old_sink
            logging.setLogRecordFactory(old_factory)

    def test_content_and_exception_canaries_never_reach_handlers(self):
        old_factory = logging.getLogRecordFactory()
        stream = io.StringIO()
        logger = logging.getLogger("synthetic_security_test")
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        marker = "PATIENT_CANARY_FIO_PHONE_DIAGNOSIS_BOT_TOKEN"
        try:
            install_privacy_logging()
            install_privacy_logging()
            logger.info("Incoming message: %s", marker)
            logger.debug(f"Raw response: {marker}")
            try:
                raise ValueError(marker)
            except ValueError:
                logger.exception("API failed %s", marker, stack_info=True)
            self.assertNotIn(marker, stream.getvalue())
            self.assertIn("error_type=ValueError", stream.getvalue())
        finally:
            logging.setLogRecordFactory(old_factory)
            logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
