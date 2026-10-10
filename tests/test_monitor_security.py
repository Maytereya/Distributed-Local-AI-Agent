import importlib.util
import json
import os
import tempfile
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parents[1] / "openclaw/monitor"))
spec = importlib.util.spec_from_file_location("monitor_under_test", Path(__file__).parents[1] / "openclaw/monitor/openclaw_monitor.py")
monitor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor_module)


class MonitorSecurityTests(unittest.TestCase):
    def test_no_log_content_in_result_even_on_errors(self):
        canary = "PATIENT_CANARY_FIO_PHONE_DIAGNOSIS"
        with tempfile.TemporaryDirectory() as directory:
            monitor = monitor_module.Monitor({"data_dir": directory, "containers": [{"key": "bot", "name": "bot", "container": "synthetic"}]})
            with patch.object(monitor_module, "run_cmd", return_value=(0, "ERROR " + canary, "WARNING " + canary)):
                result = monitor.check_logs()
                self.assertNotIn(canary, json.dumps(result))
                self.assertGreater(result[0]["data"]["matches"], 0)
            with patch.object(monitor_module, "run_cmd", return_value=(1, "", canary)):
                self.assertNotIn(canary, json.dumps(monitor.check_logs()))

    def test_all_http_actions_require_authentication(self):
        fake = Mock()
        fake.health_status.return_value = {"status": "ok"}
        monitor_module.Handler.monitor = fake
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / "key"
            key.write_text("synthetic-monitor-token")
            with patch.dict(os.environ, {"MONITOR_API_TOKEN_FILE": str(key)}):
                server = ThreadingHTTPServer(("127.0.0.1", 0), monitor_module.Handler)
                worker = threading.Thread(target=server.serve_forever, daemon=True)
                worker.start()
                base = "http://127.0.0.1:" + str(server.server_port)
                try:
                    for path in ("/healthz", "/report", "/telegram_pulse", "/check", "/telegram/test", "/telegram/menu"):
                        request = urllib.request.Request(base + path, method="POST" if path.startswith("/telegram/") else "GET")
                        with self.assertRaises(urllib.error.HTTPError) as error:
                            urllib.request.urlopen(request)
                        self.assertEqual(error.exception.code, 401)
                    self.assertEqual(fake.mock_calls, [])
                    request = urllib.request.Request(base + "/healthz", headers={"Authorization": "Bearer synthetic-monitor-token"})
                    self.assertEqual(urllib.request.urlopen(request).status, 200)
                    key.unlink()
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        urllib.request.urlopen(request)
                    self.assertEqual(error.exception.code, 503)
                finally:
                    server.shutdown()
                    server.server_close()
                    worker.join()


if __name__ == "__main__":
    unittest.main()
