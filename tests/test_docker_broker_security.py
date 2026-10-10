import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "openclaw/monitor"))
import docker_broker as broker
from probe_contract import decode_probe_output, telegram_probe_only


class DockerBrokerSecurityTests(unittest.TestCase):
    def test_dangerous_commands_and_sql_are_denied(self):
        for command, stdin in (
            (["docker", "exec", broker.GATEWAY, "sh", "-c", "cat /home/node/.openclaw/openclaw.json"], None),
            (["docker", "run", "--privileged", "alpine"], None),
            (["docker", "inspect", "arbitrary-container"], None),
            (broker.STATS, "SELECT text FROM core_message;"),
            (broker.STATS, broker.SQL + ";SELECT text FROM core_message;"),
            (broker.PROBE, "extra-input"),
            (["docker", "logs", "--since", "24h", "--tail", "999999", broker.GATEWAY], None),
        ):
            self.assertIsNone(broker.operation(command, stdin))
        self.assertEqual(broker.operation(broker.STATS, "\n" + broker.SQL), "stats")
        self.assertEqual(broker.operation(broker.PROBE), "probe")

    def test_no_secrets_or_historical_log_content_leave_broker(self):
        marker = "PATIENT_CANARY_SECRET"
        fixture = [{"State": {"Running": True, "Health": {"Status": "healthy", "Log": [marker]}}, "RestartCount": 1, "Config": {"Env": [marker]}, "Mounts": [marker]}]
        self.assertNotIn(marker, broker.scrub_output("inspect", json.dumps(fixture), ""))
        logs = broker.scrub_output("logs", "ERROR " + marker, "WARNING " + marker)
        self.assertNotIn(marker, logs)
        self.assertEqual(len(logs.splitlines()), 2)
        with tempfile.TemporaryDirectory() as directory:
            service = broker.Broker(Path(directory))
            with patch.object(broker.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout=marker, stderr=marker)):
                self.assertNotIn(marker, json.dumps(service.run({"args": broker.PROBE})))

    def test_probe_warnings_and_diagnostics_are_not_telegram_failures(self):
        raw = '\u001b[33mPlugin warning\u001b[0m\n{"channels":{"telegram":{"running":true,"probe":{"ok":true},"secret":"CANARY"}}}\nCLI footer'
        result = telegram_probe_only(decode_probe_output(raw))
        self.assertTrue(result["telegram"]["running"])
        self.assertTrue(result["telegram"]["probe"]["ok"])
        self.assertNotIn("CANARY", json.dumps(result))

    def test_broker_enforces_restart_cooldown_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            service = broker.Broker(Path(directory))
            with patch.object(broker.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as run:
                payload = {"args": ["docker", "restart", broker.GATEWAY]}
                self.assertEqual(service.run(payload)["code"], 0)
                self.assertEqual(service.run(payload)["code"], 429)
                self.assertEqual(run.call_count, 1)

    def test_dashboard_metrics_have_fixed_fields_and_commands(self):
        self.assertEqual(broker.operation(broker.CPU_COUNT), "cpu_count")
        self.assertEqual(broker.operation(broker.CONTAINER_STATS_PREFIX+["ollama"]), "container_stats")
        for args in (["docker", "info"], broker.CONTAINER_STATS_PREFIX+["--all"],
                     ["docker", "stats", "--format", "{{json .}}", "ollama"]):
            self.assertIsNone(broker.operation(args))
        row = dict(Name="ollama", CPUPerc="0.5%", MemPerc="1%", MemUsage="1GiB / 32GiB",
                   NetIO="1kB / 2kB", BlockIO="0B / 0B", PIDs="12", Secrets="PATIENT_CANARY")
        self.assertNotIn("PATIENT_CANARY", broker.scrub_output("container_stats",json.dumps(row),""))
        row["MemUsage"] = "PATIENT_CANARY"
        with self.assertRaises(ValueError): broker.scrub_output("container_stats",json.dumps(row),"")
        for raw in ('true', '0', '2049', '{"patient":"CANARY"}'):
            with self.assertRaises(ValueError): broker.scrub_output("cpu_count",raw,"")
        self.assertEqual(broker.scrub_output("cpu_count","32",""),"32")

    def test_ollama_restart_cannot_restart_other_containers(self):
        self.assertIsNone(broker.operation(["docker", "restart", "agent-api"]))
        with tempfile.TemporaryDirectory() as directory:
            service = broker.Broker(Path(directory))
            with patch.object(broker.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as run:
                for name in (broker.GATEWAY,"ollama"):
                    payload = {"args":["docker","restart",name]}
                    self.assertEqual(service.run(payload)["code"],0)
                    self.assertEqual(service.run(payload)["code"],429)
                self.assertEqual(run.call_count,2)


if __name__ == "__main__":
    unittest.main()
