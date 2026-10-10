"""Local, fixed-operation broker. Only this container mounts docker.sock.

No Docker HTTP passthrough, shell arguments, arbitrary SQL, files, environment
or container creation are exposed. The monitor connects over a Unix socket;
the gateway has no mount or network route to this broker.
"""
from __future__ import annotations

import json
import os
import re
import socketserver
import subprocess
import threading
import time
from pathlib import Path

from probe_contract import decode_probe_output, telegram_probe_only
from reporting_projection import validate_snapshot, validate_cases
from datetime import datetime

GATEWAY = "openclaw-openclaw-gateway-1"
POSTGRES = "support-messenger-aggregator-postgres-1"
CONTAINERS = frozenset({GATEWAY, POSTGRES, "singbox_telegram_gateway", "ollama", "whisper-gpu", "support-messenger-aggregator-web-1", "support-messenger-aggregator-telegram_bot-1", "agent-api", "bookworm-agent", "chroma_container", "meili_server", "searxng", "nginx_proxy"})
CONTAINER_STATS_PREFIX = ["docker","stats","--no-stream","--format","{{json .}}"]
CPU_COUNT = ["docker","info","--format","{{json .NCPU}}"]
SNAPSHOT = ["docker","exec","reporting-worker","python","manage.py","reporting_snapshot"]
FAILURES = ["docker","exec","reporting-worker","python","manage.py","reporting_failures"]
PROBE = ["docker", "exec", GATEWAY, "node", "dist/index.js", "channels", "status", "--probe", "--json"]
GPU = ["docker", "exec", "ollama", "nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"]
STATS = ["docker", "exec", "-i", POSTGRES, "sh", "-lc", 'psql -X -qAt -F "\t" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f -']
SQL = Path(__file__).with_name("aggregator_stats.sql").read_text(encoding="utf-8")
ERROR = re.compile(r"error|warn|exception|failed|fatal|panic|critical|traceback|unhealthy", re.I)
CRITICAL = re.compile(r"fatal|panic|critical|traceback|unhealthy", re.I)


def operation(args: object, input_text: object = None) -> str | None:
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        return None
    if args == STATS:
        return "stats" if isinstance(input_text, str) and input_text.strip() == SQL.strip() else None
    if input_text is not None:
        return None
    if args == PROBE:
        return "probe"
    if args == GPU:
        return "gpu"
    if args == CPU_COUNT:
        return "cpu_count"
    if args[:6] == FAILURES and len(args) == 8:
        if args[6] == "--id" and re.fullmatch(r"[1-9][0-9]{0,17}",args[7]): return "failures"
        if args[6] == "--period" and operation(SNAPSHOT+args[6:]) == "snapshot": return "failures"
    if args[:6] == SNAPSHOT and len(args) in {8,12} and args[6] == "--period":
        period = args[7]
        if period not in {"day","today","week","month","all"}:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:\.\.\d{4}-\d{2}-\d{2})?",period): return None
            try:
                dates = [datetime.strptime(p,"%Y-%m-%d") for p in period.split("..")]
                if dates[-1] < dates[0]: return None
            except ValueError: return None
        if len(args) == 12:
            if args[8] != "--start" or args[10] != "--end": return None
            try:
                start,end = map(datetime.fromisoformat,(args[9],args[11]))
                if start.tzinfo is None or end.tzinfo is None or end < start: return None
            except ValueError: return None
        return "snapshot"
    if args[:5] == CONTAINER_STATS_PREFIX and 6 <= len(args) <= 5+len(CONTAINERS) and all(name in CONTAINERS for name in args[5:]):
        return "container_stats"
    if args in (["docker", "restart", GATEWAY], ["docker","restart","ollama"]):
        return "restart"
    if len(args) == 3 and args[:2] == ["docker", "inspect"] and args[2] in CONTAINERS:
        return "inspect"
    if (len(args) == 7 and args[:3] == ["docker", "logs", "--since"]
            and args[4] == "--tail" and args[6] in CONTAINERS
            and re.fullmatch(r"[1-9][0-9]{0,3}[smhd]", args[3])
            and args[5].isdigit() and 1 <= int(args[5]) <= 1000):
        return "logs"
    return None


def scrub_output(kind: str, stdout: str, stderr: str) -> str:
    if kind == "failures": return json.dumps(validate_cases(json.loads(stdout)))
    if kind == "snapshot":
        return json.dumps(validate_snapshot(json.loads(stdout)))
    if kind == "cpu_count":
        count = json.loads(stdout)
        if type(count) is not int or not 1 <= count <= 2048: raise ValueError("invalid_cpu_count")
        return str(count)
    if kind == "container_stats":
        safe = []
        for line in stdout.splitlines():
            row = json.loads(line)
            if row.get("Name") not in CONTAINERS: raise ValueError("invalid_container_metric")
            selected = {"Name":row["Name"]}
            for key in ("CPUPerc","MemPerc","MemUsage","NetIO","BlockIO","PIDs"):
                value = row.get(key,"")
                if not isinstance(value,str) or not re.fullmatch(r"[0-9. %/kMGTPEiBb-]{1,80}",value):
                    raise ValueError("invalid_container_metric")
                selected[key] = value
            safe.append(selected)
        return json.dumps(safe)
    if kind == "inspect":
        value = json.loads(stdout)[0]
        state = value.get("State") or {}
        return json.dumps([{"State": {"Running": bool(state.get("Running")), "Health": {"Status": (state.get("Health") or {}).get("Status")}}, "RestartCount": int(value.get("RestartCount", 0))}])
    if kind == "logs":
        # Historical logs are counted inside the privileged boundary.
        return "".join(("CRITICAL" if CRITICAL.search(line) else "ERROR") + " event=container_log\n" for line in (stdout + "\n" + stderr).splitlines() if ERROR.search(line))
    if kind == "probe":
        return json.dumps(telegram_probe_only(decode_probe_output(stdout)))
    if kind == "stats":
        if not all(re.fullmatch(r"[a-z_]+\t[0-9]+", line) for line in stdout.splitlines() if line):
            raise ValueError("invalid_metric")
        return stdout
    if kind == "gpu":
        if not all(re.fullmatch(r"[A-Za-z0-9 ._-]+(?:,\s*[0-9]+(?:\.[0-9]+)?){4}", line) for line in stdout.splitlines() if line):
            raise ValueError("invalid_gpu_metric")
        return stdout
    return "operation_completed"


class Broker:
    def __init__(self, directory: Path):
        self.directory = directory
        self.restart_lock = threading.Lock()

    def run(self, payload: dict) -> dict:
        args = payload.get("args")
        input_text = payload.get("input_text")
        kind = operation(args, input_text)
        if kind is None:
            return {"code": 403, "stdout": "", "stderr": "operation_denied"}
        if kind == "restart":
            with self.restart_lock:
                stamp = self.directory / ("last_restart" if args[-1] == GATEWAY else "last_ollama_restart")
                last = float(stamp.read_text()) if stamp.exists() else 0
                if time.time() - last < 900:
                    return {"code": 429, "stdout": "", "stderr": "restart_cooldown"}
                stamp.write_text(str(time.time()))
                stamp.chmod(0o600)
        try:
            process = subprocess.run(args, input=input_text, capture_output=True, text=True, timeout=65, check=False)
            if process.returncode:
                return {"code": process.returncode, "stdout": "", "stderr": "operation_failed"}
            return {"code": 0, "stdout": scrub_output(kind, process.stdout, process.stderr), "stderr": ""}
        except (OSError, ValueError, subprocess.TimeoutExpired, KeyError, IndexError):
            return {"code": 1, "stdout": "", "stderr": "operation_unavailable"}


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.request.settimeout(70)
        try:
            payload = json.loads(self.rfile.readline(16385))
            if not isinstance(payload, dict):
                raise ValueError
            result = self.server.broker.run(payload)
        except (ValueError, OSError):
            result = {"code": 400, "stdout": "", "stderr": "invalid_request"}
        self.wfile.write((json.dumps(result) + "\n").encode("utf-8"))


def main():
    path = Path(os.environ.get("BROKER_SOCKET", "/run/monitor-broker/docker.sock"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o755)
    path.unlink(missing_ok=True)
    with socketserver.ThreadingUnixStreamServer(str(path), Handler) as server:
        path.chmod(0o660)
        server.broker = Broker(path.parent)
        server.serve_forever()


if __name__ == "__main__":
    main()
