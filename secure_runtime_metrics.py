"""Fixed broker operations for the authenticated legacy local dashboard."""
import json
import os
import re
import socket
import threading
import time

ALIASES = {"chroma":"chroma_container","meilisearch":"meili_server"}
LOCK = threading.Lock()
CACHE = {}


def configured():
    return bool(os.environ.get("DOCKER_BROKER_SOCKET"))


def run(args):
    request = {"args":args,"input_text":None}
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
        connection.settimeout(70)
        connection.connect(os.environ["DOCKER_BROKER_SOCKET"])
        connection.sendall((json.dumps(request)+"\n").encode())
        with connection.makefile("rb") as reader:
            response = json.loads(reader.readline(1_000_000))
    if response.get("code") != 0:
        raise RuntimeError("fixed_operation_unavailable")
    return response.get("stdout","")


def cpu_count():
    try: return int(run(["docker","info","--format","{{json .NCPU}}"] ))
    except Exception: return 1


def _mb(value):
    match = re.fullmatch(r"([0-9.]+)\s*([kMGTPE]?i?B)",value.strip())
    if not match: raise ValueError("invalid_metric_unit")
    units = {"B":1/1024**2,"kB":1000/1024**2,"MB":1000**2/1024**2,"GB":1000**3/1024**2,
             "TB":1000**4/1024**2,"KiB":1/1024,"MiB":1,"GiB":1024,"TiB":1024**2}
    return round(float(match[1])*units[match[2]],1)


def container_stats(names):
    key = tuple(sorted(set(ALIASES.get(n,n) for n in names)))
    try:
        with LOCK:
            cached = CACHE.get(key)
            if not cached or time.monotonic()-cached[0] > 15:
                rows = json.loads(run(["docker","stats","--no-stream","--format","{{json .}}",*key]))
                CACHE[key] = time.monotonic(),rows
            else: rows = cached[1]
        by_name = {r["Name"]:r for r in rows}
        result = {}
        for name in names:
            row = by_name[ALIASES.get(name,name)]
            used,limit = map(_mb,row["MemUsage"].split("/"))
            rx,tx = map(_mb,row["NetIO"].split("/"))
            read,write = map(_mb,row["BlockIO"].split("/"))
            result[name] = {"container_id":None,"cpu_%":float(row["CPUPerc"].rstrip("%")),
                "ram_used_mb":used,"ram_limit_mb":limit,"ram_%":float(row["MemPerc"].rstrip("%")),
                "net_in_mb":rx,"net_out_mb":tx,"disk_read_mb":read,"disk_write_mb":write,
                "pids":int(row["PIDs"])}
        return result
    except Exception:
        return {name:{"error":"Безопасная диагностика временно недоступна."} for name in names}


def diagnostics(name, log_tail=400):
    target = ALIASES.get(name,name)
    base = {"available":False,"container_name":name,"container_id":None,"status":"unknown",
        "health":None,"restart_count":None,"oom_killed":None,"metrics":{},"important_logs":[],
        "last_log":None,"runtime_memory":{},"log_error":None,"error":None,"hint":None}
    try:
        inspection = json.loads(run(["docker","inspect",target]))[0]
        metrics = container_stats([target])[target]
        if "error" in metrics: raise RuntimeError
        base.update(available=True,status="running" if inspection["State"]["Running"] else "stopped",
            health=inspection["State"]["Health"]["Status"],restart_count=inspection["RestartCount"],metrics=metrics)
        logs = run(["docker","logs","--since","60m","--tail",str(max(1,min(1000,int(log_tail)))),target])
        base["important_logs"] = ["Событие ошибки; содержимое скрыто."]*min(30,len(logs.splitlines()))
        base["hint"] = "Содержимое логов и секреты исключены из диагностики."
    except Exception:
        base["error"] = "Безопасная диагностика временно недоступна."
    return base


def restart_ollama(name="ollama"):
    if name != "ollama": raise ValueError("only_fixed_ollama_restart_allowed")
    run(["docker","restart","ollama"])
    return "Команда перезапуска Ollama принята. Состояние проверяется мониторингом."
