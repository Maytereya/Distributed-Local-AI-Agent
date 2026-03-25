from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

import time
import re
import docker


IMPORTANT_LOG_KEYWORDS = (
    "error",
    "err",
    "exception",
    "traceback",
    "fatal",
    "critical",
    "panic",
    "fail",
    "failed",
    "unable",
    "cannot",
    "denied",
    "timeout",
    "killed",
    "oom",
    "warn",
    "warning",
)


_TOTAL_MEMORY_RE = re.compile(r'msg="total memory"\s+size="([^"]+)"')
_SYSTEM_MEMORY_RE = re.compile(r'msg="system memory"\s+total="([^"]+)"\s+free="([^"]+)"\s+free_swap="([^"]+)"')
_GPU_MEMORY_RE = re.compile(r'msg="gpu memory".*id=([^\s]+).*available="([^"]+)".*free="([^"]+)"')

def _bytes_to_mb(x: int) -> float:
    return round(x / 1024 / 1024, 1)

def get_host_cpu_count() -> int:
    # docker info знает NCPU хоста
    client = None
    try:
        client = docker.DockerClient(base_url="unix://var/run/docker.sock")
        return int(client.info().get("NCPU", 1) or 1)
    except Exception:
        return 1
    finally:
        if client:
            try:
                client.close()
            except Exception:
                pass

def compute_alerts(payload: Dict[str, Any], thresholds: Dict[str, Any]) -> List[str]:
    warns: List[str] = []

    s = payload.get("summary", {})
    host_cpu = payload.get("host", {}).get("cpu_count", 1) or 1

    cpu_host = s.get("cpu_host_%")
    ram_total_mb = s.get("total_ram_used_mb")

    # CPU
    if cpu_host is not None:
        if cpu_host >= thresholds["cpu_critical"]:
            warns.append(f"CPU CRITICAL: {cpu_host}% ≥ {thresholds['cpu_critical']}%")
        elif cpu_host >= thresholds["cpu_warning"]:
            warns.append(f"CPU WARNING: {cpu_host}% ≥ {thresholds['cpu_warning']}%")

    # RAM (сумма контейнеров — это не “вся RAM хоста”, но как индикатор нагрузки системы ок)
    if ram_total_mb is not None:
        if ram_total_mb >= thresholds["ram_critical_mb"]:
            warns.append(f"RAM CRITICAL: {ram_total_mb} MB ≥ {thresholds['ram_critical_mb']} MB")
        elif ram_total_mb >= thresholds["ram_warning_mb"]:
            warns.append(f"RAM WARNING: {ram_total_mb} MB ≥ {thresholds['ram_warning_mb']} MB")

    # GPU
    gpu = payload.get("gpu", {})
    if isinstance(gpu, dict) and "gpus" in gpu:
        for g in gpu["gpus"]:
            free = g.get("vram_free_mb")
            if free is None:
                continue
            if free <= thresholds["vram_critical_free_mb"]:
                warns.append(f"GPU{g['gpu']} VRAM CRITICAL: free {free} MB ≤ {thresholds['vram_critical_free_mb']} MB")
            elif free <= thresholds["vram_warning_free_mb"]:
                warns.append(f"GPU{g['gpu']} VRAM WARNING: free {free} MB ≤ {thresholds['vram_warning_free_mb']} MB")

    return warns

def update_history(history: List[Dict[str, Any]], payload: Dict[str, Any], max_points: int = 120) -> List[Dict[str, Any]]:
    """
    history: список точек для графика (например, 120 точек = ~2 минуты при 1с интервале)
    """
    ts = time.strftime("%H:%M:%S")
    s = payload.get("summary", {})

    point = {
        "time": ts,
        "cpu_host_%": s.get("cpu_host_%", 0.0),
        "ram_mb": s.get("total_ram_used_mb", 0.0),
    }

    # можно добавить GPU: например, max vram used или min free
    gpu = payload.get("gpu", {})
    if isinstance(gpu, dict) and "gpus" in gpu and gpu["gpus"]:
        # минимальный free по всем GPU
        free_vals = [g.get("vram_free_mb") for g in gpu["gpus"] if g.get("vram_free_mb") is not None]
        point["vram_free_mb_min"] = min(free_vals) if free_vals else None

    history.append(point)
    if len(history) > max_points:
        history = history[-max_points:]
    return history

def history_to_df(history: List[Dict[str, Any]]) -> pd.DataFrame:
    if not history:
        return pd.DataFrame({"time": [], "cpu_host_%": [], "ram_mb": [], "vram_free_mb_min": []})
    return pd.DataFrame(history)


def _calc_cpu_percent(stats: dict) -> float:
    cpu_stats = stats.get("cpu_stats", {})
    precpu = stats.get("precpu_stats", {})

    cpu_delta = cpu_stats.get("cpu_usage", {}).get("total_usage", 0) - precpu.get("cpu_usage", {}).get("total_usage", 0)
    sys_delta = cpu_stats.get("system_cpu_usage", 0) - precpu.get("system_cpu_usage", 0)

    online_cpus = cpu_stats.get("online_cpus") or len(cpu_stats.get("cpu_usage", {}).get("percpu_usage") or []) or 1
    if sys_delta > 0 and cpu_delta > 0:
        return round((cpu_delta / sys_delta) * online_cpus * 100.0, 1)
    return 0.0


def _decode_log_lines(raw_logs: bytes | str | None) -> List[str]:
    if not raw_logs:
        return []
    if isinstance(raw_logs, (bytes, bytearray)):
        text = raw_logs.decode("utf-8", errors="replace")
    else:
        text = str(raw_logs)
    return [line.rstrip() for line in text.splitlines() if line and line.strip()]


def _extract_important_lines(log_lines: List[str], limit: int = 30) -> List[str]:
    if not log_lines:
        return []
    selected: List[str] = []
    for line in log_lines:
        low = line.lower()
        if any(keyword in low for keyword in IMPORTANT_LOG_KEYWORDS):
            selected.append(line)
    if limit <= 0:
        return selected
    return selected[-limit:]


def _extract_runtime_memory_from_logs(log_lines: List[str]) -> Dict[str, Any]:
    runtime: Dict[str, Any] = {
        "model_total_memory": None,
        "system_total": None,
        "system_free": None,
        "system_free_swap": None,
        "gpu_min_free": None,
        "gpu_free_by_id": {},
    }

    gpu_map: Dict[str, str] = {}
    for line in log_lines:
        total_match = _TOTAL_MEMORY_RE.search(line)
        if total_match:
            runtime["model_total_memory"] = total_match.group(1)

        system_match = _SYSTEM_MEMORY_RE.search(line)
        if system_match:
            runtime["system_total"] = system_match.group(1)
            runtime["system_free"] = system_match.group(2)
            runtime["system_free_swap"] = system_match.group(3)

        gpu_match = _GPU_MEMORY_RE.search(line)
        if gpu_match:
            gpu_id = gpu_match.group(1)
            gpu_free = gpu_match.group(3)
            gpu_map[gpu_id] = gpu_free

    if gpu_map:
        runtime["gpu_free_by_id"] = gpu_map
        # Значения в логах вида "23.5 GiB", берём минимальный free как консервативный индикатор.
        min_gpu: tuple[float, str] | None = None
        for gpu_id, free_txt in gpu_map.items():
            num_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", free_txt)
            if not num_match:
                continue
            val = float(num_match.group(1))
            if min_gpu is None or val < min_gpu[0]:
                min_gpu = (val, f"{gpu_id}: {free_txt}")
        if min_gpu:
            runtime["gpu_min_free"] = min_gpu[1]

    return runtime

def get_nvidia_gpu_summary() -> Optional[Dict[str, Any]]:
    """
    Возвращает VRAM/UTIL по GPU, если NVML доступен в текущем контейнере.
    """
    try:
        from pynvml import (
            nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetName
        )
        nvmlInit()

        gpus = []
        for i in range(nvmlDeviceGetCount()):
            h = nvmlDeviceGetHandleByIndex(i)
            util = nvmlDeviceGetUtilizationRates(h)
            mem = nvmlDeviceGetMemoryInfo(h)
            name = nvmlDeviceGetName(h)
            gpus.append({
                "gpu": i,
                "name": name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name),
                "util_gpu_%": int(util.gpu),
                "util_mem_%": int(util.memory),
                "vram_used_mb": round(mem.used / 1024 / 1024, 1),
                "vram_total_mb": round(mem.total / 1024 / 1024, 1),
                "vram_free_mb": round(mem.free / 1024 / 1024, 1),
            })

        return {"gpus": gpus}
    except Exception:
        return None

def get_docker_containers_stats(container_names: List[str]) -> Dict[str, Any]:
    client = None
    out: Dict[str, Any] = {}

    try:
        client = docker.DockerClient(base_url="unix://var/run/docker.sock")
    except Exception as e:
        err = f"Docker недоступен: {e}"
        for name in container_names:
            out[name] = {"error": err}
        return out

    try:
        for name in container_names:
            try:
                c = client.containers.get(name)
                s = c.stats(stream=False)

                mem = s.get("memory_stats", {})
                mem_usage = int(mem.get("usage", 0) or 0)
                mem_limit = int(mem.get("limit", 0) or 0)

                net = s.get("networks", {}) or {}
                net_rx = sum(int(v.get("rx_bytes", 0) or 0) for v in net.values())
                net_tx = sum(int(v.get("tx_bytes", 0) or 0) for v in net.values())

                blkio = s.get("blkio_stats", {}).get("io_service_bytes_recursive", []) or []
                blk_r = sum(int(x.get("value", 0) or 0) for x in blkio if x.get("op") == "Read")
                blk_w = sum(int(x.get("value", 0) or 0) for x in blkio if x.get("op") == "Write")

                out[name] = {
                    "container_id": c.short_id,
                    "cpu_%": _calc_cpu_percent(s), # Убрать!
                    "ram_used_mb": _bytes_to_mb(mem_usage),
                    "ram_limit_mb": _bytes_to_mb(mem_limit) if mem_limit else None,
                    "ram_%": round((mem_usage / mem_limit * 100.0), 1) if mem_limit else None,
                    "net_in_mb": _bytes_to_mb(net_rx),
                    "net_out_mb": _bytes_to_mb(net_tx),
                    "disk_read_mb": _bytes_to_mb(blk_r),
                    "disk_write_mb": _bytes_to_mb(blk_w),
                    "pids": (s.get("pids_stats", {}) or {}).get("current"),
                }
            except Exception as e:
                out[name] = {"error": str(e)}
    finally:
        if client:
            try:
                client.close()
            except Exception:
                pass

    return out


def get_container_diagnostics(
    container_name: str = "ollama",
    *,
    log_tail: int = 400,
    important_limit: int = 30,
) -> Dict[str, Any]:
    """
    Диагностика одного docker-контейнера:
    - статус/health/restart/OOM;
    - CPU/RAM/PIDs;
    - важные строки из последних логов (error/warn/fatal/oom/...).
    """
    base: Dict[str, Any] = {
        "available": False,
        "container_name": container_name,
        "container_id": None,
        "status": "unknown",
        "health": None,
        "restart_count": 0,
        "oom_killed": False,
        "metrics": {
            "cpu_%": None,
            "ram_used_mb": None,
            "ram_limit_mb": None,
            "ram_%": None,
            "pids": None,
        },
        "important_logs": [],
        "last_log": None,
        "runtime_memory": {},
        "log_error": None,
        "error": None,
        "hint": None,
    }

    client = None
    try:
        client = docker.DockerClient(base_url="unix://var/run/docker.sock")
        container = client.containers.get(container_name)
    except docker.errors.NotFound:
        base["error"] = f"Контейнер '{container_name}' не найден."
        base["hint"] = "Локальный запуск возможен без контейнера; в этом режиме docker-метрики недоступны."
        if client:
            client.close()
        return base
    except Exception as e:
        base["error"] = f"Docker недоступен: {e}"
        base["hint"] = "Проверьте, что Docker запущен и доступен socket /var/run/docker.sock."
        if client:
            client.close()
        return base

    try:
        stats = container.stats(stream=False)
        mem = stats.get("memory_stats", {}) or {}
        mem_usage = int(mem.get("usage", 0) or 0)
        mem_limit = int(mem.get("limit", 0) or 0)
        base["metrics"] = {
            "cpu_%": _calc_cpu_percent(stats),
            "ram_used_mb": _bytes_to_mb(mem_usage),
            "ram_limit_mb": _bytes_to_mb(mem_limit) if mem_limit else None,
            "ram_%": round((mem_usage / mem_limit * 100.0), 1) if mem_limit else None,
            "pids": (stats.get("pids_stats", {}) or {}).get("current"),
        }
    except Exception as e:
        base["error"] = f"Не удалось получить stats контейнера '{container_name}': {e}"

    try:
        attrs = container.attrs or {}
        state = attrs.get("State", {}) or {}
        health = (state.get("Health", {}) or {}).get("Status")
        restart_count = attrs.get("RestartCount", 0)
        oom_killed = bool(state.get("OOMKilled", False))
        base["container_id"] = container.short_id
        base["status"] = state.get("Status") or container.status or "unknown"
        base["health"] = health
        base["restart_count"] = int(restart_count or 0)
        base["oom_killed"] = oom_killed
    except Exception:
        pass

    try:
        raw_logs = container.logs(
            stdout=True,
            stderr=True,
            timestamps=True,
            tail=max(1, int(log_tail)),
        )
        lines = _decode_log_lines(raw_logs)
        base["runtime_memory"] = _extract_runtime_memory_from_logs(lines)
        base["important_logs"] = _extract_important_lines(lines, limit=important_limit)
        base["last_log"] = lines[-1] if lines else None
    except Exception as e:
        base["log_error"] = str(e)

    base["available"] = True
    if client:
        client.close()
    return base

def make_human_monitor_payload(container_names: List[str], top_k: int = 5) -> Dict[str, Any]:
    containers = get_docker_containers_stats(container_names)

    # totals (только по тем, у кого нет error)
    ok_items: List[Tuple[str, Dict[str, Any]]] = [
        (name, v) for name, v in containers.items() if isinstance(v, dict) and "error" not in v
    ]

    total_ram_mb = round(sum(v.get("ram_used_mb", 0) or 0 for _, v in ok_items), 1)
    total_cpu = round(sum(v.get("cpu_%", 0) or 0 for _, v in ok_items), 1)
    host_cpus = get_host_cpu_count()
    total_net_in = round(sum(v.get("net_in_mb", 0) or 0 for _, v in ok_items), 1)
    total_net_out = round(sum(v.get("net_out_mb", 0) or 0 for _, v in ok_items), 1)
    total_pids = sum(int(v.get("pids", 0) or 0) for _, v in ok_items)

    top_ram = sorted(ok_items, key=lambda x: x[1].get("ram_used_mb", 0) or 0, reverse=True)[:top_k]
    top_cpu = sorted(ok_items, key=lambda x: x[1].get("cpu_%", 0) or 0, reverse=True)[:top_k]

    total_cpu_raw = round(total_cpu, 1)  # например 640%
    cpu_cores_used = round(total_cpu_raw / 100, 2)  # 6.4 ядра
    cpu_host = round(total_cpu_raw / host_cpus, 1)  # 20% от сервера


    gpu = get_nvidia_gpu_summary()

    gpu_block: Dict[str, Any]
    if gpu:
        gpu_block = gpu
    else:
        gpu_block = {
            "note": "GPU метрики недоступны (NVML не виден из bookworm-agent). "
                    "Решение: дать контейнеру доступ к /dev/nvidia* (runtime NVIDIA) или вынести монитор в отдельный контейнер с --gpus all."
        }

    # пояснения
    legend = {
        "cpu_%": "Загрузка CPU контейнера (может быть >100% при использовании нескольких ядер).",
        "ram_used_mb": "Фактически использованная RAM контейнером (RSS/usage по docker).",
        "ram_limit_mb": "Лимит RAM для контейнера (если не задан — обычно = память хоста/лимит cgroup).",
        "ram_%": "Процент использования RAM от лимита контейнера.",
        "net_in_mb/net_out_mb": "Сетевой трафик с момента старта контейнера (вход/выход).",
        "disk_read_mb/disk_write_mb": "Дисковый I/O с момента старта контейнера (чтение/запись).",
        "pids": "Количество процессов/потоков в контейнере.",
        "container_id": "Короткий Docker ID контейнера (для диагностики).",
    }


    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": {
            "cpu_count": host_cpus,
        },
        "summary": {
            "cpu_raw_%_sum": total_cpu_raw,
            "cpu_cores_used": cpu_cores_used,
            "cpu_host_%": cpu_host,
            "total_cpu_%_sum": total_cpu,
            "total_ram_used_mb": total_ram_mb,
            "total_net_in_mb": total_net_in,
            "total_net_out_mb": total_net_out,
            "total_pids": total_pids,
        },
        "top_consumers": {
            "by_ram": [{ "container": n, "ram_used_mb": v["ram_used_mb"], "cpu_%": v["cpu_%"] } for n, v in top_ram],
            "by_cpu": [{ "container": n, "cpu_%": v["cpu_%"], "ram_used_mb": v["ram_used_mb"] } for n, v in top_cpu],
        },
        "gpu": gpu_block,
        "containers": containers,
        "legend": legend,
    }
