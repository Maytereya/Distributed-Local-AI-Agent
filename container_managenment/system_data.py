from typing import Dict, Any, List, Optional, Tuple
import time
import docker

def _bytes_to_mb(x: int) -> float:
    return round(x / 1024 / 1024, 1)

def _calc_cpu_percent(stats: dict) -> float:
    cpu_stats = stats.get("cpu_stats", {})
    precpu = stats.get("precpu_stats", {})

    cpu_delta = cpu_stats.get("cpu_usage", {}).get("total_usage", 0) - precpu.get("cpu_usage", {}).get("total_usage", 0)
    sys_delta = cpu_stats.get("system_cpu_usage", 0) - precpu.get("system_cpu_usage", 0)

    online_cpus = cpu_stats.get("online_cpus") or len(cpu_stats.get("cpu_usage", {}).get("percpu_usage") or []) or 1
    if sys_delta > 0 and cpu_delta > 0:
        return round((cpu_delta / sys_delta) * online_cpus * 100.0, 1)
    return 0.0

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
    client = docker.DockerClient(base_url="unix://var/run/docker.sock")
    out: Dict[str, Any] = {}

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
                "cpu_%": _calc_cpu_percent(s),
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

    return out

def make_human_monitor_payload(container_names: List[str], top_k: int = 3) -> Dict[str, Any]:
    containers = get_docker_containers_stats(container_names)

    # totals (только по тем, у кого нет error)
    ok_items: List[Tuple[str, Dict[str, Any]]] = [
        (name, v) for name, v in containers.items() if isinstance(v, dict) and "error" not in v
    ]

    total_ram_mb = round(sum(v.get("ram_used_mb", 0) or 0 for _, v in ok_items), 1)
    total_cpu = round(sum(v.get("cpu_%", 0) or 0 for _, v in ok_items), 1)
    total_net_in = round(sum(v.get("net_in_mb", 0) or 0 for _, v in ok_items), 1)
    total_net_out = round(sum(v.get("net_out_mb", 0) or 0 for _, v in ok_items), 1)
    total_pids = sum(int(v.get("pids", 0) or 0) for _, v in ok_items)

    top_ram = sorted(ok_items, key=lambda x: x[1].get("ram_used_mb", 0) or 0, reverse=True)[:top_k]
    top_cpu = sorted(ok_items, key=lambda x: x[1].get("cpu_%", 0) or 0, reverse=True)[:top_k]

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
        "summary": {
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