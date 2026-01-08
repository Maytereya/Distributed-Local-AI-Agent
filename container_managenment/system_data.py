import docker
from typing import Dict, Any, List

def _calc_cpu_percent(stats: dict) -> float:
    # Формула как в docker CLI: delta(container_cpu) / delta(system_cpu) * online_cpus * 100
    cpu_stats = stats.get("cpu_stats", {})
    precpu = stats.get("precpu_stats", {})

    cpu_delta = cpu_stats.get("cpu_usage", {}).get("total_usage", 0) - precpu.get("cpu_usage", {}).get("total_usage", 0)
    sys_delta = cpu_stats.get("system_cpu_usage", 0) - precpu.get("system_cpu_usage", 0)

    online_cpus = cpu_stats.get("online_cpus") or len(cpu_stats.get("cpu_usage", {}).get("percpu_usage") or []) or 1
    if sys_delta > 0 and cpu_delta > 0:
        return round((cpu_delta / sys_delta) * online_cpus * 100.0, 1)
    return 0.0

def _bytes_to_mb(x: int) -> float:
    return round(x / 1024 / 1024, 1)

def get_docker_containers_stats(container_names: List[str]) -> Dict[str, Any]:
    client = docker.DockerClient(base_url="unix://var/run/docker.sock")
    out: Dict[str, Any] = {}

    for name in container_names:
        try:
            c = client.containers.get(name)
            s = c.stats(stream=False)
            mem = s.get("memory_stats", {})
            mem_usage = mem.get("usage", 0)
            mem_limit = mem.get("limit", 0)

            net = s.get("networks", {}) or {}
            net_rx = sum(v.get("rx_bytes", 0) for v in net.values())
            net_tx = sum(v.get("tx_bytes", 0) for v in net.values())

            blkio = s.get("blkio_stats", {}).get("io_service_bytes_recursive", []) or []
            blk_r = sum(x.get("value", 0) for x in blkio if x.get("op") == "Read")
            blk_w = sum(x.get("value", 0) for x in blkio if x.get("op") == "Write")

            out[name] = {
                "id": c.short_id,
                "cpu_percent": _calc_cpu_percent(s),
                "mem_used_mb": _bytes_to_mb(mem_usage),
                "mem_limit_mb": _bytes_to_mb(mem_limit) if mem_limit else None,
                "mem_percent": round((mem_usage / mem_limit * 100.0), 1) if mem_limit else None,
                "net_rx_mb": _bytes_to_mb(net_rx),
                "net_tx_mb": _bytes_to_mb(net_tx),
                "blk_read_mb": _bytes_to_mb(blk_r),
                "blk_write_mb": _bytes_to_mb(blk_w),
                "pids": (s.get("pids_stats", {}) or {}).get("current"),
            }
        except Exception as e:
            out[name] = {"error": str(e)}

    return out