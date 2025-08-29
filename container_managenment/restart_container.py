import subprocess


def restart_ollama_container(container_name="ollama") -> str:
    try:
        # stdout/stderr выводим в UI
        res = subprocess.run(
            ["docker", "restart", container_name],
            check=True,
            capture_output=True,
            text=True,
        )
        out = res.stdout.strip() or "ok"
        return f"✔ Перезапущен: {out}"
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or "").strip()
        return f"✘ Ошибка: {err or str(e)}"

