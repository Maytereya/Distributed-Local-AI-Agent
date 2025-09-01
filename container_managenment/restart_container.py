import time

import docker


def restart_ollama_container(container_name="ollama") -> str:
    # подключаемся к сокету, примонтированному из хоста
    client = docker.DockerClient(base_url="unix:///var/run/docker.sock")
    c = client.containers.get(container_name)
    c.restart()
    time.sleep(7)
    status = client.containers.get(container_name).status
    return f"Контейнер {container_name}: {status}"
