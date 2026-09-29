"""Тестовый экземпляр бота не может пересечься с продом (29.09).

docker-compose.test.yml разворачивается на том же хосте, что и прод. Совпадение
имени контейнера или порта хоста — и тестовый запуск подменит или уронит прод.
"""

from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return yaml.safe_load((_ROOT / name).read_text(encoding="utf-8"))


def _container_names(compose: dict) -> set[str]:
    return {svc.get("container_name") for svc in compose["services"].values()}


def _host_ports(compose: dict) -> set[str]:
    return {str(p).split(":")[0] for svc in compose["services"].values() for p in svc.get("ports") or []}


def test_container_names_differ_from_prod():
    prod, test = _load("docker-compose.yml"), _load("docker-compose.test.yml")
    assert not (_container_names(prod) & _container_names(test))


def test_host_ports_differ_from_prod():
    prod_ports = _host_ports(_load("docker-compose.yml")) | _host_ports(_load("docker-compose.ollama.yml"))
    assert not (prod_ports & _host_ports(_load("docker-compose.test.yml")))


def test_only_the_api_is_duplicated():
    # Тестовому нужен только agent-api: LLM, поиск и GPU-сервисы общие с продом.
    assert list(_load("docker-compose.test.yml")["services"]) == ["agent-api-test"]
