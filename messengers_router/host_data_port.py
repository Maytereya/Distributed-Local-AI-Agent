"""Локальный порт к данным хоста: папка журналов и загрузчики справочников МИС.

Изолирует прямые импорты `agent_logic_2.persist` и `agent_logic_2.nayka_api` от инфраструктурных
модулей `messengers_router` (`turn_telemetry`, `warm_up`) — архитектурный чек CI
(`scripts/check_architecture_imports.py`) разрешает импорты хоста только портам.

Импорты хоста — внутри функций, как было в модулях до порта: журнал хода и прогрев не тянут
загрузчики МИС при импорте пакета, а тесты подменяют их по-прежнему.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def journals_dir() -> Path:
    """`<APP_DATA_DIR>/journals` — общая папка журналов (телеметрия хода, журнал синонимов)."""
    from agent_logic_2.persist import DATA_DIR, ensure_dir

    return ensure_dir(DATA_DIR / "journals", "journals")


def mis_warm_up_steps() -> tuple[tuple[str, Callable[[], Any]], ...]:
    """Загрузчики справочников МИС для прогрева после старта: (название шага, вызов)."""
    from agent_logic_2.nayka_api import api_nayka, api_price, api_service_info

    return (
        ("прайс Самары", lambda: api_price.load_price_by_region(api_price.BOT_PRICE_REGION_ID)),
        ("справочник услуг МИС", api_service_info.load_service_info),
        ("цены врачей", api_price.load_doctor_prices),
        ("врачи", api_nayka.get_cached_doctors_data),
    )
