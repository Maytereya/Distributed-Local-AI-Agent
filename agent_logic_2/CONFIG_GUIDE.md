# Config Guide

Этот файл описывает текущий подход к настройкам в проекте:
- источник значений: `agent_logic_2/config.ini`
- точка чтения и типизации: `agent_logic_2/config.py`

## 1) Где что находится

- `agent_logic_2/config.ini`:
  - хранит секции настроек
  - содержит environment-overrides
- `agent_logic_2/config.py`:
  - читает `config.ini`
  - применяет приоритет секций
  - приводит типы (`str/int/bool/enum`)
  - экспортирует `settings` и legacy-алиасы

## 2) Как выбрать окружение

Переключение окружения делается в `config.ini`:

```ini
[APP]
environment = PRODUCTION
```

Поддерживаемые значения сейчас:
- `DEVELOPMENT`
- `PRODUCTION`
- `DOCKER_PRODUCTION`
- `LOCAL`

## 3) Приоритет поиска значения

Для ключа из секции `SECTION` при активном `ENV` используется порядок:

1. `SECTION.ENV`
2. `SECTION`
3. legacy-секция `[ENV]` + legacy-имя ключа (если задан `legacy_key`)
4. `DEFAULT` + legacy-имя ключа (если задан `legacy_key`)
5. `DEFAULT` + новое имя ключа
6. `default`, переданный в getter

Это позволяет мигрировать ключи без резких поломок.

## 4) Текущая структура секций

Основные секции в `config.ini`:
- `APP`
- `MESSENGER_ROUTER`
- `ROUTER`
- `NLU`
- `NAUKA`
- `OLLAMA` + `OLLAMA.<ENV>`
- `CHROMA` + `CHROMA.<ENV>`
- `MEILI` + `MEILI.<ENV>`
- `ASR_WHISPER` + `ASR_WHISPER.<ENV>`
- `ASR_VOSK.<ENV>`
- `SBER_CLOUD` (сейчас минимальный набор под GigaChat)

## 5) Как добавить новый ключ (алгоритм)

1. Выберите доменную секцию в `config.ini`.
2. Добавьте ключ в базовую секцию `SECTION`.
3. Если значение зависит от окружения, добавьте `SECTION.<ENV>`.
4. В `config.py` добавьте поле в соответствующий dataclass.
5. В `settings = Settings(...)` прочитайте ключ через typed-getter:
   - `_get_str(...)`
   - `_get_int(...)`
   - `_get_bool(...)`
   - `_get_enum(...)`
6. Если нужна обратная совместимость со старым именем, укажите `legacy_key=...`.
7. Если старый код использует глобальный alias (`c.SOME_KEY`), добавьте/обновите alias в нижней части `config.py`.
8. Проверьте компиляцию и тесты.

## 6) Пример добавления bool-флага

Пример: добавить `enable_fast_mode` в `ROUTER`.

Шаг в `config.ini`:

```ini
[ROUTER]
enable_fast_mode = false
```

Шаг в `config.py` dataclass:

```python
@dataclass(frozen=True)
class RouterConfig:
    router_v2_enable: bool
    router_v2_shadow: bool
    default_city: str
    samara_only_operator_text: str
    enable_fast_mode: bool
```

Шаг в `settings`:

```python
router=RouterConfig(
    router_v2_enable=_get_bool("ROUTER", "router_v2_enable", default=True, legacy_key="MR_ROUTER_V2_ENABLE"),
    router_v2_shadow=_get_bool("ROUTER", "router_v2_shadow", default=False, legacy_key="MR_ROUTER_V2_SHADOW"),
    default_city=_get_str("ROUTER", "default_city", default="Самара"),
    samara_only_operator_text=_get_str("ROUTER", "samara_only_operator_text", default="..."),
    enable_fast_mode=_get_bool("ROUTER", "enable_fast_mode", default=False),
),
```

Использование:

```python
from agent_logic_2 import config as c
if c.settings.router.enable_fast_mode:
    ...
```

## 7) Когда можно удалять legacy_key

`legacy_key` можно убрать, когда:
- в `config.ini` больше не используется старое имя ключа
- в коде нет ссылок на старый alias
- тесты/смоук проходят в нужных окружениях

## 8) Минимальные проверки после правок

Рекомендуемый чек:

```bash
.agent_venv/bin/python -m py_compile agent_logic_2/config.py
.agent_venv/bin/python -m pytest -k "not procedure_queries"
```

Если меняли критичные маршруты/интенты, запускайте полный `pytest`.

