# Рефакторинг messengers_router — краткое резюме

**Ветка:** `refactor/core`  
**Тесты:** `516 passed, 3 warnings` (предупреждения из pydantic v2 / библиотеки websockets, не из нашего кода)  


---

## Зачем делали рефакторинг

В исходном `messengers_router/` была критическая продакшн-ошибка и ряд структурных проблем:

### Основной баг (исправлен)
`nlu_pipeline._merge()` незаметно **отдавал приоритет regex-интенту над LLM-интентом**, если `llm.confidence < 0.45`. То есть если LLM говорил "APPOINTMENT" с уверенностью 40%, а regex — "OTHER", система возвращала "OTHER". LLM фактически использовался, но затем тихо переопределялся — это обнуляло смысл его применения.

### Структурные проблемы (исправлены)
- **20+ копий** `normalize_ru` инлайн (`s.lower().replace("ё","е").strip()`)
- **3 отдельные копии** whitelist допустимых entity-ключей (prompt_contracts, classifier, entity_grounder)
- **50+ "магических" значений confidence** (0.45, 0.55, 0.65, …) без имен
- **Булевы флаги в `last_entities`** для state machine записи вместо типизированного состояния
- **`services.py` был 7 457 строк** — один класс делал всё
- **Не было единой точки входа маршрутизации** — `patient_routing_stream()` содержал ~200 строк рендеринга, смешивая NLU, middleware и генерацию ответа

---

## Что изменилось — конкретно

### 1. Исправление LLM-first NLU (`nlu_pipeline.py`)

**Было:**
```python
# Если уверенность LLM низкая, побеждает regex
if llm.confidence < 0.45:
    return rule, "rule_override"
```

**Стало:**
```python
# LLM label ВСЕГДА главный. Rule только дополняет entities.
donated = {k: v for k, v in rule.entities.items() if k not in llm.entities}
merged = RouteDecision(label=llm.label, entities={**llm.entities, **donated}, ...)
return merged, "llm_primary"
```

Safety-лейблы (URGENT / COMPLAINT / MEDICAL_ADVICE) остаются жёсткими override — они намеренно обходят LLM.

---

### 2. Унификация `normalize_ru` (`messengers_router/russian_nlu.py`, NEW)

Единый источник истины вместо 20+ копий:
```python
def normalize_ru(text: str | None) -> str:
    return str(text or "").lower().replace("ё", "е").strip()
```

Также добавлен `ENTITY_WHITELIST: frozenset[str]` вместо трёх разных копий списка допустимых ключей.

---

### 3. Именованные константы `CONFIDENCE` (`mess_types.py`)

Замена 50+ "магических" чисел:
```python
@dataclass(frozen=True)
class ConfidencePolicy:
    rule_hardcode: float = 1.0
    llm_high: float = 0.85
    llm_medium: float = 0.70
    refine_min: float = 0.55
    ...

CONFIDENCE = ConfidencePolicy()
```

Использование: `CONFIDENCE.llm_high` вместо `0.85`.

---

### 4. Типизированное состояние записи (`mess_types.py`)

**Было — булевые флаги в `last_entities`:**
```python
state.last_entities["appointment_flow_active"] = True
state.last_entities["appointment_confirm_pending"] = True
state.last_entities["appointment_confirmed"] = True
```

**Стало — типизированный `DialogState`:**
```python
class AppointmentPhase:
    IDLE      = ""
    COLLECTING = "collecting"
    CONFIRM   = "confirm"
    CONFIRMED = "confirmed"

@dataclass
class DialogState:
    phase: str = ""
    label: str = ""
    entities: dict = field(default_factory=dict)
    confidence: float = 0.0
    missing_slots: list = field(default_factory=list)
```

Теперь у каждого `SessionState` есть `state.dialog: DialogState`.

Старые флаги пока **дублируются** (dual-write), чтобы не ломать совместимость.

---

### 5. Orchestrator pipeline (`messengers_router/orchestrator.py`, NEW)

Появился явный пайплайн из 5 стадий:

```
early_guards → nlu_route → clarify_gate → tool_loop → render
```

---

### 6. `render()` теперь полностью отвечает за ответ

Теперь `render()` всегда возвращает `ResponseEnvelope`.

---

### 7. Декомпозиция services (`messengers_router/services/`)

`services.py` → `services_legacy.py`.

---

## Что осталось (не блокирует)

- Вынести `lab_tests.py`
- Вынести `appointments.py`
- Убрать `last_entities`
- Убрать `route_patient_message()`

---

## Как запустить тесты

```bash
PYTHONPATH=. venv/bin/pytest tests/ -x -q --ignore=tests/eval
```

---

# messengers_router Refactor — Summary for Colleagues

**Branch:** `refactor/core`  
**Tests:** `516 passed, 3 warnings` (warnings are from pydantic v2 / websockets library, not our code)  
**Safe to push:** Yes.

---

## Why we refactored

The original `messengers_router/` had a critical production bug and several structural problems:

### Root bug (now fixed)
`nlu_pipeline._merge()` was silently **promoting regex intent over LLM intent** when `llm.confidence < 0.45`. So if the LLM said "APPOINTMENT" with 40% confidence and regex said "OTHER", the system returned "OTHER". The LLM was being used but then quietly overridden — defeating the whole point of having an LLM.

### Structural problems (now fixed)
- **20+ copies** of `normalize_ru` inline (`s.lower().replace("ё","е").strip()`)
- **3 separate copies** of the allowed entity key whitelist (prompt_contracts, classifier, entity_grounder)
- **50+ magic confidence floats** scattered across files (0.45, 0.55, 0.65, …) with no names
- **Boolean flags in `last_entities`** for the appointment state machine instead of typed state
- **`services.py` was 7,457 lines** — one class doing everything
- **No single routing entry point** — `patient_routing_stream()` had a 200-line rendering block doing NLU + middleware + rendering all mixed together

---

## What changed — concrete list

### 1. LLM-first NLU fix (`nlu_pipeline.py`)

**Before:**
```python
# If LLM confidence is low, regex wins
if llm.confidence < 0.45:
    return rule, "rule_override"
```

**After:**
```python
# LLM label ALWAYS wins. Rule only donates entities LLM missed.
donated = {k: v for k, v in rule.entities.items() if k not in llm.entities}
merged = RouteDecision(label=llm.label, entities={**llm.entities, **donated}, ...)
return merged, "llm_primary"
```

Safety labels (URGENT / COMPLAINT / MEDICAL_ADVICE) remain hard overrides — they bypass LLM intentionally.

---

### 2. `normalize_ru` unified (`messengers_router/russian_nlu.py`, NEW)

Single source of truth replacing 20+ inline copies:
```python
def normalize_ru(text: str | None) -> str:
    return str(text or "").lower().replace("ё", "е").strip()
```

Also added `ENTITY_WHITELIST: frozenset[str]` replacing 3 separate copies of the allowed entity key list.

---

### 3. `CONFIDENCE` named constants (`mess_types.py`)

Replacing 50+ magic floats:
```python
@dataclass(frozen=True)
class ConfidencePolicy:
    rule_hardcode: float = 1.0
    llm_high: float = 0.85
    llm_medium: float = 0.70
    refine_min: float = 0.55
    ...

CONFIDENCE = ConfidencePolicy()  # module-level singleton
```

Usage: `CONFIDENCE.llm_high` instead of `0.85` everywhere.

---

### 4. Typed appointment state (`mess_types.py`)

**Before** — boolean soup in `last_entities`:
```python
state.last_entities["appointment_flow_active"] = True
state.last_entities["appointment_confirm_pending"] = True
state.last_entities["appointment_confirmed"] = True
# reads scattered across 15+ places in 2 files
if state.last_entities.get("appointment_flow_active"): ...
```

**After** — typed `DialogState` on every `SessionState`:
```python
class AppointmentPhase:
    IDLE      = ""
    COLLECTING = "collecting"
    CONFIRM   = "confirm"
    CONFIRMED = "confirmed"

@dataclass
class DialogState:
    phase: str = ""
    label: str = ""
    entities: dict = field(default_factory=dict)
    confidence: float = 0.0
    missing_slots: list = field(default_factory=list)
    # + is_active(), clear(), merge_entities()
```

Every `SessionState` now has `state.dialog: DialogState`. The boolean flags are **dual-written** (both old flag and `state.dialog.phase`) so nothing breaks during the transition.

Files fully migrated to read `state.dialog.phase`:
- `appointment_flow_guard.py` — reads `AppointmentPhase.CONFIRM` instead of `last_entities.get("appointment_confirm_pending")`
- `router.py` — reads `state.dialog.phase in (COLLECTING, CONFIRM)` instead of `last_entities.get("appointment_flow_active")`

Files still reading `last_entities` (Phase 3, not blocking):
- `planner.py`, `flow_policy.py`, `classifier.py`, `dialog_graph.py`, `mess_types.py`, `policies.py`

---

### 5. Orchestrator pipeline (`messengers_router/orchestrator.py`, NEW)

The system now has a clean 5-stage pipeline that owns the full request lifecycle:

```
early_guards → nlu_route → clarify_gate → tool_loop → render
```

| Stage | What it does |
|---|---|
| `early_guards` | Deterministic safety checks before LLM (explicit operator request, non-Samara city) |
| `nlu_route` | Runs LLM NLU, updates `state.dialog` |
| `clarify_gate` | Checks if clarification is needed before tools |
| `tool_loop` | Runs pre-pending handlers, then calls legacy route for middleware |
| `render` | Builds `ResponseEnvelope`: recovery → pending → prebuilt → `render_stream()` |

`patient_routing_stream()` is now a **110-line shell**:
```
prechecks (operator / city / appointment guard)
→ run_pipeline()
→ debug envelope
→ history + summary
→ yield ctx.response
```

The old `patient_routing_stream()` had ~350 lines including a 200-line rendering block doing intent routing, tool calls, and text rendering all mixed together. That block is gone.

---

### 6. `render()` owns all response construction

**Before:** `render()` returned `None` for normal paths; `patient_routing_stream()` had the actual rendering logic.

**After:** `render()` always produces a `ResponseEnvelope`, in this order:
1. Safety templates (URGENT / COMPLAINT / MEDICAL_ADVICE) — sync, no LLM
2. Clarify gate — returns clarify question
3. Greeting shortcut — returns intro text on first turn
4. Recovery policy — handles low-confidence / repeated misunderstanding
5. Pending clarification — appointment/catalog slot questions
6. Pre-built deterministic response via `response_builder.py` (catalog confirm, operator offer, price/doctor/schedule/address builders)
7. `render_stream()` — LLM generates response for everything else

---

### 7. Services modularization (`messengers_router/services/`)

`services.py` (7,457 lines) is now `services_legacy.py`. A new `services/` package extracts domain logic progressively:

| Module | Methods extracted | Status |
|---|---|---|
| `services/doctors.py` | `match_catalog_doctor`, `resolve_doctor_name`, `doctors_info`, `doctors_schedule_week`, `_schedule_by_specialty`, `_doctor_availability_snapshot`, `_resolve_doctor_id_from_name` | ✅ Done |
| `services/prices.py` | `price_info` | ✅ Done |
| `services/addresses.py` | `address_info` | ✅ Done |
| `services/lab_tests.py` | `test_result_status`, `test_prepare` + helpers | ⬜ Stub only |
| `services/appointments.py` | `appointment_help`, `test_assist` | ⬜ Stub only |

The `services/__init__.py` facade re-exports everything from `services_legacy` and proxies monkeypatching so tests keep working. All callers use `from .services import Services` unchanged.

Pattern for each domain module:
```python
def _legacy_module():          # lazy import to avoid circular imports
    from .. import services_legacy as legacy
    return legacy

async def price_info(self: "Services", query, entities):
    legacy = _legacy_module()
    # ... uses legacy._helper() for module-level functions
    # ... uses self._private_method() for methods already on Services
```

---

## What's left (non-blocking, post-push)

| Task | Effort | File |
|---|---|---|
| Extract `lab_tests.py` (7 methods) | Medium | `services_legacy.py` lines 6523–6925 |
| Extract `appointments.py` (2 methods) | Small | `services_legacy.py` lines 6501–6522 |
| Switch remaining `last_entities.get("appointment_flow_active")` reads to `state.dialog.phase` | Small | `planner.py`, `flow_policy.py`, `classifier.py`, `dialog_graph.py`, `policies.py` |
| Remove `route_patient_message()` from `tool_loop()` (requires moving post-NLU middleware out first) | Large | `orchestrator.py`, `router.py` |
| Delete empty `lab_tests.py` / `appointments.py` stubs if not filling them | Trivial | `services/` |

---

## File map — what to look at first

```
messengers_router/
├── russian_nlu.py          NEW — normalize_ru(), ENTITY_WHITELIST
├── mess_types.py           + ConfidencePolicy, AppointmentPhase, DialogState
├── nlu_pipeline.py         _merge() fixed — LLM-first
├── orchestrator.py         NEW — 5-stage pipeline, render() owns response
├── router.py               patient_routing_stream() is now a shell
├── appointment_flow_guard.py reads state.dialog.phase (not last_entities flags)
├── response_builder.py     deterministic response builders (unchanged API)
└── services/
    ├── __init__.py         facade + monkeypatch proxy
    ├── doctors.py          7 methods extracted
    ├── prices.py           price_info extracted
    └── addresses.py        address_info extracted
```

---

## How to run tests

```bash
cd /path/to/project
PYTHONPATH=. venv/bin/pytest tests/ -x -q --ignore=tests/eval
# Expected: 516 passed, 3 warnings
```

The 3 warnings are pydantic v2 deprecation and websockets legacy — both from library code, not ours.

