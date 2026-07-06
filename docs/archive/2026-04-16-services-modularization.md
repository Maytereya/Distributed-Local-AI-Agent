# Services Modularization: Extract Remaining Domains from services_legacy.py

**Branch:** `refactor/core` → merges into `origin/release`  
**Test command:** `cd /Users/maxten/Dev/Distributed-Local-AI-Agent2 && PYTHONPATH=. venv/bin/pytest tests/ -x -q --ignore=tests/eval`  
**Current baseline:** 514 passed, 0 failed — must stay green after every task.

---

## Background

`messengers_router/services_legacy.py` is a 7,457-line monolith. The `Services` class (starts at line 4945) holds all business-domain methods. The plan is to extract each domain into a submodule under `messengers_router/services/`.

### What's already done

`messengers_router/services/doctors.py` (672 lines) — pilot extraction. It contains:
- `match_catalog_doctor`, `_schedule_by_specialty`, `_doctor_availability_snapshot`
- `resolve_doctor_name`, `_resolve_doctor_id_from_name`
- `doctors_info`, `doctors_schedule_week`

At the **bottom** of `services_legacy.py` (lines 7430–7448), the rebind block already exists:
```python
from .services.doctors import (
    _doctor_availability_snapshot as _doctor_availability_snapshot_impl,
    _resolve_doctor_id_from_name as _resolve_doctor_id_from_name_impl,
    _schedule_by_specialty as _schedule_by_specialty_impl,
    doctors_info as _doctors_info_impl,
    doctors_schedule_week as _doctors_schedule_week_impl,
    match_catalog_doctor as _match_catalog_doctor_impl,
    resolve_doctor_name as _resolve_doctor_name_impl,
)

Services.match_catalog_doctor = _match_catalog_doctor_impl
Services._schedule_by_specialty = _schedule_by_specialty_impl
Services._doctor_availability_snapshot = _doctor_availability_snapshot_impl
Services.resolve_doctor_name = _resolve_doctor_name_impl
Services._resolve_doctor_id_from_name = _resolve_doctor_id_from_name_impl
Services.doctors_info = _doctors_info_impl
Services.doctors_schedule_week = _doctors_schedule_week_impl
```

The facade at `messengers_router/services/__init__.py` re-exports everything from `services_legacy` and proxies monkeypatching. Do NOT modify it.

---

## The pattern — follow exactly

Each domain submodule uses this structure:

```python
"""Domain docstring."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..services_legacy import Services


def _legacy_module():
    """Lazy import to avoid circular imports."""
    from .. import services_legacy as legacy
    return legacy


async def method_name(self: "Services", query: str, entities: dict[str, Any]) -> dict[str, Any]:
    legacy = _legacy_module()
    # ... implementation using legacy.helper_function(...)
    # "self" is the Services instance — all self._xxx calls work unchanged
```

At the **bottom of `services_legacy.py`** (before the `if __name__ == "__main__":` block), add:
```python
from .services.domain import fn1 as _fn1_impl, fn2 as _fn2_impl
Services.method1 = _fn1_impl
Services.method2 = _fn2_impl
```

**Critical:** The methods inside the domain module call `self._xxx()` for private helpers that remain in `services_legacy.py`. Those calls go through the live `Services` instance — they still work because `self` IS the `Services` instance. Module-level helpers (`_normalise_input`, `_service_fallback`, etc.) must be accessed via `legacy = _legacy_module(); legacy.helper(...)`.

---

## Domain extraction tasks

Do them **one at a time**, run the test suite after each.

---

### Task 1 — `services/prices.py`

Extract `price_info` from `services_legacy.py`.

**Method in services_legacy.py:** `async def price_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]` at line ~6988.

**The method ends at line ~7163** (next method is `address_info` at ~7164).

**Helper functions used by price_info** (stay in services_legacy.py, access via `_legacy_module()`):
- `_get_first_present` — module-level function
- `_as_int` — module-level function
- `_normalise_input` — module-level function
- `_resolve_doctor_id_from_name` — already extracted into doctors.py, accessed as `self._resolve_doctor_id_from_name()` — that call works unchanged
- `_rank_price_rows` — module-level function
- `_service_fallback` — module-level function
- `_is_service_query_type` — module-level function
- `SAMARA_PRICE_REGION_ID` — module-level constant

Create `messengers_router/services/prices.py`:

```python
"""Модуль домена цен."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..services_legacy import Services


def _legacy_module():
    from .. import services_legacy as legacy
    return legacy


async def price_info(self: "Services", query: str, entities: dict[str, Any]) -> dict[str, Any]:
    legacy = _legacy_module()
    # paste the full body of price_info here, replacing every
    # module-level name with legacy.name
    # e.g.:  _get_first_present(...)  →  legacy._get_first_present(...)
    #        SAMARA_PRICE_REGION_ID   →  legacy.SAMARA_PRICE_REGION_ID
    #        _service_fallback(...)   →  legacy._service_fallback(...)
    # self.xxx() calls stay as-is — they resolve on the Services instance
    ...
```

Then in `services_legacy.py`, at the rebind block (right before `if __name__ == "__main__":`), append:
```python
from .services.prices import price_info as _price_info_impl
Services.price_info = _price_info_impl
```

**Test:** Run full suite. Add one smoke test in `tests/test_services_modular.py` (create if it doesn't exist):
```python
def test_services_has_price_info():
    from messengers_router.services import Services
    assert callable(getattr(Services, "price_info", None))
```

---

### Task 2 — `services/addresses.py`

Extract `address_info`.

**Method:** `async def address_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]` at line ~7164.

**The method ends at line ~7354** (next is `news_info` at ~7355).

**Helper functions (access via legacy):**
- `_ensure_regions_loaded` — private method on self, call as `await self._ensure_regions_loaded()` (unchanged)
- `_is_samara_city_value` — module-level function
- `_normalise_input` — module-level
- `_samara_region_tokens` — private method on self: `await self._samara_region_tokens()` (unchanged)
- `_ensure_doctors_cache_loaded` — private method on self (unchanged)
- `_service_fallback` — module-level

Create `messengers_router/services/addresses.py` following the same pattern.

Rebind in `services_legacy.py`:
```python
from .services.addresses import address_info as _address_info_impl
Services.address_info = _address_info_impl
```

---

### Task 3 — `services/lab_tests.py`

Extract these methods:
- `async def test_result_status(self, query, entities)` — line ~6926
- `async def test_prepare(self, query, entities)` — line ~6816
- `async def _prepare_from_analysis_api_cache(self, query, entities)` — line ~6911
- `async def _prepare_llm_validate_candidate(self, query, candidate)` — line ~6523
- `async def _pick_prepare_candidate(self, query, candidates)` — line ~6574
- `async def _prepare_candidates_from_analysis_api_cache(self, query, entities)` — line ~6671
- `async def _maybe_compact_prepare_text(self, query, source_text)` — line ~6729

These methods reference each other (`self._prepare_llm_validate_candidate()`, `self._pick_prepare_candidate()`, etc.) — those calls all work unchanged since they're on the same `self`.

**Important:** `_PrepareCandidate` is a dataclass defined at line 896 of `services_legacy.py`. Reference it via `legacy._PrepareCandidate` in type annotations.

**Module-level helpers (access via legacy):**
- `_prepare_query_variants`, `_prepare_service_info_queries`, `_prepare_relevance_prompt`
- `_parse_prepare_relevance_validator`, `_dedupe_prepare_candidates`
- `_prepare_fast_relevance_score`, `_prepare_relevance_thresholds`, `_prepare_relevance_gate`
- `_maybe_compact_prepare_text` — actually a method, not module-level
- `_is_meili_error_text`, `_service_fallback`, `SAMARA_PRICE_REGION_ID`
- `_normalise_input`, `_get_first_present`
- `_runtime_bool` — module-level

Create `messengers_router/services/lab_tests.py`.

Rebind in `services_legacy.py`:
```python
from .services.lab_tests import (
    test_result_status as _test_result_status_impl,
    test_prepare as _test_prepare_impl,
    _prepare_from_analysis_api_cache as _prepare_from_analysis_api_cache_impl,
    _prepare_llm_validate_candidate as _prepare_llm_validate_candidate_impl,
    _pick_prepare_candidate as _pick_prepare_candidate_impl,
    _prepare_candidates_from_analysis_api_cache as _prepare_candidates_from_analysis_api_cache_impl,
    _maybe_compact_prepare_text as _maybe_compact_prepare_text_impl,
)
Services.test_result_status = _test_result_status_impl
Services.test_prepare = _test_prepare_impl
Services._prepare_from_analysis_api_cache = _prepare_from_analysis_api_cache_impl
Services._prepare_llm_validate_candidate = _prepare_llm_validate_candidate_impl
Services._pick_prepare_candidate = _pick_prepare_candidate_impl
Services._prepare_candidates_from_analysis_api_cache = _prepare_candidates_from_analysis_api_cache_impl
Services._maybe_compact_prepare_text = _maybe_compact_prepare_text_impl
```

---

### Task 4 — `services/appointments.py`

Extract:
- `async def appointment_help(self, query, entities)` — line ~6501
- `async def test_assist(self, query, entities)` — line ~6523 (check exact line)

Wait — `test_assist` is a lab-test concept, not really appointments. But it starts right after `appointment_help` in the source. You may group them together here or move `test_assist` to `lab_tests.py`. Choose whichever keeps related code together — just be consistent.

**Module-level helpers:**
- `meilisearch.search_meili` — accessed via `import` at top of services_legacy.py; copy the import
- `html_cleaner.strip_html` — same
- `_is_meili_error_text` — access via legacy
- `_service_fallback` — access via legacy
- `_rank_price_rows` (for test_assist) — access via legacy
- `_normalise_input` (for test_assist) — access via legacy
- `_get_first_present` (for test_assist) — access via legacy
- `SAMARA_PRICE_REGION_ID` (for test_assist) — access via legacy

Note: `meilisearch` and `html_cleaner` are external packages already imported in services_legacy.py. In the new domain module, import them directly at the top:
```python
from localragagent import meilisearch
from localragagent import html_cleaner
```
Check the exact import paths by looking at the top of `services_legacy.py`.

Rebind:
```python
from .services.appointments import (
    appointment_help as _appointment_help_impl,
    test_assist as _test_assist_impl,
)
Services.appointment_help = _appointment_help_impl
Services.test_assist = _test_assist_impl
```

---

### Task 5 (optional, do only if tasks 1–4 are green) — `services/catalog.py`

Extract:
- `async def get_catalog_health(self)` — line ~5269
- `async def match_catalog_service(self, ...)` — line ~5355
- `async def service_bundle_info(self, ...)` — line ~5660
- `async def _procedure_branches_from_index(self, ...)` — line ~5424

These are more complex (200–300 lines each) and call many private methods on `self`. Extract them last. Follow the same pattern.

---

## What NOT to do

1. Do NOT modify `services/__init__.py` — the facade is already correct.
2. Do NOT delete anything from `services_legacy.py` — only add the rebind lines at the bottom and leave the original method bodies in place. The rebind replaces them on the class at import time.
3. Do NOT import `services_legacy` at the top of a domain module — always use `_legacy_module()` for lazy import.
4. Do NOT add `asyncio` imports to domain modules unless the method body explicitly calls `asyncio.to_thread()`. Check the original method body first.
5. Read the full method body from `services_legacy.py` before writing the domain module — don't guess.

---

## Verification after all tasks

```
PYTHONPATH=. venv/bin/pytest tests/ -x -q --ignore=tests/eval
```

Must still show 514+ passed, 0 failed.
