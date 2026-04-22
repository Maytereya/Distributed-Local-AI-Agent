"""Runtime-политика режимов LLM для мессенджерного роутера.

Ответственность модуля: нормализовать входные runtime-опции (`strict/hybrid/rich`)
и дать единый объект настроек для pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

LLMMode = Literal["strict", "hybrid", "rich"]


@dataclass(frozen=True)
class RuntimeOptions:
    llm_mode: LLMMode = "hybrid"
    self_check: bool = False
    self_check_max_retries: int = 1
    self_check_threshold: float = 0.8
    queue_timeout_ms: int = 30000

    @property
    def uses_llm_primary_nlu(self) -> bool:
        return self.llm_mode in {"hybrid", "rich"}

    @property
    def allows_refine_pass(self) -> bool:
        return self.llm_mode == "rich"


def _norm_mode(value: Any) -> LLMMode:
    raw = str(value or "").strip().lower()
    if raw in {"strict", "hybrid", "rich"}:
        return raw  # type: ignore[return-value]
    return "hybrid"


def normalize_runtime_options(
    *,
    llm_mode: Any = None,
    self_check: Any = None,
    self_check_max_retries: Any = None,
    self_check_threshold: Any = None,
    queue_timeout_ms: Any = None,
) -> RuntimeOptions:
    mode = _norm_mode(llm_mode)
    check = bool(self_check) if self_check is not None else False
    try:
        retries = int(self_check_max_retries if self_check_max_retries is not None else 1)
    except Exception:
        retries = 1
    retries = max(0, min(retries, 2))

    try:
        threshold = float(self_check_threshold if self_check_threshold is not None else 0.8)
    except Exception:
        threshold = 0.8
    threshold = max(0.0, min(threshold, 1.0))

    try:
        timeout_ms = int(queue_timeout_ms if queue_timeout_ms is not None else 30000)
    except Exception:
        timeout_ms = 30000
    timeout_ms = max(1000, min(timeout_ms, 120000))

    return RuntimeOptions(
        llm_mode=mode,
        self_check=check and mode == "rich",
        self_check_max_retries=retries,
        self_check_threshold=threshold,
        queue_timeout_ms=timeout_ms,
    )
