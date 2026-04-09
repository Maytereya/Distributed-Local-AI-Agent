#!/usr/bin/env python3
"""In-process FT eval runner with CI quality gate."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from localragagent.freetalk.agent import FreeTalkAgent
from localragagent.freetalk.clinical_router import parse_clinical_decision
from localragagent.freetalk.config import FreeTalkConfig
from localragagent.freetalk.contracts import SessionContext
from localragagent.ports import freetalk_llm_port


def _norm(value: str) -> str:
    return str(value or "").strip().lower()


def _contains_all(text: str, patterns: list[str]) -> bool:
    hay = _norm(text)
    for pattern in patterns:
        if _norm(pattern) not in hay:
            return False
    return True


def _contains_any(text: str, patterns: list[str]) -> bool:
    hay = _norm(text)
    return any(_norm(pattern) in hay for pattern in patterns)


def _looks_like_clarification(text: str) -> bool:
    value = _norm(text)
    if not value:
        return False
    if "уточните" in value:
        return True
    if "укажите" in value and "пожалуйста" in value:
        return True
    return value.endswith("?")


def _config() -> FreeTalkConfig:
    return FreeTalkConfig(
        mode_key="Free-talk-Ai",
        redis_url="redis://redis:6379/0",
        redis_prefix="ft_eval",
        session_ttl_sec=86400,
        history_tail_turns=14,
        compaction_trigger_turns=9999,
        summary_keep_turns=8,
        max_tool_steps=3,
        llm_timeout_s=5,
        llm_queue_timeout_ms=2000,
        include_meili_tools=True,
        enable_web_search_tool=True,
        web_search_url="http://searxng:8080",
        web_search_timeout_s=4,
        web_search_max_results=5,
        web_search_language="ru-RU",
        web_search_healthcheck_timeout_s=2,
        web_search_healthcheck_ttl_s=5,
        context_window_tokens=24576,
        context_warn_ratio=0.82,
        context_estimate_chars_per_token=4,
        context_response_reserve_tokens=2048,
        persistent_memory_path=PROJECT_ROOT / "app_data/free_talk_memory/dialog_summaries.jsonl",
        system_prompt_path=PROJECT_ROOT / "src/localragagent/freetalk/system_prompt.txt",
    )


class InMemoryMemory:
    def __init__(self) -> None:
        self.turns: dict[str, list[dict[str, Any]]] = {}
        self.summaries: dict[str, str] = {}
        self.meta: dict[str, dict[str, str]] = {}

    async def load_context(self, session_id: str, *, history_tail_turns: int) -> SessionContext:
        sid = str(session_id or "").strip()
        all_turns = list(self.turns.get(sid, []))
        tail = all_turns[-history_tail_turns:] if history_tail_turns > 0 else all_turns
        return SessionContext(session_id=sid, summary=self.summaries.get(sid, ""), turns=tail)

    async def append_exchange(
        self,
        session_id: str,
        *,
        user_text: str,
        assistant_text: str,
        source: str,
        source_fragments: list[dict[str, str]] | None = None,
    ) -> None:
        sid = str(session_id or "").strip()
        bucket = self.turns.setdefault(sid, [])
        bucket.append({"role": "user", "content": str(user_text or "")})
        assistant_turn: dict[str, Any] = {
            "role": "assistant",
            "content": str(assistant_text or ""),
            "source": str(source or ""),
        }
        if source_fragments:
            assistant_turn["source_fragments"] = list(source_fragments)
        bucket.append(assistant_turn)

    async def get_turn_count(self, session_id: str) -> int:
        sid = str(session_id or "").strip()
        return len(self.turns.get(sid, []))

    async def save_summary(self, session_id: str, summary: str) -> None:
        sid = str(session_id or "").strip()
        self.summaries[sid] = str(summary or "")

    async def get_meta_int(self, session_id: str, key: str, default: int = 0) -> int:
        sid = str(session_id or "").strip()
        raw = self.meta.get(sid, {}).get(str(key or "").strip(), "")
        try:
            return int(raw)
        except Exception:
            return int(default)

    async def set_meta_int(self, session_id: str, key: str, value: int) -> None:
        sid = str(session_id or "").strip()
        self.meta.setdefault(sid, {})[str(key or "").strip()] = str(int(value))

    async def get_meta_str(self, session_id: str, key: str, default: str = "") -> str:
        sid = str(session_id or "").strip()
        return str(self.meta.get(sid, {}).get(str(key or "").strip(), default) or "")

    async def set_meta_str(self, session_id: str, key: str, value: str) -> None:
        sid = str(session_id or "").strip()
        self.meta.setdefault(sid, {})[str(key or "").strip()] = str(value or "")

    async def clear_session(self, session_id: str) -> None:
        sid = str(session_id or "").strip()
        self.turns.pop(sid, None)
        self.summaries.pop(sid, None)
        self.meta.pop(sid, None)


class InMemoryPersist:
    def __init__(self) -> None:
        self.snapshots: list[dict[str, Any]] = []

    async def append_snapshot(
        self,
        *,
        session_id: str,
        summary: str,
        key_facts: list[str],
        open_loops: list[str],
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.snapshots.append(
            {
                "session_id": session_id,
                "summary": summary,
                "key_facts": list(key_facts or []),
                "open_loops": list(open_loops or []),
                "extra": dict(extra or {}),
            }
        )


class StubServices:
    def __init__(self) -> None:
        self.doctors: dict[str, dict[str, str]] = {
            "дразнин": {"fio": "Дразнин Антон Владимирович", "specialization": "Эндоскопист"},
            "трубин": {"fio": "Трубин Алексей Юрьевич", "specialization": "Уролог"},
        }
        self.services: dict[str, dict[str, Any]] = {
            "фкс": {
                "name": "ФКС с наркозом",
                "cost": 7200,
                "prepare": "Подготовка к ФКС: диета за 3 дня, очищение по инструкции врача.",
            },
            "общий анализ крови": {
                "name": "Общий анализ крови",
                "cost": 650,
                "prepare": "Сдавать натощак, утром, допускается вода.",
            },
            "узи шеи": {
                "name": "УЗИ шеи",
                "cost": 1500,
                "prepare": "Специальной подготовки не требуется.",
            },
        }

    async def get_catalog_health(self) -> dict[str, Any]:
        return {"ok": True}

    def _resolve_doctor(self, value: str) -> tuple[str, dict[str, str] | None]:
        text = _norm(value)
        for key, doctor in self.doctors.items():
            if key in text or _norm(doctor["fio"]) in text:
                return key, doctor
        return "", None

    def _resolve_service(self, value: str) -> tuple[str, dict[str, Any] | None]:
        text = _norm(value)
        for key, service in self.services.items():
            if key in text or _norm(service["name"]) in text:
                return key, service
        return "", None

    async def match_catalog_service(self, raw_text_or_name: str, *, current_service_name: str = "") -> dict[str, Any]:
        probe = str(current_service_name or raw_text_or_name or "")
        key, service = self._resolve_service(probe)
        if service:
            return {"status": "exact", "canonical": service["name"], "query": raw_text_or_name}
        return {"status": "miss", "canonical": "", "query": raw_text_or_name}

    async def match_catalog_doctor(self, raw_text_or_name: str) -> dict[str, Any]:
        key, doctor = self._resolve_doctor(raw_text_or_name)
        if doctor:
            return {"status": "exact", "canonical": doctor["fio"], "query": raw_text_or_name}
        return {"status": "miss", "canonical": "", "query": raw_text_or_name}

    async def service_bundle_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        probe = str(entities.get("service_name") or query)
        _key, service = self._resolve_service(probe)
        if not service:
            return {"retail_prices": [], "doctors": [], "prepare": "", "note": "service_bundle_info"}
        return {
            "retail_prices": [{"serviceName": service["name"], "cost": service["cost"]}],
            "doctors": [{"fio": self.doctors["дразнин"]["fio"]}],
            "prepare": str(service["prepare"]),
            "note": "service_bundle_info",
        }

    async def price_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        probe = str(entities.get("service_name") or query)
        _key, service = self._resolve_service(probe)
        if not service:
            return {"prices": [], "note": "price_info"}
        return {
            "prices": [{"serviceName": service["name"], "cost": service["cost"]}],
            "entities_used": {"service_name_effective": service["name"]},
            "note": "price_info",
        }

    async def test_prepare(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        probe = str(entities.get("service_name") or entities.get("test_name") or query)
        _key, service = self._resolve_service(probe)
        if not service:
            return {"prepare": "", "note": "test_prepare"}
        return {"prepare": str(service["prepare"]), "note": "test_prepare"}

    async def test_assist(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        probe = str(entities.get("service_name") or entities.get("test_name") or query)
        _key, service = self._resolve_service(probe)
        if not service:
            return {"tests": [], "note": "test_assist"}
        return {"tests": [{"serviceName": service["name"], "cost": service["cost"]}], "note": "test_assist"}

    async def doctors_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        probe = str(entities.get("doctor_name") or query)
        _key, doctor = self._resolve_doctor(probe)
        if not doctor:
            return {"doctors": [], "note": "doctors_info"}
        return {"doctors": [doctor], "note": "doctors_info"}

    async def doctors_schedule_week(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        probe = str(entities.get("doctor_name") or query)
        _key, doctor = self._resolve_doctor(probe)
        if not doctor:
            return {"schedule": [], "schedule_unavailable_reason": "", "note": "doctors_schedule_week"}
        return {
            "schedule": [
                {
                    "fio": doctor["fio"],
                    "schedule": {
                        "г. Самара, пр. Ленина, 5": [
                            {"date": "2026-04-10", "slots": ["09:00", "09:30", "10:00"], "start": "09:00", "end": "18:00"},
                        ]
                    },
                }
            ],
            "entities_used": {"doctor_name_resolved": doctor["fio"]},
            "schedule_unavailable_reason": "",
            "note": "doctors_schedule_week",
        }

    async def address_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        _ = query, entities
        return {"addresses": ["г. Самара, пр. Ленина, 5"], "note": "address_info"}

    async def test_result_status(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        _ = query
        if str(entities.get("order_number") or "").strip():
            return {"ready": True, "result_links": ["https://example.org/result"], "note": "test_result_status"}
        return {"ready": False, "missing_fields": ["номер заказа"], "note": "test_result_status"}

    async def main_index_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        _ = entities
        if "фкс" in _norm(query):
            return {"content": "Из документов клиники: подготовка к ФКС с наркозом включает диету и очищение.", "note": "main_index_info"}
        return {"content": "", "note": "main_index_info"}

    async def news_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        _ = query, entities
        return {"news": [{"title": "Новости клиники", "url": "https://example.org/news"}], "note": "news_info"}

    def tool_handlers(self, *, include_meili_tools: bool) -> dict[str, Any]:
        handlers: dict[str, Any] = {
            "service_bundle_info": self.service_bundle_info,
            "price_info": self.price_info,
            "test_prepare": self.test_prepare,
            "test_assist": self.test_assist,
            "doctors_info": self.doctors_info,
            "doctors_schedule_week": self.doctors_schedule_week,
            "address_info": self.address_info,
            "test_result_status": self.test_result_status,
        }
        if include_meili_tools:
            handlers["main_index_info"] = self.main_index_info
            handlers["news_info"] = self.news_info
        return handlers


class StubWebSearch:
    async def search(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        _ = entities
        if not str(query or "").strip():
            return {"results": [], "note": "web_search: empty query"}
        return {
            "results": [
                {
                    "title": "Свежие новости по теме",
                    "url": "https://example.org/web-news",
                    "snippet": "Краткая сводка из интернета по запросу пользователя.",
                    "source": "stub",
                }
            ],
            "note": "web_search",
        }


class EvalFreeTalkAgent(FreeTalkAgent):
    __slots__ = ("_eval_router_payload", "_eval_verifier_payload", "_eval_general_answer")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._eval_router_payload: dict[str, Any] | None = None
        self._eval_verifier_payload: dict[str, Any] | None = None
        self._eval_general_answer: str = ""

    def set_eval_step(
        self,
        *,
        router_payload: dict[str, Any] | None,
        verifier_payload: dict[str, Any] | None,
        general_answer: str,
    ) -> None:
        self._eval_router_payload = router_payload if isinstance(router_payload, dict) else None
        self._eval_verifier_payload = verifier_payload if isinstance(verifier_payload, dict) else None
        self._eval_general_answer = str(general_answer or "")

    async def _route_clinical_decision(
        self,
        *,
        user_message: str,
        context: SessionContext,
        pending_state: dict[str, Any],
        remembered_doctor: str,
    ) -> Any:
        _ = user_message, context, pending_state, remembered_doctor
        payload = self._eval_router_payload
        if isinstance(payload, dict):
            decision = parse_clinical_decision(
                payload,
                include_meili_tools=self.config.include_meili_tools,
            )
            decision.source = "eval_dataset_router"
            return decision
        return await super()._route_clinical_decision(
            user_message=user_message,
            context=context,
            pending_state=pending_state,
            remembered_doctor=remembered_doctor,
        )

    async def _llm_json(self, prompt: str) -> dict[str, Any]:
        if "post-tool verifier" in str(prompt or "").lower():
            payload = self._eval_verifier_payload
            if isinstance(payload, dict):
                return payload
        return {}

    async def _llm_text(self, prompt: str) -> str:
        _ = prompt
        return str(self._eval_general_answer or "").strip()


@dataclass(slots=True)
class StepEval:
    session_id: str
    step_index: int
    message: str
    reply_text: str
    tool_called: bool
    clarification: bool
    source_tags: list[str]
    pass_step: bool
    hallucination: bool
    grounded: bool
    expected: dict[str, Any]
    failures: list[str]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            items.append(value)
    return items


async def _run_eval(dataset_path: Path, *, profile: str) -> list[StepEval]:
    sessions = _load_jsonl(dataset_path)
    memory = InMemoryMemory()
    persist = InMemoryPersist()
    services = StubServices()
    web = StubWebSearch()
    if profile == "deterministic":
        agent: FreeTalkAgent = EvalFreeTalkAgent(
            config=_config(),
            services=services,  # type: ignore[arg-type]
            memory=memory,  # type: ignore[arg-type]
            persist=persist,  # type: ignore[arg-type]
            system_prompt="FT eval",
            web_search=web,  # type: ignore[arg-type]
        )
    else:
        agent = FreeTalkAgent(
            config=_config(),
            services=services,  # type: ignore[arg-type]
            memory=memory,  # type: ignore[arg-type]
            persist=persist,  # type: ignore[arg-type]
            system_prompt="FT eval",
            web_search=web,  # type: ignore[arg-type]
        )
    results: list[StepEval] = []
    for session in sessions:
        sid = str(session.get("session_id") or "").strip()
        if not sid:
            continue
        steps = session.get("steps") if isinstance(session.get("steps"), list) else []
        for idx, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            message = str(step.get("message") or "").strip()
            expected = step.get("expected") if isinstance(step.get("expected"), dict) else {}
            if isinstance(agent, EvalFreeTalkAgent):
                agent.set_eval_step(
                    router_payload=step.get("router_decision") if isinstance(step.get("router_decision"), dict) else None,
                    verifier_payload=step.get("verifier_decision") if isinstance(step.get("verifier_decision"), dict) else None,
                    general_answer=str(step.get("general_answer") or ""),
                )
            reply = await agent.chat(message, sid)
            pending_raw = await memory.get_meta_str(sid, "clinical_pending_state", "")
            clarification = bool(str(pending_raw or "").strip()) or _looks_like_clarification(reply.text)
            tool_called = bool(str(reply.tool_name or "").strip())
            source_tags = []
            for frag in reply.source_fragments or []:
                if not isinstance(frag, dict):
                    continue
                tag = _norm(str(frag.get("source") or ""))
                if tag and tag not in source_tags:
                    source_tags.append(tag)

            must_contain = [str(x) for x in (expected.get("must_contain") or []) if str(x).strip()]
            must_not = [str(x) for x in (expected.get("must_not_contain") or []) if str(x).strip()]
            expected_tags = [_norm(str(x)) for x in (expected.get("source_tags") or []) if _norm(str(x))]
            clarification_expected = expected.get("clarification_expected")
            tool_expected = expected.get("tool_called_expected")
            grounded_required = bool(expected.get("grounded_required", False))

            failures: list[str] = []
            if must_contain and not _contains_all(reply.text, must_contain):
                failures.append("missing_required_phrases")
            if must_not and _contains_any(reply.text, must_not):
                failures.append("contains_forbidden_phrase")
            if expected_tags and not set(expected_tags).issubset(set(source_tags)):
                failures.append("source_tags_mismatch")
            if clarification_expected is not None and clarification != bool(clarification_expected):
                failures.append("clarification_mismatch")
            if tool_expected is not None and tool_called != bool(tool_expected):
                failures.append("tool_called_mismatch")

            grounded = True
            if grounded_required:
                grounded = bool("clinic_data" in source_tags and "general_knowledge" not in source_tags and "web_search" not in source_tags and tool_called)
                if not grounded:
                    failures.append("grounding_mismatch")

            hallucination = False
            if grounded_required and (not grounded):
                hallucination = True
            if _contains_any(reply.text, must_not):
                hallucination = True

            results.append(
                StepEval(
                    session_id=sid,
                    step_index=idx,
                    message=message,
                    reply_text=str(reply.text or ""),
                    tool_called=tool_called,
                    clarification=clarification,
                    source_tags=source_tags,
                    pass_step=not failures,
                    hallucination=hallucination,
                    grounded=grounded,
                    expected=expected,
                    failures=failures,
                )
            )
    return results


def _metric_ratio(num: int, den: int) -> float:
    if den <= 0:
        return 1.0
    return float(num) / float(den)


def _compute_metrics(results: list[StepEval]) -> dict[str, float]:
    tool_required = [r for r in results if bool(r.expected.get("tool_call_required", False))]
    tool_called = [r for r in tool_required if r.tool_called]

    clarify_cases = [r for r in results if "clarification_expected" in r.expected]
    clarify_ok = [r for r in clarify_cases if r.clarification == bool(r.expected.get("clarification_expected"))]

    grounded_cases = [r for r in results if bool(r.expected.get("grounded_required", False))]
    grounded_ok = [r for r in grounded_cases if r.grounded]

    hallucination_cases = grounded_cases
    hallucinated = [r for r in hallucination_cases if r.hallucination]

    pass_cases = [r for r in results if r.pass_step]

    return {
        "tool_call_recall": _metric_ratio(len(tool_called), len(tool_required)),
        "clarification_appropriateness": _metric_ratio(len(clarify_ok), len(clarify_cases)),
        "answer_grounded_rate": _metric_ratio(len(grounded_ok), len(grounded_cases)),
        "hallucination_rate_clinic": _metric_ratio(len(hallucinated), len(hallucination_cases)),
        "step_pass_rate": _metric_ratio(len(pass_cases), len(results)),
        "steps_total": float(len(results)),
    }


def _load_thresholds(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _check_gate(metrics: dict[str, float], thresholds: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    min_steps = int(thresholds.get("min_steps", 1))
    if int(metrics.get("steps_total", 0)) < min_steps:
        errors.append(f"steps_total < min_steps ({int(metrics.get('steps_total', 0))} < {min_steps})")

    tool_min = float(thresholds.get("tool_call_recall_min", 0.0))
    if metrics["tool_call_recall"] < tool_min:
        errors.append(f"tool_call_recall {metrics['tool_call_recall']:.3f} < {tool_min:.3f}")

    clarify_min = float(thresholds.get("clarification_appropriateness_min", 0.0))
    if metrics["clarification_appropriateness"] < clarify_min:
        errors.append(
            f"clarification_appropriateness {metrics['clarification_appropriateness']:.3f} < {clarify_min:.3f}"
        )

    grounded_min = float(thresholds.get("answer_grounded_rate_min", 0.0))
    if metrics["answer_grounded_rate"] < grounded_min:
        errors.append(f"answer_grounded_rate {metrics['answer_grounded_rate']:.3f} < {grounded_min:.3f}")

    hallucination_max = float(thresholds.get("hallucination_rate_clinic_max", 1.0))
    if metrics["hallucination_rate_clinic"] > hallucination_max:
        errors.append(
            f"hallucination_rate_clinic {metrics['hallucination_rate_clinic']:.3f} > {hallucination_max:.3f}"
        )

    step_pass_min = float(thresholds.get("step_pass_rate_min", 0.0))
    if metrics["step_pass_rate"] < step_pass_min:
        errors.append(f"step_pass_rate {metrics['step_pass_rate']:.3f} < {step_pass_min:.3f}")

    return errors


async def _probe_llm() -> tuple[bool, str]:
    try:
        text, _usage = await freetalk_llm_port.generate_text_with_usage(
            "Ответь одним словом: ok",
            timeout_s=10,
            queue_timeout_ms=5000,
            fmt=None,
            think=False,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if not str(text or "").strip():
        return False, "empty_llm_response"
    return True, "ok"


def _print_report(results: list[StepEval], metrics: dict[str, float], gate_errors: list[str]) -> None:
    print("FT EVAL REPORT")
    print(f"steps_total={int(metrics['steps_total'])}")
    print(f"tool_call_recall={metrics['tool_call_recall']:.3f}")
    print(f"clarification_appropriateness={metrics['clarification_appropriateness']:.3f}")
    print(f"answer_grounded_rate={metrics['answer_grounded_rate']:.3f}")
    print(f"hallucination_rate_clinic={metrics['hallucination_rate_clinic']:.3f}")
    print(f"step_pass_rate={metrics['step_pass_rate']:.3f}")
    failures = [r for r in results if not r.pass_step]
    if failures:
        print("")
        print("STEP FAILURES:")
        for row in failures:
            print(
                f"- {row.session_id}#{row.step_index}: failures={','.join(row.failures)} "
                f"tool_called={row.tool_called} clarify={row.clarification} tags={','.join(row.source_tags)}"
            )
    if gate_errors:
        print("")
        print("QUALITY GATE: FAILED")
        for err in gate_errors:
            print(f"- {err}")
    else:
        print("")
        print("QUALITY GATE: PASSED")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FreeTalk in-process eval and enforce quality gate thresholds.")
    parser.add_argument(
        "--profile",
        choices=["deterministic", "production-like"],
        default="deterministic",
        help="deterministic: uses router_decision from dataset; production-like: uses live LLM router/verifier.",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Path to FT eval dataset (.jsonl).",
    )
    parser.add_argument(
        "--thresholds",
        default="",
        help="Path to quality gate thresholds (json).",
    )
    args = parser.parse_args()

    if str(args.profile) == "production-like":
        dataset_arg = str(args.dataset or "tests/eval/freetalk_dialogs_production_like.jsonl")
        thresholds_arg = str(args.thresholds or "tests/eval/freetalk_quality_gate_production_like.json")
    else:
        dataset_arg = str(args.dataset or "tests/eval/freetalk_dialogs.jsonl")
        thresholds_arg = str(args.thresholds or "tests/eval/freetalk_quality_gate.json")

    dataset_path = (PROJECT_ROOT / dataset_arg).resolve() if not Path(dataset_arg).is_absolute() else Path(dataset_arg)
    thresholds_path = (PROJECT_ROOT / thresholds_arg).resolve() if not Path(thresholds_arg).is_absolute() else Path(thresholds_arg)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 2
    if not thresholds_path.exists():
        print(f"Thresholds file not found: {thresholds_path}")
        return 2

    profile = str(args.profile)
    if profile == "production-like":
        ok, probe = asyncio.run(_probe_llm())
        if not ok:
            print("LLM probe failed for production-like profile.")
            print(f"Reason: {probe}")
            print("Run this profile inside runtime environment where messengers_router.llm_runtime can access Ollama.")
            return 2

    results = asyncio.run(_run_eval(dataset_path, profile=profile))
    metrics = _compute_metrics(results)
    thresholds = _load_thresholds(thresholds_path)
    gate_errors = _check_gate(metrics, thresholds)
    _print_report(results, metrics, gate_errors)
    return 1 if gate_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
