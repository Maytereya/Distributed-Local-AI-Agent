#!/usr/bin/env python3
"""Architecture guardrails for messengers_router.

Checks:
1. No import cycles inside `messengers_router` top-level modules.
2. Cross-layer imports follow whitelist rules.
3. Direct `agent_logic_2.*` imports are allowed only in adapter/port modules.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


PACKAGE_NAME = "messengers_router"
PACKAGE_DIR = Path(__file__).resolve().parents[1]


LAYER_BY_MODULE: dict[str, str] = {
    # Interface
    "endpoint": "interface",
    # Orchestration
    "router": "orchestration",
    "nlu_pipeline": "orchestration",
    "classifier": "orchestration",
    "planner": "orchestration",
    "executor": "orchestration",
    "response_builder": "orchestration",
    "renderer": "orchestration",
    # Policy / decision helpers
    "appointment_flow_guard": "policy",
    "entity_grounder": "policy",
    "flow_policy": "policy",
    "dialog_graph": "policy",
    "recovery_policy": "policy",
    "policies": "policy",
    "context_summary": "policy",
    "self_check": "policy",
    "topic_registry": "policy",
    # Domain primitives
    "mess_types": "domain",
    "city": "domain",
    "text_templates": "domain",
    "llm_mode_policy": "domain",
    "prompt_contracts": "domain",
    "service_phrase": "domain",
    # Infrastructure / adapters
    "services": "infrastructure",
    "memory": "infrastructure",
    "llm_runtime": "infrastructure",
    "prompt_registry": "infrastructure",
    # Ports to host project
    "runtime_config": "port",
    "doctor_name_port": "port",
}


ALLOWED_LAYER_IMPORTS: dict[str, set[str]] = {
    "interface": {"orchestration", "policy", "domain", "infrastructure", "port"},
    "orchestration": {"policy", "domain", "infrastructure", "port"},
    "policy": {"domain", "infrastructure", "port"},
    "domain": set(),
    "infrastructure": {"domain", "port"},
    "port": set(),
}


ALLOWED_DIRECT_AGENT_LOGIC_IMPORTS = {
    "runtime_config",
    "doctor_name_port",
    "services",
    "llm_runtime",
    "prompt_registry",
}


@dataclass(frozen=True)
class InternalImport:
    src: str
    dst: str
    lineno: int


@dataclass(frozen=True)
class ExternalImport:
    src: str
    target: str
    lineno: int


def discover_modules() -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for path in PACKAGE_DIR.glob("*.py"):
        if path.name == "__init__.py":
            continue
        modules[path.stem] = path
    return modules


def parse_imports(module_name: str, path: Path, module_names: set[str]) -> tuple[list[InternalImport], list[ExternalImport]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    internal: list[InternalImport] = []
    external: list[ExternalImport] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # Relative import from same package.
            if node.level >= 1:
                if node.module:
                    target = node.module.split(".")[0]
                    if target in module_names and target != module_name:
                        internal.append(InternalImport(module_name, target, node.lineno))
                else:
                    # from . import foo, bar
                    for alias in node.names:
                        target = alias.name.split(".")[0]
                        if target in module_names and target != module_name:
                            internal.append(InternalImport(module_name, target, node.lineno))

            # Absolute import from same package.
            if node.level == 0 and node.module:
                if node.module == PACKAGE_NAME:
                    for alias in node.names:
                        target = alias.name.split(".")[0]
                        if target in module_names and target != module_name:
                            internal.append(InternalImport(module_name, target, node.lineno))
                elif node.module.startswith(f"{PACKAGE_NAME}."):
                    target = node.module.split(".")[1]
                    if target in module_names and target != module_name:
                        internal.append(InternalImport(module_name, target, node.lineno))

            # Direct host imports.
            if node.module and (
                node.module == "agent_logic_2" or node.module.startswith("agent_logic_2.")
            ):
                external.append(ExternalImport(module_name, node.module, node.lineno))

        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(f"{PACKAGE_NAME}."):
                    target = alias.name.split(".")[1]
                    if target in module_names and target != module_name:
                        internal.append(InternalImport(module_name, target, node.lineno))
                if alias.name == "agent_logic_2" or alias.name.startswith("agent_logic_2."):
                    external.append(ExternalImport(module_name, alias.name, node.lineno))

    return internal, external


def strongly_connected_components(graph: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    components: list[list[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in graph.get(node, set()):
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif neighbor in on_stack:
                lowlink[node] = min(lowlink[node], indices[neighbor])

        if lowlink[node] == indices[node]:
            component: list[str] = []
            while stack:
                top = stack.pop()
                on_stack.remove(top)
                component.append(top)
                if top == node:
                    break
            components.append(sorted(component))

    for node in sorted(graph):
        if node not in indices:
            strongconnect(node)

    return components


def find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    cycles: list[list[str]] = []
    for component in strongly_connected_components(graph):
        if len(component) > 1:
            cycles.append(component)
    return sorted(cycles)


def unique_internal_imports(imports: Iterable[InternalImport]) -> list[InternalImport]:
    seen: set[tuple[str, str]] = set()
    out: list[InternalImport] = []
    for imp in sorted(imports, key=lambda x: (x.src, x.dst, x.lineno)):
        key = (imp.src, imp.dst)
        if key in seen:
            continue
        seen.add(key)
        out.append(imp)
    return out


def check_layer_map(modules: dict[str, Path]) -> list[str]:
    errors: list[str] = []
    unknown = sorted(set(modules) - set(LAYER_BY_MODULE))
    stale = sorted(set(LAYER_BY_MODULE) - set(modules))
    if unknown:
        errors.append(f"Layer map is missing modules: {', '.join(unknown)}")
    if stale:
        errors.append(f"Layer map has stale modules: {', '.join(stale)}")
    return errors


def run_checks() -> int:
    modules = discover_modules()
    layer_errors = check_layer_map(modules)

    internal_imports: list[InternalImport] = []
    external_imports: list[ExternalImport] = []
    module_names = set(modules)

    for module_name, path in sorted(modules.items()):
        internal, external = parse_imports(module_name, path, module_names)
        internal_imports.extend(internal)
        external_imports.extend(external)

    internal_imports = unique_internal_imports(internal_imports)

    graph: dict[str, set[str]] = {module_name: set() for module_name in modules}
    for imp in internal_imports:
        graph[imp.src].add(imp.dst)

    violations: list[str] = []
    violations.extend(layer_errors)

    # Cycles
    cycles = find_cycles(graph)
    for component in cycles:
        violations.append(f"Import cycle detected: {' -> '.join(component)}")

    # Cross-layer rules
    for imp in internal_imports:
        src_layer = LAYER_BY_MODULE.get(imp.src)
        dst_layer = LAYER_BY_MODULE.get(imp.dst)
        if not src_layer or not dst_layer:
            continue
        if src_layer == dst_layer:
            continue
        allowed = ALLOWED_LAYER_IMPORTS.get(src_layer, set())
        if dst_layer not in allowed:
            violations.append(
                f"Forbidden cross-layer import: {imp.src}.py:{imp.lineno} imports {imp.dst} "
                f"({src_layer} -> {dst_layer})"
            )

    # Direct host imports
    for imp in sorted(external_imports, key=lambda x: (x.src, x.lineno, x.target)):
        if imp.src not in ALLOWED_DIRECT_AGENT_LOGIC_IMPORTS:
            violations.append(
                f"Forbidden host import: {imp.src}.py:{imp.lineno} imports {imp.target}; "
                f"use internal ports instead"
            )

    if violations:
        print("ARCHITECTURE CHECK FAILED")
        for idx, message in enumerate(violations, start=1):
            print(f"{idx}. {message}")
        return 1

    print("ARCHITECTURE CHECK PASSED")
    print(f"Modules checked: {len(modules)}")
    print(f"Internal import edges: {len(internal_imports)}")
    print("No cycles, no forbidden cross-layer imports, no forbidden host imports.")
    return 0


def main() -> int:
    return run_checks()


if __name__ == "__main__":
    raise SystemExit(main())
