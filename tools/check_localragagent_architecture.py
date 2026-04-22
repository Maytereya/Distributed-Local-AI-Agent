#!/usr/bin/env python3
"""Architecture fitness checks for `src/localragagent`."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


PACKAGE_NAME = "localragagent"
PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / PACKAGE_NAME

LAYER_BY_PREFIX: dict[str, str] = {
    "interfaces": "interface",
    "orchestration": "orchestration",
    "freetalk": "orchestration",
    "domain": "domain",
    "infrastructure": "infrastructure",
    "ports": "port",
}

ALLOWED_LAYER_IMPORTS: dict[str, set[str]] = {
    "interface": {"orchestration", "domain", "infrastructure", "port"},
    "orchestration": {"domain", "infrastructure", "port"},
    "domain": set(),
    "infrastructure": {"domain", "port"},
    "port": set(),
}

HOST_PREFIXES: tuple[str, ...] = (
    "agent_api",
    "gradio_interface",
    "messenger_simulator",
    "schedule_ttl_cache",
    "agent_logic_1",
    "agent_logic_2",
    "messengers_router",
    "converters",
    "container_managenment",
    "openclaw",
    "VOSK",
    "whisper",
)


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
    for path in PACKAGE_DIR.rglob("*.py"):
        rel = path.relative_to(PACKAGE_DIR)
        if "__pycache__" in rel.parts:
            continue
        if rel.name == "__init__.py":
            continue
        modules[".".join(rel.with_suffix("").parts)] = path
    return modules


def module_layer(module_name: str) -> str | None:
    top = module_name.split(".")[0]
    return LAYER_BY_PREFIX.get(top)


def _is_host_import(target: str) -> bool:
    return any(target == p or target.startswith(f"{p}.") for p in HOST_PREFIXES)


def _resolve_relative_import(module_name: str, node: ast.ImportFrom, module_names: set[str]) -> list[str]:
    if node.level <= 0:
        return []

    src_parts = module_name.split(".")
    base_parts = src_parts[:-node.level]

    if node.module:
        target_parts = base_parts + node.module.split(".")
        candidate = ".".join(target_parts)
        return [candidate] if candidate in module_names and candidate != module_name else []

    resolved: list[str] = []
    for alias in node.names:
        candidate = ".".join(base_parts + [alias.name.split(".")[0]])
        if candidate in module_names and candidate != module_name:
            resolved.append(candidate)
    return resolved


def _resolve_absolute_internal_import(module_name: str, node: ast.ImportFrom, module_names: set[str]) -> list[str]:
    if node.level != 0 or not node.module:
        return []

    resolved: list[str] = []

    if node.module == PACKAGE_NAME:
        for alias in node.names:
            candidate = alias.name.split(".")[0]
            if candidate in module_names and candidate != module_name:
                resolved.append(candidate)
        return resolved

    pkg_prefix = f"{PACKAGE_NAME}."
    if not node.module.startswith(pkg_prefix):
        return []

    candidate = node.module[len(pkg_prefix):]
    if candidate in module_names and candidate != module_name:
        resolved.append(candidate)

    for alias in node.names:
        nested = f"{candidate}.{alias.name.split('.')[0]}"
        if nested in module_names and nested != module_name:
            resolved.append(nested)
    return resolved


def parse_imports(module_name: str, path: Path, module_names: set[str]) -> tuple[list[InternalImport], list[ExternalImport]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    internal: list[InternalImport] = []
    external: list[ExternalImport] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for target in _resolve_relative_import(module_name, node, module_names):
                internal.append(InternalImport(module_name, target, node.lineno))
            for target in _resolve_absolute_internal_import(module_name, node, module_names):
                internal.append(InternalImport(module_name, target, node.lineno))
            if node.level == 0 and node.module and _is_host_import(node.module):
                external.append(ExternalImport(module_name, node.module, node.lineno))
            continue

        if isinstance(node, ast.Import):
            for alias in node.names:
                target = alias.name
                pkg_prefix = f"{PACKAGE_NAME}."
                if target.startswith(pkg_prefix):
                    candidate = target[len(pkg_prefix):]
                    if candidate in module_names and candidate != module_name:
                        internal.append(InternalImport(module_name, candidate, node.lineno))
                if _is_host_import(target):
                    external.append(ExternalImport(module_name, target, node.lineno))

    return unique_internal_imports(internal), external


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
    unknown = sorted(module for module in modules if module_layer(module) is None)
    if unknown:
        errors.append(
            "Modules outside configured layers: " + ", ".join(unknown)
        )
    return errors


def run_checks() -> int:
    if not PACKAGE_DIR.exists():
        print(f"ARCHITECTURE CHECK FAILED\n1. Package dir not found: {PACKAGE_DIR}")
        return 1

    modules = discover_modules()
    module_names = set(modules)

    violations: list[str] = []
    violations.extend(check_layer_map(modules))

    internal_imports: list[InternalImport] = []
    external_imports: list[ExternalImport] = []
    for module_name, path in sorted(modules.items()):
        internal, external = parse_imports(module_name, path, module_names)
        internal_imports.extend(internal)
        external_imports.extend(external)

    internal_imports = unique_internal_imports(internal_imports)

    graph: dict[str, set[str]] = {module_name: set() for module_name in modules}
    for imp in internal_imports:
        graph[imp.src].add(imp.dst)

    for cycle in find_cycles(graph):
        violations.append(f"Import cycle detected: {' -> '.join(cycle)}")

    for imp in internal_imports:
        src_layer = module_layer(imp.src)
        dst_layer = module_layer(imp.dst)
        if not src_layer or not dst_layer or src_layer == dst_layer:
            continue
        allowed = ALLOWED_LAYER_IMPORTS.get(src_layer, set())
        if dst_layer not in allowed:
            violations.append(
                f"Forbidden cross-layer import: {imp.src}.py:{imp.lineno} imports {imp.dst} "
                f"({src_layer} -> {dst_layer})"
            )

    for imp in sorted(external_imports, key=lambda x: (x.src, x.lineno, x.target)):
        src_layer = module_layer(imp.src)
        if src_layer != "port":
            violations.append(
                f"Forbidden host import: {imp.src}.py:{imp.lineno} imports {imp.target}; "
                f"host imports are allowed only in ports/*"
            )

    if violations:
        print("LOCALRAGAGENT ARCHITECTURE CHECK FAILED")
        for idx, message in enumerate(violations, start=1):
            print(f"{idx}. {message}")
        return 1

    print("LOCALRAGAGENT ARCHITECTURE CHECK PASSED")
    print(f"Modules checked: {len(modules)}")
    print(f"Internal import edges: {len(internal_imports)}")
    print("No cycles, no forbidden cross-layer imports, no forbidden host imports.")
    return 0


def main() -> int:
    return run_checks()


if __name__ == "__main__":
    raise SystemExit(main())
