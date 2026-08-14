"""Enforce the Runtime lower-layer dependency direction."""

from __future__ import annotations

import ast
from pathlib import Path


RUNTIME_ROOT = Path(__file__).parents[2] / "src" / "matrix" / "runtime"
LAYER_ORDER = {"domain": 0, "ports": 1, "core": 2}
FORBIDDEN_PREFIXES = (
    "matrix.chat",
    "matrix.server",
    "matrix.orchestration",
    "matrix.agent",
    "langgraph",
    "fastapi",
)


def _module_name(path: Path) -> str:
    relative = path.relative_to(RUNTIME_ROOT.parent.parent.parent / "src")
    return ".".join(relative.with_suffix("").parts)


def _resolve_import(current: Path, module: str | None, level: int) -> str:
    current_module = _module_name(current)
    current_package = current_module.split(".")[:-1]
    if level:
        base = current_package[: len(current_package) - level + 1]
        parts = base + (module.split(".") if module else [])
        return ".".join(parts)
    return module or ""


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(_resolve_import(path, node.module, node.level))
    return modules


def test_runtime_lower_layers_have_no_upward_or_application_imports() -> None:
    for layer, layer_number in LAYER_ORDER.items():
        layer_root = RUNTIME_ROOT / layer
        for path in layer_root.rglob("*.py"):
            for imported in _imports(path):
                assert not imported.startswith(FORBIDDEN_PREFIXES), (
                    f"{path} imports forbidden application module {imported}"
                )
                if imported.startswith("matrix.runtime."):
                    imported_layer = imported.split(".")[2]
                    if imported_layer in LAYER_ORDER:
                        assert LAYER_ORDER[imported_layer] <= layer_number, (
                            f"{path} imports upward Runtime layer {imported}"
                        )
