"""The wheel's only-include list covers every local module the packaged code imports."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _only_include() -> set[str]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return set(pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["only-include"])


def _local_imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return {f"{n}.py" for n in names if (ROOT / f"{n}.py").exists()}


def test_wheel_includes_local_imports():
    included = _only_include()
    imported = set().union(*(_local_imports(ROOT / f) for f in included if f.endswith(".py")))

    assert imported <= included


def test_wheel_includes_mid_table():
    assert "MID_Lkp.xlsx" in _only_include()
