#!/usr/bin/env python3
"""Stdlib-only test runner for environments without pytest installed.


Usage
-----
    python scripts/run_tests.py [optional/path/to/test_file.py ...]

With no arguments, discovers every ``tests/unit/test_*.py`` and
``tests/integration/test_*.py`` file.

Design decisions
-----------------
* A minimal ``pytest.raises`` shim is provided so test files can use the
  familiar ``with pytest.raises(SomeError):`` idiom without importing the
  real pytest package.
* Each test function runs in isolation; a failure in one does not stop the
  others (matches pytest's default behavior), and a summary is printed at
  the end.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"


class _RaisesContext:
    """Minimal stand-in for ``pytest.raises`` used by test modules."""

    def __init__(self, expected_exception: type[BaseException] | tuple) -> None:
        self.expected_exception = expected_exception
        self.value: BaseException | None = None  # matches real pytest's ExceptionInfo.value

    def __enter__(self) -> "_RaisesContext":
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> bool:
        if exc_type is None:
            raise AssertionError(
                f"Expected {self.expected_exception!r} to be raised, but no "
                "exception was raised."
            )
        if not issubclass(exc_type, self.expected_exception):
            return False  # re-raise: wrong exception type
        self.value = exc_value
        return True  # suppress the expected exception


def _parametrize(argnames, argvalues):  # noqa: ANN001 - shim signature
    """Minimal stand-in for ``pytest.mark.parametrize``.

    Stores the parametrization spec as an attribute on the decorated
    function (``__parametrize__``) instead of actually generating separate
    test objects (which requires pytest's collection machinery). The
    runner's ``main()`` reads this attribute and expands the call itself.
    """
    names = [n.strip() for n in argnames.split(",")] if isinstance(argnames, str) else list(argnames)

    def decorator(fn):  # noqa: ANN001, ANN202 - shim signature
        fn.__parametrize__ = (names, argvalues)
        return fn

    return decorator


class _FakeMark:
    """Minimal stand-in for ``pytest.mark`` exposing only ``parametrize``."""

    parametrize = staticmethod(_parametrize)


def _fake_pytest_module() -> types.ModuleType:
    """Build a fake 'pytest' module exposing just what our tests need."""
    fake = types.ModuleType("pytest")

    def raises(expected_exception):  # noqa: ANN001 - shim signature
        return _RaisesContext(expected_exception)

    fake.raises = raises  # type: ignore[attr-defined]
    fake.mark = _FakeMark()  # type: ignore[attr-defined]
    return fake


def _install_fake_pytest_if_absent() -> None:
    try:
        import pytest  # noqa: F401

        return  # real pytest is available; do not shadow it
    except ImportError:
        sys.modules["pytest"] = _fake_pytest_module()


def _discover_test_files() -> list[Path]:
    if len(sys.argv) > 1:
        return [Path(p).resolve() for p in sys.argv[1:]]
    return sorted(TESTS_DIR.glob("unit/test_*.py")) + sorted(
        TESTS_DIR.glob("integration/test_*.py")
    )


def _load_module(path: Path) -> types.ModuleType:
    module_name = f"_manual_test_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Could not load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    sys.path.insert(0, str(SRC_DIR))
    _install_fake_pytest_if_absent()

    test_files = _discover_test_files()
    if not test_files:
        print("No test files discovered.")
        return 1

    total_passed = 0
    total_failed = 0
    failures: list[tuple[str, str]] = []

    for path in test_files:
        print(f"\n=== {path.relative_to(PROJECT_ROOT)} ===")
        try:
            module = _load_module(path)
        except Exception:  # noqa: BLE001 - report and continue
            print(f"  [IMPORT ERROR] {path}")
            traceback.print_exc()
            total_failed += 1
            continue

        test_functions = [
            getattr(module, name)
            for name in dir(module)
            if name.startswith("test_") and callable(getattr(module, name))
        ]
        for fn in test_functions:
            spec = getattr(fn, "__parametrize__", None)
            if spec is None:
                calls = [((), {})]
                labels = [fn.__name__]
            else:
                names, argvalues = spec
                calls = []
                labels = []
                for values in argvalues:
                    values_tuple = values if isinstance(values, tuple) else (values,)
                    calls.append(((), dict(zip(names, values_tuple))))
                    labels.append(f"{fn.__name__}[{values_tuple}]")

            for (args, kwargs), label in zip(calls, labels):
                qualified_name = f"{path.stem}::{label}"
                try:
                    fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - collect and continue
                    total_failed += 1
                    failures.append((qualified_name, f"{type(exc).__name__}: {exc}"))
                    print(f"  FAIL  {label}  ({type(exc).__name__}: {exc})")
                else:
                    total_passed += 1
                    print(f"  PASS  {label}")

    print("\n" + "=" * 60)
    print(f"Results: {total_passed} passed, {total_failed} failed")
    if failures:
        print("\nFailures:")
        for name, reason in failures:
            print(f"  - {name}: {reason}")
    print("=" * 60)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
