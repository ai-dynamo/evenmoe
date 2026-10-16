# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import pathlib

import pytest

from evenmoe.kernels.jit.utils import (
    INCLUDE_PATH,
    _find_package_root,
    get_mathdx_root,
)


@pytest.mark.unit
def test_find_package_root_handles_regular_packages():
    root = _find_package_root("evenmoe")
    assert root is not None
    assert (root / "kernels").is_dir()


@pytest.mark.unit
def test_shared_jit_headers_ship_next_to_the_loader():
    """``load_jit`` puts ``INCLUDE_PATH`` on every build; the ``evenmoe_jit``
    headers every kernel package includes must be there."""
    for name in ("tensor.h", "utils.h", "utils.cuh", "source_location.h"):
        assert (INCLUDE_PATH / "evenmoe_jit" / name).is_file(), name


@pytest.mark.unit
def test_find_package_root_handles_namespace_packages():
    """A namespace package has no ``__init__.py`` and therefore no
    ``spec.origin``; its directory only appears in
    ``submodule_search_locations``.

    ``nvidia.mathdx`` -- the package that carries the Math-DX headers the LPLB
    kernels compile against -- is distributed this way, so resolving only via
    ``origin`` reports a pip-installed ``nvidia-mathdx`` as absent.
    """
    spec = importlib.util.find_spec("evenmoe")
    assert spec is not None and spec.origin is not None, (
        "precondition: this test compares against a regular package"
    )

    namespace = _namespace_package_candidate()
    if namespace is None:
        pytest.skip("no namespace package available in this environment")
    name, expected = namespace
    resolved = _find_package_root(name)
    assert resolved == expected


@pytest.mark.unit
def test_find_package_root_returns_none_for_missing_package():
    assert _find_package_root("evenmoe_does_not_exist") is None
    # A missing parent makes find_spec raise rather than return None.
    assert _find_package_root("no_such_parent.child") is None


@pytest.mark.unit
def test_get_mathdx_root_prefers_env_home(tmp_path, monkeypatch):
    (tmp_path / "include").mkdir()
    monkeypatch.setenv("MATHDX_HOME", str(tmp_path))
    assert get_mathdx_root() == tmp_path.resolve()


@pytest.mark.unit
def test_get_mathdx_root_ignores_env_home_without_headers(tmp_path, monkeypatch):
    monkeypatch.setenv("MATHDX_HOME", str(tmp_path))
    # Falls through to the package/bundled search instead of returning a root
    # whose headers do not exist.
    assert get_mathdx_root() != tmp_path.resolve()


def _namespace_package_candidate() -> tuple[str, pathlib.Path] | None:
    """Find any installed namespace package to exercise the fallback with."""
    for name in ("nvidia.mathdx", "nvidia"):
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ModuleNotFoundError, ValueError):
            continue
        if spec is None or spec.origin is not None:
            continue
        for location in spec.submodule_search_locations or ():
            candidate = pathlib.Path(location).resolve()
            if candidate.exists():
                return name, candidate
    return None
