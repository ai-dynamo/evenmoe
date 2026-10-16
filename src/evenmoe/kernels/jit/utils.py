# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Shared JIT-kernel build helpers.

This module hosts the small set of helpers SDK CUDA kernels need to
compile through ``tvm-ffi``:

- ``cache_once``: simple memoization decorator (compatible with ``torch.compile``).
- ``make_cpp_args`` + ``CPPArgList``: C++ template argument formatting.
- ``ArchInfo`` + ``get_jit_cuda_arch`` + ``override_jit_cuda_arch``: CUDA arch
  detection.
- ``load_jit``: thin wrapper around ``tvm_ffi.cpp.load`` / ``load_inline``
  that injects the SDK's default include / cflag / target-arch setup;
  accepts a ``source_root`` so each kernel subpackage can host its own
  ``csrc/``.
- ``register_dependency`` + ``make_mathdx_include_paths_resolver``:
  declare build-time include-path dependencies (e.g. Math-DX) that
  resolve at JIT compile time rather than at module import.

``include/evenmoe_jit/`` next to this module holds the C++ helper headers
adapted from SGLang's ``jit_kernel`` (``TensorMatcher`` validation, the
stream-aware ``LaunchKernel`` wrapper, error reporting); ``load_jit`` puts
it on every build's include path, so a kernel just ``#include
<evenmoe_jit/tensor.h>``.

Nothing here imports any inference framework. Kernel subpackages
(:mod:`evenmoe.kernels.<algo>`) ship their own ``.cuh`` /
``.hpp`` files and pass ``source_root=Path(__file__).parent`` to
``load_jit``.
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
import pathlib
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAlias,
    TypeVar,
)

import torch

if TYPE_CHECKING:
    from tvm_ffi import Module

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def cache_once(fn: F) -> F:
    """Memoize a function once per (args, kwargs) tuple.

    ``functools.lru_cache`` is incompatible with ``torch.compile``; this
    plain dict-backed cache is a torch.compile-safe alternative.
    """
    result_map: dict[tuple[Any, ...], Any] = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in result_map:
            result_map[key] = fn(*args, **kwargs)
        return result_map[key]

    return wrapper  # type: ignore[return-value]


def _make_wrapper(tup: tuple[str, str]) -> str:
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"


@cache_once
def _resolve_kernel_path() -> pathlib.Path:
    """Locate this module's directory; default search root for ``load_jit``.

    Kernel subpackages typically ship their own ``csrc/`` directory and
    pass ``source_root=Path(__file__).parent`` to ``load_jit``; this
    default exists so callers that don't override it land in the same
    package as ``utils.py``. Existence of any ``csrc/`` beneath it is
    not validated here — ``load_jit`` surfaces the missing-file error
    at compile time.
    """
    return pathlib.Path(__file__).parent.resolve()


KERNEL_PATH = _resolve_kernel_path()
# The shared ``evenmoe_jit/*`` headers; on the include path of every build.
INCLUDE_PATH = KERNEL_PATH / "include"
DEFAULT_CFLAGS = ["-std=c++20", "-O3"]
DEFAULT_LDFLAGS: list[str] = []
CPP_TEMPLATE_TYPE: TypeAlias = int | float | str | bool | torch.dtype


class CPPArgList(list[str]):
    def __str__(self) -> str:
        return ", ".join(self)


CPP_DTYPE_MAP = {
    torch.float: "fp32_t",
    torch.float16: "fp16_t",
    torch.float8_e4m3fn: "fp8_e4m3_t",
    torch.bfloat16: "bf16_t",
    torch.int8: "int8_t",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
}


@cache_once
def is_hip_runtime() -> bool:
    return bool(torch.version.hip)


def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CPPArgList:
    """Render Python scalars into a C++ template argument list.

    Used to key JIT-compiled kernels by their template parameters.
    """

    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, str, float)):
            return str(arg)
        if isinstance(arg, torch.dtype):
            return CPP_DTYPE_MAP[arg]
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")

    return CPPArgList(_convert(arg) for arg in args)


@dataclass
class ArchInfo:
    major: int
    minor: int
    suffix: str

    @property
    def target_name(self) -> str:
        return f"{self.major}.{self.minor}{self.suffix}"

    @property
    def jit_flag(self) -> str:
        # Vendored ``.cuh`` files key conditional code paths on this define
        # (e.g. SM-version-specific shmem budgets); keep the name stable.
        return f"-DSGL_CUDA_ARCH={self.major * 100 + self.minor * 10}"


_CUDA_ARCH: ArchInfo | None = None


@cache_once
def _init_jit_cuda_arch_once() -> None:
    global _CUDA_ARCH
    try:
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
    # Whatever torch raises here (no device, CUDA-less build, driver error)
    # means the same thing: no architecture to target.
    except Exception:  # noqa: BLE001
        logger.warning("Cannot detect CUDA architecture.")
        major, minor = 0, 0  # invalid value to trigger compile error if used
    _CUDA_ARCH = ArchInfo(major, minor, "")


def get_jit_cuda_arch() -> ArchInfo:
    """Return the current CUDA architecture info."""
    _init_jit_cuda_arch_once()
    assert _CUDA_ARCH is not None
    return _CUDA_ARCH


@contextmanager
def override_jit_cuda_arch(major: int, minor: int, suffix: str = ""):
    """Temporarily override the detected CUDA architecture (mostly for tests)."""
    global _CUDA_ARCH
    old_value = get_jit_cuda_arch()
    _CUDA_ARCH = ArchInfo(major, minor, suffix)
    try:
        yield
    finally:
        _CUDA_ARCH = old_value


@contextmanager
def _jit_compile_context():
    if is_hip_runtime():
        yield  # TODO: ROCm `TVM_FFI_ROCM_ARCH_LIST` support if needed.
        return
    env_key = "TVM_FFI_CUDA_ARCH_LIST"
    old_value = os.environ.get(env_key, None)
    os.environ[env_key] = get_jit_cuda_arch().target_name
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = old_value


def _get_default_target_flags() -> list[str]:
    if is_hip_runtime():
        return ["-DUSE_ROCM", "-std=c++20", "-O3"]
    return [
        get_jit_cuda_arch().jit_flag,
        "-std=c++20",
        "-O3",
        "--expt-relaxed-constexpr",
    ]


def load_jit(
    *args: str,
    cpp_files: list[str] | None = None,
    cuda_files: list[str] | None = None,
    cpp_wrappers: list[tuple[str, str]] | None = None,
    cuda_wrappers: list[tuple[str, str]] | None = None,
    extra_cflags: list[str] | None = None,
    extra_cuda_cflags: list[str] | None = None,
    extra_ldflags: list[str] | None = None,
    extra_include_paths: list[str] | None = None,
    extra_dependencies: list[str] | None = None,
    build_directory: str | None = None,
    header_only: bool = True,
    source_root: pathlib.Path | None = None,
) -> Module:
    """Load a JIT module from C++/CUDA source files.

    By default, ``cpp_files`` and ``cuda_files`` are resolved against
    ``KERNEL_PATH / "csrc" / <file>`` (the SDK's top-level kernel sources).
    Kernel subpackages that ship their own ``csrc/`` directory (e.g.
    ``kernels/lplb/csrc/``) pass ``source_root=<package dir>`` so
    resolution happens against ``source_root / "csrc" / <file>``
    instead. ``INCLUDE_PATH`` (the shared ``evenmoe_jit/*`` headers) is
    always on the include path; ``extra_include_paths`` adds to it.
    """

    from tvm_ffi.cpp import load, load_inline

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    # A fresh list: the dependency paths appended below must not leak into
    # the caller's.
    extra_include_paths = [str(INCLUDE_PATH), *(extra_include_paths or [])]

    root = source_root.resolve() if source_root is not None else KERNEL_PATH
    cpp_files = [str((root / "csrc" / f).resolve()) for f in cpp_files]
    cuda_files = [str((root / "csrc" / f).resolve()) for f in cuda_files]

    for dep in set(extra_dependencies or []):
        if dep not in _REGISTERED_DEPENDENCIES:
            raise ValueError(f"Dependency {dep!r} is not registered.")
        extra_include_paths += _REGISTERED_DEPENDENCIES[dep]()

    module_name = "evenmoe_jit_" + "_".join(str(arg) for arg in args)
    if header_only:
        cpp_wrappers = cpp_wrappers or []
        cuda_wrappers = cuda_wrappers or []
        cpp_sources = [f'#include "{path}"' for path in cpp_files]
        cpp_sources += [_make_wrapper(tup) for tup in cpp_wrappers]

        cuda_sources = [f'#include "{path}"' for path in cuda_files]
        cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]
        with _jit_compile_context():
            return load_inline(
                module_name,
                cpp_sources=cpp_sources,
                cuda_sources=cuda_sources,
                extra_cflags=DEFAULT_CFLAGS + extra_cflags,
                extra_cuda_cflags=_get_default_target_flags() + extra_cuda_cflags,
                extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
                extra_include_paths=extra_include_paths,
                build_directory=build_directory,
            )
    assert cpp_wrappers is None and cuda_wrappers is None
    with _jit_compile_context():
        return load(
            module_name,
            cpp_files=cpp_files,
            cuda_files=cuda_files,
            extra_cflags=DEFAULT_CFLAGS + extra_cflags,
            extra_cuda_cflags=_get_default_target_flags() + extra_cuda_cflags,
            extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
            extra_include_paths=extra_include_paths,
            build_directory=build_directory,
        )


def _find_package_root(package: str) -> pathlib.Path | None:
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ModuleNotFoundError, ValueError):
        # find_spec raises (not returns None) when a parent package along
        # the dotted path is missing — e.g. find_spec("nvidia.mathdx") with
        # no "nvidia" namespace package installed. Treat any such failure
        # as "package absent" so dependency resolvers can give a clean
        # descriptive RuntimeError downstream.
        return None
    if spec is None:
        return None
    if spec.origin is not None:
        return pathlib.Path(spec.origin).resolve().parent
    # Namespace packages (no __init__.py) have no origin, but do carry their
    # directory in submodule_search_locations. `nvidia.mathdx` ships this way,
    # so without this branch a pip-installed `nvidia-mathdx` looks absent and
    # the caller reports "Cannot find NVIDIA Math-DX headers" while the headers
    # are sitting in site-packages.
    for location in spec.submodule_search_locations or ():
        candidate = pathlib.Path(location).resolve()
        if candidate.exists():
            return candidate
    return None


_REGISTERED_DEPENDENCIES: dict[str, Callable[[], list[str]]] = {}


def register_dependency(name: str):
    """Register a build-time include-path resolver under ``name``.

    ``load_jit(..., extra_dependencies=[name])`` looks up the resolver and
    appends the returned paths to the kernel's include list. Resolvers must
    not import the framework module they describe; they should only consult
    ``importlib.util.find_spec`` and the file system.
    """

    def decorator(f: Callable[[], list[str]]) -> Callable[[], list[str]]:
        if name in _REGISTERED_DEPENDENCIES:
            raise ValueError(f"Dependency {name!r} already registered")
        _REGISTERED_DEPENDENCIES[name] = f
        return f

    return decorator


def get_mathdx_root(*extra_bundled_paths: pathlib.Path) -> pathlib.Path | None:
    """Locate the NVIDIA Math-DX install (cuBLASDx + cuSolverDx).

    Searches, in order:
      1. ``$MATHDX_HOME`` env var (extracted Math-DX archive root).
      2. The ``nvidia-mathdx`` PyPI package, if installed.
      3. Any caller-supplied ``extra_bundled_paths``.
    """
    env_home = os.environ.get("MATHDX_HOME")
    if env_home:
        candidate = pathlib.Path(env_home).expanduser().resolve()
        if (candidate / "include").exists():
            return candidate

    pkg = _find_package_root("nvidia.mathdx")
    if pkg is not None and (pkg / "include").exists():
        return pkg

    for bundled in extra_bundled_paths:
        if (bundled / "include").exists():
            return bundled.resolve()

    return None


def make_mathdx_include_paths_resolver(
    *bundled_search_paths: pathlib.Path,
    bundled_hint: str | None = None,
) -> Callable[[], list[str]]:
    """Build an include-path resolver suitable for ``@register_dependency``.

    Adapters call this with their own bundled-mathdx location(s) and
    register the result. Keeps Math-DX lookup centralized in
    ``kernels.jit.utils`` while letting each kernel package point at its own
    ``resources/mathdx`` (rather than a hard-coded SDK-relative path).
    """

    def _resolver() -> list[str]:
        root = get_mathdx_root(*bundled_search_paths)
        if root is None:
            raise RuntimeError(
                "Cannot find NVIDIA Math-DX (cuBLASDx + cuSolverDx) headers. "
                "Set MATHDX_HOME to the extracted Math-DX archive or install "
                "the `nvidia-mathdx` package."
                + (f" Alternatively, run {bundled_hint}." if bundled_hint else "")
            )
        candidates = [root / "include"]
        cutlass = root / "external" / "cutlass" / "include"
        if cutlass.exists():
            candidates.append(cutlass)
        return [str(p) for p in candidates]

    return _resolver


__all__ = [
    "CPP_DTYPE_MAP",
    "DEFAULT_CFLAGS",
    "DEFAULT_LDFLAGS",
    "INCLUDE_PATH",
    "KERNEL_PATH",
    "ArchInfo",
    "CPPArgList",
    "cache_once",
    "get_jit_cuda_arch",
    "get_mathdx_root",
    "is_hip_runtime",
    "load_jit",
    "make_cpp_args",
    "make_mathdx_include_paths_resolver",
    "override_jit_cuda_arch",
    "register_dependency",
]
