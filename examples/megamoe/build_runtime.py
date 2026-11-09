# SPDX-License-Identifier: Apache-2.0
"""Install pinned external sources inside the example's disposable build image."""

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


def main():
    lock = json.loads((ROOT / "dependencies.lock.json").read_text())
    dg = lock["deepgemm"]
    if importlib.metadata.version(dg["distribution"]) != dg["version"]:
        raise RuntimeError("DeepGEMM version differs from the pinned base image")
    source_root = Path("/opt/megamoe-sources")
    source_root.mkdir(exist_ok=True)
    for name in ("sglang", "flashinfer"):
        dep = lock[name]
        dest = source_root / name
        run("git", "init", str(dest))
        run("git", "-C", str(dest), "remote", "add", "origin", dep["url"])
        run("git", "-C", str(dest), "fetch", "--depth=1", "origin", dep["commit"])
        run("git", "-C", str(dest), "checkout", "--detach", "FETCH_HEAD")
        actual = subprocess.check_output(
            ["git", "-C", str(dest), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != dep["commit"]:
            raise RuntimeError(f"Wrong {name} revision")
        if name == "flashinfer":
            run("git", "-C", str(dest), "submodule", "update", "--init", "--recursive", "--depth=1")
        package = dest / "python" if name == "sglang" else dest
        env = dict(os.environ)
        env.update(
            SGLANG_BUILD_RUST_EXTS="all",
            CARGO_BUILD_JOBS="8",
            MAX_JOBS="8",
            FLASHINFER_CUDA_ARCH_LIST="9.0a",
            SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="0.0.0.dev1+g" + lock["sglang"]["commit"][:9],
        )
        run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--force-reinstall",
            str(package),
            env=env,
        )
    run(sys.executable, str(ROOT / "install_overlay.py"), "--apply")


if __name__ == "__main__":
    main()
