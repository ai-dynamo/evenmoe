# SPDX-License-Identifier: Apache-2.0
"""Verify (default) or install the example in an isolated, pinned runtime."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sysconfig
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(site, apply=False):
    lock = json.loads((ROOT / "dependencies.lock.json").read_text())
    manifest = json.loads((ROOT / "sources.sha256.json").read_text())
    for name, expected in manifest.items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f"Example source changed: {name}; update the reviewed manifest")
    pending = {}
    # Check all inputs before writing any installed file. No fuzzy patching of
    # a different SGLang/FlashInfer version is permitted.
    with tempfile.TemporaryDirectory() as directory:
        stage = Path(directory)
        for item in lock["patches"]:
            target = site / item["target"]
            actual = digest(target)
            if actual == item["result_sha256"]:
                continue
            if actual != item["base_sha256"]:
                raise RuntimeError(f"Unsupported base: {target}")
            copy = stage / item["target"]
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, copy)
            subprocess.run(
                [
                    "patch",
                    "--batch",
                    "--forward",
                    "--fuzz=0",
                    "-p1",
                    "-d",
                    str(stage),
                    "-i",
                    str(ROOT / item["file"]),
                ],
                check=True,
                capture_output=True,
            )
            if digest(copy) != item["result_sha256"]:
                raise RuntimeError(f"Unexpected patched content: {target}")
            pending[target] = copy.read_bytes()
        package = site / "sglang/srt/layers/moe"
        if not package.is_dir():
            raise RuntimeError("SGLang MoE package missing")
        for source in sorted((ROOT / "runtime").iterdir()):
            if source.suffix not in (".py", ".cu"):
                continue
            target = package / source.name
            if target.exists():
                if digest(target) != digest(source):
                    raise RuntimeError(f"Conflicting overlay: {target}; use a fresh image")
            else:
                pending[target] = source.read_bytes()
        if apply:
            for target, data in pending.items():
                target.write_bytes(data)
    return {"status": "applied" if apply else "verified_only", "files": len(pending)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-packages", type=Path, default=Path(sysconfig.get_paths()["purelib"])
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(install(args.site_packages, args.apply)))


if __name__ == "__main__":
    main()
