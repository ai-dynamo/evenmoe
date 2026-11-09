# SPDX-License-Identifier: Apache-2.0
"""Installer preflight must reject all incompatible inputs before writing."""

import hashlib
import json

import pytest

from examples.megamoe import install_overlay as installer


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root, site = tmp_path / "example", tmp_path / "site"
    (root / "runtime").mkdir(parents=True)
    target = site / "sglang/srt/layers/moe/base.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"value = 1\n")
    (root / "runtime/adapter.py").write_bytes(b"enabled = True\n")
    patch = b"--- a/sglang/srt/layers/moe/base.py\n+++ b/sglang/srt/layers/moe/base.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    (root / "change.patch").write_bytes(patch)
    (root / "dependencies.lock.json").write_text(
        json.dumps(
            {
                "patches": [
                    {
                        "target": "sglang/srt/layers/moe/base.py",
                        "file": "change.patch",
                        "base_sha256": sha(b"value = 1\n"),
                        "result_sha256": sha(b"value = 2\n"),
                    }
                ]
            }
        )
    )
    (root / "sources.sha256.json").write_text(
        json.dumps(
            {
                p.relative_to(root).as_posix(): sha(p.read_bytes())
                for p in root.rglob("*")
                if p.is_file()
            }
        )
    )
    monkeypatch.setattr(installer, "ROOT", root)
    return root, site, target


def test_dry_run_and_idempotent_apply(fixture):
    _, site, target = fixture
    assert installer.install(site)["status"] == "verified_only"
    assert target.read_bytes() == b"value = 1\n"
    assert not (target.parent / "adapter.py").exists()
    assert installer.install(site, True)["files"] == 2
    assert target.read_bytes() == b"value = 2\n"
    assert installer.install(site, True)["files"] == 0


def test_unknown_base_is_unchanged(fixture):
    _, site, target = fixture
    target.write_bytes(b"unknown = True\n")
    with pytest.raises(RuntimeError, match="Unsupported base"):
        installer.install(site, True)
    assert target.read_bytes() == b"unknown = True\n"
    assert not (target.parent / "adapter.py").exists()


def test_conflicting_overlay_prevents_patch_write(fixture):
    _, site, target = fixture
    (target.parent / "adapter.py").write_bytes(b"existing = True\n")
    with pytest.raises(RuntimeError, match="Conflicting overlay"):
        installer.install(site, True)
    assert target.read_bytes() == b"value = 1\n"


def test_changed_sources_are_rejected(fixture):
    root, site, target = fixture
    (root / "runtime/adapter.py").write_bytes(b"changed = True\n")
    with pytest.raises(RuntimeError, match="Example source changed"):
        installer.install(site, True)
    assert target.read_bytes() == b"value = 1\n"
