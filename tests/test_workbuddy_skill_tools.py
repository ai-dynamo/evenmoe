# SPDX-License-Identifier: Apache-2.0
"""Keep the portable skill's CPU regression checks in the repository test gate."""
import subprocess
import sys
from pathlib import Path


def test_workbuddy_skill_tools():
    script = (Path(__file__).resolve().parents[1] / '.agents' / 'skills'
              / 'workbuddy-moe-benchmark' / 'scripts' / 'test_tools.py')
    subprocess.run([sys.executable, str(script)], check=True, capture_output=True, text=True)
