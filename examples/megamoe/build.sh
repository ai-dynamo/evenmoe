#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
docker build --file "$repo_root/examples/megamoe/Dockerfile" --tag "${1:-evenmoe-megamoe:dev}" "$repo_root"
