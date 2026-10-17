#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Usage: check_signoff.sh <commit-msg-file>   (pre-commit commit-msg hook)
#        check_signoff.sh --range <base> <head> (CI: every non-merge commit in base..head)
set -euo pipefail

has_signoff() { grep -q -E '^Signed-off-by: [^<[:space:]][^<]* <[^>[:space:]]+@[^>[:space:]]+>[[:space:]]*$' "$1"; }

if [ "${1:-}" != "--range" ]; then
    has_signoff "$1" || { echo "missing Signed-off-by: Name <email>, use git commit -s"; exit 1; }
    exit 0
fi

base="$2"; head="$3"
if [ -z "$base" ] || [ "$base" = "0000000000000000000000000000000000000000" ]; then
    range=(-1 "$head")
else
    range=("$base..$head")
fi
commits=$(git rev-list --no-merges "${range[@]}") || { echo "cannot resolve commit range ${range[*]}"; exit 1; }
msg=$(mktemp); trap 'rm -f "$msg"' EXIT
status=0
for c in $commits; do
    git log -1 --format=%B "$c" > "$msg"
    has_signoff "$msg" || { echo "missing Signed-off-by: $(git log -1 --format='%h %s' "$c")"; status=1; }
done
exit $status
