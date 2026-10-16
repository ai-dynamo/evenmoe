#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Fails any file whose first 15 lines carry no SPDX-License-Identifier, or a license term outside
# SPDX_ALLOWED. A "<id> WITH <exc>" term passes only when SPDX_ALLOWED lists "<id>_WITH_<exc>".
set -euo pipefail

allowed="${SPDX_ALLOWED:-Apache-2.0}"
is_allowed() { case "|${allowed// /|}|" in *"|$1|"*) return 0 ;; *) return 1 ;; esac; }

status=0
for f in "$@"; do
    ids=$(head -n 15 "$f" | grep -o 'SPDX-License-Identifier: [A-Za-z0-9.+ ()-]*' | sed 's/SPDX-License-Identifier: //; s/[[:space:]]*$//' || true)
    if [ -z "$ids" ]; then
        echo "missing SPDX header: $f"
        status=1
        continue
    fi
    while IFS= read -r expr; do
        terms=$(echo "$expr" | tr '()' '  ' | tr -s ' ' | awk '{gsub(/ WITH /, "_WITH_"); gsub(/ (AND|OR) /, " "); print}')
        for term in $terms; do
            is_allowed "$term" || { echo "license ${term//_WITH_/ WITH } not in SPDX_ALLOWED ($allowed): $f"; status=1; }
        done
    done <<< "$ids"
done
exit $status
