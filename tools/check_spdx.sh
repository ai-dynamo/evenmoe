#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Fails any file whose first 15 lines carry no SPDX-License-Identifier, or one outside SPDX_ALLOWED.
set -euo pipefail

allowed="${SPDX_ALLOWED:-Apache-2.0}"
status=0
for f in "$@"; do
    ids=$(head -n 15 "$f" | grep -o 'SPDX-License-Identifier: [A-Za-z0-9.+ ()-]*' | sed 's/SPDX-License-Identifier: //; s/[[:space:]]*$//' || true)
    if [ -z "$ids" ]; then
        echo "missing SPDX header: $f"
        status=1
        continue
    fi
    while IFS= read -r expr; do
        for id in $(echo "$expr" | tr '()' '  ' | sed 's/ AND / /g; s/ OR / /g; s/ WITH [^ ]*//g'); do
            case " $allowed " in
                *" $id "*) ;;
                *) echo "license $id not in SPDX_ALLOWED ($allowed): $f"; status=1 ;;
            esac
        done
    done <<< "$ids"
done
exit $status
