# SPDX-License-Identifier: Apache-2.0
"""Create a frozen V3 EPLB layout from independent calibration expert counts."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from evenmoe import MoELoadBalancer
from evenmoe.adapters.sglang import to_placement_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts",
        type=Path,
        required=True,
        help="JSON array [61, 256] of globally aggregated logical counts",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = torch.tensor(json.loads(args.counts.read_text()), dtype=torch.float64)
    if counts.shape != (61, 256) or not torch.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Expected finite nonnegative logical counts [61, 256]")
    result = MoELoadBalancer().plan_placement(
        to_placement_request(
            counts,
            num_physical_experts=288,
            num_local_physical_experts=36,
            num_groups=8,
            num_nodes=1,
            algorithm="deepseek",
        )
    )
    with args.output.open("x") as stream:
        json.dump(
            {
                "physical_to_logical_map": result.physical_to_logical_map.tolist(),
                "calibration_counts_sha256": hashlib.sha256(args.counts.read_bytes()).hexdigest(),
            },
            stream,
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
