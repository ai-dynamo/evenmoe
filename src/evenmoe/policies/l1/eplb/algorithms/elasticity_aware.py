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


import torch

from evenmoe.policies.l1.eplb.algorithms.deepseek import (
    rebalance_experts_hierarchical,
)


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    enable_hierarchical: bool,
    active_ranks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the Elastic-EP-aware DeepSeek placement policy."""

    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    num_active_ranks = int(active_ranks.sum().item())
    num_local_experts = num_replicas // num_gpus
    if num_active_ranks < num_gpus:
        # Fall back to global balancing over only the active ranks.
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight,
            num_local_experts * num_active_ranks,
            1,
            1,
            num_active_ranks,
        )
    elif enable_hierarchical:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus
        )
    else:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus
        )
    maxlogcnt = logcnt.max().item()
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(
            num_local_experts * num_active_ranks,
            dtype=torch.int64,
            device=log2phy.device,
        ).expand(num_layers, -1),
    )
    if num_active_ranks < num_gpus:
        phy2log_slices = list(phy2log.view(num_layers, num_active_ranks, -1).unbind(dim=1))
        active_ranks_list = active_ranks.tolist()
        for idx, active_rank in enumerate(active_ranks_list):
            if not active_rank:
                phy2log_slices.insert(idx, torch.zeros_like(phy2log_slices[0]))
                log2phy = torch.where(
                    log2phy >= idx * num_local_experts,
                    log2phy + num_local_experts,
                    log2phy,
                )
        phy2log = torch.stack(phy2log_slices, dim=1).contiguous().view(num_layers, -1)
    return phy2log, log2phy, logcnt
