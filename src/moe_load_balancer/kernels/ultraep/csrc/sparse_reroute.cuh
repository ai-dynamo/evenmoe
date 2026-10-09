// SPDX-FileCopyrightText: Copyright (c) 2026 PKU-DASYS and Dots-Infra
// SPDX-License-Identifier: MIT
//
// Modifications:
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Adapted from UltraEP v1.0.0 (Dots-Infra/UltraEP @ 94cab099):
//   csrc/kernels/reroute.cu (sparse quota path)
// Copyright (c) 2026 PKU-DASYS and Dots-Infra. MIT License -- see
// LICENSE.ultraep.

#include <cstdint>

#include "launch.cuh"

namespace ultra_ep::kernels {

namespace {

__device__ __forceinline__ int quota_replica_index(
    int ordinal,
    int logical_id,
    int replica_count,
    int max_replicas,
    const int32_t* __restrict__ rank_quota_prefix) {
    constexpr int prefetch_replicas = 8;
    const int row_offset = logical_id * max_replicas;
    const int prefetch_count = min(replica_count, prefetch_replicas);
    int replica_index = 0;
    int local_prefix[prefetch_replicas];

#pragma unroll
    for (int index = 0; index < prefetch_replicas; ++index) {
        local_prefix[index] =
            index < prefetch_count ? rank_quota_prefix[row_offset + index] : 0;
    }
    // Two cases upstream's sparse_quota_replica_index never meets: in training
    // the quota is solved from the very microbatch that is rerouted, so every
    // ordinal is below quota_total. In serving the quota describes an earlier
    // batch, so tokens past it are routine; without these branches the prefix
    // comparison would send every such token to the last replica.
    const int quota_total = replica_count <= prefetch_replicas
        ? local_prefix[replica_count - 1]
        : rank_quota_prefix[row_offset + replica_count - 1];
    if (quota_total <= 0) {
        // No quota solved for this expert yet: spread evenly rather than
        // pinning everything to one replica.
        return ordinal % replica_count;
    }
    if (ordinal >= quota_total) {
        // Wrap back into the quota's own range so the proportions it encodes
        // still apply. One subtraction covers up to 2x overflow, which is the
        // common case; beyond that fall back to the modulo.
        ordinal = ordinal < quota_total * 2 ? ordinal - quota_total
                                           : ordinal % quota_total;
    }
#pragma unroll
    for (int index = 0; index < prefetch_replicas; ++index) {
        if (index < prefetch_count) {
            replica_index += ordinal >= local_prefix[index] ? 1 : 0;
        }
    }
    for (int index = prefetch_replicas; index < replica_count; ++index) {
        replica_index += ordinal >= rank_quota_prefix[row_offset + index] ? 1 : 0;
    }
    return min(replica_index, max(replica_count - 1, 0));
}

template <typename IdType>
__global__ __launch_bounds__(256) void sparse_quota_reroute_kernel(
    const IdType* __restrict__ logical_topk_ids,
    int64_t* __restrict__ routed_topk_ids,
    const int32_t* __restrict__ logical_to_physical_map,
    const int32_t* __restrict__ logical_replica_counts,
    const int32_t* __restrict__ rank_quota_prefix,
    int32_t* __restrict__ counters,
    int num_entries,
    int max_replicas,
    int num_experts) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_entries;
         idx += gridDim.x * blockDim.x) {
        const int64_t logical_id = static_cast<int64_t>(logical_topk_ids[idx]);
        if (logical_id < 0 || logical_id >= num_experts) {
            routed_topk_ids[idx] = logical_id;
            continue;
        }
        const int replica_count =
            min(max(logical_replica_counts[logical_id], 0), max_replicas);
        if (replica_count <= 0) {
            routed_topk_ids[idx] = logical_id;
            continue;
        }
        const int ordinal = atomicAdd(&counters[logical_id], 1);
        const int replica_index = quota_replica_index(
            ordinal,
            static_cast<int>(logical_id),
            replica_count,
            max_replicas,
            rank_quota_prefix);
        routed_topk_ids[idx] =
            logical_to_physical_map[logical_id * max_replicas + replica_index];
    }
}

template <typename IdType>
void run_sparse_reroute_quota_impl(const IdType* logical_topk_ids,
                                   int64_t* routed_topk_ids,
                                   const int32_t* logical_to_physical_map,
                                   const int32_t* logical_replica_counts,
                                   const int32_t* rank_quota_prefix,
                                   int32_t* counters,
                                   int num_tokens,
                                   int top_k,
                                   int num_global_logical_experts,
                                   int max_replicas,
                                   cudaStream_t stream) {
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(
        counters, 0, num_global_logical_experts * sizeof(int32_t), stream));
    const int num_entries = num_tokens * top_k;
    if (num_entries == 0) {
        return;
    }
    constexpr int block_size = 256;
    const int num_blocks = min(256, (num_entries - 1) / block_size + 1);
    const auto config = make_launch_config(dim3(num_blocks), dim3(block_size), stream);
    launch_kernel(sparse_quota_reroute_kernel<IdType>,
                  config,
                  logical_topk_ids,
                  routed_topk_ids,
                  logical_to_physical_map,
                  logical_replica_counts,
                  rank_quota_prefix,
                  counters,
                  num_entries,
                  max_replicas,
                  num_global_logical_experts);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
}

}  // namespace

void run_sparse_reroute_quota(const int32_t* logical_topk_ids,
                              int64_t* routed_topk_ids,
                              const int32_t* logical_to_physical_map,
                              const int32_t* logical_replica_counts,
                              const int32_t* rank_quota_prefix,
                              int32_t* counters,
                              int num_tokens,
                              int top_k,
                              int num_global_logical_experts,
                              int max_replicas,
                              cudaStream_t stream) {
    run_sparse_reroute_quota_impl(logical_topk_ids,
                                  routed_topk_ids,
                                  logical_to_physical_map,
                                  logical_replica_counts,
                                  rank_quota_prefix,
                                  counters,
                                  num_tokens,
                                  top_k,
                                  num_global_logical_experts,
                                  max_replicas,
                                  stream);
}

void run_sparse_reroute_quota(const int64_t* logical_topk_ids,
                              int64_t* routed_topk_ids,
                              const int32_t* logical_to_physical_map,
                              const int32_t* logical_replica_counts,
                              const int32_t* rank_quota_prefix,
                              int32_t* counters,
                              int num_tokens,
                              int top_k,
                              int num_global_logical_experts,
                              int max_replicas,
                              cudaStream_t stream) {
    run_sparse_reroute_quota_impl(logical_topk_ids,
                                  routed_topk_ids,
                                  logical_to_physical_map,
                                  logical_replica_counts,
                                  rank_quota_prefix,
                                  counters,
                                  num_tokens,
                                  top_k,
                                  num_global_logical_experts,
                                  max_replicas,
                                  stream);
}

}  // namespace ultra_ep::kernels

// ---------------------------------------------------------------------------
// MLB tvm-ffi entry point (see placement.cuh for the pattern). One template
// instantiation per id dtype; ``cuda_placement.py`` builds the module that
// matches ``logical_topk_ids``.
// ---------------------------------------------------------------------------

#include "entry.cuh"

namespace {

template <typename IdType>
void reroute_sparse(tvm::ffi::TensorView logical_topk_ids,
                    tvm::ffi::TensorView routed_topk_ids,
                    tvm::ffi::TensorView logical_to_physical_map,
                    tvm::ffi::TensorView logical_replica_counts,
                    tvm::ffi::TensorView rank_quota_prefix,
                    tvm::ffi::TensorView counters) {
    using namespace host;

    SymbolicDevice device_;
    auto num_tokens = SymbolicSize{"num_tokens"};
    auto top_k = SymbolicSize{"top_k"};
    auto num_experts = SymbolicSize{"num_logical_experts"};
    auto max_replicas = SymbolicSize{"max_replicas"};
    // The dependent ``with_dtype<IdType>()`` goes last in its chain: after it
    // the expression is type-dependent and a following ``with_device<...>``
    // would need the ``template`` keyword to parse as a call.
    TensorMatcher({num_tokens, top_k}).with_device<kDLCUDA>(device_).with_dtype<IdType>().verify(logical_topk_ids);
    TensorMatcher({num_tokens, top_k}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(routed_topk_ids);
    TensorMatcher({num_experts, max_replicas})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device_)
        .verify(logical_to_physical_map);
    TensorMatcher({num_experts}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(logical_replica_counts);
    TensorMatcher({num_experts, max_replicas})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device_)
        .verify(rank_quota_prefix);
    TensorMatcher({num_experts}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(counters);
    RuntimeCheck(!mlb_ultraep::overlaps(logical_topk_ids, routed_topk_ids),
                 "logical_topk_ids and routed_topk_ids must not share memory");

    const int64_t tokens = num_tokens.unwrap();
    const int64_t k = top_k.unwrap();
    const int64_t experts = num_experts.unwrap();
    const int64_t replicas = max_replicas.unwrap();
    RuntimeCheck(experts > 0 && replicas > 0,
                 "logical_to_physical_map must contain experts and replica slots");
    RuntimeCheck(experts <= INT_MAX / replicas, "the flattened placement exceeds the CUDA kernel limit");
    RuntimeCheck(k <= INT_MAX && (k == 0 || tokens <= INT_MAX / k),
                 "the TopK shape exceeds the CUDA kernel limit");

    const DLDevice device = device_.unwrap();
    const mlb_ultraep::CurrentDeviceGuard guard(device.device_id);
    ultra_ep::kernels::run_sparse_reroute_quota(static_cast<const IdType*>(logical_topk_ids.data_ptr()),
                                                static_cast<int64_t*>(routed_topk_ids.data_ptr()),
                                                static_cast<const int32_t*>(logical_to_physical_map.data_ptr()),
                                                static_cast<const int32_t*>(logical_replica_counts.data_ptr()),
                                                static_cast<const int32_t*>(rank_quota_prefix.data_ptr()),
                                                static_cast<int32_t*>(counters.data_ptr()),
                                                static_cast<int>(tokens),
                                                static_cast<int>(k),
                                                static_cast<int>(experts),
                                                static_cast<int>(replicas),
                                                mlb_ultraep::current_stream(device));
}

}  // namespace
