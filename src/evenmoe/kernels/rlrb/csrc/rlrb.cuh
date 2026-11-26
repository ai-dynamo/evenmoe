// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <evenmoe_jit/tensor.h>
#include <evenmoe_jit/utils.cuh>
#include <evenmoe_jit/utils.h>

#include <cstdint>

namespace {

constexpr int kLogical = 128;
constexpr int kPhysical = 160;
constexpr int kTopK = 4;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr uint32_t kTokenMultiplier = 2654435769u;
constexpr uint32_t kRankMultiplier = 2246822519u;
constexpr unsigned kWarpMask = 0xffffffffu;

// Placement table offsets. The group and replica lengths are GPU values so
// graph replays see them along with the tables after an in-place update.
constexpr int kCandidates = 0;
constexpr int kSlotGroup = kCandidates + kPhysical;
constexpr int kGroupRank = kSlotGroup + kPhysical;
constexpr int kGroupExpert = kGroupRank + kPhysical;
constexpr int kGroupCopies = kGroupExpert + kPhysical;
constexpr int kGroupOffsets = kGroupCopies + kPhysical;
constexpr int kCopies = kGroupOffsets + kLogical + 1;
constexpr int kReplicated = kCopies + kLogical;
constexpr int kRankGroups = kReplicated + kPhysical - kLogical;
constexpr int kCandidateOffsets = kRankGroups + kPhysical;
constexpr int kNumGroups = kCandidateOffsets + kLogical + 1;
constexpr int kNumReplicated = kNumGroups + 1;
constexpr int kMetadataSize = kNumReplicated + 1;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(kWarpMask, value, offset);
  }
  return __shfl_sync(kWarpMask, value, 0);
}

__device__ __forceinline__ float warp_min(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fminf(value, __shfl_down_sync(kWarpMask, value, offset));
  }
  return __shfl_sync(kWarpMask, value, 0);
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(kWarpMask, value, offset));
  }
  return __shfl_sync(kWarpMask, value, 0);
}

template <int EP, int N>
__global__ void rlrb_kernel(
    const int32_t* __restrict__ ids,
    const int32_t* __restrict__ metadata,
    int32_t* __restrict__ output,
    void* __restrict__ expert_load,
    const void* __restrict__ record_enabled,
    const int32_t* __restrict__ valid_tokens,
    int source_rank, int flag_bytes, int load_bytes) {
  constexpr int kSlotsPerRank = kPhysical / EP;
  __shared__ int meta[kMetadataSize], histogram[kLogical];
  __shared__ float share[kPhysical], rank_load[EP], cdf[kPhysical];
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int warp = tid / 32;
  const int tokens = valid_tokens ? min(max(*valid_tokens, 0), N) : N;

  for (int i = tid; i < kMetadataSize; i += kThreads) {
    meta[i] = metadata[i];
  }
  if (tid < kLogical) histogram[tid] = 0;
  __syncthreads();

  // Count current valid Top-K assignments once per source rank.
#pragma unroll
  for (int i = tid; i < N * kTopK; i += kThreads) {
    const int expert = ids[i];
    const bool valid = i < tokens * kTopK && expert >= 0 && expert < kLogical;
    const unsigned active = __ballot_sync(kWarpMask, valid);
    if (valid) {
      const unsigned same = __match_any_sync(active, expert);
      if (lane == __ffs(same) - 1) atomicAdd(histogram + expert, __popc(same));
    }
  }
  __syncthreads();

  // Start with uniform shares across physical replicas. Several copies of
  // one expert on the same rank contribute to one (expert, rank) group.
  if (tid < meta[kNumGroups]) {
    const int expert = meta[kGroupExpert + tid];
    share[tid] =
        float(histogram[expert]) * float(meta[kGroupCopies + tid]) / float(meta[kCopies + expert]);
  }
  __syncthreads();
  for (int rank = warp; rank < EP; rank += kWarps) {
    const int group = lane < kSlotsPerRank ? meta[kRankGroups + rank * kSlotsPerRank + lane] : -1;
    const float total = warp_sum(group >= 0 ? share[group] : 0.f);
    if (lane == 0) rank_load[rank] = total;
  }
  __syncthreads();

  const float my_load = lane < EP ? rank_load[lane] : 0.f;
  const float peak = warp_max(my_load);
  const float total = warp_sum(my_load);
  const float update = peak * float(EP) >= 1.65f * total ? 1.f : 0.5f;
  for (int index = warp; index < meta[kNumReplicated]; index += kWarps) {
    const int expert = meta[kReplicated + index];
    const float mass = float(histogram[expert]);
    if (mass == 0.f) continue;

    const int begin = meta[kGroupOffsets + expert];
    const int count = meta[kGroupOffsets + expert + 1] - begin;
    const int group = begin + lane;
    const float old = lane < count ? share[group] : 0.f;
    const float remaining = lane < count ? fmaxf(rank_load[meta[kGroupRank + group]] - old, 0.f)
                                         : 1.e30f;
    // One headroom projection. Renormalizing the proposal conserves this
    // expert's mass even when some ranks have no room under the target.
    const float level = warp_min(remaining) + mass / float(count);
    const float proposed = lane < count ? fmaxf(level - remaining, 0.f) : 0.f;
    const float sum = warp_sum(proposed);
    if (lane < count) {
      share[group] = (1.f - update) * old + update * proposed * (mass / fmaxf(sum, 1.e-20f));
    }
  }
  __syncthreads();

  if (tid < kLogical) {
    const int copies = meta[kCopies + tid];
    float cumulative = 0.f;
#pragma unroll 1
    for (int j = 0; j < copies; ++j) {
      const int physical = meta[kCandidates + meta[kCandidateOffsets + tid] + j];
      const int group = meta[kSlotGroup + physical];
      cumulative += histogram[tid] > 0
                        ? share[group] / float(meta[kGroupCopies + group] * histogram[tid])
                        : 1.f / float(copies);
      cdf[physical] = cumulative;
    }
  }
  __syncthreads();

  const bool record = expert_load &&
      (flag_bytes == 1 ? *static_cast<const uint8_t*>(record_enabled) != 0
                       : *static_cast<const int32_t*>(record_enabled) != 0);
#pragma unroll
  for (int i = tid; i < N * kTopK; i += kThreads) {
    const int expert = ids[i];
    int chosen = -1;
    if (expert >= 0 && expert < kLogical) {
      const int copies = meta[kCopies + expert];
      const int* slots = meta + kCandidates + meta[kCandidateOffsets + expert];
      const uint32_t hash = uint32_t(i / kTopK) * kTokenMultiplier;
      if (copies == 1) {
        chosen = slots[0];
      } else if (i >= tokens * kTopK) {
        chosen = slots[hash % uint32_t(copies)];
      } else {
        // Use one low-discrepancy draw per token, rotated by source rank.
        const uint32_t seed = hash + uint32_t(source_rank) * kRankMultiplier;
        const float draw = float(seed >> 8) * (1.f / 16777216.f);
        chosen = slots[copies - 1];
#pragma unroll 1
        for (int j = 0; j < copies; ++j) {
          if (draw < cdf[slots[j]]) {
            chosen = slots[j];
            break;
          }
        }
      }
      if (record && i < tokens * kTopK) {
        if (load_bytes == 4) {
          atomicAdd(static_cast<int32_t*>(expert_load) + chosen, 1);
        } else {
          atomicAdd(static_cast<unsigned long long*>(expert_load) + chosen, 1ull);
        }
      }
    }
    output[i] = chosen;
  }
}

template <int EP, int N>
void run_rlrb(
    tvm::ffi::TensorView ids,
    tvm::ffi::TensorView metadata,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView expert_load,
    tvm::ffi::TensorView record_enabled,
    tvm::ffi::TensorView valid_tokens,
    int64_t source_rank) {
  using namespace host;
  SymbolicDevice device;
  TensorMatcher({N, kTopK}).with_dtype<int32_t>().with_device<kDLCUDA>(device)
      .verify(ids).verify(output);
  TensorMatcher({kMetadataSize}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(metadata);
  const int load_bytes = expert_load.dtype().bits / 8;
  const int flag_bytes = record_enabled.dtype().bits == 32 ? 4 : 1;
  if (expert_load.numel()) {
    TensorMatcher({kPhysical}).with_dtype<int32_t, int64_t>().with_device<kDLCUDA>(device)
        .verify(expert_load);
    TensorMatcher({1}).with_device<kDLCUDA>(device).verify(record_enabled);
  }
  if (valid_tokens.numel()) {
    TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(valid_tokens);
  }

  LaunchKernel(1, kThreads, device.unwrap())(
      rlrb_kernel<EP, N>,
      static_cast<const int32_t*>(ids.data_ptr()),
      static_cast<const int32_t*>(metadata.data_ptr()),
      static_cast<int32_t*>(output.data_ptr()),
      expert_load.numel() ? expert_load.data_ptr() : nullptr,
      record_enabled.numel() ? record_enabled.data_ptr() : nullptr,
      valid_tokens.numel() ? static_cast<const int32_t*>(valid_tokens.data_ptr()) : nullptr,
      static_cast<int>(source_rank), flag_bytes, load_bytes);
}

}  // namespace
