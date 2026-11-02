// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// tile_repair decode routing: one launch, one CTA, per MoE layer per forward.
//
// Inputs are this rank's logical TopK ids and the EP-wide per-source-rank
// logical counts ``counts[R, L]`` (gathered by the orchestrator). Every rank
// evaluates the same deterministic function of ``counts`` and of a
// rank-invariant placement table, so all ranks compute the same plan.
//
//   1. Fixed first: every logical expert's ``G[e] = sum_s counts[s, e]`` tokens
//      start on its primary copy. A rank's GEMM work is its tile count
//      ``sum ceil(m / block_m)`` over the experts it runs.
//   2. Repair (warp 0, lane r holds rank r): up to ``max_moves`` times, take the
//      busiest rank (max tiles, then max tokens, then lowest rank), and move one
//      chunk -- the last partial tile ``m - block_m * (ceil(m / block_m) - 1)``
//      of a replicated expert it runs -- to another copy's rank r' when
//      ``tiles[r'] + add < tiles[busiest]``. The move minimising
//      ``(tiles[r'] + add, tokens[r'], block_m - chunk, rep index, copy index)``
//      wins. Stop at the lower bound ``ceil(sum_e ceil(G_e / block_m) / R)`` or
//      when no move strictly helps. The busiest rank loses exactly one tile per
//      move, so the peak never exceeds primary-only routing.
//   3. Exact-quota rewrite: the tokens of logical ``e`` are ordered globally by
//      (source rank, local ordinal); copy j takes the contiguous range
//      ``[sum_{i<j} m_i, sum_{i<=j} m_i)`` of that order. The local ordinal is
//      the entry's index among this rank's entries of ``e`` in flattened
//      ``[T, K]`` order, so the output is deterministic.
//
// Ids outside ``[0, L)`` come back as -1. No host synchronisation: the launch is
// CUDA-graph capturable; ``num_replicated`` is read on device so a same-shape
// placement update stays valid inside a captured graph.

#pragma once

#include <evenmoe_jit/tensor.h>
#include <evenmoe_jit/utils.h>

#include <evenmoe_jit/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <climits>
#include <cstdint>

namespace {

constexpr int kTileThreads = 256;
constexpr int kTileWarps = kTileThreads / 32;
constexpr int kTileMaxLogical = 512;
constexpr int kTileMaxCopies = 4;
constexpr int kTileMaxRanks = 32;
constexpr int kTileStats = 4;  // moves, lower bound, primary-only peak tiles, final peak tiles

__device__ __forceinline__ int tile_ceil(int m, int block_m) {
  return (m + block_m - 1) / block_m;
}

template <typename InT, typename OutT>
__global__ void __launch_bounds__(kTileThreads) tile_repair_route_kernel(
    const InT* __restrict__ logical_topk_ids,
    OutT* __restrict__ routed,
    const float* __restrict__ counts,          // [R, L]
    const int32_t* __restrict__ copy_physical,  // [L, C]
    const int32_t* __restrict__ copy_rank,      // [L, C]
    const int32_t* __restrict__ num_copies,     // [L]
    const int32_t* __restrict__ replicated,     // [L], ascending, -1 padded
    const int32_t* __restrict__ num_replicated,  // [1]
    int32_t* __restrict__ plan_out,             // [L, C] or nullptr
    int32_t* __restrict__ stats_out,            // [kTileStats + R] or nullptr
    int64_t num_ids,
    int num_logical,
    int num_ranks,
    int source_rank,
    int block_m,
    int max_moves) {
  __shared__ int total_s[kTileMaxLogical];   // G[e]
  __shared__ int prefix_s[kTileMaxLogical];  // tokens of e from ranks < source_rank
  __shared__ int seen_s[kTileMaxLogical];    // local entries of e already rewritten
  __shared__ int rep_index_s[kTileMaxLogical];
  __shared__ int warp_count_s[kTileWarps][kTileMaxLogical];
  __shared__ int rep_tokens_s[kTileMaxLogical][kTileMaxCopies];
  __shared__ int rep_rank_s[kTileMaxLogical][kTileMaxCopies];
  __shared__ int tiles_s[kTileMaxRanks];
  __shared__ int tokens_s[kTileMaxRanks];
  __shared__ int tile_sum_s;
  __shared__ int moves_s;
  __shared__ int primary_peak_s;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int num_rep = min(max(num_replicated[0], 0), num_logical);

  for (int e = tid; e < num_logical; e += kTileThreads) {
    seen_s[e] = 0;
    rep_index_s[e] = -1;
  }
  for (int i = tid; i < kTileWarps * kTileMaxLogical; i += kTileThreads) {
    (&warp_count_s[0][0])[i] = 0;
  }
  if (tid < kTileMaxRanks) {
    tiles_s[tid] = 0;
    tokens_s[tid] = 0;
  }
  if (tid == 0) {
    tile_sum_s = 0;
    moves_s = 0;
  }
  __syncthreads();

  // 1. EP-wide totals, this rank's prefix, primary-only tiles.
  for (int e = tid; e < num_logical; e += kTileThreads) {
    int total = 0;
    int prefix = 0;
    for (int s = 0; s < num_ranks; ++s) {
      const int c = __float2int_rn(counts[s * num_logical + e]);
      total += c;
      prefix += s < source_rank ? c : 0;
    }
    total_s[e] = total;
    prefix_s[e] = prefix;
    if (total > 0) {
      const int rank = copy_rank[e * kTileMaxCopies];
      const int t = tile_ceil(total, block_m);
      atomicAdd(&tiles_s[rank], t);
      atomicAdd(&tokens_s[rank], total);
      atomicAdd(&tile_sum_s, t);
    }
  }
  __syncthreads();
  for (int i = tid; i < num_rep; i += kTileThreads) {
    const int e = replicated[i];
    rep_index_s[e] = i;
    const int nc = num_copies[e];
#pragma unroll
    for (int j = 0; j < kTileMaxCopies; ++j) {
      rep_tokens_s[i][j] = j == 0 ? total_s[e] : 0;
      rep_rank_s[i][j] = j < nc ? copy_rank[e * kTileMaxCopies + j] : -1;
    }
  }
  __syncthreads();

  // 2. Bounded repair on warp 0.
  if (warp == 0) {
    const int lower_bound = (tile_sum_s + num_ranks - 1) / num_ranks;
    {
      const unsigned t = lane < num_ranks ? static_cast<unsigned>(tiles_s[lane]) : 0u;
      const unsigned peak = __reduce_max_sync(0xffffffffu, t);
      if (lane == 0) primary_peak_s = static_cast<int>(peak);
    }
    int moves = 0;
    for (; num_rep > 0 && moves < max_moves; ++moves) {
      const bool live = lane < num_ranks;
      const unsigned t = live ? static_cast<unsigned>(tiles_s[lane]) : 0u;
      const unsigned k = live ? static_cast<unsigned>(tokens_s[lane]) : 0u;
      const unsigned peak = __reduce_max_sync(0xffffffffu, t);
      const unsigned peak_tokens = __reduce_max_sync(0xffffffffu, (live && t == peak) ? k : 0u);
      const unsigned busiest_lanes = __ballot_sync(0xffffffffu, live && t == peak && k == peak_tokens);
      const int busiest = __ffs(busiest_lanes) - 1;
      const int busiest_tiles = static_cast<int>(peak);
      if (busiest_tiles <= lower_bound) break;

      unsigned key_a = UINT_MAX, key_b = UINT_MAX, key_c = UINT_MAX, key_d = UINT_MAX;
      int best_i = -1, best_from = -1, best_to = -1, best_chunk = 0, best_add = 0;
      for (int i = lane; i < num_rep; i += 32) {
        int from = -1;
#pragma unroll
        for (int j = 0; j < kTileMaxCopies; ++j) {
          if (rep_rank_s[i][j] == busiest && rep_tokens_s[i][j] > 0) from = j;
        }
        if (from < 0) continue;
        const int m = rep_tokens_s[i][from];
        const int chunk = m - block_m * (tile_ceil(m, block_m) - 1);
#pragma unroll
        for (int j = 0; j < kTileMaxCopies; ++j) {
          const int dst = rep_rank_s[i][j];
          if (j == from || dst < 0) continue;
          const int have = rep_tokens_s[i][j];
          const int add = tile_ceil(have + chunk, block_m) - tile_ceil(have, block_m);
          const int after = tiles_s[dst] + add;
          if (after >= busiest_tiles) continue;
          const unsigned a = static_cast<unsigned>(after);
          const unsigned b = static_cast<unsigned>(tokens_s[dst]);
          const unsigned c = static_cast<unsigned>(block_m - chunk);
          const unsigned d = static_cast<unsigned>(i * kTileMaxCopies + j);
          const bool better = a < key_a || (a == key_a && (b < key_b || (b == key_b && (c < key_c || (c == key_c && d < key_d)))));
          if (better) {
            key_a = a; key_b = b; key_c = c; key_d = d;
            best_i = i; best_from = from; best_to = j; best_chunk = chunk; best_add = add;
          }
        }
      }
      const unsigned win_a = __reduce_min_sync(0xffffffffu, key_a);
      if (win_a == UINT_MAX) break;
      bool tie = key_a == win_a;
      const unsigned win_b = __reduce_min_sync(0xffffffffu, tie ? key_b : UINT_MAX);
      tie = tie && key_b == win_b;
      const unsigned win_c = __reduce_min_sync(0xffffffffu, tie ? key_c : UINT_MAX);
      tie = tie && key_c == win_c;
      const unsigned win_d = __reduce_min_sync(0xffffffffu, tie ? key_d : UINT_MAX);
      if (tie && key_d == win_d) {  // the unique winning lane applies the move
        const int dst = rep_rank_s[best_i][best_to];
        rep_tokens_s[best_i][best_from] -= best_chunk;
        rep_tokens_s[best_i][best_to] += best_chunk;
        tiles_s[busiest] -= 1;
        tiles_s[dst] += best_add;
        tokens_s[busiest] -= best_chunk;
        tokens_s[dst] += best_chunk;
      }
      __syncwarp();
    }
    if (lane == 0) moves_s = moves;
  }
  __syncthreads();

  if (plan_out != nullptr) {
    for (int idx = tid; idx < num_logical * kTileMaxCopies; idx += kTileThreads) {
      const int e = idx / kTileMaxCopies;
      const int j = idx % kTileMaxCopies;
      const int ri = rep_index_s[e];
      int v;
      if (j >= num_copies[e]) {
        v = 0;
      } else if (ri < 0) {
        v = j == 0 ? total_s[e] : 0;
      } else {
        v = rep_tokens_s[ri][j];
      }
      plan_out[idx] = v;
    }
  }
  if (stats_out != nullptr && warp == 0) {
    const unsigned t = lane < num_ranks ? static_cast<unsigned>(tiles_s[lane]) : 0u;
    const unsigned peak = __reduce_max_sync(0xffffffffu, t);
    if (lane == 0) {
      stats_out[0] = moves_s;
      stats_out[1] = (tile_sum_s + num_ranks - 1) / num_ranks;
      stats_out[2] = primary_peak_s;
      stats_out[3] = static_cast<int>(peak);
    }
    if (lane < num_ranks) stats_out[kTileStats + lane] = tiles_s[lane];
  }

  // 3. Exact-quota rewrite with a deterministic local ordinal.
  const unsigned lane_lt = (1u << lane) - 1u;
  for (int64_t base = 0; base < num_ids; base += kTileThreads) {
    const int64_t i = base + tid;
    int e = -1;
    if (i < num_ids) {
      const int64_t id = static_cast<int64_t>(logical_topk_ids[i]);
      e = (id >= 0 && id < num_logical) ? static_cast<int>(id) : -1;
    }
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    const bool leader = (peers & lane_lt) == 0u;
    if (e >= 0 && leader) warp_count_s[warp][e] = __popc(peers);
    __syncthreads();
    if (i < num_ids) {
      OutT physical = static_cast<OutT>(-1);
      if (e >= 0) {
        const int ri = rep_index_s[e];
        if (ri < 0) {
          physical = static_cast<OutT>(copy_physical[e * kTileMaxCopies]);
        } else {
          int ordinal = seen_s[e] + __popc(peers & lane_lt);
          for (int w = 0; w < warp; ++w) ordinal += warp_count_s[w][e];
          const int position = prefix_s[e] + ordinal;
          const int nc = min(num_copies[e], kTileMaxCopies);
          int chosen = 0;  // unreachable with consistent counts; the primary is always valid
          int end = 0;
          for (int j = 0; j < nc; ++j) {
            end += rep_tokens_s[ri][j];
            if (position < end) {
              chosen = j;
              break;
            }
          }
          physical = static_cast<OutT>(copy_physical[e * kTileMaxCopies + chosen]);
        }
      }
      routed[i] = physical;
    }
    __syncthreads();
    if (e >= 0 && leader) {
      atomicAdd(&seen_s[e], __popc(peers));
      warp_count_s[warp][e] = 0;
    }
    __syncthreads();
  }
}

// tvm-ffi entry point. ``plan_out`` is ``[L, C]`` int32 or empty; ``stats_out``
// is ``[4 + R]`` int32 or empty (moves, lower bound, primary-only peak, final
// peak, per-rank tiles).
template <typename InT, typename OutT>
void tile_repair_route(tvm::ffi::TensorView logical_topk_ids,
                       tvm::ffi::TensorView routed,
                       tvm::ffi::TensorView counts,
                       tvm::ffi::TensorView copy_physical,
                       tvm::ffi::TensorView copy_rank,
                       tvm::ffi::TensorView num_copies,
                       tvm::ffi::TensorView replicated,
                       tvm::ffi::TensorView num_replicated,
                       tvm::ffi::TensorView plan_out,
                       tvm::ffi::TensorView stats_out,
                       int64_t source_rank,
                       int64_t block_m,
                       int64_t max_moves) {
  using namespace host;

  SymbolicDevice device_;
  auto num_tokens = SymbolicSize{"num_tokens"};
  auto top_k = SymbolicSize{"top_k"};
  auto num_logical = SymbolicSize{"num_logical_experts"};
  auto num_ranks = SymbolicSize{"num_ranks"};
  auto max_copies = SymbolicSize{"max_copies"};
  auto num_plan = SymbolicSize{"num_plan"};
  auto num_stats = SymbolicSize{"num_stats"};
  TensorMatcher({num_tokens, top_k}).with_device<kDLCUDA>(device_).with_dtype<InT>().verify(logical_topk_ids);
  TensorMatcher({num_tokens, top_k}).with_device<kDLCUDA>(device_).with_dtype<OutT>().verify(routed);
  TensorMatcher({num_ranks, num_logical}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(counts);
  TensorMatcher({num_logical, max_copies}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(copy_physical);
  TensorMatcher({num_logical, max_copies}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(copy_rank);
  TensorMatcher({num_logical}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(num_copies);
  TensorMatcher({num_logical}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(replicated);
  TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(num_replicated);
  TensorMatcher({num_plan}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(plan_out);
  TensorMatcher({num_stats}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(stats_out);

  const int64_t logical = num_logical.unwrap();
  const int64_t ranks = num_ranks.unwrap();
  const int64_t tokens = num_tokens.unwrap();
  const int64_t k = top_k.unwrap();
  RuntimeCheck(logical > 0 && logical <= kTileMaxLogical,
               "tile_repair supports 1..", kTileMaxLogical, " logical experts, got ", logical);
  RuntimeCheck(ranks > 0 && ranks <= kTileMaxRanks, "tile_repair supports 1..", kTileMaxRanks, " ranks, got ", ranks);
  RuntimeCheck(max_copies.unwrap() == kTileMaxCopies, "copy tables must have ", kTileMaxCopies, " columns");
  RuntimeCheck(num_plan.unwrap() == 0 || num_plan.unwrap() == logical * kTileMaxCopies,
               "plan_out must be empty or [num_logical * ", kTileMaxCopies, "]");
  RuntimeCheck(num_stats.unwrap() == 0 || num_stats.unwrap() == kTileStats + ranks,
               "stats_out must be empty or [", kTileStats, " + num_ranks]");
  RuntimeCheck(source_rank >= 0 && source_rank < ranks, "source_rank out of range: ", source_rank);
  RuntimeCheck(block_m > 0 && block_m <= (1 << 20), "block_m out of range: ", block_m);
  RuntimeCheck(max_moves >= 0 && max_moves <= (1 << 20), "max_moves out of range: ", max_moves);
  RuntimeCheck(tokens * k < (int64_t{1} << 31), "too many TopK entries for one launch");
  RuntimeCheck(logical_topk_ids.data_ptr() != routed.data_ptr() || tokens * k == 0,
               "logical_topk_ids and routed must not alias");

  const DLDevice device = device_.unwrap();
  LaunchKernel(/*grid=*/1, /*block=*/kTileThreads, device)(
      tile_repair_route_kernel<InT, OutT>,
      static_cast<const InT*>(logical_topk_ids.data_ptr()),
      static_cast<OutT*>(routed.data_ptr()),
      static_cast<const float*>(counts.data_ptr()),
      static_cast<const int32_t*>(copy_physical.data_ptr()),
      static_cast<const int32_t*>(copy_rank.data_ptr()),
      static_cast<const int32_t*>(num_copies.data_ptr()),
      static_cast<const int32_t*>(replicated.data_ptr()),
      static_cast<const int32_t*>(num_replicated.data_ptr()),
      num_plan.unwrap() > 0 ? static_cast<int32_t*>(plan_out.data_ptr()) : nullptr,
      num_stats.unwrap() > 0 ? static_cast<int32_t*>(stats_out.data_ptr()) : nullptr,
      tokens * k,
      static_cast<int>(logical),
      static_cast<int>(ranks),
      static_cast<int>(source_rank),
      static_cast<int>(block_m),
      static_cast<int>(max_moves));
}

}  // namespace
