// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// tile_repair decode routing: one launch, one CTA, per MoE layer per forward.
//
// Every rank evaluates the same deterministic function of the EP-wide
// per-source-rank logical counts ``counts[R, L]`` and of a rank-invariant
// placement table, so all ranks compute the same plan.
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
// ``block_m == 0`` selects the tile height per call from the expected rows per
// physical slot ``em = sum_e G[e] / num_slots`` (integer compares, identical on
// every rank): em < 5.5 -> 8, < 12.7 -> 16, < 28.4 -> 32, < 80 -> 48, else 64.
//
// Counts [R, L] are gathered by the framework before this kernel runs.
// Ids outside [0, L) become -1. Device-resident placement tables support
// same-shape updates under CUDA Graph replay.

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
// moves, lower bound, primary-only peak tiles, final peak tiles, block_m used.
constexpr int kTileStats = 5;
static_assert(kTileMaxLogical % kTileThreads == 0, "table staging assumes whole per-thread slices");

struct TileShared {
  int total[kTileMaxLogical];      // G[e]
  int prefix[kTileMaxLogical];     // tokens of e from ranks < source_rank
  int seen[kTileMaxLogical];       // local entries of e already rewritten
  int rep_index[kTileMaxLogical];  // index into the replicated list, -1 if not replicated
  int warp_count[kTileWarps][kTileMaxLogical];
  int rep_tokens[kTileMaxLogical][kTileMaxCopies];
  // Placement tables, staged into shared memory while the counts are gathered.
  int physical[kTileMaxLogical][kTileMaxCopies];
  int8_t rep_rank[kTileMaxLogical][kTileMaxCopies];
  int8_t primary_rank[kTileMaxLogical];
  int8_t copies[kTileMaxLogical];
  int16_t rep_logical[kTileMaxLogical];
  int tiles[kTileMaxRanks];
  int tokens[kTileMaxRanks];
  int tile_sum;
  int grand_total;
  int moves;
  int primary_peak;
  int block_m;
};

// ceil(m / d) through a float reciprocal with exact integer correction
// (the repair loop's hot path); exact for 0 <= m < 2^23.
struct TileCeil {
  int d;
  float inv;
  __device__ __forceinline__ explicit TileCeil(int divisor) : d(divisor), inv(1.0f / static_cast<float>(divisor)) {}
  __device__ __forceinline__ int operator()(int m) const {
    const int x = m + d - 1;
    if (x >= (1 << 23)) return x / d;
    int q = __float2int_rz(static_cast<float>(x) * inv);
    if (q * d > x) --q;
    if ((q + 1) * d <= x) ++q;
    return q;
  }
};

__device__ __forceinline__ int tile_adaptive_block_m(long long rows, long long slots) {
  if (2 * rows < 11 * slots) return 8;
  if (10 * rows < 127 * slots) return 16;
  if (10 * rows < 284 * slots) return 32;
  if (rows < 80 * slots) return 48;
  return 64;
}

// Adds ``value`` to ``*target`` once per warp (lanes outside the loop add 0).
__device__ __forceinline__ void tile_warp_add(int* target, int value) {
  const int sum = __reduce_add_sync(0xffffffffu, value);
  if ((threadIdx.x & 31) == 0 && sum != 0) atomicAdd(target, sum);
}

__device__ __forceinline__ void tile_init(TileShared& s, int num_logical) {
  const int tid = threadIdx.x;
  for (int e = tid; e < num_logical; e += kTileThreads) {
    s.seen[e] = 0;
    s.rep_index[e] = -1;
    s.total[e] = 0;
    s.prefix[e] = 0;
  }
  for (int i = tid; i < kTileWarps * kTileMaxLogical; i += kTileThreads) {
    (&s.warp_count[0][0])[i] = 0;
  }
  if (tid < kTileMaxRanks) {
    s.tiles[tid] = 0;
    s.tokens[tid] = 0;
  }
  if (tid == 0) {
    s.tile_sum = 0;
    s.grand_total = 0;
    s.moves = 0;
    s.primary_peak = 0;
    s.block_m = 1;
  }
}

// Stage the placement tables (independent of the counts) into shared memory.
// Runs right after tile_init's barrier; the next barrier publishes it.
__device__ __forceinline__ void tile_load_tables(TileShared& s,
                                                 const int32_t* __restrict__ copy_physical,
                                                 const int32_t* __restrict__ copy_rank,
                                                 const int32_t* __restrict__ num_copies,
                                                 const int32_t* __restrict__ replicated,
                                                 int num_rep,
                                                 int num_logical) {
  const int tid = threadIdx.x;
  // Issue every load before the first shared store so their latencies overlap.
  constexpr int kPhysPerThread = kTileMaxLogical * kTileMaxCopies / kTileThreads;
  constexpr int kLogicalPerThread = kTileMaxLogical / kTileThreads;
  int phys[kPhysPerThread];
  int prim[kLogicalPerThread];
  int ncop[kLogicalPerThread];
#pragma unroll
  for (int k = 0; k < kPhysPerThread; ++k) {
    const int idx = tid + k * kTileThreads;
    phys[k] = idx < num_logical * kTileMaxCopies ? copy_physical[idx] : -1;
  }
#pragma unroll
  for (int k = 0; k < kLogicalPerThread; ++k) {
    const int e = tid + k * kTileThreads;
    prim[k] = e < num_logical ? copy_rank[e * kTileMaxCopies] : 0;
    ncop[k] = e < num_logical ? num_copies[e] : 1;
  }
#pragma unroll
  for (int k = 0; k < kPhysPerThread; ++k) {
    const int idx = tid + k * kTileThreads;
    if (idx < num_logical * kTileMaxCopies) (&s.physical[0][0])[idx] = phys[k];
  }
#pragma unroll
  for (int k = 0; k < kLogicalPerThread; ++k) {
    const int e = tid + k * kTileThreads;
    if (e < num_logical) {
      s.primary_rank[e] = static_cast<int8_t>(prim[k]);
      s.copies[e] = static_cast<int8_t>(min(max(ncop[k], 1), kTileMaxCopies));
    }
  }
  for (int i = tid; i < num_rep; i += kTileThreads) {
    const int e = replicated[i];
    s.rep_index[e] = i;
    s.rep_logical[i] = static_cast<int16_t>(e);
    const int nc = num_copies[e];
#pragma unroll
    for (int j = 0; j < kTileMaxCopies; ++j) {
      s.rep_rank[i][j] = static_cast<int8_t>(j < nc ? copy_rank[e * kTileMaxCopies + j] : -1);
    }
  }
}

// Steps 1-2 given ``s.total`` / ``s.prefix`` / ``s.grand_total``. Leaves
// ``s.rep_index`` / ``s.rep_tokens`` holding the plan.
__device__ __forceinline__ void tile_plan(TileShared& s,
                                          int num_rep,
                                          int num_logical,
                                          int num_ranks,
                                          int block_m_arg,
                                          int num_slots,
                                          int max_moves) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (tid == 0) {
    s.block_m = block_m_arg > 0 ? block_m_arg : tile_adaptive_block_m(s.grand_total, num_slots);
  }
  __syncthreads();
  const int block_m = s.block_m;
  const TileCeil ceil_div(block_m);
  for (int base = 0; base < num_logical; base += kTileThreads) {
    const int e = base + tid;
    const int total = e < num_logical ? s.total[e] : 0;
    const int t = total > 0 ? ceil_div(total) : 0;
    if (total > 0) {
      const int rank = s.primary_rank[e];
      atomicAdd(&s.tiles[rank], t);
      atomicAdd(&s.tokens[rank], total);
    }
    tile_warp_add(&s.tile_sum, t);
  }
  for (int i = tid; i < num_rep; i += kTileThreads) {
    const int total = s.total[s.rep_logical[i]];
#pragma unroll
    for (int j = 0; j < kTileMaxCopies; ++j) s.rep_tokens[i][j] = j == 0 ? total : 0;
  }
  __syncthreads();

  // Bounded repair on warp 0.
  if (warp == 0) {
    const int lower_bound = (s.tile_sum + num_ranks - 1) / num_ranks;
    {
      const unsigned t = lane < num_ranks ? static_cast<unsigned>(s.tiles[lane]) : 0u;
      const unsigned peak = __reduce_max_sync(0xffffffffu, t);
      if (lane == 0) s.primary_peak = static_cast<int>(peak);
    }
    int moves = 0;
    for (; num_rep > 0 && moves < max_moves; ++moves) {
      const bool live = lane < num_ranks;
      const unsigned t = live ? static_cast<unsigned>(s.tiles[lane]) : 0u;
      const unsigned k = live ? static_cast<unsigned>(s.tokens[lane]) : 0u;
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
          if (s.rep_rank[i][j] == busiest && s.rep_tokens[i][j] > 0) from = j;
        }
        if (from < 0) continue;
        const int m = s.rep_tokens[i][from];
        const int chunk = m - block_m * (ceil_div(m) - 1);
#pragma unroll
        for (int j = 0; j < kTileMaxCopies; ++j) {
          const int dst = s.rep_rank[i][j];
          if (j == from || dst < 0) continue;
          const int have = s.rep_tokens[i][j];
          const int add = ceil_div(have + chunk) - ceil_div(have);
          const int after = s.tiles[dst] + add;
          if (after >= busiest_tiles) continue;
          const unsigned a = static_cast<unsigned>(after);
          const unsigned b = static_cast<unsigned>(s.tokens[dst]);
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
        const int dst = s.rep_rank[best_i][best_to];
        s.rep_tokens[best_i][best_from] -= best_chunk;
        s.rep_tokens[best_i][best_to] += best_chunk;
        s.tiles[busiest] -= 1;
        s.tiles[dst] += best_add;
        s.tokens[busiest] -= best_chunk;
        s.tokens[dst] += best_chunk;
      }
      __syncwarp();
    }
    if (lane == 0) s.moves = moves;
  }
  __syncthreads();
}

__device__ __forceinline__ void tile_write_plan(const TileShared& s,
                                                int32_t* __restrict__ plan_out,
                                                int32_t* __restrict__ stats_out,
                                                int num_logical,
                                                int num_ranks) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (plan_out != nullptr) {
    for (int idx = tid; idx < num_logical * kTileMaxCopies; idx += kTileThreads) {
      const int e = idx / kTileMaxCopies;
      const int j = idx % kTileMaxCopies;
      const int ri = s.rep_index[e];
      int v;
      if (j >= s.copies[e]) {
        v = 0;
      } else if (ri < 0) {
        v = j == 0 ? s.total[e] : 0;
      } else {
        v = s.rep_tokens[ri][j];
      }
      plan_out[idx] = v;
    }
  }
  if (stats_out != nullptr && warp == 0) {
    const unsigned t = lane < num_ranks ? static_cast<unsigned>(s.tiles[lane]) : 0u;
    const unsigned peak = __reduce_max_sync(0xffffffffu, t);
    if (lane == 0) {
      stats_out[0] = s.moves;
      stats_out[1] = (s.tile_sum + num_ranks - 1) / num_ranks;
      stats_out[2] = s.primary_peak;
      stats_out[3] = static_cast<int>(peak);
      stats_out[4] = s.block_m;
    }
    if (lane < num_ranks) stats_out[kTileStats + lane] = s.tiles[lane];
  }
}

// Step 3. Entries of logical experts with ``rep_index < 0`` go to the primary.
template <typename InT, typename OutT>
__device__ __forceinline__ void tile_rewrite(TileShared& s,
                                             const InT* __restrict__ logical_topk_ids,
                                             OutT* __restrict__ routed,
                                             int64_t num_ids,
                                             int num_logical) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const unsigned lane_lt = (1u << lane) - 1u;
  for (int64_t base = 0; base < num_ids; base += kTileThreads) {
    const int64_t i = base + tid;
    int e = -1;
    if (i < num_ids) {
      const int64_t id = static_cast<int64_t>(logical_topk_ids[i]);
      e = (id >= 0 && id < num_logical) ? static_cast<int>(id) : -1;
    }
    const int ri = e >= 0 ? s.rep_index[e] : -1;
    // Ordinals are only needed for replicated experts: group the rest under -1
    // and skip the match entirely in a warp without any.
    const int key = ri >= 0 ? e : -1;
    const unsigned peers = __any_sync(0xffffffffu, key >= 0) ? __match_any_sync(0xffffffffu, key) : 0u;
    const bool leader = key >= 0 && (peers & lane_lt) == 0u;
    if (leader) s.warp_count[warp][e] = __popc(peers);
    __syncthreads();
    if (i < num_ids) {
      OutT physical = static_cast<OutT>(-1);
      if (e >= 0) {
        if (ri < 0) {
          physical = static_cast<OutT>(s.physical[e][0]);
        } else {
          int ordinal = s.seen[e] + __popc(peers & lane_lt);
          for (int w = 0; w < warp; ++w) ordinal += s.warp_count[w][e];
          const int position = s.prefix[e] + ordinal;
          const int nc = s.copies[e];
          int chosen = 0;  // unreachable with consistent counts; the primary is always valid
          int end = 0;
          for (int j = 0; j < nc; ++j) {
            end += s.rep_tokens[ri][j];
            if (position < end) {
              chosen = j;
              break;
            }
          }
          physical = static_cast<OutT>(s.physical[e][chosen]);
        }
      }
      routed[i] = physical;
    }
    __syncthreads();
    if (leader) {
      atomicAdd(&s.seen[e], __popc(peers));
      s.warp_count[warp][e] = 0;
    }
    __syncthreads();
  }
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
    int num_slots,
    int max_moves) {
  __shared__ TileShared s;
  const int tid = threadIdx.x;
  const int num_rep = min(max(num_replicated[0], 0), num_logical);
  tile_init(s, num_logical);
  __syncthreads();
  tile_load_tables(s, copy_physical, copy_rank, num_copies, replicated, num_rep, num_logical);
  for (int base = 0; base < num_logical; base += kTileThreads) {
    const int e = base + tid;
    int total = 0;
    if (e < num_logical) {
      int prefix = 0;
      for (int r = 0; r < num_ranks; ++r) {
        const int c = __float2int_rn(counts[r * num_logical + e]);
        total += c;
        prefix += r < source_rank ? c : 0;
      }
      s.total[e] = total;
      s.prefix[e] = prefix;
    }
    tile_warp_add(&s.grand_total, total);
  }
  __syncthreads();
  tile_plan(s, num_rep, num_logical, num_ranks, block_m, num_slots, max_moves);
  tile_write_plan(s, plan_out, stats_out, num_logical, num_ranks);
  tile_rewrite<InT, OutT>(s, logical_topk_ids, routed, num_ids, num_logical);
}

// tvm-ffi entry point. ``plan_out`` is ``[L, C]`` int32 or empty; ``stats_out``
// is ``[kTileStats + R]`` int32 or empty. ``block_m == 0`` is adaptive.
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
                       int64_t num_slots,
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
  RuntimeCheck(block_m >= 0 && block_m <= (1 << 20), "block_m out of range: ", block_m);
  RuntimeCheck(block_m > 0 || (num_slots > 0 && num_slots <= (1 << 24)), "adaptive block_m needs num_slots > 0");
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
      static_cast<int>(num_slots),
      static_cast<int>(max_moves));
}

}  // namespace
