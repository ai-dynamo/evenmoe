// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// METRO-repair decode routing: one launch, one CTA.
//
//   1. Active set: ``count[l] > 0`` from the EP-summed count, or -- when no
//      count is given -- every in-range id of the (already EP-global) TopK.
//   2. METRO fixed-first (identical to ``metro_assign_reference``): an active
//      expert held by one rank loads that rank; replicated active experts go,
//      in ascending logical id, to the least-loaded holder (ties -> lowest rank).
//   3. Bounded repair: up to ``max_moves`` single moves, each relieving one
//      overloaded rank ``src`` towards an underloaded rank ``dst`` (load gap >=
//      2) along a direct edge ``src -> dst`` or a two-hop path
//      ``src -> mid -> dst``. An edge ``a -> b`` exists when some active expert
//      owned by ``a`` also has a copy on ``b``; each hop moves the lowest such
//      logical id. The move maximizes ``load[src]``, then minimizes ``load[dst]``,
//      then ``(src, dst, mid)`` with direct edges last, which is exactly the host
//      oracle ``metro_repair_assign_reference``. With ``early_exit`` the search
//      stops once the rank maximum reaches the lower bound
//      ``max(ceil(active / ranks), max forced single-copy load)``.
//   4. Every ``(token, slot)`` of logical ``l`` maps to the copy of ``l`` on its
//      owner rank. Ids outside ``[0, L)`` and rows at or past the non-padded
//      token count come back as -1.
//
// Warp 0 runs the serial part; lane ``r`` holds rank ``r``'s load. The edge set
// is kept incrementally as per-(src, dst) 256-bit masks of movable experts plus
// a per-src adjacency bitmap, so a move costs O(ranks) instead of O(L x ranks).

#pragma once

#include <evenmoe_jit/tensor.h>
#include <evenmoe_jit/utils.h>

#include <evenmoe_jit/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <climits>
#include <cstdint>

namespace {

constexpr int kMetroRepairThreads = 256;
constexpr int kMetroRepairMaxLogical = 256;
constexpr int kMetroRepairWords = kMetroRepairMaxLogical / 32;

__device__ __forceinline__ unsigned long long metro_repair_warp_min(unsigned long long v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const unsigned long long other = __shfl_xor_sync(0xffffffffu, v, offset);
    v = other < v ? other : v;
  }
  return v;
}

template <int MAX_RANKS>
struct MetroRepairEdges {
  // bits[src][dst][word]: active experts owned by ``src`` that also have a copy on ``dst``.
  unsigned int bits[MAX_RANKS * MAX_RANKS * kMetroRepairWords];
  // adj[src] bit dst: the (src, dst) mask is non-empty.
  unsigned int adj[MAX_RANKS];
};

template <int MAX_RANKS>
__device__ __forceinline__ unsigned int* metro_repair_cell(MetroRepairEdges<MAX_RANKS>& edges, int src, int dst) {
  return &edges.bits[(src * MAX_RANKS + dst) * kMetroRepairWords];
}

template <int MAX_RANKS>
__device__ __forceinline__ int metro_repair_lowest(MetroRepairEdges<MAX_RANKS>& edges, int src, int dst) {
  const unsigned int* cell = metro_repair_cell(edges, src, dst);
#pragma unroll
  for (int word = 0; word < kMetroRepairWords; ++word) {
    if (cell[word] != 0u) return word * 32 + __ffs(cell[word]) - 1;
  }
  return INT_MAX;
}

// Move expert ``e`` from rank ``from`` to rank ``to`` (whole warp).
template <int MAX_RANKS>
__device__ __forceinline__ void metro_repair_move(MetroRepairEdges<MAX_RANKS>& edges,
                                                  const unsigned int* holders,
                                                  int* owner,
                                                  int e,
                                                  int from,
                                                  int to,
                                                  int lane,
                                                  int num_ranks) {
  const unsigned int mask = holders[e];
  const unsigned int bit = 1u << (e & 31);
  const int word = e >> 5;
  if (lane < num_ranks && ((mask >> lane) & 1u)) {
    if (lane != from) atomicAnd(&metro_repair_cell(edges, from, lane)[word], ~bit);
    if (lane != to) atomicOr(&metro_repair_cell(edges, to, lane)[word], bit);
  }
  __syncwarp();
  bool from_live = false;
  bool to_live = false;
  if (lane < num_ranks) {
    const unsigned int* f = metro_repair_cell(edges, from, lane);
    const unsigned int* t = metro_repair_cell(edges, to, lane);
#pragma unroll
    for (int w = 0; w < kMetroRepairWords; ++w) {
      from_live |= f[w] != 0u;
      to_live |= t[w] != 0u;
    }
  }
  const unsigned int from_row = __ballot_sync(0xffffffffu, from_live);
  const unsigned int to_row = __ballot_sync(0xffffffffu, to_live);
  if (lane == 0) {
    edges.adj[from] = from_row;
    edges.adj[to] = to_row;
    owner[e] = to;
  }
  __syncwarp();
}

template <typename InT, typename OutT, int RANKS>
__global__ void __launch_bounds__(kMetroRepairThreads) metro_repair_route_kernel(
    const InT* __restrict__ logical_topk_ids,
    OutT* __restrict__ routed,
    const float* __restrict__ counts,  // nullptr: active set from the ids
    const int32_t* __restrict__ token_gate,  // nullptr: every row is live
    const int32_t* __restrict__ holder_mask,
    const int32_t* __restrict__ physical_by_rank,
    const int32_t* __restrict__ default_physical,
    int32_t* __restrict__ rank_load,
    int64_t num_ids,
    int top_k,
    int num_logical,
    int max_moves,
    bool early_exit) {
  // The rank count is a compile-time constant so the pair loops unroll and
  // ``/`` and ``%`` by it fold; one module is built per rank count.
  constexpr int num_ranks = RANKS;
  // Stage the [L, ranks] physical table while the active set is scanned, when it fits.
  constexpr bool kStagePhysical = RANKS <= 16;
  __shared__ int physical_s[kStagePhysical ? kMetroRepairMaxLogical * RANKS : 1];
  __shared__ unsigned int active[kMetroRepairWords];
  __shared__ unsigned int holders[kMetroRepairMaxLogical];
  __shared__ int owner[kMetroRepairMaxLogical];
  __shared__ int fixed_load[32];
  __shared__ int load_snapshot[32];
  __shared__ unsigned int column[32];
  __shared__ MetroRepairEdges<RANKS> edges;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  int64_t limit = num_ids;
  if (token_gate != nullptr) {
    const int64_t live_rows = token_gate[0] < 0 ? 0 : static_cast<int64_t>(token_gate[0]);
    limit = live_rows * top_k < num_ids ? live_rows * top_k : num_ids;
  }

  // One thread per logical expert: holder mask, count-based activity and the
  // physical-table staging are all loaded in a single pass.
  static_assert(kMetroRepairThreads == kMetroRepairMaxLogical, "one thread per logical expert");
  const int expert = tid;
  const unsigned int mask = expert < num_logical ? static_cast<unsigned int>(holder_mask[expert]) : 0u;
  bool live = counts != nullptr && expert < num_logical && counts[expert] > 0.f;
  holders[expert] = mask;
  owner[expert] = -1;
  if (tid < 32) fixed_load[tid] = 0;
  if constexpr (kStagePhysical) {
    for (int i = tid; i < num_logical * RANKS; i += kMetroRepairThreads) physical_s[i] = physical_by_rank[i];
  }
  if (counts != nullptr) {
    const unsigned int word = __ballot_sync(0xffffffffu, live);
    if (lane == 0) active[warp] = word;
  } else if (tid < kMetroRepairWords) {
    active[tid] = 0u;
  }
  __syncthreads();

  if (counts == nullptr) {  // EP-global TopK: every in-range live id is active
    for (int64_t i = tid; i < limit; i += kMetroRepairThreads) {
      const int64_t e = static_cast<int64_t>(logical_topk_ids[i]);
      if (e >= 0 && e < num_logical) atomicOr(&active[e >> 5], 1u << (e & 31));
    }
    __syncthreads();
    live = (active[warp] >> lane) & 1u;
  }

  // Fixed load: active experts held by exactly one rank.
  if (live && __popc(mask) == 1) {
    const int rank = __ffs(mask) - 1;
    owner[expert] = rank;
    atomicAdd(&fixed_load[rank], 1);
  }
  __syncthreads();

  if (warp == 0) {
    int load = lane < num_ranks ? fixed_load[lane] : 0;
    int num_active = 0;
#pragma unroll
    for (int w = 0; w < kMetroRepairWords; ++w) num_active += __popc(active[w]);
    const int forced = __reduce_max_sync(0xffffffffu, load);
    const int lower_bound = max((num_active + num_ranks - 1) / num_ranks, forced);

    // METRO greedy over replicated active experts in ascending logical id.
    for (int w = 0; w < kMetroRepairWords; ++w) {
      const int e0 = w * 32 + lane;
      const bool replicated = ((active[w] >> lane) & 1u) && __popc(holders[e0]) > 1;
      unsigned int pending = __ballot_sync(0xffffffffu, replicated);
      while (pending) {
        const int e = w * 32 + __ffs(pending) - 1;
        pending &= pending - 1;
        const unsigned int key =
            (lane < num_ranks && ((holders[e] >> lane) & 1u)) ? static_cast<unsigned int>(load) * 32u + lane : UINT_MAX;
        const int rank = static_cast<int>(__reduce_min_sync(0xffffffffu, key) & 31u);
        if (lane == rank) ++load;
        if (lane == 0) owner[e] = rank;
      }
    }
    __syncwarp();

    int peak = __reduce_max_sync(0xffffffffu, load);
    if (max_moves > 0 && !(early_exit && peak <= lower_bound)) {
      // Build the edge state once.
      for (int i = lane; i < num_ranks * num_ranks * kMetroRepairWords; i += 32) {
        const int src = i / (num_ranks * kMetroRepairWords);
        const int rest = i % (num_ranks * kMetroRepairWords);
        metro_repair_cell(edges, src, rest / kMetroRepairWords)[rest % kMetroRepairWords] = 0u;
      }
      __syncwarp();
      for (int w = 0; w < kMetroRepairWords; ++w) {
        const int e = w * 32 + lane;
        if ((active[w] >> lane) & 1u) {
          const int src = owner[e];
          unsigned int others = holders[e] & ~(1u << src);
          while (others) {
            const int dst = __ffs(others) - 1;
            others &= others - 1;
            atomicOr(&metro_repair_cell(edges, src, dst)[w], 1u << lane);
          }
        }
      }
      __syncwarp();
      for (int src = 0; src < num_ranks; ++src) {
        bool live = false;
        if (lane < num_ranks) {
          const unsigned int* cell = metro_repair_cell(edges, src, lane);
#pragma unroll
          for (int w = 0; w < kMetroRepairWords; ++w) live |= cell[w] != 0u;
        }
        const unsigned int row = __ballot_sync(0xffffffffu, live);
        if (lane == 0) edges.adj[src] = row;
      }
      __syncwarp();

      for (int step = 0; step < max_moves; ++step) {
        if (lane < num_ranks) {
          load_snapshot[lane] = load;
          unsigned int into = 0u;  // ranks m with an edge m -> lane
          for (int m = 0; m < num_ranks; ++m) into |= ((edges.adj[m] >> lane) & 1u) << m;
          column[lane] = into;
        }
        __syncwarp();
        unsigned long long best = ULLONG_MAX;
        for (int pair = lane; pair < num_ranks * num_ranks; pair += 32) {
          const int src = pair / num_ranks;
          const int dst = pair % num_ranks;
          if (load_snapshot[src] - load_snapshot[dst] < 2) continue;
          const unsigned int row = edges.adj[src];
          int mid = -1;
          if ((row >> dst) & 1u) {
            mid = 32;
          } else {
            const unsigned int both = row & column[dst];
            if (both) mid = __ffs(both) - 1;
          }
          if (mid >= 0) {
            const unsigned long long key =
                ((256ull - load_snapshot[src]) * 257ull + load_snapshot[dst]) * 32ull * 32ull * 33ull +
                static_cast<unsigned long long>(src) * 32ull * 33ull + static_cast<unsigned long long>(dst) * 33ull +
                static_cast<unsigned long long>(mid);
            best = key < best ? key : best;
          }
        }
        best = metro_repair_warp_min(best);
        if (best == ULLONG_MAX) break;
        const int mid = static_cast<int>(best % 33ull);
        const int dst = static_cast<int>((best / 33ull) % 32ull);
        const int src = static_cast<int>((best / (33ull * 32ull)) % 32ull);
        if (mid == 32) {
          const int e = metro_repair_lowest(edges, src, dst);
          metro_repair_move(edges, holders, owner, e, src, dst, lane, num_ranks);
        } else {
          // Both hops are chosen before either expert moves.
          const int first = metro_repair_lowest(edges, src, mid);
          const int second = metro_repair_lowest(edges, mid, dst);
          metro_repair_move(edges, holders, owner, first, src, mid, lane, num_ranks);
          metro_repair_move(edges, holders, owner, second, mid, dst, lane, num_ranks);
        }
        if (lane == src) --load;
        if (lane == dst) ++load;
        peak = __reduce_max_sync(0xffffffffu, load);
        if (early_exit && peak <= lower_bound) break;
      }
    }
    if (lane < num_ranks) rank_load[lane] = load;
  }
  __syncthreads();

  for (int64_t i = tid; i < num_ids; i += kMetroRepairThreads) {
    OutT physical = static_cast<OutT>(-1);
    if (i < limit) {
      const int64_t e = static_cast<int64_t>(logical_topk_ids[i]);
      if (e >= 0 && e < num_logical) {
        const int rank = owner[e];
        int copy;
        if (rank < 0) {
          copy = default_physical[e];
        } else if constexpr (kStagePhysical) {
          copy = physical_s[e * RANKS + rank];
        } else {
          copy = physical_by_rank[e * RANKS + rank];
        }
        physical = static_cast<OutT>(copy);
      }
    }
    routed[i] = physical;
  }
}

// tvm-ffi entry point. ``counts`` is ``[L]`` float32 or empty (active set from
// the ids); ``token_gate`` is a one-element int32 non-padded row count or
// empty. ``rank_load`` receives the per-rank active-expert count.
template <typename InT, typename OutT, int RANKS>
void metro_repair_route(tvm::ffi::TensorView logical_topk_ids,
                        tvm::ffi::TensorView routed,
                        tvm::ffi::TensorView counts,
                        tvm::ffi::TensorView token_gate,
                        tvm::ffi::TensorView holder_mask,
                        tvm::ffi::TensorView physical_by_rank,
                        tvm::ffi::TensorView default_physical,
                        tvm::ffi::TensorView rank_load,
                        int64_t max_moves,
                        bool early_exit) {
  using namespace host;

  SymbolicDevice device_;
  auto num_tokens = SymbolicSize{"num_tokens"};
  auto top_k = SymbolicSize{"top_k"};
  auto num_logical = SymbolicSize{"num_logical_experts"};
  auto num_ranks = SymbolicSize{"num_ranks"};
  auto num_counts = SymbolicSize{"num_counts"};
  auto num_gate = SymbolicSize{"num_gate"};
  TensorMatcher({num_tokens, top_k}).with_device<kDLCUDA>(device_).with_dtype<InT>().verify(logical_topk_ids);
  TensorMatcher({num_tokens, top_k}).with_device<kDLCUDA>(device_).with_dtype<OutT>().verify(routed);
  TensorMatcher({num_logical}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(holder_mask);
  TensorMatcher({num_logical, num_ranks}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(physical_by_rank);
  TensorMatcher({num_logical}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(default_physical);
  TensorMatcher({num_ranks}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(rank_load);
  TensorMatcher({num_counts}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(counts);
  TensorMatcher({num_gate}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(token_gate);

  const int64_t logical = num_logical.unwrap();
  const int64_t ranks = num_ranks.unwrap();
  const int64_t tokens = num_tokens.unwrap();
  const int64_t k = top_k.unwrap();
  RuntimeCheck(logical > 0 && logical <= kMetroRepairMaxLogical,
               "METRO-repair supports 1..", kMetroRepairMaxLogical, " logical experts, got ", logical);
  RuntimeCheck(ranks == RANKS, "num_ranks ", ranks, " does not match the module compiled for ", RANKS);
  RuntimeCheck(num_counts.unwrap() == 0 || num_counts.unwrap() == logical, "counts must be empty or [num_logical]");
  RuntimeCheck(num_gate.unwrap() <= 1, "token_gate must be empty or one element");
  RuntimeCheck(max_moves >= 0 && max_moves <= 4096, "max_moves out of range: ", max_moves);
  RuntimeCheck(k > 0 && k <= INT_MAX, "invalid top_k");
  RuntimeCheck(logical_topk_ids.data_ptr() != routed.data_ptr() || tokens == 0,
               "logical_topk_ids and routed must not alias");

  const DLDevice device = device_.unwrap();
  LaunchKernel(/*grid=*/1, /*block=*/kMetroRepairThreads, device)(
      metro_repair_route_kernel<InT, OutT, RANKS>,
      static_cast<const InT*>(logical_topk_ids.data_ptr()),
      static_cast<OutT*>(routed.data_ptr()),
      num_counts.unwrap() > 0 ? static_cast<const float*>(counts.data_ptr()) : nullptr,
      num_gate.unwrap() > 0 ? static_cast<const int32_t*>(token_gate.data_ptr()) : nullptr,
      static_cast<const int32_t*>(holder_mask.data_ptr()),
      static_cast<const int32_t*>(physical_by_rank.data_ptr()),
      static_cast<const int32_t*>(default_physical.data_ptr()),
      static_cast<int32_t*>(rank_load.data_ptr()),
      tokens * k,
      static_cast<int>(k),
      static_cast<int>(logical),
      static_cast<int>(max_moves),
      early_exit);
}

}  // namespace
