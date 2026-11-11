// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tile-aware Waterfill: each token's shared-expert rank, decided in whole GEMM
// tiles from the EP-wide per-expert counts of this layer (host mirror for the
// tests: tests/_waterfill_tile_reference.py).
//
// K1 twf_count_exchange: one CTA counts this rank's routed rows per physical
//   expert into its payload row -- [0, P) counts of the first nv tokens, [P]
//   nv, [P + 1] tile rows, zero padding to W = round_up(P + 2, 4) -- and, with
//   peer buffers, all-gathers every rank's row over NVLink (protocol in
//   evenmoe/kernels/p2p.py); without them it only writes the row. After
//   kTwfMaxFailures consecutive failed exchanges (a rank that keeps skipping
//   calls) the process stops exchanging and keeps rows local.
// K2 twf_plan_assign: every CTA computes the same plan, then assigns its
//   tokens. T = the largest tile reported by a rank with tokens (this rank's
//   own after a failed exchange). Rank r costs 4T * tiles_r + wr4 * rows_r
//   (routed tiles sum_p ceil(g_p / T) and routed rows; slot p is on rank
//   p / per). The sum(nv) shared rows become full tiles plus at most one
//   partial tile; each goes to the smallest key (cls, cost, rank): cls 0 =
//   stays within the busiest rank's routed tiles and rows, 1 = within its
//   tiles, 2 = anyone. The resulting quotas cut all shared rows, interleaved
//   round-robin by source rank (s0 row 0, s1 row 0, ..., s0 row 1, ...), in
//   rank order, so every source spreads its rows over the receivers (balanced
//   rank-pair messages under DeepEP). A failed exchange, or a cost beyond the
//   25-bit key, keeps every row local, which is always valid.

#pragma once

#include <evenmoe_jit/tensor.h>
#include <evenmoe_jit/utils.h>

#include <evenmoe_jit/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int kTwfThreads = 512;      // K1
constexpr int kTwfPlanThreads = 256;  // K2
constexpr int kTwfMaxRanks = 32;      // one warp lane per rank
constexpr int kTwfValueMax = 0xFFFF;  // 16-bit payload values on the wire
constexpr int kTwfFailTimeout = 1, kTwfFailOverwritten = 2, kTwfFailOverflow = 4, kTwfFailDisabled = 8;
constexpr int kTwfMaxFailures = 8;  // consecutive failed exchanges before this process stops exchanging
constexpr unsigned kTwfCostLimit = 1u << 25;  // key = cls(2) | cost(25) | rank(5)
constexpr int64_t kTwfMaxSharedBytes = 48 * 1024;
constexpr unsigned kFull = 0xFFFFFFFFu;

__host__ __device__ constexpr int twf_payload_width(int num_physical) {
  return (num_physical + 2 + 3) / 4 * 4;
}

__device__ __forceinline__ void twf_st4_relaxed_sys(int* ptr, int4 v) {
  asm volatile("st.relaxed.sys.global.v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}

__device__ __forceinline__ int4 twf_ld4_relaxed_sys(const int* ptr) {
  int4 v;
  asm volatile("ld.relaxed.sys.global.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr)
               : "memory");
  return v;
}

__device__ __forceinline__ unsigned long long twf_globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ bool twf_overflows(int4 v) {
  return v.x >= kTwfValueMax || v.y >= kTwfValueMax || v.z >= kTwfValueMax || v.w >= kTwfValueMax;
}

// 0 when ``word`` carries ``epoch``, -1 while still stale, kTwfFailOverwritten when the peer is past this call.
__device__ __forceinline__ int twf_word_state(int word, int epoch) {
  const unsigned e = static_cast<unsigned>(word) >> 16;
  if (e == static_cast<unsigned>(epoch)) return 0;
  return static_cast<int16_t>(static_cast<uint16_t>(e - static_cast<unsigned>(epoch))) > 0 ? kTwfFailOverwritten : -1;
}

// ------------------------------------------------------- K1: count + exchange
template <typename IdT>
__global__ void __launch_bounds__(kTwfThreads) twf_count_exchange_kernel(const IdT* __restrict__ ids,
                                                                         int64_t num_tokens,
                                                                         int top_k,
                                                                         const int32_t* __restrict__ token_count,
                                                                         int64_t token_count_value,
                                                                         int32_t* __restrict__ payload_out,
                                                                         int32_t* __restrict__ gathered,
                                                                         const int64_t* __restrict__ peer_buffers,
                                                                         int32_t* __restrict__ state,
                                                                         int32_t* __restrict__ status,
                                                                         int num_ranks,
                                                                         int rank,
                                                                         int num_physical,
                                                                         int tile_rows,
                                                                         long long timeout_ns) {
  device::PDLWaitPrimary<true>();  // PDL: no-ops unless the launch attribute is set
  device::PDLTriggerSecondary<true>();
  const int tid = threadIdx.x;
  const int width = twf_payload_width(num_physical);
  extern __shared__ __align__(16) int twf_smem[];
  int* payload = twf_smem;       // [width], 16-byte aligned
  int* hist = twf_smem + width;  // [num_physical]
  __shared__ int failed;
  for (int p = tid; p < num_physical; p += kTwfThreads) hist[p] = 0;
  if (tid == 0) failed = 0;
  __syncthreads();
  int64_t nv = token_count != nullptr ? token_count[0] : token_count_value;
  nv = nv < 0 ? 0 : (nv > num_tokens ? num_tokens : nv);
  for (int64_t e = tid; e < nv * top_k; e += kTwfThreads) {
    const int64_t p = static_cast<int64_t>(ids[e]);
    if (p >= 0 && p < num_physical) atomicAdd(&hist[p], 1);
  }
  __syncthreads();
  for (int j = tid; j < width; j += kTwfThreads) {
    const int w = j < num_physical ? hist[j] : (j == num_physical ? static_cast<int>(nv) : (j == num_physical + 1 ? tile_rows : 0));
    payload[j] = w;
    if (payload_out != nullptr) payload_out[j] = w;
  }
  if (peer_buffers == nullptr) return;
  __syncthreads();
  if (state[1] >= kTwfMaxFailures) {  // peers keep missing calls: stop waiting, keep rows local
    for (int j = tid; j < width; j += kTwfThreads) gathered[rank * width + j] = payload[j];
    if (tid == 0) status[0] = kTwfFailDisabled;
    return;
  }
  const int epoch = (state[0] + 1) & 0xFFFF;
  const int parity = epoch & 1;
  const int groups = width / 4;
  // Publish row ``rank`` into every peer's buffer (and keep it locally).
  for (int idx = tid; idx < num_ranks * groups; idx += kTwfThreads) {
    const int dst = idx / groups, g = idx - dst * groups;
    int4 v = reinterpret_cast<const int4*>(payload)[g];
    if (dst == rank) {
      reinterpret_cast<int4*>(gathered + rank * width)[g] = v;
      if (twf_overflows(v)) atomicOr(&failed, kTwfFailOverflow);
      continue;
    }
    const int tag = epoch << 16;
    v = make_int4(tag | min(max(v.x, 0), kTwfValueMax), tag | min(max(v.y, 0), kTwfValueMax),
                  tag | min(max(v.z, 0), kTwfValueMax), tag | min(max(v.w, 0), kTwfValueMax));
    twf_st4_relaxed_sys(reinterpret_cast<int*>(peer_buffers[dst]) + (parity * num_ranks + rank) * width + 4 * g, v);
  }
  // Gather: poll this rank's own buffer until every peer's words carry the epoch.
  const int* inbox = reinterpret_cast<const int*>(peer_buffers[rank]) + parity * num_ranks * width;
  volatile int* failed_v = &failed;
  const unsigned long long start = twf_globaltimer();
  for (int idx = tid; idx < num_ranks * groups; idx += kTwfThreads) {
    const int src = idx / groups, g = idx - src * groups;
    if (src == rank) continue;
    int4 v;
    int bad = 0;
    while (true) {
      v = twf_ld4_relaxed_sys(inbox + src * width + 4 * g);
      const int a = twf_word_state(v.x, epoch), b = twf_word_state(v.y, epoch);
      const int c = twf_word_state(v.z, epoch), d = twf_word_state(v.w, epoch);
      if ((a | b | c | d) == 0) break;
      if (a > 0 || b > 0 || c > 0 || d > 0) {
        bad = kTwfFailOverwritten;
      } else if (*failed_v != 0) {
        bad = -1;  // another thread gave up: the call keeps every row local anyway
      } else if (static_cast<long long>(twf_globaltimer() - start) > timeout_ns) {
        bad = kTwfFailTimeout;
      }
      if (bad != 0) break;
    }
    if (bad > 0 && atomicOr(&failed, bad) == 0) {
      printf("evenmoe waterfill: P2P count exchange failed on rank %d (epoch %d, peer %d, reason %d); "
             "this call keeps every shared row local\n",
             rank, epoch, src, bad);
    }
    if (bad != 0) continue;
    v = make_int4(v.x & 0xFFFF, v.y & 0xFFFF, v.z & 0xFFFF, v.w & 0xFFFF);
    if (twf_overflows(v)) atomicOr(&failed, kTwfFailOverflow);
    reinterpret_cast<int4*>(gathered + src * width)[g] = v;
  }
  __syncthreads();
  if (tid == 0) {
    status[0] = failed;
    state[0] = epoch;
    state[1] = failed ? state[1] + 1 : 0;
    if (state[1] == kTwfMaxFailures) {
      printf("evenmoe waterfill: WARNING rank %d stops the P2P count exchange after %d consecutive failures "
             "(a rank keeps missing routing calls); shared rows stay local from now on\n",
             rank, kTwfMaxFailures);
    }
  }
}

// ------------------------------------------------------- K2: plan + assign
template <typename OutT>
__global__ void __launch_bounds__(kTwfPlanThreads) twf_plan_assign_kernel(const int32_t* __restrict__ gathered,
                                                                          const int32_t* __restrict__ status,
                                                                          OutT* __restrict__ shared_rank,
                                                                          int64_t num_tokens,
                                                                          int num_ranks,
                                                                          int rank,
                                                                          int width,
                                                                          int num_physical,
                                                                          int per,
                                                                          int wr4) {
  device::PDLWaitPrimary<true>();
  device::PDLTriggerSecondary<true>();
  __shared__ int rt[kTwfMaxRanks], rr[kTwfMaxRanks], nvs[kTwfMaxRanks], quota_end[kTwfMaxRanks + 1];
  __shared__ int tile, local_only;
  const int tid = threadIdx.x;
  const bool failed = status != nullptr && status[0] != 0;
  if (tid < 32) {
    const bool act = tid < num_ranks;
    const int nv = act ? gathered[tid * width + num_physical] : 0;
    const int t = act ? gathered[tid * width + num_physical + 1] : 0;
    rt[tid] = rr[tid] = 0;
    nvs[tid] = nv;
    const unsigned m = __reduce_max_sync(kFull, (act && nv > 0 && t > 0) ? static_cast<unsigned>(t) : 0u);
    if (tid == 0) tile = (failed || m == 0) ? max(gathered[rank * width + num_physical + 1], 1) : static_cast<int>(m);
  }
  __syncthreads();
  const int block_m = tile;
  // Routed tiles and rows per rank.
  for (int base = 0; base < num_physical; base += kTwfPlanThreads) {
    const int p = base + tid;
    int g = 0;
    if (p < num_physical) {
      for (int s = 0; s < num_ranks; ++s) g += gathered[s * width + p];
    }
    const int t = (g + block_m - 1) / block_m;
    if (per % 32 == 0) {  // a whole warp's slots belong to one rank
      const int t_sum = static_cast<int>(__reduce_add_sync(kFull, static_cast<unsigned>(t)));
      const int g_sum = static_cast<int>(__reduce_add_sync(kFull, static_cast<unsigned>(g)));
      if ((tid & 31) == 0 && p < num_physical) {
        if (t_sum) atomicAdd(&rt[p / per], t_sum);
        if (g_sum) atomicAdd(&rr[p / per], g_sum);
      }
    } else if (p < num_physical) {
      if (t) atomicAdd(&rt[p / per], t);
      if (g) atomicAdd(&rr[p / per], g);
    }
  }
  __syncthreads();
  if (tid < 32) {
    // Quotas: lane r is rank r.
    const int lane = tid;
    const bool act = lane < num_ranks;
    int tiles = act ? rt[lane] : 0, rows = act ? rr[lane] : 0;
    const int nv = act ? nvs[lane] : 0;
    const long long wt = 4LL * block_m, inc = wt + static_cast<long long>(block_m) * wr4;
    const int max_tiles = static_cast<int>(__reduce_max_sync(kFull, act ? static_cast<unsigned>(tiles) : 0u));
    const int max_rows = static_cast<int>(__reduce_max_sync(kFull, act ? static_cast<unsigned>(rows) : 0u));
    const int total = static_cast<int>(__reduce_add_sync(kFull, static_cast<unsigned>(nv)));
    const int num_full = total / block_m, rem = total - num_full * block_m;
    const long long cost = act ? wt * tiles + static_cast<long long>(wr4) * rows : 0;
    const long long reach = cost + static_cast<long long>(num_full + 1) * inc;
    const bool too_big = __any_sync(kFull, act && (reach >= static_cast<long long>(kTwfCostLimit) || nv < 0));
    int quota = 0;
    if (failed || too_big) {
      quota = nv;
    } else {
      unsigned c = static_cast<unsigned>(cost);
      for (int it = 0; it < num_full + (rem > 0); ++it) {
        const int q = it < num_full ? block_m : rem;
        const unsigned cls = (tiles + 1 <= max_tiles && rows + q <= max_rows) ? 0u : (tiles + 1 <= max_tiles ? 1u : 2u);
        const unsigned key = act ? ((cls << 30) | (c << 5) | static_cast<unsigned>(lane)) : kFull;
        if (lane == static_cast<int>(__reduce_min_sync(kFull, key) & 31u)) {
          quota += q;
          c += static_cast<unsigned>(inc);
          tiles += 1;
          rows += q;
        }
      }
    }
    // Proportional split: all shared rows, interleaved round-robin by source rank, cut into the quotas in rank order.
    int q = act ? quota : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int qq = __shfl_up_sync(kFull, q, o);
      if (lane >= o) q += qq;
    }
    if (act) quota_end[lane + 1] = q;
    if (lane == 0) quota_end[0] = 0, local_only = failed || too_big;
  }
  __syncthreads();
  const int my_nv = nvs[rank];
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * kTwfPlanThreads + tid; i < num_tokens;
       i += static_cast<int64_t>(gridDim.x) * kTwfPlanThreads) {
    int dest = -1;
    if (i < my_nv && local_only) {
      dest = rank;
    } else if (i < my_nv) {
      // Position of this rank's i-th row in the interleaved order: sum_t min(nv_t, i) + #{t < rank : nv_t > i}.
      int position = 0;
      for (int t = 0; t < num_ranks; ++t) position += min(nvs[t], static_cast<int>(i)) + (t < rank && nvs[t] > i);
      dest = 0;
      for (int r = 1; r < num_ranks; ++r) dest += quota_end[r] <= position;
    }
    shared_rank[i] = static_cast<OutT>(dest);
  }
}

// ----------------------------------------------------------- host entries
template <typename IdT>
void twf_count_exchange(tvm::ffi::TensorView ids,
                        tvm::ffi::TensorView token_count,
                        tvm::ffi::TensorView payload_out,
                        tvm::ffi::TensorView gathered,
                        tvm::ffi::TensorView peer_buffers,
                        tvm::ffi::TensorView state,
                        tvm::ffi::TensorView status,
                        int64_t token_count_value,
                        int64_t num_physical,
                        int64_t tile_rows,
                        int64_t num_ranks,
                        int64_t rank,
                        int64_t buffer_words,
                        int64_t timeout_ns,
                        int64_t pdl) {
  using namespace host;
  SymbolicDevice device_;
  auto num_tokens = SymbolicSize{"num_tokens"}, top_k = SymbolicSize{"top_k"}, num_nv = SymbolicSize{"num_nv"};
  auto payload_words = SymbolicSize{"payload_words"}, num_peers = SymbolicSize{"num_peers"};
  auto gathered_rows = SymbolicSize{"gathered_rows"}, gathered_cols = SymbolicSize{"gathered_cols"};
  auto e1 = SymbolicSize{"state"}, s1 = SymbolicSize{"status"};
  TensorMatcher({num_tokens, top_k}).with_device<kDLCUDA>(device_).with_dtype<IdT>().verify(ids);
  TensorMatcher({num_nv}).with_device<kDLCUDA>(device_).with_dtype<int32_t>().verify(token_count);
  TensorMatcher({payload_words}).with_device<kDLCUDA>(device_).with_dtype<int32_t>().verify(payload_out);
  TensorMatcher({gathered_rows, gathered_cols}).with_device<kDLCUDA>(device_).with_dtype<int32_t>().verify(gathered);
  TensorMatcher({num_peers}).with_device<kDLCUDA>(device_).with_dtype<int64_t>().verify(peer_buffers);
  TensorMatcher({e1}).with_device<kDLCUDA>(device_).with_dtype<int32_t>().verify(state);
  TensorMatcher({s1}).with_device<kDLCUDA>(device_).with_dtype<int32_t>().verify(status);
  const int64_t w = twf_payload_width(static_cast<int>(num_physical));
  const int64_t shared_bytes = (w + num_physical) * static_cast<int64_t>(sizeof(int));
  const bool exchange = num_peers.unwrap() > 0;
  RuntimeCheck(shared_bytes <= kTwfMaxSharedBytes, "waterfill tile mode: ", num_physical, " physical experts need ",
               shared_bytes, " bytes of shared memory for the count, above ", kTwfMaxSharedBytes);
  RuntimeCheck(tile_rows >= 1 && tile_rows < kTwfValueMax, "tile_rows must lie in [1, ", kTwfValueMax, ")");
  RuntimeCheck(num_nv.unwrap() <= 1 && payload_words.unwrap() == (exchange ? 0 : w), "bad token_count / payload");
  RuntimeCheck(!exchange || (num_peers.unwrap() == num_ranks && rank >= 0 && rank < num_ranks &&
                             gathered_rows.unwrap() == num_ranks && gathered_cols.unwrap() == w &&
                             e1.unwrap() == 2 && s1.unwrap() == 1 && buffer_words >= 2 * num_ranks * w),
               "P2P exchange arguments do not match the EP group or the payload width");
  LaunchKernel(1, kTwfThreads, device_.unwrap(), static_cast<std::size_t>(shared_bytes)).enable_pdl(pdl != 0)(
      twf_count_exchange_kernel<IdT>, static_cast<const IdT*>(ids.data_ptr()), num_tokens.unwrap(),
      static_cast<int>(top_k.unwrap()), num_nv.unwrap() == 1 ? static_cast<const int32_t*>(token_count.data_ptr()) : nullptr,
      token_count_value, exchange ? nullptr : static_cast<int32_t*>(payload_out.data_ptr()),
      exchange ? static_cast<int32_t*>(gathered.data_ptr()) : nullptr,
      exchange ? static_cast<const int64_t*>(peer_buffers.data_ptr()) : nullptr,
      exchange ? static_cast<int32_t*>(state.data_ptr()) : nullptr,
      exchange ? static_cast<int32_t*>(status.data_ptr()) : nullptr, static_cast<int>(num_ranks), static_cast<int>(rank),
      static_cast<int>(num_physical), static_cast<int>(tile_rows), static_cast<long long>(timeout_ns));
}

template <typename OutT>
void twf_plan_assign(tvm::ffi::TensorView gathered,
                     tvm::ffi::TensorView status,
                     tvm::ffi::TensorView shared_rank,
                     int64_t num_physical,
                     int64_t per,
                     int64_t wr4,
                     int64_t rank,
                     int64_t pdl) {
  using namespace host;
  SymbolicDevice device_;
  auto width = SymbolicSize{"width"}, num_ranks = SymbolicSize{"num_ranks"};
  auto num_status = SymbolicSize{"num_status"}, num_tokens = SymbolicSize{"num_tokens"};
  TensorMatcher({num_ranks, width}).with_device<kDLCUDA>(device_).with_dtype<int32_t>().verify(gathered);
  TensorMatcher({num_status}).with_device<kDLCUDA>(device_).with_dtype<int32_t>().verify(status);
  TensorMatcher({num_tokens}).with_device<kDLCUDA>(device_).with_dtype<OutT>().verify(shared_rank);
  const int64_t ranks = num_ranks.unwrap(), n = num_tokens.unwrap();
  RuntimeCheck(ranks > 0 && ranks <= kTwfMaxRanks, "waterfill tile mode supports 1..", kTwfMaxRanks, " ranks");
  RuntimeCheck(rank >= 0 && rank < ranks && per > 0 && num_physical == ranks * per && num_status.unwrap() <= 1 &&
                   width.unwrap() == twf_payload_width(static_cast<int>(num_physical)),
               "plan arguments do not match the gathered rows");
  RuntimeCheck(wr4 >= 0 && wr4 < 4096, "tile_rho out of range");
  const int64_t blocks = std::max<int64_t>(1, std::min<int64_t>((n + kTwfPlanThreads - 1) / kTwfPlanThreads, 132));
  LaunchKernel(static_cast<int>(blocks), kTwfPlanThreads, device_.unwrap()).enable_pdl(pdl != 0)(
      twf_plan_assign_kernel<OutT>, static_cast<const int32_t*>(gathered.data_ptr()),
      num_status.unwrap() == 1 ? static_cast<const int32_t*>(status.data_ptr()) : nullptr,
      static_cast<OutT*>(shared_rank.data_ptr()), n, static_cast<int>(ranks),
      static_cast<int>(rank), static_cast<int>(width.unwrap()), static_cast<int>(num_physical), static_cast<int>(per),
      static_cast<int>(wr4));
}

}  // namespace
