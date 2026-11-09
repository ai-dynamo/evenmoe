// SPDX-License-Identifier: Apache-2.0
// Derived from SGLang Waterfill demo a85e2a43307d436477d096647066cd4899b64f87.
// Tile-aware Waterfill for MegaMoE (SM90): decides, inside the CUDA graph and without host sync, which rank's shared-expert slot
// every token's shared row goes to (column K of a K+1 top-k; global id = dest * (per+1) + per).
// ONE single-CTA kernel per MoE layer:
//   1. static logical->physical map (dm), local per-physical-expert counts (smem atomics), #valid tokens
//   2. NVLink P2P all-gather of [P counts | #valid] (LL protocol: every 32-bit word = epoch16<<16 | count16, parity double buffer,
//      128-bit st/ld.relaxed.sys), identical to wf/wf_tile.cu
//   3. every rank runs the identical integer plan (warp 0, lane r = rank r, 32-bit keys cls|cost|rank, redux.min): N = sum(valid) shared rows -> floor(N/64) full tiles +
//      one partial tile; each tile goes to the cheapest rank, cost = WT*tiles + WR*rows (routed + shared already given); with caps a
//      receiver that stays <= the routed-only max tiles AND max rows is preferred, then <= max tiles, then anyone. mode 1 (TWFC)
//      first fills receivers cheapest-first up to the caps (fewest receivers). mode 3: everything local (B).
//   4. local-first prefix realization: rank s keeps k_s = min(S_s, nv_s) of its own tokens (the first k_s), surplus tokens of all
//      sources (rank order) are matched to receivers' deficits (rank order) -> every rank knows its tokens' destinations.
//   5. writes topk ids [n, K+1] (routed p -> p + p/per, shared -> dest*(per+1)+per, invalid -> -1) and weights [n, K+1].
// Reference: ref.py (plan_twf / plan_twfc / realize), bit-exact.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

constexpr int NT = 512;
constexpr int BMR = 64;
constexpr int MAXP = 288;
constexpr int MAXL = 512;

__device__ __forceinline__ void st4_relaxed_sys(int* p, int4 v) {
  asm volatile("st.relaxed.sys.global.v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ int4 ld4_relaxed_sys(const int* p) {
  int4 v; asm volatile("ld.relaxed.sys.global.v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}

template <int R, typename IT>
__global__ void __launch_bounds__(NT, 1) tilewf_kernel(
    const int* __restrict__ topk, const float* __restrict__ w_in, int n, int K, const int* __restrict__ nvalid_ptr,
    const int64_t* __restrict__ defmap, int num_logical, int P, int per, float shared_w, int WT, int WR, int mode,
    const int64_t* __restrict__ peer_bufs, int me, int* __restrict__ epoch_ctr, int xchg, const int* __restrict__ xin,
    IT* __restrict__ ids_out, float* __restrict__ w_out, int64_t* __restrict__ phys_out, int* __restrict__ plan_out, long long* __restrict__ dbg) {
  const int W4 = (P + 1 + 3) & ~3;
  __shared__ int dm[MAXL], lcnt[MAXP];
  __shared__ __align__(16) int xs[R * ((MAXP + 1 + 3) & ~3)];
  __shared__ int rt[R], rr[R], nvs[R], Sfin[R], Ks[R], Uc[R + 1], Dc[R + 1];
  const int tid = threadIdx.x;
  long long t0 = 0;
  if (dbg && tid == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
#define STAMP(k) if (dbg && tid == 0) { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); dbg[k] = t - t0; }
  const int epoch = *epoch_ctr + 1, par = epoch & 1, tag = (epoch & 0xFFFF) << 16;
  const int nv_rows = nvalid_ptr ? max(0, min(n, *nvalid_ptr)) : n;
  const int new_per = per + 1;
  const bool ident = defmap == nullptr;

  if (!ident) for (int i = tid; i < num_logical; i += NT) dm[i] = (int)defmap[i];
  for (int i = tid; i < P; i += NT) lcnt[i] = 0;
  if (tid < R) { rt[tid] = 0; rr[tid] = 0; }
  __syncthreads();
  STAMP(0);
  for (int e = tid; e < nv_rows * K; e += NT) {  // local counts
    const int l = topk[e];
    const int p = (l >= 0 && l < num_logical) ? (ident ? l : dm[l]) : -1;
    if (p >= 0) atomicAdd(&lcnt[p], 1);
  }
  __syncthreads();
  for (int j = tid; j < W4; j += NT) xs[me * W4 + j] = j < P ? lcnt[j] : (j == P ? nv_rows : 0);
  __syncthreads();
  STAMP(1);
  if (xchg) {
    int* mybuf = reinterpret_cast<int*>(peer_bufs[me]);
    const int G = W4 / 4;
    for (int idx = tid; idx < R * G; idx += NT) {
      const int dst = idx / G, g = idx - dst * G;
      if (dst == me) continue;
      int4 v = *reinterpret_cast<const int4*>(xs + me * W4 + 4 * g);
      v.x = tag | (v.x & 0xFFFF); v.y = tag | (v.y & 0xFFFF); v.z = tag | (v.z & 0xFFFF); v.w = tag | (v.w & 0xFFFF);
      st4_relaxed_sys(reinterpret_cast<int*>(peer_bufs[dst]) + (par * R + me) * W4 + 4 * g, v);
    }
    long long tstart; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tstart));
    for (int idx = tid; idx < R * G; idx += NT) {
      const int s = idx / G, g = idx - s * G;
      if (s == me) continue;
      const int* src = mybuf + (par * R + s) * W4 + 4 * g; int4 v;
      while (true) {
        v = ld4_relaxed_sys(src);
        if ((v.x & 0xFFFF0000) == tag && (v.y & 0xFFFF0000) == tag && (v.z & 0xFFFF0000) == tag && (v.w & 0xFFFF0000) == tag) break;
        long long tn; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tn));
        if (tn - tstart > 2000000000LL) { if (g == 0) printf("tilewf: exchange timeout rank %d from %d epoch %d\n", me, s, epoch); asm volatile("trap;"); return; }
      }
      v.x &= 0xFFFF; v.y &= 0xFFFF; v.z &= 0xFFFF; v.w &= 0xFFFF;
      *reinterpret_cast<int4*>(xs + s * W4 + 4 * g) = v;
    }
  } else {  // test mode: other sources' rows from xin [R, W4]
    for (int idx = tid; idx < R * W4; idx += NT) { const int s = idx / W4; if (s != me) xs[idx] = xin[idx]; }
  }
  __syncthreads();
  STAMP(2);
  for (int p = tid; p < P; p += NT) {  // global counts -> per-rank routed tiles / rows
    int g = 0;
#pragma unroll
    for (int s = 0; s < R; ++s) g += xs[s * W4 + p];
    const int tl = (g + BMR - 1) / BMR;
    if ((per & 31) == 0 && (P & 31) == 0) {
      const int wt = (int)__reduce_add_sync(0xFFFFFFFFu, (unsigned)tl), wr = (int)__reduce_add_sync(0xFFFFFFFFu, (unsigned)g);
      if ((tid & 31) == 0) { if (wt) atomicAdd(&rt[p / per], wt); if (wr) atomicAdd(&rr[p / per], wr); }
    } else { if (tl) atomicAdd(&rt[p / per], tl); if (g) atomicAdd(&rr[p / per], g); }
  }
  if (tid < R) nvs[tid] = xs[tid * W4 + P];
  __syncthreads();
  STAMP(3);
  if (tid < 32) {  // plan: warp 0, lane r = rank r; 32-bit keys + redux.min (integer-exact with ref.py)
    const unsigned FULL = 0xFFFFFFFFu;
    const int lane = tid;
    const bool act = lane < R;
    int T = act ? rt[lane] : 0, rows = act ? rr[lane] : 0;
    int c = WT * T + WR * rows;                          // < 2^26 for any realistic shape (checked on the host side)
    const int inc = WT + BMR * WR;
    const int Mt = (int)__reduce_max_sync(FULL, act ? (unsigned)T : 0u), Mr = (int)__reduce_max_sync(FULL, act ? (unsigned)rows : 0u);
    const int nv = act ? nvs[lane] : 0;
    const int N = (int)__reduce_add_sync(FULL, (unsigned)nv);
    int Kt = N / BMR; const int rem = N - Kt * BMR;
    int S = 0;
    if (mode == 3) {
      S = nv;
    } else {
      if (mode == 1) {  // TWFC: cheapest-first (ties: lower rank), fill each up to the caps
        bool done = !act;
        for (int j = 0; j < R && Kt > 0; ++j) {
          const unsigned key = done ? 0xFFFFFFFFu : (((unsigned)c << 3) | (unsigned)lane);
          const int r = (int)(__reduce_min_sync(FULL, key) & 7u);
          int k = 0;
          if (lane == r) { k = min(Kt, min(Mt - T, (Mr - rows) / BMR)); k = max(k, 0); S += BMR * k; c += inc * k; T += k; rows += BMR * k; done = true; }
          Kt -= __shfl_sync(FULL, k, r);
        }
      }
      for (int it = 0; it < Kt; ++it) {
        const unsigned cls = mode == 2 ? 0u : ((T + 1 <= Mt && rows + BMR <= Mr) ? 0u : (T + 1 <= Mt ? 1u : 2u));
        const unsigned key = act ? ((cls << 29) | ((unsigned)c << 3) | (unsigned)lane) : 0xFFFFFFFFu;
        const int r = (int)(__reduce_min_sync(FULL, key) & 7u);
        if (lane == r) { S += BMR; c += inc; T += 1; rows += BMR; }
      }
      if (rem) {
        const unsigned cls = mode == 2 ? 0u : ((T + 1 <= Mt && rows + rem <= Mr) ? 0u : (T + 1 <= Mt ? 1u : 2u));
        const unsigned key = act ? ((cls << 29) | ((unsigned)c << 3) | (unsigned)lane) : 0xFFFFFFFFu;
        const int r = (int)(__reduce_min_sync(FULL, key) & 7u);
        if (lane == r) S += rem;
      }
    }
    // prefix sums for the local-first realization (R <= 8 lanes)
    const int k = min(S, nv);
    int u = act ? nv - k : 0, d = act ? S - k : 0;
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) { const int uu = __shfl_up_sync(FULL, u, o), dd = __shfl_up_sync(FULL, d, o); if (lane >= o) { u += uu; d += dd; } }
    if (act) { Sfin[lane] = S; Ks[lane] = k; Uc[lane + 1] = u; Dc[lane + 1] = d; }
    if (lane == 0) { Uc[0] = 0; Dc[0] = 0; }
  }
  __syncthreads();
  STAMP(4);
  if (plan_out && tid < R) plan_out[tid] = Sfin[tid];
  for (int e = tid; e < n * K; e += NT) {  // routed part
    const int i = e / K, k = e - i * K;
    const int l = topk[e];
    const int p = (i < nv_rows && l >= 0 && l < num_logical) ? (ident ? l : dm[l]) : -1;
    if (phys_out) phys_out[e] = p;
    ids_out[(long long)i * (K + 1) + k] = p >= 0 ? (IT)(p + p / per) : (IT)-1;
    w_out[(long long)i * (K + 1) + k] = p >= 0 ? w_in[e] : 0.f;
  }
  const int kme = Ks[me];
  for (int i = tid; i < n; i += NT) {  // shared slot
    int dest = -1;
    if (i < nv_rows) {
      if (i < kme) dest = me;
      else {
        const int p = Uc[me] + (i - kme);
        dest = 0;
#pragma unroll
        for (int r = 1; r < R; ++r) dest += (Dc[r] <= p);
        // receivers with zero deficit have Dc[r]==Dc[r+1]; "count of r>=1 with Dc[r] <= p" = last r with Dc[r] <= p (Dc monotone)
      }
    }
    ids_out[(long long)i * (K + 1) + K] = dest >= 0 ? (IT)(dest * new_per + per) : (IT)-1;
    w_out[(long long)i * (K + 1) + K] = dest >= 0 ? shared_w : 0.f;
  }
  __syncthreads();
  STAMP(5);
  if (tid == 0) *epoch_ctr = epoch;
}

int64_t tilewf_row_words(int64_t P) { return (P + 1 + 3) & ~3LL; }

void tilewf(torch::Tensor topk, torch::Tensor w_in, torch::Tensor nvalid, torch::Tensor defmap, int64_t num_logical, int64_t P, int64_t per,
            double shared_w, int64_t WT, int64_t WR, int64_t mode, torch::Tensor peer_bufs, int64_t me, int64_t R, torch::Tensor epoch_ctr,
            int64_t xchg, torch::Tensor xin, torch::Tensor ids_out, torch::Tensor w_out, torch::Tensor phys_out, torch::Tensor plan_out,
            torch::Tensor dbg) {
  auto stream = c10::cuda::getCurrentCUDAStream();
  const int n = topk.size(0), K = topk.size(1);
  TORCH_CHECK(P <= MAXP && num_logical <= MAXL && P % per == 0 && P / per == R, "tilewf: unsupported shape");
  // key = cls(2b) | cost(26b) | rank(3b); cost = WT*tiles + WR*rows <= 1024*4640 + 64*8*4096*9 < 2^26 for n <= 4096 tokens/rank
  TORCH_CHECK(R <= 8 && WT >= 0 && WT <= 1024 && WR >= 0 && WR <= 64 && n <= 4096, "tilewf: WT/WR/n out of range");
  TORCH_CHECK(topk.is_contiguous() && topk.dtype() == torch::kInt32 && w_in.is_contiguous() && w_in.dtype() == torch::kFloat32);
  TORCH_CHECK(ids_out.is_contiguous() && w_out.is_contiguous() && ids_out.size(1) == K + 1);
  const int* nv = nvalid.numel() ? nvalid.data_ptr<int>() : nullptr;
  const int64_t* dm = defmap.numel() ? defmap.data_ptr<int64_t>() : nullptr;
  int64_t* po = phys_out.numel() ? phys_out.data_ptr<int64_t>() : nullptr;
  int* pl = plan_out.numel() ? plan_out.data_ptr<int>() : nullptr;
  const int* xi = xin.numel() ? xin.data_ptr<int>() : nullptr;
  long long* db = dbg.numel() ? (long long*)dbg.data_ptr<int64_t>() : nullptr;
  const int64_t* pb = peer_bufs.numel() ? peer_bufs.data_ptr<int64_t>() : nullptr;
#define TWF(RR, TT) tilewf_kernel<RR, TT><<<1, NT, 0, stream>>>(topk.data_ptr<int>(), w_in.data_ptr<float>(), n, K, nv, dm, (int)num_logical, \
      (int)P, (int)per, (float)shared_w, (int)WT, (int)WR, (int)mode, pb, (int)me, epoch_ctr.data_ptr<int>(), (int)xchg, xi, \
      ids_out.data_ptr<TT>(), w_out.data_ptr<float>(), po, pl, db)
  const bool i64 = ids_out.dtype() == torch::kInt64;
  if (R == 8) { if (i64) TWF(8, int64_t); else TWF(8, int); }
  else if (R == 4) { if (i64) TWF(4, int64_t); else TWF(4, int); }
  else TORCH_CHECK(false, "tilewf: R must be 4 or 8");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("tilewf", &tilewf); m.def("row_words", &tilewf_row_words); }
