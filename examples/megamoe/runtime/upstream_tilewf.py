# SPDX-License-Identifier: Apache-2.0
# Derived from SGLang Waterfill demo a85e2a43307d436477d096647066cd4899b64f87.
"""Tile-aware Waterfill (TWF: whole 64-row shared tiles, any receiver) for SGLang + MegaMoE (any static placement, incl. EPLB).
Selected from evenmoe_wf_fused.route with EVM_WF_TWF=1. Input topk_ids/weights [n, K] (logical ids), output top-(K+1):
expanded ids [n, K+1] (routed physical p -> p + p // per, shared column -> dest_rank * (per + 1) + per, -1 = invalid) and weights.
One CUDA kernel per MoE layer (tilewf.cu): counts -> NVLink P2P all-gather -> identical plan on every EP rank -> layout.
Every EP rank must call route() once per MoE layer per forward (also with 0 tokens) because of the exchange.

env: EVM_TWF_MODE  0 = TWF (default; cheapest receiver, caps "stay <= routed max tiles & rows"), 1 = TWFC (fewest receivers),
                    2 = TWF without caps, 3 = all local (B)
     EVM_TWF_RHO   weight of 64 rows in tile units (default 1.5 -> WT=128, WR=3)
     EVM_WF_BUILD  JIT build dir"""

import logging
import os

import torch

logger = logging.getLogger(__name__)
MODE = int(os.environ.get("EVM_TWF_MODE", "0"))
RHO = float(os.environ.get("EVM_TWF_RHO", "1.5"))
WT, WR = 128, round(RHO * 2)
_S = None
_DM = {}


class _State:
    def __init__(self, group=None, me=None, R=None):
        import torch.distributed._symmetric_memory as symm_mem
        from torch.utils.cpp_extension import load

        if group is None:
            from sglang.srt.distributed.parallel_state import get_moe_ep_group

            grp = get_moe_ep_group()
            group, me, R = grp.device_group, grp.rank_in_group, grp.world_size
        self.group, self.me, self.R = group, me, R
        dev = torch.device("cuda", torch.cuda.current_device())
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tilewf.cu")
        bdir = os.environ.get("EVM_WF_BUILD", "/tmp/evm_twf_build")
        os.makedirs(bdir, exist_ok=True)
        self.mod = load(
            "tilewf",
            [src],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"],
            build_directory=bdir,
            verbose=False,
        )
        try:
            symm_mem.enable_symm_mem_for_group(group.group_name)
        except (AttributeError, RuntimeError) as ex:
            logger.info(f"tilewf: enable_symm_mem_for_group skipped ({ex})")
        self.symm_mem = symm_mem
        self.dev = dev
        self.buf = None
        self.W4 = None
        self.epoch = torch.zeros(1, dtype=torch.int32, device=dev)
        self.e32 = torch.empty(0, dtype=torch.int32, device=dev)
        self.e64 = torch.empty(0, dtype=torch.int64, device=dev)

    def ensure_buf(self, P):
        if self.buf is None:
            self.W4 = int(self.mod.row_words(P))
            self.buf = self.symm_mem.empty(2 * self.R * self.W4, dtype=torch.int32, device=self.dev)
            self.buf.zero_()
            hdl = self.symm_mem.rendezvous(self.buf, self.group)
            self.ptrs = torch.tensor(
                [int(x) for x in hdl.buffer_ptrs], dtype=torch.int64, device=self.dev
            )
            torch.cuda.synchronize()
            torch.distributed.barrier(group=self.group)
            logger.info(
                f"tilewf: ready R={self.R} rank={self.me} P={P} row_words={self.W4} mode={MODE} WT={WT} WR={WR}"
            )

    def call(
        self,
        tk,
        tw,
        nv,
        dm,
        num_logical,
        P,
        per,
        sw,
        ids,
        w,
        phys=None,
        mode=None,
        plan=None,
        dbg=None,
        xchg=1,
        xin=None,
    ):
        self.mod.tilewf(
            tk,
            tw,
            nv,
            dm,
            num_logical,
            P,
            per,
            sw,
            WT,
            WR,
            MODE if mode is None else mode,
            self.ptrs,
            self.me,
            self.R,
            self.epoch,
            xchg,
            self.e32 if xin is None else xin,
            ids,
            w,
            self.e64 if phys is None else phys,
            self.e32 if plan is None else plan,
            self.e64 if dbg is None else dbg,
        )


def state(**kw):
    global _S
    if _S is None:
        _S = _State(**kw)
    return _S


def route(
    *,
    live_metadata,
    layer_id,
    ep_size,
    ep_rank,
    topk_ids,
    topk_weights,
    num_token_non_padded,
    routed_scaling_factor,
    K=8,
    ids_dtype=torch.int64,
):
    """returns (expanded_ids [n,K+1], expanded_weights [n,K+1] fp32, physical_ids [n,K] int64 (routed p, -1 invalid))"""
    S = state()
    dm = _DM.get(layer_id)
    if dm is None or dm.device != topk_ids.device:
        rd = getattr(live_metadata, "logical_to_rank_dispatch_physical_map", None)
        src_map = (
            rd[layer_id]
            if rd is not None
            else live_metadata.logical_to_all_physical_map[layer_id][:, 0]
        )
        dm = src_map.to(device=topk_ids.device, dtype=torch.int64).contiguous()
        _DM[layer_id] = dm
    P = int(live_metadata.num_physical_experts)
    per = P // ep_size
    NL = int(dm.shape[0])
    S.ensure_buf(P)
    n = topk_ids.shape[0]
    dev = topk_ids.device
    if n == 0:  # still take part in the exchange
        tk = torch.empty((0, K), dtype=torch.int32, device=dev)
        tw = torch.empty((0, K), dtype=torch.float32, device=dev)
        ids = torch.empty((0, K + 1), dtype=ids_dtype, device=dev)
        w = torch.empty((0, K + 1), dtype=torch.float32, device=dev)
        S.call(tk, tw, S.e32, dm, NL, P, per, 1.0, ids, w)
        return ids, w, torch.empty((0, K), dtype=torch.int64, device=dev)
    ids = torch.empty((n, K + 1), dtype=ids_dtype, device=dev)
    w = torch.empty((n, K + 1), dtype=torch.float32, device=dev)
    tk = (
        topk_ids
        if (topk_ids.dtype == torch.int32 and topk_ids.is_contiguous())
        else topk_ids.to(torch.int32).contiguous()
    )
    tw = (
        topk_weights
        if (topk_weights.dtype == torch.float32 and topk_weights.is_contiguous())
        else topk_weights.float().contiguous()
    )
    nv = (
        num_token_non_padded.to(torch.int32).reshape(-1)[:1]
        if isinstance(num_token_non_padded, torch.Tensor)
        else S.e32
    )
    rsf = routed_scaling_factor if routed_scaling_factor else 1.0
    phys = torch.empty((n, K), dtype=torch.int64, device=dev)
    S.call(tk, tw, nv, dm, NL, P, per, 1.0 / rsf, ids, w, phys)
    return ids, w, phys
