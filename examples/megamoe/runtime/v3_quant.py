# SPDX-License-Identifier: Apache-2.0
"""Experimental V3 FP8 -> NVFP4 routed-expert conversion (no calibration).

The source is the stored V3 FP8 checkpoint, not unavailable BF16 master weights.
Per 16 K values: E2M1 codes and E4M3 scales. One power-of-two FP32 global
scale per expert/matrix; gate/up share it. Scale range fits FlashInfer's SM90
decoder. Conversion is load-time only; attention/shared experts stay unchanged.
"""

import torch

LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequant_fp8(weight, scales):
    n, k = weight.shape
    assert tuple(scales.shape) == (n // 128, k // 128)
    return (
        weight.float().reshape(n // 128, 128, k // 128, 128) * scales.float()[:, None, :, None]
    ).reshape(n, k)


def quantize_matrix(weight):
    """Return Marlin-order packed bytes, raw E4M3 scales and scalar global."""
    n, k = weight.shape
    assert n % 128 == k % 128 == 0 and torch.isfinite(weight).all()
    w = weight.float().reshape(n, k // 16, 16)
    # Max local scale <=64, leaving FP4*scale <=384<448.
    g = torch.exp2(torch.ceil(torch.log2(w.abs().amax().clamp_min(1e-30) / 384)))
    local = (w.abs().amax(-1) / (6 * g)).to(torch.float8_e4m3fn)
    # Nonzero underflow groups use the smallest positive E4M3 scale.
    sf = local.float()
    sf = torch.where((sf == 0) & (w.abs().amax(-1) != 0), 2.0**-9, sf)
    local = sf.to(torch.float8_e4m3fn)
    z = w / (g * sf.clamp_min(2.0**-9))[..., None]
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=w.device)
    mag = torch.bucketize(z.abs().contiguous(), mids)
    # E2M1 round-to-nearest ties-to-even, including the 0/.5 tie.
    tie = (z.abs() == mids[mag.clamp_max(6)]) & (mag < 7)
    mag = mag + (tie & ((mag & 1) != 0)).long()
    codes = (mag.to(torch.uint8) | ((z < 0).to(torch.uint8) << 3)).reshape(n, k // 8, 8)
    packed = ((codes[..., :4] << 4) | codes[..., 4:]).reshape(n, k // 2)
    assert (local.float() <= 64).all()
    return packed.contiguous(), local.view(torch.uint8).contiguous(), g


def dequant_nvfp4(packed, scales, global_scale, decoder_round=True):
    """Independent byte decode; include SM90 E4M3 intermediate rounding."""
    n, halfk = packed.shape
    words = packed.reshape(n, -1, 4)
    codes = torch.cat((words >> 4, words & 15), -1).reshape(n, halfk * 2)
    levels = torch.tensor(LEVELS, device=packed.device)
    vals = levels[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0)
    vals *= scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    if decoder_round:
        vals = vals.to(torch.float8_e4m3fn).float()
    return vals * global_scale


def convert_experts(weight, scales):
    e, n, k = weight.shape
    packed = torch.empty((e, n, k // 2), device=weight.device, dtype=torch.uint8)
    sf = torch.empty((e, n, k // 16), device=weight.device, dtype=torch.uint8)
    gs = torch.empty(e, device=weight.device, dtype=torch.float32)
    err = torch.zeros(2, device=weight.device, dtype=torch.float64)
    for i in range(e):
        original = dequant_fp8(weight[i], scales[i])
        packed[i], sf[i], gs[i] = quantize_matrix(original)
        decoded = dequant_nvfp4(packed[i], sf[i], gs[i])
        err[0] += (decoded - original).double().square().sum()
        err[1] += original.double().square().sum()
    return (packed, sf), gs, float((err[0] / err[1].clamp_min(1e-30)).sqrt())
