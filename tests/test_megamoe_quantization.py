# SPDX-License-Identifier: Apache-2.0
"""CPU gates for lossy format conversion and V3 activation semantics."""

import unittest

import torch

from examples.megamoe.runtime.v3_quant import dequant_fp8, dequant_nvfp4, quantize_matrix


class Conversion(unittest.TestCase):
    def test_exact_representable_signed_codes_and_zero(self):
        values = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        )
        w = values.repeat(128, 8)
        p, s, g = quantize_matrix(w)
        self.assertTrue(torch.equal(w, dequant_nvfp4(p, s, g)))
        self.assertEqual(p[0, :4].tolist(), [0x04, 0x15, 0x26, 0x37])
        p, s, g = quantize_matrix(torch.zeros(128, 128))
        self.assertTrue(torch.equal(dequant_nvfp4(p, s, g), torch.zeros(128, 128)))

    def test_fp8_block_scale_axes(self):
        w = torch.ones((256, 256)).to(torch.float8_e4m3fn)
        sf = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = dequant_fp8(w, sf)
        self.assertEqual(out[127, 128].item(), 2.0)
        self.assertEqual(out[128, 127].item(), 3.0)

    def test_rne_ties_and_decoder_bounds(self):
        w = torch.zeros(128, 128)
        w[0, :8] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
        p, s, g = quantize_matrix(w)
        actual = dequant_nvfp4(p, s, g)[0, :8]
        self.assertTrue(torch.equal(actual, torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0])))
        self.assertTrue((s.view(torch.float8_e4m3fn).float() * 6 <= 448).all())

    def test_random_weight_error_is_bounded(self):
        torch.manual_seed(21)
        w = torch.randn(256, 512)
        p, s, g = quantize_matrix(w)
        out = dequant_nvfp4(p, s, g)
        rms = ((out - w).square().sum() / w.square().sum()).sqrt()
        self.assertLess(rms.item(), 0.15)
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
