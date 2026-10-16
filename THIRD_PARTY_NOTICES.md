# Third-Party Notices

This repository contains code copied or adapted from the projects below.
Modifications made for EvenMoE are distributed under the repository's
Apache-2.0 license; the original portions remain subject to their original
licenses.

## DeepSeek EPLB

- Source: <https://github.com/deepseek-ai/EPLB>
- License: MIT
- Local code:
  `src/evenmoe/policies/l1/eplb/algorithms/deepseek.py` and
  `src/evenmoe/policies/l1/eplb/algorithms/deepseek_vec.py`
- Copyright: Copyright (c) 2025 DeepSeek

## DeepSeek LPLB

- Source: <https://github.com/deepseek-ai/LPLB>
- License: MIT
- Local code: the Interior Point Method implementation in
  `src/evenmoe/kernels/lplb/csrc/ipm.cuh`
- Copyright: Copyright (c) 2025 DeepSeek

The MIT-licensed components above are distributed under the following terms:

> MIT License
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## SGLang

- Source: <https://github.com/sgl-project/sglang>
- License: Apache License 2.0
- Copyright: Copyright 2023-2026 SGLang Team
- Local code includes the LPLB policy and CUDA/JIT support under
  `src/evenmoe/policies/l2/lplb.py`,
  the static, dynamic, and experimental fake replica-routing behavior in
  `src/evenmoe/policies/l2/replica.py`,
  `src/evenmoe/kernels/jit/utils.py`, the JIT kernel helper headers
  in `src/evenmoe/kernels/jit/include/evenmoe_jit/`, and
  `src/evenmoe/kernels/lplb/`, plus the Elastic-EP-aware placement
  policy in
  `src/evenmoe/policies/l1/eplb/algorithms/elasticity_aware.py`.

The Apache License 2.0 text is provided in the repository's `LICENSE` file.

## UltraEP

- Source: <https://github.com/Dots-Infra/UltraEP>
- Version: v1.0.0, commit `94cab099b44fffa99a82fea99e7c12d89cf65e4f`
- License: MIT
- Copyright: Copyright (c) 2026 PKU-DASYS and Dots-Infra
- Local code: placement and sparse quota-rerouting kernels under
  `src/evenmoe/kernels/ultraep/csrc/`, JIT-compiled through EvenMoE's
  tvm-ffi loader (the entry point at the end of each kernel file and
  `csrc/entry.cuh` are EvenMoE's); the UltraEP communication runtime is not copied
  into EvenMoE.

The original MIT license is retained at
`src/evenmoe/kernels/ultraep/LICENSE.ultraep`.
