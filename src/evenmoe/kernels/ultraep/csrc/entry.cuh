// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// EvenMoE glue between the vendored UltraEP kernels and tvm-ffi; not part of
// upstream UltraEP. The kernels take raw pointers and a stream as upstream
// wrote them; the entry point at the end of each kernel header validates its
// ``TensorView`` arguments, resolves the caller's current stream, and calls
// through. What both entry points share lives here.

#pragma once

#include <evenmoe_jit/tensor.h>
#include <evenmoe_jit/utils.h>

#include <evenmoe_jit/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <climits>
#include <cstdint>

namespace mlb_ultraep {

// Make ``device_id`` current for one call and restore the previous device
// afterwards: the vendored launches use the plain runtime API on the resolved
// stream, and that addresses the current device.
class CurrentDeviceGuard {
 public:
  explicit CurrentDeviceGuard(int device_id) {
    host::RuntimeDeviceCheck(cudaGetDevice(&previous_));
    if (previous_ != device_id) {
      host::RuntimeDeviceCheck(cudaSetDevice(device_id));
      switched_ = true;
    }
  }
  ~CurrentDeviceGuard() {
    if (switched_) {
      // Best effort: a destructor has no way to report a failure here.
      static_cast<void>(cudaSetDevice(previous_));
    }
  }
  CurrentDeviceGuard(const CurrentDeviceGuard&) = delete;
  CurrentDeviceGuard& operator=(const CurrentDeviceGuard&) = delete;

 private:
  int previous_ = 0;
  bool switched_ = false;
};

// The stream torch has current for ``device``; tvm-ffi mirrors it into its
// environment on every call.
inline cudaStream_t current_stream(DLDevice device) {
  return host::LaunchKernel::resolve_device(device);
}

// Whether two tensors share storage; a kernel that reads one and writes the
// other must not be handed aliased memory.
inline bool overlaps(tvm::ffi::TensorView a, tvm::ffi::TensorView b) {
  const auto* a_begin = static_cast<const char*>(a.data_ptr());
  const auto* b_begin = static_cast<const char*>(b.data_ptr());
  const auto a_bytes = a.numel() * static_cast<int64_t>(host::dtype_bytes(a.dtype()));
  const auto b_bytes = b.numel() * static_cast<int64_t>(host::dtype_bytes(b.dtype()));
  return a_bytes > 0 && b_bytes > 0 && a_begin < b_begin + b_bytes && b_begin < a_begin + a_bytes;
}

}  // namespace mlb_ultraep
