#pragma once

#include "nn/context.hpp"

#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace nn {

struct ArenaBlock {
  void* ptr = nullptr;
  size_t bytes = 0;
  size_t used = 0;
};

struct Context::Impl {
  hipStream_t stream = nullptr;
  hipblasHandle_t blas = nullptr;
  DType activation = DType::BFloat16;
  std::vector<ArenaBlock> f32_arena;
  std::vector<ArenaBlock> bf16_arena;
};

struct ContextAccess {
  static Context::Impl& get(Context& ctx) { return *ctx.impl_; }
  static const Context::Impl& get(const Context& ctx) { return *ctx.impl_; }
};

inline hipStream_t stream_of(Context& ctx) { return ContextAccess::get(ctx).stream; }
inline hipStream_t stream_of(const Context& ctx) { return ContextAccess::get(ctx).stream; }
inline hipblasHandle_t hipblas_of(Context& ctx) { return ContextAccess::get(ctx).blas; }

}  // namespace nn
