#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace nn {

inline int blocks_for(int64_t n, int threads = 256) {
  if (n <= 0) return 1;
  const int64_t blocks = (n + threads - 1) / threads;
  if (blocks > 0x7fffffffLL) return 0x7fffffff;
  return static_cast<int>(blocks);
}

inline bool aligned16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 15u) == 0;
}

}  // namespace nn
