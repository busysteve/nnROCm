#pragma once

#include <hip/hip_runtime.h>

namespace nn {

// Width is a compile-time constant: __shfl_xor requires one, and gfx1152 is wave32.
#if defined(__AMDGCN_WAVEFRONT_SIZE)
constexpr int kWave = __AMDGCN_WAVEFRONT_SIZE;
#else
constexpr int kWave = 32;
#endif

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int offset = kWave / 2; offset > 0; offset >>= 1) v += __shfl_xor(v, offset, kWave);
  return v;
}

__device__ __forceinline__ float wave_max(float v) {
#pragma unroll
  for (int offset = kWave / 2; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_xor(v, offset, kWave));
  return v;
}

// buffer holds one float per wave. The trailing sync lets the next reduction reuse it.
__device__ __forceinline__ float block_sum(float v, float* buffer) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % kWave;
  const int wid = tid / kWave;
  const int nwarps = (static_cast<int>(blockDim.x) + kWave - 1) / kWave;
  v = wave_sum(v);
  if (lane == 0) buffer[wid] = v;
  __syncthreads();
  v = (tid < nwarps) ? buffer[tid] : 0.f;
  if (wid == 0) v = wave_sum(v);
  if (tid == 0) buffer[0] = v;
  __syncthreads();
  v = buffer[0];
  __syncthreads();
  return v;
}

__device__ __forceinline__ float block_max(float v, float* buffer) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % kWave;
  const int wid = tid / kWave;
  const int nwarps = (static_cast<int>(blockDim.x) + kWave - 1) / kWave;
  v = wave_max(v);
  if (lane == 0) buffer[wid] = v;
  __syncthreads();
  v = (tid < nwarps) ? buffer[tid] : -INFINITY;
  if (wid == 0) v = wave_max(v);
  if (tid == 0) buffer[0] = v;
  __syncthreads();
  v = buffer[0];
  __syncthreads();
  return v;
}

}  // namespace nn
