#include "activate.hpp"

#include "check.hpp"
#include "context_impl.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace nn {
namespace {

int blocks_for(int64_t n) { return static_cast<int>((n + 255) / 256); }

__device__ uint16_t dev_f32_to_bf16(float value) {
  uint32_t bits = __float_as_uint(value);
  const uint32_t exp = bits & 0x7f800000u;
  if (exp == 0x7f800000u) return static_cast<uint16_t>(bits >> 16);
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

__device__ float dev_bf16_to_f32(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__global__ void cast_to_bf16_kernel(uint16_t* dst, const float* src, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = dev_f32_to_bf16(src[i]);
}

__global__ void cast_to_f32_kernel(float* dst, const uint16_t* src, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = dev_bf16_to_f32(src[i]);
}

__global__ void add_bias_bf16_kernel(uint16_t* y, const float* bias, int rows, int cols) {
  const int64_t n = static_cast<int64_t>(rows) * cols;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  y[i] = dev_f32_to_bf16(dev_bf16_to_f32(y[i]) + bias[i % cols]);
}

__global__ void bias_grad_bf16_kernel(const uint16_t* grad_y, float* grad_bias, int rows, int cols) {
  const int col = static_cast<int>(blockIdx.x);
  if (col >= cols) return;
  float sum = 0.f;
  for (int row = static_cast<int>(threadIdx.x); row < rows; row += static_cast<int>(blockDim.x)) {
    sum += dev_bf16_to_f32(grad_y[static_cast<int64_t>(row) * cols + col]);
  }
  __shared__ float buffer[256];
  buffer[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = static_cast<int>(blockDim.x) / 2; offset > 0; offset >>= 1) {
    if (static_cast<int>(threadIdx.x) < offset) buffer[threadIdx.x] += buffer[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) grad_bias[col] += buffer[0];
}

void* bump(std::vector<ArenaBlock>& blocks, size_t bytes) {
  const size_t align = 256;
  const size_t need = (bytes + align - 1) & ~(align - 1);
  if (need == 0) return nullptr;
  for (ArenaBlock& block : blocks) {
    if (block.used + need <= block.bytes) {
      void* ptr = static_cast<char*>(block.ptr) + block.used;
      block.used += need;
      return ptr;
    }
  }
  size_t cap = 1u << 20;
  if (!blocks.empty() && blocks.back().bytes > cap) cap = blocks.back().bytes * 2;
  if (cap < need) cap = need;
  ArenaBlock block;
  check_hip(hipMalloc(&block.ptr, cap), "hipMalloc");
  block.bytes = cap;
  block.used = need;
  void* ptr = block.ptr;
  blocks.push_back(block);
  return ptr;
}

void gemm_bf16(Context& ctx, hipblasOperation_t trans_a, hipblasOperation_t trans_b, int m, int n, int k,
               const void* A, int lda, int64_t stride_a, const void* B, int ldb, int64_t stride_b,
               void* C, bool c_bf16, int ldc, int64_t stride_c, int batches, float alpha, float beta) {
  if (m <= 0 || n <= 0 || batches <= 0) return;
  const hipDataType ctype = c_bf16 ? HIP_R_16BF : HIP_R_32F;
  check_hipblas(hipblasGemmStridedBatchedEx(hipblas_of(ctx), trans_a, trans_b, m, n, k, &alpha, A,
                                            HIP_R_16BF, lda, static_cast<hipblasStride>(stride_a), B,
                                            HIP_R_16BF, ldb, static_cast<hipblasStride>(stride_b), &beta, C,
                                            ctype, ldc, static_cast<hipblasStride>(stride_c), batches,
                                            HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
                "hipblasGemmStridedBatchedEx");
}

}  // namespace

void arena_reset(Context& ctx) {
  for (ArenaBlock& block : ContextAccess::get(ctx).f32_arena) block.used = 0;
  for (ArenaBlock& block : ContextAccess::get(ctx).bf16_arena) block.used = 0;
}

float* arena_f32(Context& ctx, int64_t n) {
  if (n <= 0) return nullptr;
  return static_cast<float*>(bump(ContextAccess::get(ctx).f32_arena, static_cast<size_t>(n) * sizeof(float)));
}

void zero_f32(Context& ctx, float* dst, int64_t n) {
  if (n <= 0 || dst == nullptr) return;
  check_hip(hipMemsetAsync(dst, 0, static_cast<size_t>(n) * sizeof(float), stream_of(ctx)),
            "hipMemsetAsync");
}

uint16_t* arena_bf16(Context& ctx, int64_t n) {
  if (n <= 0) return nullptr;
  return static_cast<uint16_t*>(
      bump(ContextAccess::get(ctx).bf16_arena, static_cast<size_t>(n) * sizeof(uint16_t)));
}

void cast_f32_to_bf16(Context& ctx, uint16_t* dst, const float* src, int64_t n) {
  if (n <= 0) return;
  cast_to_bf16_kernel<<<blocks_for(n), 256, 0, stream_of(ctx)>>>(dst, src, n);
  check_hip(hipGetLastError(), "cast_f32_to_bf16");
}

void cast_bf16_to_f32(Context& ctx, float* dst, const uint16_t* src, int64_t n) {
  if (n <= 0) return;
  cast_to_f32_kernel<<<blocks_for(n), 256, 0, stream_of(ctx)>>>(dst, src, n);
  check_hip(hipGetLastError(), "cast_bf16_to_f32");
}

void load_f32(Context& ctx, float* dst, const Tensor& src) {
  if (src.dtype() == DType::Float32) {
    check_hip(hipMemcpyAsync(dst, src.data(), static_cast<size_t>(src.numel()) * sizeof(float),
                             hipMemcpyDeviceToDevice, stream_of(ctx)),
              "hipMemcpyAsync");
    return;
  }
  cast_bf16_to_f32(ctx, dst, src.bf16_data(), src.numel());
}

void store_tensor(Context& ctx, Tensor& dst, const float* src) {
  if (dst.dtype() == DType::Float32) {
    check_hip(hipMemcpyAsync(dst.data(), src, static_cast<size_t>(dst.numel()) * sizeof(float),
                             hipMemcpyDeviceToDevice, stream_of(ctx)),
              "hipMemcpyAsync");
    return;
  }
  cast_f32_to_bf16(ctx, dst.bf16_data(), src, dst.numel());
}

void gemm_nt_bf16(Context& ctx, void* C, bool c_bf16, const uint16_t* A, const uint16_t* B, int M, int N,
                  int K, int batches, int ldc, int lda, int ldb, int64_t stride_c, int64_t stride_a,
                  int64_t stride_b, float alpha, float beta) {
  gemm_bf16(ctx, HIPBLAS_OP_T, HIPBLAS_OP_N, N, M, K, B, ldb, stride_b, A, lda, stride_a, C, c_bf16, ldc,
            stride_c, batches, alpha, beta);
}

void gemm_nn_bf16(Context& ctx, void* C, bool c_bf16, const uint16_t* A, const uint16_t* B, int M, int N,
                  int K, int batches, int ldc, int lda, int ldb, int64_t stride_c, int64_t stride_a,
                  int64_t stride_b, float alpha, float beta) {
  gemm_bf16(ctx, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, B, ldb, stride_b, A, lda, stride_a, C, c_bf16, ldc,
            stride_c, batches, alpha, beta);
}

void gemm_tn_bf16(Context& ctx, void* C, bool c_bf16, const uint16_t* A, const uint16_t* B, int M, int N,
                  int K, int batches, int ldc, int lda, int ldb, int64_t stride_c, int64_t stride_a,
                  int64_t stride_b, float alpha, float beta) {
  gemm_bf16(ctx, HIPBLAS_OP_N, HIPBLAS_OP_T, N, M, K, B, ldb, stride_b, A, lda, stride_a, C, c_bf16, ldc,
            stride_c, batches, alpha, beta);
}

void launch_add_bias_bf16(Context& ctx, uint16_t* y, const float* bias, int rows, int cols) {
  const int64_t n = static_cast<int64_t>(rows) * cols;
  if (n <= 0) return;
  add_bias_bf16_kernel<<<blocks_for(n), 256, 0, stream_of(ctx)>>>(y, bias, rows, cols);
  check_hip(hipGetLastError(), "add_bias_bf16");
}

void launch_bias_grad_bf16(Context& ctx, const uint16_t* grad_y, float* grad_bias, int rows, int cols) {
  if (cols <= 0) return;
  bias_grad_bf16_kernel<<<cols, 256, 0, stream_of(ctx)>>>(grad_y, grad_bias, rows, cols);
  check_hip(hipGetLastError(), "bias_grad_bf16");
}

}  // namespace nn
