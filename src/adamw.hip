#include "kernels.hpp"

#include "check.hpp"
#include "launch.hpp"

namespace nn {
namespace {

__device__ void adamw_one(float& param, float& m, float& v, float g, float lr, float beta1, float beta2,
                          float eps, float weight_decay, float bc1, float bc2) {
  const float m_new = beta1 * m + (1.f - beta1) * g;
  const float v_new = beta2 * v + (1.f - beta2) * g * g;
  m = m_new;
  v = v_new;
  const float m_hat = m_new / bc1;
  const float v_hat = v_new / bc2;
  const float update = m_hat / (sqrtf(v_hat) + eps) + weight_decay * param;
  param = param - lr * update;
}

__global__ void __launch_bounds__(256) adamw_kernel(float* param, float* m, float* v, const float* grad,
                                                    int64_t n, float lr, float beta1, float beta2, float eps,
                                                    float weight_decay, float bc1, float bc2) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  adamw_one(param[i], m[i], v[i], grad[i], lr, beta1, beta2, eps, weight_decay, bc1, bc2);
}

__global__ void __launch_bounds__(256) adamw4_kernel(float* param, float* m, float* v, const float* grad,
                                                     int64_t n4, float lr, float beta1, float beta2, float eps,
                                                     float weight_decay, float bc1, float bc2) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  float4 p = reinterpret_cast<float4*>(param)[i];
  float4 mm = reinterpret_cast<float4*>(m)[i];
  float4 vv = reinterpret_cast<float4*>(v)[i];
  const float4 g = reinterpret_cast<const float4*>(grad)[i];
  adamw_one(p.x, mm.x, vv.x, g.x, lr, beta1, beta2, eps, weight_decay, bc1, bc2);
  adamw_one(p.y, mm.y, vv.y, g.y, lr, beta1, beta2, eps, weight_decay, bc1, bc2);
  adamw_one(p.z, mm.z, vv.z, g.z, lr, beta1, beta2, eps, weight_decay, bc1, bc2);
  adamw_one(p.w, mm.w, vv.w, g.w, lr, beta1, beta2, eps, weight_decay, bc1, bc2);
  reinterpret_cast<float4*>(param)[i] = p;
  reinterpret_cast<float4*>(m)[i] = mm;
  reinterpret_cast<float4*>(v)[i] = vv;
}

__global__ void sum_partial_kernel(const float* x, int64_t n, double* partials) {
  double sum = 0.0;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    sum += static_cast<double>(x[i]);
  }
  __shared__ double buffer[256];
  buffer[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = 128; offset > 0; offset >>= 1) {
    if (static_cast<int>(threadIdx.x) < offset) buffer[threadIdx.x] += buffer[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) partials[blockIdx.x] = buffer[0];
}

__global__ void sum_squares_partial_kernel(const float* x, int64_t n, double* partials) {
  // Square in double so a large finite grad does not overflow before the square root.
  double sum = 0.0;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    const double value = static_cast<double>(x[i]);
    sum += value * value;
  }
  __shared__ double buffer[256];
  buffer[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = 128; offset > 0; offset >>= 1) {
    if (static_cast<int>(threadIdx.x) < offset) buffer[threadIdx.x] += buffer[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) partials[blockIdx.x] = buffer[0];
}

__global__ void sum_partials_kernel(const double* partials, int n, double* out) {
  double sum = 0.0;
  for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
    sum += partials[i];
  }
  __shared__ double buffer[256];
  buffer[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = 128; offset > 0; offset >>= 1) {
    if (static_cast<int>(threadIdx.x) < offset) buffer[threadIdx.x] += buffer[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) *out = buffer[0];
}

double reduce_partials(hipStream_t stream, int blocks, double* scratch) {
  double* partials = scratch;
  double* out = scratch + blocks;
  sum_partials_kernel<<<1, 256, 0, stream>>>(partials, blocks, out);
  check_hip(hipGetLastError(), "sum_partials");
  check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize");
  double host = 0.0;
  check_hip(hipMemcpy(&host, out, sizeof(double), hipMemcpyDeviceToHost), "hipMemcpy");
  return host;
}

}  // namespace

void launch_adamw(hipStream_t stream, float* param, float* m, float* v, const float* grad, int64_t n,
                  float lr, float beta1, float beta2, float eps, float weight_decay, float bc1,
                  float bc2) {
  if (n <= 0) return;
  if ((n % 4) == 0 && aligned16(param) && aligned16(m) && aligned16(v) && aligned16(grad)) {
    adamw4_kernel<<<blocks_for(n / 4), 256, 0, stream>>>(param, m, v, grad, n / 4, lr, beta1, beta2, eps,
                                                         weight_decay, bc1, bc2);
  } else {
    adamw_kernel<<<blocks_for(n), 256, 0, stream>>>(param, m, v, grad, n, lr, beta1, beta2, eps,
                                                    weight_decay, bc1, bc2);
  }
  check_hip(hipGetLastError(), "adamw");
}

double launch_sum_squares(hipStream_t stream, const float* x, int64_t n) {
  if (n <= 0) return 0.0;
  int64_t blocks64 = (n + 255) / 256;
  if (blocks64 > 1024) blocks64 = 1024;
  if (blocks64 < 1) blocks64 = 1;
  const int blocks = static_cast<int>(blocks64);
  double* scratch = nullptr;
  check_hip(hipMalloc(&scratch, sizeof(double) * static_cast<size_t>(blocks + 1)), "hipMalloc");
  try {
    sum_squares_partial_kernel<<<blocks, 256, 0, stream>>>(x, n, scratch);
    check_hip(hipGetLastError(), "sum_squares");
    const double host = reduce_partials(stream, blocks, scratch);
    hipFree(scratch);
    return host;
  } catch (...) {
    hipFree(scratch);
    throw;
  }
}

float launch_sum(hipStream_t stream, const float* x, int64_t n) {
  if (n <= 0) return 0.f;
  int64_t blocks64 = (n + 255) / 256;
  if (blocks64 > 1024) blocks64 = 1024;
  if (blocks64 < 1) blocks64 = 1;
  const int blocks = static_cast<int>(blocks64);
  double* scratch = nullptr;
  check_hip(hipMalloc(&scratch, sizeof(double) * static_cast<size_t>(blocks + 1)), "hipMalloc");
  try {
    sum_partial_kernel<<<blocks, 256, 0, stream>>>(x, n, scratch);
    check_hip(hipGetLastError(), "sum");
    const double host = reduce_partials(stream, blocks, scratch);
    hipFree(scratch);
    return static_cast<float>(host);
  } catch (...) {
    hipFree(scratch);
    throw;
  }
}

}  // namespace nn
