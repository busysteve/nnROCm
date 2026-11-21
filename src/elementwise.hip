#include "kernels.hpp"

#include "check.hpp"
#include "context_impl.hpp"
#include "launch.hpp"
#include "nn/tensor.hpp"

namespace nn {
namespace {

__device__ float gelu_forward(float x) {
  constexpr float kInvSqrt2 = 0.7071067811865476f;
  return 0.5f * x * (1.f + erff(x * kInvSqrt2));
}

__device__ float gelu_grad(float x) {
  constexpr float kInvSqrt2 = 0.7071067811865476f;
  constexpr float kInvSqrt2Pi = 0.3989422804014327f;
  const float cdf = 0.5f * (1.f + erff(x * kInvSqrt2));
  const float pdf = kInvSqrt2Pi * expf(-0.5f * x * x);
  return cdf + x * pdf;
}

__device__ float sigmoid_stable(float x) {
  if (x >= 0.f) {
    const float z = expf(-x);
    return 1.f / (1.f + z);
  }
  const float z = expf(x);
  return z / (1.f + z);
}

__device__ float silu(float x) { return x * sigmoid_stable(x); }

__device__ float silu_grad(float x) {
  const float s = sigmoid_stable(x);
  return s * (1.f + x * (1.f - s));
}

__global__ void __launch_bounds__(256) fill4_kernel(float* x, int64_t n4, float value) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n4) reinterpret_cast<float4*>(x)[i] = make_float4(value, value, value, value);
}

__global__ void __launch_bounds__(256) fill_kernel(float* x, int64_t n, float value) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) x[i] = value;
}

__global__ void __launch_bounds__(256) scale4_kernel(float* x, int64_t n4, float scale) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  float4 v = reinterpret_cast<float4*>(x)[i];
  v.x *= scale;
  v.y *= scale;
  v.z *= scale;
  v.w *= scale;
  reinterpret_cast<float4*>(x)[i] = v;
}

__global__ void __launch_bounds__(256) scale_kernel(float* x, int64_t n, float scale) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= scale;
}

__global__ void __launch_bounds__(256) add_bias4_kernel(float* y, const float* bias, int cols, int64_t n4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  const int col = static_cast<int>((i * 4) % cols);
  float4 v = reinterpret_cast<float4*>(y)[i];
  const float4 b = *reinterpret_cast<const float4*>(bias + col);
  v.x += b.x;
  v.y += b.y;
  v.z += b.z;
  v.w += b.w;
  reinterpret_cast<float4*>(y)[i] = v;
}

__global__ void __launch_bounds__(256) add_bias_kernel(float* y, const float* bias, int cols, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] += bias[static_cast<int>(i % cols)];
}

// One thread owns one column so a row load is coalesced.
__global__ void __launch_bounds__(256) bias_grad_kernel(const float* grad_y, float* grad_bias, int rows,
                                                        int cols) {
  const int col = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
  if (col >= cols) return;
  float sum = 0.f;
  for (int row = 0; row < rows; ++row) sum += grad_y[static_cast<int64_t>(row) * cols + col];
  grad_bias[col] += sum;
}

__global__ void __launch_bounds__(256) add2_4_kernel(float* dst, const float* a, const float* b, int64_t n4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  const float4 av = reinterpret_cast<const float4*>(a)[i];
  const float4 bv = reinterpret_cast<const float4*>(b)[i];
  reinterpret_cast<float4*>(dst)[i] = make_float4(av.x + bv.x, av.y + bv.y, av.z + bv.z, av.w + bv.w);
}

__global__ void __launch_bounds__(256) add2_kernel(float* dst, const float* a, const float* b, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = a[i] + b[i];
}

__global__ void __launch_bounds__(256) mul4_kernel(float* y, const float* a, const float* b, int64_t n4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  const float4 av = reinterpret_cast<const float4*>(a)[i];
  const float4 bv = reinterpret_cast<const float4*>(b)[i];
  reinterpret_cast<float4*>(y)[i] = make_float4(av.x * bv.x, av.y * bv.y, av.z * bv.z, av.w * bv.w);
}

__global__ void __launch_bounds__(256) mul_kernel(float* y, const float* a, const float* b, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a[i] * b[i];
}

__global__ void __launch_bounds__(256) gelu4_kernel(float* y, const float* x, int64_t n4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  float4 v = reinterpret_cast<const float4*>(x)[i];
  v.x = gelu_forward(v.x);
  v.y = gelu_forward(v.y);
  v.z = gelu_forward(v.z);
  v.w = gelu_forward(v.w);
  reinterpret_cast<float4*>(y)[i] = v;
}

__global__ void __launch_bounds__(256) gelu_kernel(float* y, const float* x, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = gelu_forward(x[i]);
}

__global__ void __launch_bounds__(256) gelu_bwd4_kernel(float* grad_x, const float* x, const float* grad_y,
                                                       int64_t n4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  const float4 xv = reinterpret_cast<const float4*>(x)[i];
  const float4 gv = reinterpret_cast<const float4*>(grad_y)[i];
  reinterpret_cast<float4*>(grad_x)[i] = make_float4(gv.x * gelu_grad(xv.x), gv.y * gelu_grad(xv.y),
                                                     gv.z * gelu_grad(xv.z), gv.w * gelu_grad(xv.w));
}

__global__ void __launch_bounds__(256) gelu_bwd_kernel(float* grad_x, const float* x, const float* grad_y,
                                                      int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) grad_x[i] = grad_y[i] * gelu_grad(x[i]);
}

__global__ void __launch_bounds__(256) swiglu4_kernel(float* y, const float* x, int rows, int64_t inner) {
  const int64_t n4 = static_cast<int64_t>(rows) * (inner / 4);
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  const int64_t elem = i * 4;
  const int64_t row = elem / inner;
  const int64_t col = elem - row * inner;
  const int64_t base = row * (inner * 2) + col;
  const float4 a = *reinterpret_cast<const float4*>(x + base);
  const float4 b = *reinterpret_cast<const float4*>(x + base + inner);
  reinterpret_cast<float4*>(y)[i] =
      make_float4(silu(a.x) * b.x, silu(a.y) * b.y, silu(a.z) * b.z, silu(a.w) * b.w);
}

__global__ void __launch_bounds__(256) swiglu_kernel(float* y, const float* x, int rows, int64_t inner) {
  const int64_t n = static_cast<int64_t>(rows) * inner;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t row = i / inner;
  const int64_t col = i - row * inner;
  const int64_t base = row * (inner * 2) + col;
  y[i] = silu(x[base]) * x[base + inner];
}

__global__ void __launch_bounds__(256) swiglu_bwd4_kernel(float* grad_x, const float* x, const float* grad_y,
                                                         int rows, int64_t inner) {
  const int64_t n4 = static_cast<int64_t>(rows) * (inner / 4);
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  const int64_t elem = i * 4;
  const int64_t row = elem / inner;
  const int64_t col = elem - row * inner;
  const int64_t base = row * (inner * 2) + col;
  const float4 a = *reinterpret_cast<const float4*>(x + base);
  const float4 b = *reinterpret_cast<const float4*>(x + base + inner);
  const float4 dy = reinterpret_cast<const float4*>(grad_y)[i];
  *reinterpret_cast<float4*>(grad_x + base) =
      make_float4(dy.x * b.x * silu_grad(a.x), dy.y * b.y * silu_grad(a.y), dy.z * b.z * silu_grad(a.z),
                  dy.w * b.w * silu_grad(a.w));
  *reinterpret_cast<float4*>(grad_x + base + inner) =
      make_float4(dy.x * silu(a.x), dy.y * silu(a.y), dy.z * silu(a.z), dy.w * silu(a.w));
}

__global__ void __launch_bounds__(256) swiglu_bwd_kernel(float* grad_x, const float* x, const float* grad_y,
                                                        int rows, int64_t inner) {
  const int64_t n = static_cast<int64_t>(rows) * inner;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t row = i / inner;
  const int64_t col = i - row * inner;
  const int64_t base = row * (inner * 2) + col;
  const float gate = x[base];
  const float value = x[base + inner];
  const float dy = grad_y[i];
  grad_x[base] = dy * value * silu_grad(gate);
  grad_x[base + inner] = dy * silu(gate);
}

__global__ void __launch_bounds__(256) tanh4_kernel(float* y, const float* x, int64_t n4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  float4 v = reinterpret_cast<const float4*>(x)[i];
  v.x = tanhf(v.x);
  v.y = tanhf(v.y);
  v.z = tanhf(v.z);
  v.w = tanhf(v.w);
  reinterpret_cast<float4*>(y)[i] = v;
}

__global__ void __launch_bounds__(256) tanh_kernel(float* y, const float* x, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = tanhf(x[i]);
}

__global__ void __launch_bounds__(256) tanh_bwd4_kernel(float* grad_x, const float* y, const float* grad_y,
                                                       int64_t n4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  const float4 hv = reinterpret_cast<const float4*>(y)[i];
  const float4 gv = reinterpret_cast<const float4*>(grad_y)[i];
  reinterpret_cast<float4*>(grad_x)[i] =
      make_float4(gv.x * (1.f - hv.x * hv.x), gv.y * (1.f - hv.y * hv.y), gv.z * (1.f - hv.z * hv.z),
                  gv.w * (1.f - hv.w * hv.w));
}

__global__ void __launch_bounds__(256) tanh_bwd_kernel(float* grad_x, const float* y, const float* grad_y,
                                                      int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float h = y[i];
  grad_x[i] = grad_y[i] * (1.f - h * h);
}

__global__ void __launch_bounds__(256) gather_time_kernel2(float* dst, const float* src, int time,
                                                          int features, int index, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t row = i / features;
  const int64_t col = i - row * features;
  dst[i] = src[(row * time + index) * features + col];
}

__global__ void __launch_bounds__(256) scatter_time_kernel(float* dst, const float* src, int time,
                                                          int features, int index, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t row = i / features;
  const int64_t col = i - row * features;
  dst[(row * time + index) * features + col] = src[i];
}

bool vec4(const void* a, const void* b, int64_t n) {
  return n >= 4 && aligned16(a) && aligned16(b) && (n % 4) == 0;
}

bool vec4(const void* a, const void* b, const void* c, int64_t n) {
  return n >= 4 && aligned16(a) && aligned16(b) && aligned16(c) && (n % 4) == 0;
}

}  // namespace

void launch_fill(hipStream_t stream, float* x, int64_t n, float value) {
  if (n <= 0) return;
  if (aligned16(x) && n >= 4) {
    const int64_t n4 = n / 4;
    fill4_kernel<<<blocks_for(n4), 256, 0, stream>>>(x, n4, value);
    const int tail = static_cast<int>(n - n4 * 4);
    if (tail) fill_kernel<<<1, tail, 0, stream>>>(x + n4 * 4, tail, value);
  } else {
    fill_kernel<<<blocks_for(n), 256, 0, stream>>>(x, n, value);
  }
  check_hip(hipGetLastError(), "fill");
}

void launch_add_bias(hipStream_t stream, float* y, const float* bias, int rows, int cols) {
  const int64_t n = static_cast<int64_t>(rows) * cols;
  if (n <= 0) return;
  if ((cols % 4) == 0 && aligned16(y) && aligned16(bias)) {
    add_bias4_kernel<<<blocks_for(n / 4), 256, 0, stream>>>(y, bias, cols, n / 4);
  } else {
    add_bias_kernel<<<blocks_for(n), 256, 0, stream>>>(y, bias, cols, n);
  }
  check_hip(hipGetLastError(), "add_bias");
}

void launch_bias_grad(hipStream_t stream, const float* grad_y, float* grad_bias, int rows, int cols) {
  if (cols <= 0) return;
  bias_grad_kernel<<<blocks_for(cols), 256, 0, stream>>>(grad_y, grad_bias, rows, cols);
  check_hip(hipGetLastError(), "bias_grad");
}

void launch_add2(hipStream_t stream, float* dst, const float* a, const float* b, int64_t n) {
  if (n <= 0) return;
  if (vec4(dst, a, b, n)) {
    add2_4_kernel<<<blocks_for(n / 4), 256, 0, stream>>>(dst, a, b, n / 4);
  } else {
    add2_kernel<<<blocks_for(n), 256, 0, stream>>>(dst, a, b, n);
  }
  check_hip(hipGetLastError(), "add2");
}

void launch_mul(hipStream_t stream, float* y, const float* a, const float* b, int64_t n) {
  if (n <= 0) return;
  if (vec4(y, a, b, n)) {
    mul4_kernel<<<blocks_for(n / 4), 256, 0, stream>>>(y, a, b, n / 4);
  } else {
    mul_kernel<<<blocks_for(n), 256, 0, stream>>>(y, a, b, n);
  }
  check_hip(hipGetLastError(), "mul");
}

void Tensor::scale(float factor) { launch_scale(stream_of(*ctx_), data_, numel_, factor); }

void launch_scale(hipStream_t stream, float* x, int64_t n, float scale) {
  if (n <= 0) return;
  if (aligned16(x) && n >= 4) {
    const int64_t n4 = n / 4;
    scale4_kernel<<<blocks_for(n4), 256, 0, stream>>>(x, n4, scale);
    const int tail = static_cast<int>(n - n4 * 4);
    if (tail) scale_kernel<<<1, tail, 0, stream>>>(x + n4 * 4, tail, scale);
  } else {
    scale_kernel<<<blocks_for(n), 256, 0, stream>>>(x, n, scale);
  }
  check_hip(hipGetLastError(), "scale");
}

void launch_gelu(hipStream_t stream, float* y, const float* x, int64_t n) {
  if (n <= 0) return;
  if (vec4(y, x, n)) gelu4_kernel<<<blocks_for(n / 4), 256, 0, stream>>>(y, x, n / 4);
  else gelu_kernel<<<blocks_for(n), 256, 0, stream>>>(y, x, n);
  check_hip(hipGetLastError(), "gelu");
}

void launch_gelu_backward(hipStream_t stream, float* grad_x, const float* x, const float* grad_y,
                          int64_t n) {
  if (n <= 0) return;
  if (vec4(grad_x, x, grad_y, n))
    gelu_bwd4_kernel<<<blocks_for(n / 4), 256, 0, stream>>>(grad_x, x, grad_y, n / 4);
  else
    gelu_bwd_kernel<<<blocks_for(n), 256, 0, stream>>>(grad_x, x, grad_y, n);
  check_hip(hipGetLastError(), "gelu_backward");
}

void launch_swiglu(hipStream_t stream, float* y, const float* x, int rows, int64_t inner) {
  const int64_t n = static_cast<int64_t>(rows) * inner;
  if (n <= 0) return;
  if ((inner % 4) == 0 && aligned16(y) && aligned16(x))
    swiglu4_kernel<<<blocks_for(n / 4), 256, 0, stream>>>(y, x, rows, inner);
  else
    swiglu_kernel<<<blocks_for(n), 256, 0, stream>>>(y, x, rows, inner);
  check_hip(hipGetLastError(), "swiglu");
}

void launch_swiglu_backward(hipStream_t stream, float* grad_x, const float* x, const float* grad_y,
                            int rows, int64_t inner) {
  const int64_t n = static_cast<int64_t>(rows) * inner;
  if (n <= 0) return;
  if ((inner % 4) == 0 && aligned16(grad_x) && aligned16(x) && aligned16(grad_y))
    swiglu_bwd4_kernel<<<blocks_for(n / 4), 256, 0, stream>>>(grad_x, x, grad_y, rows, inner);
  else
    swiglu_bwd_kernel<<<blocks_for(n), 256, 0, stream>>>(grad_x, x, grad_y, rows, inner);
  check_hip(hipGetLastError(), "swiglu_backward");
}

void launch_tanh(hipStream_t stream, float* y, const float* x, int64_t n) {
  if (n <= 0) return;
  if (vec4(y, x, n)) tanh4_kernel<<<blocks_for(n / 4), 256, 0, stream>>>(y, x, n / 4);
  else tanh_kernel<<<blocks_for(n), 256, 0, stream>>>(y, x, n);
  check_hip(hipGetLastError(), "tanh");
}

void launch_tanh_backward(hipStream_t stream, float* grad_x, const float* y, const float* grad_y,
                          int64_t n) {
  if (n <= 0) return;
  if (vec4(grad_x, y, grad_y, n))
    tanh_bwd4_kernel<<<blocks_for(n / 4), 256, 0, stream>>>(grad_x, y, grad_y, n / 4);
  else
    tanh_bwd_kernel<<<blocks_for(n), 256, 0, stream>>>(grad_x, y, grad_y, n);
  check_hip(hipGetLastError(), "tanh_backward");
}

void launch_gather_time(hipStream_t stream, float* dst, const float* src, int batch, int time,
                        int features, int index) {
  const int64_t n = static_cast<int64_t>(batch) * features;
  if (n <= 0) return;
  gather_time_kernel2<<<blocks_for(n), 256, 0, stream>>>(dst, src, time, features, index, n);
  check_hip(hipGetLastError(), "gather_time");
}

void launch_scatter_time(hipStream_t stream, float* dst, const float* src, int batch, int time,
                         int features, int index) {
  const int64_t n = static_cast<int64_t>(batch) * features;
  if (n <= 0) return;
  scatter_time_kernel<<<blocks_for(n), 256, 0, stream>>>(dst, src, time, features, index, n);
  check_hip(hipGetLastError(), "scatter_time");
}

}  // namespace nn
