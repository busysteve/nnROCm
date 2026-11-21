#include "nn/layers/rms_norm.hpp"

#include "check.hpp"
#include "context_impl.hpp"
#include "ensure.hpp"
#include "nn/error.hpp"
#include "reduce_device.hpp"

#include <cmath>
#include <vector>

namespace nn {
namespace {

constexpr int kThreads = 256;

__global__ void __launch_bounds__(256) rms_norm_forward_kernel(float* y, float* inv_rms_out, const float* x,
                                                               const float* weight, int cols, float eps) {
  __shared__ float buffer[32];
  const int row = static_cast<int>(blockIdx.x);
  const int64_t base = static_cast<int64_t>(row) * cols;
  const float* row_x = x + base;
  float* row_y = y + base;

  float sq = 0.f;
  if ((cols % 4) == 0) {
    const float4* v = reinterpret_cast<const float4*>(row_x);
    const int n4 = cols / 4;
    for (int c = static_cast<int>(threadIdx.x); c < n4; c += kThreads) {
      const float4 t = v[c];
      sq += t.x * t.x + t.y * t.y + t.z * t.z + t.w * t.w;
    }
  } else {
    for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) {
      const float value = row_x[col];
      sq += value * value;
    }
  }
  sq = block_sum(sq, buffer);
  const float inv_rms = rsqrtf(sq / static_cast<float>(cols) + eps);
  if (threadIdx.x == 0) inv_rms_out[row] = inv_rms;
  if ((cols % 4) == 0) {
    const float4* vx = reinterpret_cast<const float4*>(row_x);
    const float4* vw = reinterpret_cast<const float4*>(weight);
    float4* vy = reinterpret_cast<float4*>(row_y);
    const int n4 = cols / 4;
    for (int c = static_cast<int>(threadIdx.x); c < n4; c += kThreads) {
      const float4 t = vx[c];
      const float4 w = vw[c];
      float4 out;
      out.x = t.x * inv_rms * w.x;
      out.y = t.y * inv_rms * w.y;
      out.z = t.z * inv_rms * w.z;
      out.w = t.w * inv_rms * w.w;
      vy[c] = out;
    }
  } else {
    for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) {
      row_y[col] = row_x[col] * inv_rms * weight[col];
    }
  }
}

__global__ void __launch_bounds__(256) rms_norm_backward_input_kernel(float* dx, const float* dy,
                                                                      const float* x, const float* inv_rms,
                                                                      const float* weight, int cols) {
  __shared__ float buffer[32];
  const int row = static_cast<int>(blockIdx.x);
  const int64_t base = static_cast<int64_t>(row) * cols;
  const float inv = inv_rms[row];

  float sum_dhat_xhat = 0.f;
  if ((cols % 4) == 0) {
    const float4* vx = reinterpret_cast<const float4*>(x + base);
    const float4* vdy = reinterpret_cast<const float4*>(dy + base);
    const float4* vw = reinterpret_cast<const float4*>(weight);
    const int n4 = cols / 4;
    for (int c = static_cast<int>(threadIdx.x); c < n4; c += kThreads) {
      const float4 t = vx[c];
      const float4 d = vdy[c];
      const float4 w = vw[c];
      const float x0 = t.x * inv;
      const float x1 = t.y * inv;
      const float x2 = t.z * inv;
      const float x3 = t.w * inv;
      sum_dhat_xhat += d.x * w.x * x0 + d.y * w.y * x1 + d.z * w.z * x2 + d.w * w.w * x3;
    }
  } else {
    for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) {
      const float xhat = x[base + col] * inv;
      const float dhat = dy[base + col] * weight[col];
      sum_dhat_xhat += dhat * xhat;
    }
  }
  const float mean_dhat_xhat = block_sum(sum_dhat_xhat, buffer) / static_cast<float>(cols);

  if ((cols % 4) == 0) {
    const float4* vx = reinterpret_cast<const float4*>(x + base);
    const float4* vdy = reinterpret_cast<const float4*>(dy + base);
    const float4* vw = reinterpret_cast<const float4*>(weight);
    float4* vdx = reinterpret_cast<float4*>(dx + base);
    const int n4 = cols / 4;
    for (int c = static_cast<int>(threadIdx.x); c < n4; c += kThreads) {
      const float4 t = vx[c];
      const float4 d = vdy[c];
      const float4 w = vw[c];
      float4 out;
      out.x = inv * (d.x * w.x - t.x * inv * mean_dhat_xhat);
      out.y = inv * (d.y * w.y - t.y * inv * mean_dhat_xhat);
      out.z = inv * (d.z * w.z - t.z * inv * mean_dhat_xhat);
      out.w = inv * (d.w * w.w - t.w * inv * mean_dhat_xhat);
      vdx[c] = out;
    }
  } else {
    for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) {
      const float xhat = x[base + col] * inv;
      const float dhat = dy[base + col] * weight[col];
      dx[base + col] = inv * (dhat - xhat * mean_dhat_xhat);
    }
  }
}

__global__ void __launch_bounds__(256) rms_norm_param_grad_kernel(const float* dy, const float* x,
                                                                  const float* inv_rms, float* grad_weight,
                                                                  int rows, int cols) {
  const int col = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
  if (col >= cols) return;
  float sum = 0.f;
  for (int row = 0; row < rows; ++row) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float xhat = x[index] * inv_rms[row];
    sum += dy[index] * xhat;
  }
  grad_weight[col] += sum;
}

int normalized_rows(const Tensor& input, int normalized_shape) {
  const std::vector<int>& shape = input.shape();
  if (shape.size() < 2 || shape.back() != normalized_shape) {
    throw Error("RMSNorm input shape");
  }
  return static_cast<int>(input.numel() / normalized_shape);
}

Context& require_rms_norm(Context& ctx, int normalized_shape, float eps) {
  if (normalized_shape < 1) throw Error("RMSNorm size");
  if (!(eps > 0.f) || !std::isfinite(eps)) throw Error("RMSNorm eps");
  return ctx;
}

}  // namespace

RMSNorm::RMSNorm(Context& ctx, int normalized_shape, float eps)
    : ctx_(&ctx),
      normalized_shape_(normalized_shape),
      eps_(eps),
      weight_(require_rms_norm(ctx, normalized_shape, eps), {normalized_shape}),
      grad_weight_(ctx, {normalized_shape}) {
  weight_.fill(1.f);
  grad_weight_.zero();
}

std::vector<NamedParam> RMSNorm::parameters() const {
  return {NamedParam{"weight", const_cast<Tensor*>(&weight_), const_cast<Tensor*>(&grad_weight_)}};
}

void RMSNorm::zero_grad() { grad_weight_.zero(); }

const Tensor& RMSNorm::forward(const Tensor& input) {
  const int rows = normalized_rows(input, normalized_shape_);
  const std::vector<int>& shape = input.shape();
  ensure_tensor(cache_x_, *ctx_, shape);
  ensure_tensor(output_, *ctx_, shape);
  ensure_tensor(inv_rms_, *ctx_, std::vector<int>{rows});
  cache_x_->copy_from(input);
  rms_norm_forward_kernel<<<rows, kThreads, 0, stream_of(*ctx_)>>>(
      output_->data(), inv_rms_->data(), cache_x_->data(), weight_.data(), normalized_shape_, eps_);
  check_hip(hipGetLastError(), "rms_norm_forward");
  return *output_;
}

const Tensor& RMSNorm::backward(const Tensor& grad_output) {
  if (!cache_x_ || !inv_rms_ || grad_output.shape() != cache_x_->shape()) {
    throw Error("RMSNorm backward shape");
  }
  const int rows = static_cast<int>(inv_rms_->numel());
  ensure_tensor(grad_input_, *ctx_, cache_x_->shape());
  hipStream_t stream = stream_of(*ctx_);
  rms_norm_backward_input_kernel<<<rows, kThreads, 0, stream>>>(
      grad_input_->data(), grad_output.data(), cache_x_->data(), inv_rms_->data(), weight_.data(),
      normalized_shape_);
  check_hip(hipGetLastError(), "rms_norm_backward_input");
  const int param_blocks = (normalized_shape_ + kThreads - 1) / kThreads;
  rms_norm_param_grad_kernel<<<param_blocks, kThreads, 0, stream>>>(
      grad_output.data(), cache_x_->data(), inv_rms_->data(), grad_weight_.data(), rows,
      normalized_shape_);
  check_hip(hipGetLastError(), "rms_norm_backward_param");
  return *grad_input_;
}

}  // namespace nn
