#include "nn/layers/layer_norm.hpp"

#include "activate.hpp"
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

__device__ float row_sum(const float* row, int cols) {
  float sum = 0.f;
  if ((cols % 4) == 0) {
    const float4* v = reinterpret_cast<const float4*>(row);
    const int n4 = cols / 4;
    for (int c = static_cast<int>(threadIdx.x); c < n4; c += kThreads) {
      const float4 t = v[c];
      sum += t.x + t.y + t.z + t.w;
    }
  } else {
    for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) sum += row[col];
  }
  return sum;
}

__global__ void __launch_bounds__(256) layer_norm_forward_kernel(float* y, float* mean_out,
                                                                 float* inv_std_out, const float* x,
                                                                 const float* gamma, const float* beta,
                                                                 int cols, float eps) {
  __shared__ float buffer[32];
  const int row = static_cast<int>(blockIdx.x);
  const int64_t base = static_cast<int64_t>(row) * cols;
  const float* row_x = x + base;
  float* row_y = y + base;

  const float mean = block_sum(row_sum(row_x, cols), buffer) / static_cast<float>(cols);

  float var_sum = 0.f;
  if ((cols % 4) == 0) {
    const float4* v = reinterpret_cast<const float4*>(row_x);
    const int n4 = cols / 4;
    for (int c = static_cast<int>(threadIdx.x); c < n4; c += kThreads) {
      const float4 t = v[c];
      const float d0 = t.x - mean;
      const float d1 = t.y - mean;
      const float d2 = t.z - mean;
      const float d3 = t.w - mean;
      var_sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
  } else {
    for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) {
      const float diff = row_x[col] - mean;
      var_sum += diff * diff;
    }
  }
  // Population second moment: divide by D, not D - 1.
  var_sum = block_sum(var_sum, buffer);
  const float inv_std = rsqrtf(var_sum / static_cast<float>(cols) + eps);

  if (threadIdx.x == 0) {
    mean_out[row] = mean;
    inv_std_out[row] = inv_std;
  }
  if ((cols % 4) == 0) {
    const float4* vx = reinterpret_cast<const float4*>(row_x);
    const float4* vg = reinterpret_cast<const float4*>(gamma);
    const float4* vb = reinterpret_cast<const float4*>(beta);
    float4* vy = reinterpret_cast<float4*>(row_y);
    const int n4 = cols / 4;
    for (int c = static_cast<int>(threadIdx.x); c < n4; c += kThreads) {
      const float4 t = vx[c];
      const float4 g = vg[c];
      const float4 b = vb[c];
      float4 out;
      out.x = (t.x - mean) * inv_std * g.x + b.x;
      out.y = (t.y - mean) * inv_std * g.y + b.y;
      out.z = (t.z - mean) * inv_std * g.z + b.z;
      out.w = (t.w - mean) * inv_std * g.w + b.w;
      vy[c] = out;
    }
  } else {
    for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) {
      const float xhat = (row_x[col] - mean) * inv_std;
      row_y[col] = xhat * gamma[col] + beta[col];
    }
  }
}

__global__ void __launch_bounds__(256) layer_norm_backward_input_kernel(float* dx, const float* dy,
                                                                        const float* x, const float* mean,
                                                                        const float* inv_std,
                                                                        const float* gamma, int cols) {
  __shared__ float buffer[32];
  const int row = static_cast<int>(blockIdx.x);
  const int64_t base = static_cast<int64_t>(row) * cols;
  const float mean_v = mean[row];
  const float inv = inv_std[row];

  float sum_dhat = 0.f;
  float sum_dhat_xhat = 0.f;
  if ((cols % 4) == 0) {
    const float4* vx = reinterpret_cast<const float4*>(x + base);
    const float4* vdy = reinterpret_cast<const float4*>(dy + base);
    const float4* vg = reinterpret_cast<const float4*>(gamma);
    const int n4 = cols / 4;
    for (int c = static_cast<int>(threadIdx.x); c < n4; c += kThreads) {
      const float4 t = vx[c];
      const float4 g = vg[c];
      const float4 d = vdy[c];
      const float x0 = (t.x - mean_v) * inv;
      const float x1 = (t.y - mean_v) * inv;
      const float x2 = (t.z - mean_v) * inv;
      const float x3 = (t.w - mean_v) * inv;
      const float d0 = d.x * g.x;
      const float d1 = d.y * g.y;
      const float d2 = d.z * g.z;
      const float d3 = d.w * g.w;
      sum_dhat += d0 + d1 + d2 + d3;
      sum_dhat_xhat += d0 * x0 + d1 * x1 + d2 * x2 + d3 * x3;
    }
  } else {
    for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) {
      const float xhat = (x[base + col] - mean_v) * inv;
      const float dhat = dy[base + col] * gamma[col];
      sum_dhat += dhat;
      sum_dhat_xhat += dhat * xhat;
    }
  }
  sum_dhat = block_sum(sum_dhat, buffer);
  sum_dhat_xhat = block_sum(sum_dhat_xhat, buffer);
  const float mean_dhat = sum_dhat / static_cast<float>(cols);
  const float mean_dhat_xhat = sum_dhat_xhat / static_cast<float>(cols);

  if ((cols % 4) == 0) {
    const float4* vx = reinterpret_cast<const float4*>(x + base);
    const float4* vdy = reinterpret_cast<const float4*>(dy + base);
    const float4* vg = reinterpret_cast<const float4*>(gamma);
    float4* vdx = reinterpret_cast<float4*>(dx + base);
    const int n4 = cols / 4;
    for (int c = static_cast<int>(threadIdx.x); c < n4; c += kThreads) {
      const float4 t = vx[c];
      const float4 g = vg[c];
      const float4 d = vdy[c];
      float4 out;
      const float x0 = (t.x - mean_v) * inv;
      const float x1 = (t.y - mean_v) * inv;
      const float x2 = (t.z - mean_v) * inv;
      const float x3 = (t.w - mean_v) * inv;
      out.x = inv * (d.x * g.x - mean_dhat - x0 * mean_dhat_xhat);
      out.y = inv * (d.y * g.y - mean_dhat - x1 * mean_dhat_xhat);
      out.z = inv * (d.z * g.z - mean_dhat - x2 * mean_dhat_xhat);
      out.w = inv * (d.w * g.w - mean_dhat - x3 * mean_dhat_xhat);
      vdx[c] = out;
    }
  } else {
    for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) {
      const float xhat = (x[base + col] - mean_v) * inv;
      const float dhat = dy[base + col] * gamma[col];
      dx[base + col] = inv * (dhat - mean_dhat - xhat * mean_dhat_xhat);
    }
  }
}

// One thread per column. Consecutive columns are consecutive addresses.
__global__ void __launch_bounds__(256) layer_norm_param_grad_kernel(const float* dy, const float* x,
                                                                    const float* mean, const float* inv_std,
                                                                    float* grad_gamma, float* grad_beta,
                                                                    int rows, int cols) {
  const int col = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
  if (col >= cols) return;
  float sum_gamma = 0.f;
  float sum_beta = 0.f;
  for (int row = 0; row < rows; ++row) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float xhat = (x[index] - mean[row]) * inv_std[row];
    const float dy_v = dy[index];
    sum_gamma += dy_v * xhat;
    sum_beta += dy_v;
  }
  grad_gamma[col] += sum_gamma;
  grad_beta[col] += sum_beta;
}

int normalized_rows(const Tensor& input, int normalized_shape) {
  const std::vector<int>& shape = input.shape();
  if ((shape.size() != 2 && shape.size() != 3) || shape.back() != normalized_shape) {
    throw Error("LayerNorm input shape");
  }
  return static_cast<int>(input.numel() / normalized_shape);
}

Context& require_layer_norm(Context& ctx, int normalized_shape, float eps) {
  if (normalized_shape < 1) throw Error("LayerNorm size");
  if (!(eps > 0.f) || !std::isfinite(eps)) throw Error("LayerNorm eps");
  return ctx;
}

}  // namespace

LayerNorm::LayerNorm(Context& ctx, int normalized_shape, float eps)
    : ctx_(&ctx),
      normalized_shape_(normalized_shape),
      eps_(eps),
      weight_(require_layer_norm(ctx, normalized_shape, eps), {normalized_shape}),
      grad_weight_(ctx, {normalized_shape}),
      bias_(ctx, {normalized_shape}),
      grad_bias_(ctx, {normalized_shape}) {
  weight_.fill(1.f);
  bias_.zero();
  grad_weight_.zero();
  grad_bias_.zero();
}

std::vector<NamedParam> LayerNorm::parameters() const {
  return {NamedParam{"weight", const_cast<Tensor*>(&weight_), const_cast<Tensor*>(&grad_weight_)},
          NamedParam{"bias", const_cast<Tensor*>(&bias_), const_cast<Tensor*>(&grad_bias_)}};
}

void LayerNorm::zero_grad() {
  grad_weight_.zero();
  grad_bias_.zero();
}

const Tensor& LayerNorm::forward(const Tensor& input) {
  const int rows = normalized_rows(input, normalized_shape_);
  const std::vector<int>& shape = input.shape();
  ensure_activation(cache_x_, *ctx_, shape);
  ensure_activation(output_, *ctx_, shape);
  ensure_tensor(mean_, *ctx_, std::vector<int>{rows});
  ensure_tensor(inv_std_, *ctx_, std::vector<int>{rows});
  if (output_->dtype() == DType::BFloat16) {
    arena_reset(*ctx_);
    float* x = stage_f32(*ctx_, input);
    float* y = arena_f32(*ctx_, input.numel());
    layer_norm_forward_kernel<<<rows, kThreads, 0, stream_of(*ctx_)>>>(
        y, mean_->data(), inv_std_->data(), x, weight_.data(), bias_.data(), normalized_shape_, eps_);
    check_hip(hipGetLastError(), "layer_norm_forward");
    keep_f32(*ctx_, *cache_x_, x);
    keep_f32(*ctx_, *output_, y);
    return *output_;
  }
  cache_x_->copy_from(input);
  layer_norm_forward_kernel<<<rows, kThreads, 0, stream_of(*ctx_)>>>(
      output_->data(), mean_->data(), inv_std_->data(), cache_x_->data(), weight_.data(),
      bias_.data(), normalized_shape_, eps_);
  check_hip(hipGetLastError(), "layer_norm_forward");
  return *output_;
}

const Tensor& LayerNorm::backward(const Tensor& grad_output) {
  if (!cache_x_ || !mean_ || !inv_std_ || grad_output.shape() != cache_x_->shape()) {
    throw Error("LayerNorm backward shape");
  }
  const int rows = static_cast<int>(mean_->numel());
  ensure_activation(grad_input_, *ctx_, cache_x_->shape());
  hipStream_t stream = stream_of(*ctx_);
  if (grad_input_->dtype() == DType::BFloat16) {
    arena_reset(*ctx_);
    float* x = stage_f32(*ctx_, *cache_x_);
    float* dy = stage_f32(*ctx_, grad_output);
    float* dx = arena_f32(*ctx_, cache_x_->numel());
    layer_norm_backward_input_kernel<<<rows, kThreads, 0, stream>>>(
        dx, dy, x, mean_->data(), inv_std_->data(), weight_.data(), normalized_shape_);
    check_hip(hipGetLastError(), "layer_norm_backward_input");
    const int param_blocks = (normalized_shape_ + kThreads - 1) / kThreads;
    layer_norm_param_grad_kernel<<<param_blocks, kThreads, 0, stream>>>(
        dy, x, mean_->data(), inv_std_->data(), grad_weight_.data(), grad_bias_.data(), rows,
        normalized_shape_);
    check_hip(hipGetLastError(), "layer_norm_backward_param");
    keep_f32(*ctx_, *grad_input_, dx);
    return *grad_input_;
  }
  layer_norm_backward_input_kernel<<<rows, kThreads, 0, stream>>>(
      grad_input_->data(), grad_output.data(), cache_x_->data(), mean_->data(), inv_std_->data(),
      weight_.data(), normalized_shape_);
  check_hip(hipGetLastError(), "layer_norm_backward_input");
  const int param_blocks = (normalized_shape_ + kThreads - 1) / kThreads;
  layer_norm_param_grad_kernel<<<param_blocks, kThreads, 0, stream>>>(
      grad_output.data(), cache_x_->data(), mean_->data(), inv_std_->data(), grad_weight_.data(),
      grad_bias_.data(), rows, normalized_shape_);
  check_hip(hipGetLastError(), "layer_norm_backward_param");
  return *grad_input_;
}

}  // namespace nn
