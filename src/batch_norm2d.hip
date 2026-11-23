#include "nn/layers/batch_norm2d.hpp"

#include "activate.hpp"
#include "check.hpp"
#include "context_impl.hpp"
#include "ensure.hpp"
#include "launch.hpp"
#include "nn/error.hpp"
#include "reduce_device.hpp"

namespace nn {

struct BatchNorm2d::Impl {};

namespace {

__device__ float plane_sum(const float* plane, int spatial) {
  float sum = 0.f;
  if ((spatial % 4) == 0) {
    const float4* v = reinterpret_cast<const float4*>(plane);
    for (int s = static_cast<int>(threadIdx.x); s < spatial / 4; s += static_cast<int>(blockDim.x)) {
      const float4 t = v[s];
      sum += t.x + t.y + t.z + t.w;
    }
  } else {
    for (int s = static_cast<int>(threadIdx.x); s < spatial; s += static_cast<int>(blockDim.x)) sum += plane[s];
  }
  return sum;
}

__global__ void __launch_bounds__(256) bn_train_kernel(float* y, const float* x, const float* gamma,
                                                       const float* beta, float* running_mean,
                                                       float* running_var, float* saved_mean,
                                                       float* saved_inv, int n, int channels, int spatial,
                                                       float factor, float eps) {
  __shared__ float buffer[32];
  const int ch = static_cast<int>(blockIdx.x);
  if (ch >= channels) return;
  const int m = n * spatial;
  float sum = 0.f;
  for (int i = 0; i < n; ++i) sum += plane_sum(x + (static_cast<int64_t>(i) * channels + ch) * spatial, spatial);
  const float mean = block_sum(sum, buffer) / static_cast<float>(m);

  float var_sum = 0.f;
  for (int i = 0; i < n; ++i) {
    const float* plane = x + (static_cast<int64_t>(i) * channels + ch) * spatial;
    if ((spatial % 4) == 0) {
      const float4* v = reinterpret_cast<const float4*>(plane);
      for (int s = static_cast<int>(threadIdx.x); s < spatial / 4; s += static_cast<int>(blockDim.x)) {
        const float4 t = v[s];
        const float d0 = t.x - mean;
        const float d1 = t.y - mean;
        const float d2 = t.z - mean;
        const float d3 = t.w - mean;
        var_sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
      }
    } else {
      for (int s = static_cast<int>(threadIdx.x); s < spatial; s += static_cast<int>(blockDim.x)) {
        const float d = plane[s] - mean;
        var_sum += d * d;
      }
    }
  }
  var_sum = block_sum(var_sum, buffer);
  const float var_b = var_sum / static_cast<float>(m);
  const float inv = rsqrtf(var_b + eps);
  if (threadIdx.x == 0) {
    saved_mean[ch] = mean;
    saved_inv[ch] = inv;
    // cuDNN stores the unbiased variance in the running statistic and normalizes with the biased one.
    const float var_u = var_b * static_cast<float>(m) / static_cast<float>(m - 1);
    running_mean[ch] = (1.f - factor) * running_mean[ch] + factor * mean;
    running_var[ch] = (1.f - factor) * running_var[ch] + factor * var_u;
  }
  const float scale = gamma[ch];
  const float shift = beta[ch];
  for (int i = 0; i < n; ++i) {
    const float* plane = x + (static_cast<int64_t>(i) * channels + ch) * spatial;
    float* out = y + (static_cast<int64_t>(i) * channels + ch) * spatial;
    for (int s = static_cast<int>(threadIdx.x); s < spatial; s += static_cast<int>(blockDim.x)) {
      out[s] = scale * (plane[s] - mean) * inv + shift;
    }
  }
}

__global__ void __launch_bounds__(256) bn_eval_kernel(float* y, const float* x, const float* gamma,
                                                      const float* beta, const float* running_mean,
                                                      const float* running_var, int n, int channels,
                                                      int spatial, float eps) {
  const int ch = static_cast<int>(blockIdx.x);
  if (ch >= channels) return;
  const float mean = running_mean[ch];
  const float inv = rsqrtf(running_var[ch] + eps);
  const float scale = gamma[ch];
  const float shift = beta[ch];
  for (int i = 0; i < n; ++i) {
    const float* plane = x + (static_cast<int64_t>(i) * channels + ch) * spatial;
    float* out = y + (static_cast<int64_t>(i) * channels + ch) * spatial;
    for (int s = static_cast<int>(threadIdx.x); s < spatial; s += static_cast<int>(blockDim.x)) {
      out[s] = scale * (plane[s] - mean) * inv + shift;
    }
  }
}

__global__ void __launch_bounds__(256) bn_backward_kernel(float* dx, const float* x, const float* dy,
                                                          const float* gamma, float* grad_gamma,
                                                          float* grad_beta, const float* saved_mean,
                                                          const float* saved_inv, int n, int channels,
                                                          int spatial) {
  __shared__ float buffer[32];
  const int ch = static_cast<int>(blockIdx.x);
  if (ch >= channels) return;
  const int m = n * spatial;
  const float mean = saved_mean[ch];
  const float inv = saved_inv[ch];
  float sum_dy = 0.f;
  float sum_dy_xhat = 0.f;
  for (int i = 0; i < n; ++i) {
    const float* plane = x + (static_cast<int64_t>(i) * channels + ch) * spatial;
    const float* gplane = dy + (static_cast<int64_t>(i) * channels + ch) * spatial;
    for (int s = static_cast<int>(threadIdx.x); s < spatial; s += static_cast<int>(blockDim.x)) {
      const float xhat = (plane[s] - mean) * inv;
      const float g = gplane[s];
      sum_dy += g;
      sum_dy_xhat += g * xhat;
    }
  }
  sum_dy = block_sum(sum_dy, buffer);
  sum_dy_xhat = block_sum(sum_dy_xhat, buffer);
  const float mean_dy = sum_dy / static_cast<float>(m);
  const float mean_dy_xhat = sum_dy_xhat / static_cast<float>(m);
  if (threadIdx.x == 0) {
    grad_gamma[ch] += sum_dy_xhat;
    grad_beta[ch] += sum_dy;
  }
  const float scale = gamma[ch] * inv;
  for (int i = 0; i < n; ++i) {
    const float* plane = x + (static_cast<int64_t>(i) * channels + ch) * spatial;
    const float* gplane = dy + (static_cast<int64_t>(i) * channels + ch) * spatial;
    float* out = dx + (static_cast<int64_t>(i) * channels + ch) * spatial;
    for (int s = static_cast<int>(threadIdx.x); s < spatial; s += static_cast<int>(blockDim.x)) {
      const float xhat = (plane[s] - mean) * inv;
      out[s] = scale * (gplane[s] - mean_dy - xhat * mean_dy_xhat);
    }
  }
}

}  // namespace

BatchNorm2d::BatchNorm2d(Context& ctx, int channels)
    : ctx_(&ctx),
      channels_(channels),
      weight_(ctx, {channels}),
      grad_weight_(ctx, {channels}),
      bias_(ctx, {channels}),
      grad_bias_(ctx, {channels}),
      running_mean_(ctx, {channels}),
      running_var_(ctx, {channels}),
      saved_mean_(ctx, {channels}),
      saved_inv_variance_(ctx, {channels}),
      impl_(std::make_unique<Impl>()) {
  if (channels < 1) throw Error("BatchNorm2d channels");
  weight_.fill(1.f);
  bias_.zero();
  grad_weight_.zero();
  grad_bias_.zero();
  running_mean_.zero();
  running_var_.fill(1.f);
  saved_mean_.zero();
  saved_inv_variance_.zero();
}

BatchNorm2d::~BatchNorm2d() = default;

std::vector<NamedParam> BatchNorm2d::parameters() const {
  return {NamedParam{"weight", const_cast<Tensor*>(&weight_), const_cast<Tensor*>(&grad_weight_)},
          NamedParam{"bias", const_cast<Tensor*>(&bias_), const_cast<Tensor*>(&grad_bias_)}};
}

std::vector<NamedBuffer> BatchNorm2d::buffers() const {
  auto* self = const_cast<BatchNorm2d*>(this);
  return {NamedBuffer{"running_mean", &self->running_mean_, &self->batches_},
          NamedBuffer{"running_var", &self->running_var_, &self->batches_}};
}

void BatchNorm2d::zero_grad() {
  grad_weight_.zero();
  grad_bias_.zero();
}

void BatchNorm2d::set_training(bool training) { training_ = training; }

void BatchNorm2d::bind(const Tensor& input) {
  if (input.rank() != 4 || input.shape()[1] != channels_) throw Error("BatchNorm2d input shape");
  ensure_activation(output_, *ctx_, input.shape());
  ensure_activation(grad_input_, *ctx_, input.shape());
}

const Tensor& BatchNorm2d::forward(const Tensor& input) {
  bind(input);
  ensure_activation(cache_x_, *ctx_, input.shape());
  const int n = input.shape()[0];
  const int spatial = input.shape()[2] * input.shape()[3];
  hipStream_t stream = stream_of(*ctx_);
  if (output_->dtype() == DType::BFloat16) {
    arena_reset(*ctx_);
    float* x = stage_f32(*ctx_, input);
    float* y = arena_f32(*ctx_, output_->numel());
    if (training_) {
      const int64_t count = static_cast<int64_t>(n) * spatial;
      if (count <= 1) throw Error("BatchNorm train mode needs more than one value per channel");
      ++batches_;
      const float factor = 1.f / static_cast<float>(batches_);
      bn_train_kernel<<<channels_, 256, 0, stream>>>(
          y, x, weight_.data(), bias_.data(), running_mean_.data(), running_var_.data(), saved_mean_.data(),
          saved_inv_variance_.data(), n, channels_, spatial, factor, 1e-5f);
      check_hip(hipGetLastError(), "batch_norm_forward");
    } else {
      bn_eval_kernel<<<channels_, 256, 0, stream>>>(y, x, weight_.data(), bias_.data(), running_mean_.data(),
                                                    running_var_.data(), n, channels_, spatial, 1e-5f);
      check_hip(hipGetLastError(), "batch_norm_forward");
    }
    keep_f32(*ctx_, *cache_x_, x);
    keep_f32(*ctx_, *output_, y);
    return *output_;
  }
  cache_x_->copy_from(input);
  if (training_) {
    const int64_t count = static_cast<int64_t>(n) * spatial;
    if (count <= 1) throw Error("BatchNorm train mode needs more than one value per channel");
    ++batches_;
    const float factor = 1.f / static_cast<float>(batches_);
    bn_train_kernel<<<channels_, 256, 0, stream>>>(
        output_->data(), cache_x_->data(), weight_.data(), bias_.data(), running_mean_.data(),
        running_var_.data(), saved_mean_.data(), saved_inv_variance_.data(), n, channels_, spatial, factor,
        1e-5f);
    check_hip(hipGetLastError(), "batch_norm_forward");
  } else {
    bn_eval_kernel<<<channels_, 256, 0, stream>>>(
        output_->data(), cache_x_->data(), weight_.data(), bias_.data(), running_mean_.data(),
        running_var_.data(), n, channels_, spatial, 1e-5f);
    check_hip(hipGetLastError(), "batch_norm_forward");
  }
  return *output_;
}

const Tensor& BatchNorm2d::backward(const Tensor& grad_output) {
  if (!training_) throw Error("BatchNorm backward requires train mode");
  if (!cache_x_ || grad_output.shape() != cache_x_->shape()) throw Error("BatchNorm2d backward shape");
  const int n = cache_x_->shape()[0];
  const int spatial = cache_x_->shape()[2] * cache_x_->shape()[3];
  if (grad_input_->dtype() == DType::BFloat16) {
    arena_reset(*ctx_);
    float* x = stage_f32(*ctx_, *cache_x_);
    float* dy = stage_f32(*ctx_, grad_output);
    float* dx = arena_f32(*ctx_, cache_x_->numel());
    bn_backward_kernel<<<channels_, 256, 0, stream_of(*ctx_)>>>(
        dx, x, dy, weight_.data(), grad_weight_.data(), grad_bias_.data(), saved_mean_.data(),
        saved_inv_variance_.data(), n, channels_, spatial);
    check_hip(hipGetLastError(), "batch_norm_backward");
    keep_f32(*ctx_, *grad_input_, dx);
    return *grad_input_;
  }
  bn_backward_kernel<<<channels_, 256, 0, stream_of(*ctx_)>>>(
      grad_input_->data(), cache_x_->data(), grad_output.data(), weight_.data(), grad_weight_.data(),
      grad_bias_.data(), saved_mean_.data(), saved_inv_variance_.data(), n, channels_, spatial);
  check_hip(hipGetLastError(), "batch_norm_backward");
  return *grad_input_;
}

}  // namespace nn
