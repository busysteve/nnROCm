#include "nn/layers/relu.hpp"

#include "activate.hpp"
#include "check.hpp"
#include "context_impl.hpp"
#include "ensure.hpp"
#include "launch.hpp"
#include "nn/error.hpp"

namespace nn {

struct ReLU::Impl {};

namespace {

void require_rank(const std::vector<int>& shape) {
  if (shape.size() != 2 && shape.size() != 3 && shape.size() != 4) throw Error("ReLU rank");
}

__global__ void __launch_bounds__(256) relu4_kernel(float* y, const float* x, int64_t n4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  float4 v = reinterpret_cast<const float4*>(x)[i];
  v.x = fmaxf(v.x, 0.f);
  v.y = fmaxf(v.y, 0.f);
  v.z = fmaxf(v.z, 0.f);
  v.w = fmaxf(v.w, 0.f);
  reinterpret_cast<float4*>(y)[i] = v;
}

__global__ void __launch_bounds__(256) relu_kernel(float* y, const float* x, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = fmaxf(x[i], 0.f);
}

__global__ void __launch_bounds__(256) relu_bwd4_kernel(float* dx, const float* x, const float* dy, int64_t n4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  const float4 xv = reinterpret_cast<const float4*>(x)[i];
  const float4 gv = reinterpret_cast<const float4*>(dy)[i];
  reinterpret_cast<float4*>(dx)[i] =
      make_float4(xv.x > 0.f ? gv.x : 0.f, xv.y > 0.f ? gv.y : 0.f, xv.z > 0.f ? gv.z : 0.f,
                  xv.w > 0.f ? gv.w : 0.f);
}

__global__ void __launch_bounds__(256) relu_bwd_kernel(float* dx, const float* x, const float* dy, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dx[i] = x[i] > 0.f ? dy[i] : 0.f;
}

}  // namespace

ReLU::ReLU(Context& ctx) : ctx_(&ctx), impl_(std::make_unique<Impl>()) {}

ReLU::~ReLU() = default;

std::vector<NamedParam> ReLU::parameters() const { return {}; }

void ReLU::zero_grad() {}

const Tensor& ReLU::forward(const Tensor& input) {
  require_rank(input.shape());
  ensure_activation(cache_x_, *ctx_, input.shape());
  ensure_activation(output_, *ctx_, input.shape());
  const int64_t n = input.numel();
  hipStream_t stream = stream_of(*ctx_);
  if (output_->dtype() == DType::BFloat16) {
    arena_reset(*ctx_);
    float* x = stage_f32(*ctx_, input);
    float* y = arena_f32(*ctx_, n);
    if (aligned16(y) && aligned16(x) && (n % 4) == 0) {
      relu4_kernel<<<blocks_for(n / 4), 256, 0, stream>>>(y, x, n / 4);
    } else {
      relu_kernel<<<blocks_for(n), 256, 0, stream>>>(y, x, n);
    }
    check_hip(hipGetLastError(), "relu");
    keep_f32(*ctx_, *cache_x_, x);
    keep_f32(*ctx_, *output_, y);
    return *output_;
  }
  cache_x_->copy_from(input);
  if (aligned16(output_->data()) && aligned16(cache_x_->data()) && (n % 4) == 0) {
    relu4_kernel<<<blocks_for(n / 4), 256, 0, stream>>>(output_->data(), cache_x_->data(), n / 4);
  } else {
    relu_kernel<<<blocks_for(n), 256, 0, stream>>>(output_->data(), cache_x_->data(), n);
  }
  check_hip(hipGetLastError(), "relu");
  return *output_;
}

const Tensor& ReLU::backward(const Tensor& grad_output) {
  if (!cache_x_ || !output_ || grad_output.shape() != output_->shape()) throw Error("ReLU backward shape");
  ensure_activation(grad_input_, *ctx_, grad_output.shape());
  const int64_t n = cache_x_->numel();
  hipStream_t stream = stream_of(*ctx_);
  if (grad_input_->dtype() == DType::BFloat16) {
    arena_reset(*ctx_);
    float* x = stage_f32(*ctx_, *cache_x_);
    float* dy = stage_f32(*ctx_, grad_output);
    float* dx = arena_f32(*ctx_, n);
    if (aligned16(dx) && aligned16(x) && aligned16(dy) && (n % 4) == 0) {
      relu_bwd4_kernel<<<blocks_for(n / 4), 256, 0, stream>>>(dx, x, dy, n / 4);
    } else {
      relu_bwd_kernel<<<blocks_for(n), 256, 0, stream>>>(dx, x, dy, n);
    }
    check_hip(hipGetLastError(), "relu_backward");
    keep_f32(*ctx_, *grad_input_, dx);
    return *grad_input_;
  }
  if (aligned16(grad_input_->data()) && aligned16(cache_x_->data()) && aligned16(grad_output.data()) &&
      (n % 4) == 0) {
    relu_bwd4_kernel<<<blocks_for(n / 4), 256, 0, stream>>>(grad_input_->data(), cache_x_->data(),
                                                           grad_output.data(), n / 4);
  } else {
    relu_bwd_kernel<<<blocks_for(n), 256, 0, stream>>>(grad_input_->data(), cache_x_->data(), grad_output.data(),
                                                      n);
  }
  check_hip(hipGetLastError(), "relu_backward");
  return *grad_input_;
}

}  // namespace nn
