#include "nn/layers/global_avg_pool2d.hpp"

#include "activate.hpp"
#include "check.hpp"
#include "context_impl.hpp"
#include "ensure.hpp"
#include "nn/error.hpp"
#include "reduce_device.hpp"

namespace nn {

struct GlobalAvgPool2d::Impl {};

namespace {

__global__ void __launch_bounds__(256) gap_forward_kernel(float* y, const float* x, int planes, int spatial) {
  __shared__ float buffer[32];
  const int plane = static_cast<int>(blockIdx.x);
  if (plane >= planes) return;
  const float* src = x + static_cast<int64_t>(plane) * spatial;
  float sum = 0.f;
  if ((spatial % 4) == 0) {
    const float4* v = reinterpret_cast<const float4*>(src);
    for (int s = static_cast<int>(threadIdx.x); s < spatial / 4; s += static_cast<int>(blockDim.x)) {
      const float4 t = v[s];
      sum += t.x + t.y + t.z + t.w;
    }
  } else {
    for (int s = static_cast<int>(threadIdx.x); s < spatial; s += static_cast<int>(blockDim.x)) sum += src[s];
  }
  sum = block_sum(sum, buffer);
  if (threadIdx.x == 0) y[plane] = sum / static_cast<float>(spatial);
}

__global__ void __launch_bounds__(256) gap_backward_kernel(float* dx, const float* dy, int planes, int spatial) {
  const int plane = static_cast<int>(blockIdx.x);
  if (plane >= planes) return;
  const float scale = dy[plane] / static_cast<float>(spatial);
  float* dst = dx + static_cast<int64_t>(plane) * spatial;
  if ((spatial % 4) == 0) {
    float4* v = reinterpret_cast<float4*>(dst);
    const float4 filled = make_float4(scale, scale, scale, scale);
    for (int s = static_cast<int>(threadIdx.x); s < spatial / 4; s += static_cast<int>(blockDim.x)) v[s] = filled;
  } else {
    for (int s = static_cast<int>(threadIdx.x); s < spatial; s += static_cast<int>(blockDim.x)) dst[s] = scale;
  }
}

}  // namespace

GlobalAvgPool2d::GlobalAvgPool2d(Context& ctx) : ctx_(&ctx), impl_(std::make_unique<Impl>()) {}

GlobalAvgPool2d::~GlobalAvgPool2d() = default;

std::vector<NamedParam> GlobalAvgPool2d::parameters() const { return {}; }

void GlobalAvgPool2d::zero_grad() {}

const Tensor& GlobalAvgPool2d::forward(const Tensor& input) {
  if (input.rank() != 4) throw Error("GlobalAvgPool2d input rank");
  const int n = input.shape()[0];
  const int c = input.shape()[1];
  const int h = input.shape()[2];
  const int w = input.shape()[3];
  const int spatial = h * w;
  ensure_activation(cache_x_, *ctx_, input.shape());
  ensure_activation(output_, *ctx_, {n, c, 1, 1});
  ensure_activation(cache_y_, *ctx_, output_->shape());
  if (output_->dtype() == DType::BFloat16) {
    arena_reset(*ctx_);
    float* x = stage_f32(*ctx_, input);
    float* y = arena_f32(*ctx_, output_->numel());
    gap_forward_kernel<<<n * c, 256, 0, stream_of(*ctx_)>>>(y, x, n * c, spatial);
    check_hip(hipGetLastError(), "global_avg_pool_forward");
    keep_f32(*ctx_, *cache_x_, x);
    keep_f32(*ctx_, *output_, y);
    keep_f32(*ctx_, *cache_y_, y);
    return *output_;
  }
  cache_x_->copy_from(input);
  gap_forward_kernel<<<n * c, 256, 0, stream_of(*ctx_)>>>(output_->data(), cache_x_->data(), n * c, spatial);
  check_hip(hipGetLastError(), "global_avg_pool_forward");
  cache_y_->copy_from(*output_);
  return *output_;
}

const Tensor& GlobalAvgPool2d::backward(const Tensor& grad_output) {
  if (!cache_x_ || !cache_y_ || grad_output.shape() != cache_y_->shape()) {
    throw Error("GlobalAvgPool2d backward shape");
  }
  ensure_activation(grad_input_, *ctx_, cache_x_->shape());
  const int n = cache_x_->shape()[0];
  const int c = cache_x_->shape()[1];
  const int spatial = cache_x_->shape()[2] * cache_x_->shape()[3];
  if (grad_input_->dtype() == DType::BFloat16) {
    arena_reset(*ctx_);
    float* dy = stage_f32(*ctx_, grad_output);
    float* dx = arena_f32(*ctx_, grad_input_->numel());
    gap_backward_kernel<<<n * c, 256, 0, stream_of(*ctx_)>>>(dx, dy, n * c, spatial);
    check_hip(hipGetLastError(), "global_avg_pool_backward");
    keep_f32(*ctx_, *grad_input_, dx);
    return *grad_input_;
  }
  gap_backward_kernel<<<n * c, 256, 0, stream_of(*ctx_)>>>(grad_input_->data(), grad_output.data(), n * c,
                                                          spatial);
  check_hip(hipGetLastError(), "global_avg_pool_backward");
  return *grad_input_;
}

}  // namespace nn
