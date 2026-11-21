#include "nn/layers/max_pool2d.hpp"

#include "check.hpp"
#include "context_impl.hpp"
#include "ensure.hpp"
#include "launch.hpp"
#include "nn/config.hpp"
#include "nn/error.hpp"

namespace nn {

struct MaxPool2d::Impl {
  int* index = nullptr;
  int64_t capacity = 0;
};

namespace {

__global__ void __launch_bounds__(256) maxpool_forward_kernel(float* y, int* index, const float* x, int n,
                                                              int c, int h, int w, int ho, int wo, int k,
                                                              int stride) {
  const int64_t total = static_cast<int64_t>(n) * c * ho * wo;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int ow = static_cast<int>(idx % wo);
  const int oh = static_cast<int>((idx / wo) % ho);
  const int ch = static_cast<int>((idx / (static_cast<int64_t>(wo) * ho)) % c);
  const int ni = static_cast<int>(idx / (static_cast<int64_t>(wo) * ho * c));
  const int ih0 = oh * stride;
  const int iw0 = ow * stride;
  float best = x[(((static_cast<int64_t>(ni) * c + ch) * h + ih0) * w) + iw0];
  int best_s = ih0 * w + iw0;
  for (int ky = 0; ky < k; ++ky) {
    const int ih = ih0 + ky;
    if (ih >= h) continue;
    for (int kx = 0; kx < k; ++kx) {
      const int iw = iw0 + kx;
      if (iw >= w) continue;
      const float value = x[(((static_cast<int64_t>(ni) * c + ch) * h + ih) * w) + iw];
      if (value > best) {
        best = value;
        best_s = ih * w + iw;
      }
    }
  }
  y[idx] = best;
  index[idx] = best_s;
}

__global__ void __launch_bounds__(256) maxpool_backward_kernel(float* dx, const float* dy, const int* index,
                                                               int n, int c, int h, int w, int ho, int wo) {
  const int64_t total = static_cast<int64_t>(n) * c * ho * wo;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int ow = static_cast<int>(idx % wo);
  const int oh = static_cast<int>((idx / wo) % ho);
  const int ch = static_cast<int>((idx / (static_cast<int64_t>(wo) * ho)) % c);
  const int ni = static_cast<int>(idx / (static_cast<int64_t>(wo) * ho * c));
  const int spatial = index[idx];
  const int ih = spatial / w;
  const int iw = spatial - ih * w;
  atomicAdd(dx + (((static_cast<int64_t>(ni) * c + ch) * h + ih) * w) + iw, dy[idx]);
  (void)oh;
  (void)ow;
}

}  // namespace

MaxPool2d::MaxPool2d(Context& ctx, int kernel, int stride)
    : ctx_(&ctx), kernel_(kernel), stride_(stride), impl_(std::make_unique<Impl>()) {
  conv_out_size(kernel, kernel, stride, 0);
}

MaxPool2d::~MaxPool2d() {
  if (impl_ && impl_->index) {
    if (ctx_) hipStreamSynchronize(stream_of(*ctx_));
    hipFree(impl_->index);
  }
}

std::vector<NamedParam> MaxPool2d::parameters() const { return {}; }

void MaxPool2d::zero_grad() {}

const Tensor& MaxPool2d::forward(const Tensor& input) {
  if (input.rank() != 4) throw Error("MaxPool2d input rank");
  const int n = input.shape()[0];
  const int c = input.shape()[1];
  const int h = input.shape()[2];
  const int w = input.shape()[3];
  const int ho = conv_out_size(h, kernel_, stride_, 0);
  const int wo = conv_out_size(w, kernel_, stride_, 0);
  ensure_tensor(cache_x_, *ctx_, input.shape());
  ensure_tensor(output_, *ctx_, {n, c, ho, wo});
  ensure_tensor(cache_y_, *ctx_, output_->shape());
  cache_x_->copy_from(input);
  const int64_t outs = output_->numel();
  if (impl_->capacity < outs) {
    if (impl_->index) {
      hipStreamSynchronize(stream_of(*ctx_));
      hipFree(impl_->index);
      impl_->index = nullptr;
      impl_->capacity = 0;
    }
    check_hip(hipMalloc(reinterpret_cast<void**>(&impl_->index), static_cast<size_t>(outs) * sizeof(int)),
              "hipMalloc");
    impl_->capacity = outs;
  }
  maxpool_forward_kernel<<<blocks_for(outs), 256, 0, stream_of(*ctx_)>>>(
      output_->data(), impl_->index, cache_x_->data(), n, c, h, w, ho, wo, kernel_, stride_);
  check_hip(hipGetLastError(), "max_pool_forward");
  cache_y_->copy_from(*output_);
  return *output_;
}

const Tensor& MaxPool2d::backward(const Tensor& grad_output) {
  if (!cache_x_ || !cache_y_ || grad_output.shape() != cache_y_->shape()) {
    throw Error("MaxPool2d backward shape");
  }
  ensure_tensor(grad_input_, *ctx_, cache_x_->shape());
  grad_input_->zero();
  const int n = cache_x_->shape()[0];
  const int c = cache_x_->shape()[1];
  const int h = cache_x_->shape()[2];
  const int w = cache_x_->shape()[3];
  const int ho = cache_y_->shape()[2];
  const int wo = cache_y_->shape()[3];
  maxpool_backward_kernel<<<blocks_for(grad_output.numel()), 256, 0, stream_of(*ctx_)>>>(
      grad_input_->data(), grad_output.data(), impl_->index, n, c, h, w, ho, wo);
  check_hip(hipGetLastError(), "max_pool_backward");
  return *grad_input_;
}

}  // namespace nn
