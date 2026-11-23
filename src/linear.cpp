#include "nn/layers/linear.hpp"

#include "activate.hpp"
#include "check.hpp"
#include "context_impl.hpp"
#include "ensure.hpp"
#include "kernels.hpp"
#include "nn/error.hpp"
#include "nn/init.hpp"

#include <hipblas/hipblas.h>

#include <cstdint>

namespace nn {
namespace {

int gemm_rows(const std::vector<int>& shape) {
  if (shape.size() == 2) return shape[0];
  // Rank 3 stays (N, T, F). rows is N*T on data(); reshape() would copy.
  if (shape.size() != 3) throw Error("Linear input shape");
  return static_cast<int>(static_cast<int64_t>(shape[0]) * shape[1]);
}

}  // namespace

Linear::Linear(Context& ctx, int in_features, int out_features, std::mt19937& rng)
    : Linear(ctx, in_features, out_features, true, rng) {}

Linear::Linear(Context& ctx, int in_features, int out_features, bool use_bias, std::mt19937& rng)
    : ctx_(&ctx),
      in_features_(in_features),
      out_features_(out_features),
      use_bias_(use_bias),
      weight_(ctx, {out_features, in_features}),
      grad_weight_(ctx, {out_features, in_features}) {
  if (in_features < 1 || out_features < 1) throw Error("Linear size");
  kaiming_normal(weight_, in_features, rng);
  grad_weight_.zero();
  if (use_bias_) {
    bias_.emplace(ctx, std::vector<int>{out_features});
    grad_bias_.emplace(ctx, std::vector<int>{out_features});
    bias_->zero();
    grad_bias_->zero();
  }
}

std::vector<NamedParam> Linear::parameters() const {
  std::vector<NamedParam> params;
  params.push_back(
      NamedParam{"weight", const_cast<Tensor*>(&weight_), const_cast<Tensor*>(&grad_weight_)});
  if (bias_) {
    params.push_back(
        NamedParam{"bias", const_cast<Tensor*>(&*bias_), const_cast<Tensor*>(&*grad_bias_)});
  }
  return params;
}

void Linear::zero_grad() {
  grad_weight_.zero();
  if (grad_bias_) grad_bias_->zero();
}

const Tensor& Linear::forward(const Tensor& input) {
  const std::vector<int>& shape = input.shape();
  if ((shape.size() != 2 && shape.size() != 3) || shape.back() != in_features_) {
    throw Error("Linear input shape");
  }
  const int rows = gemm_rows(shape);
  std::vector<int> out_shape = shape;
  out_shape.back() = out_features_;
  ensure_activation(output_, *ctx_, out_shape);
  ensure_activation(cache_x_, *ctx_, shape);
  cache_x_->copy_from(input);

  if (output_->dtype() == DType::BFloat16) {
    arena_reset(*ctx_);
    const uint16_t* weight_bf = stage_bf16(*ctx_, weight_);
    const int64_t stride_c = static_cast<int64_t>(rows) * out_features_;
    const int64_t stride_a = static_cast<int64_t>(rows) * in_features_;
    const int64_t stride_b = static_cast<int64_t>(out_features_) * in_features_;
    gemm_nt_bf16(*ctx_, output_->bf16_data(), true, cache_x_->bf16_data(), weight_bf, rows, out_features_,
                 in_features_, 1, out_features_, in_features_, in_features_, stride_c, stride_a, stride_b,
                 1.f, 0.f);
    if (bias_) launch_add_bias_bf16(*ctx_, output_->bf16_data(), bias_->data(), rows, out_features_);
    return *output_;
  }

  // Row-major y = x W^T. cuBLAS is column-major, so this is C = W x^T
  // with the leading dimensions equal to the row-major inner sizes.
  const float one = 1.f;
  const float zero = 0.f;
  check_hipblas(hipblasGemmEx(hipblas_of(*ctx_), HIPBLAS_OP_T, HIPBLAS_OP_N, out_features_, rows,
                            in_features_, &one, weight_.data(), HIP_R_32F, in_features_,
                            input.data(), HIP_R_32F, in_features_, &zero, output_->data(),
                            HIP_R_32F, out_features_, HIPBLAS_COMPUTE_32F,
                            HIPBLAS_GEMM_DEFAULT),
               "hipblasGemmEx");
  if (bias_) launch_add_bias(stream_of(*ctx_), output_->data(), bias_->data(), rows, out_features_);
  return *output_;
}

const Tensor& Linear::backward(const Tensor& grad_output) {
  if (!cache_x_ || grad_output.shape().size() != cache_x_->shape().size() ||
      grad_output.shape().back() != out_features_) {
    throw Error("Linear backward shape");
  }
  const std::vector<int>& in_shape = cache_x_->shape();
  for (size_t axis = 0; axis + 1 < in_shape.size(); ++axis) {
    if (grad_output.shape()[axis] != in_shape[axis]) throw Error("Linear backward shape");
  }
  if (in_shape.size() != 2 && in_shape.size() != 3) throw Error("Linear backward shape");
  const int rows = gemm_rows(in_shape);
  ensure_activation(grad_input_, *ctx_, in_shape);
  if (grad_input_->dtype() == DType::BFloat16) {
    arena_reset(*ctx_);
    const uint16_t* grad_bf = stage_bf16(*ctx_, grad_output);
    const uint16_t* weight_bf = stage_bf16(*ctx_, weight_);
    const int64_t stride_g = static_cast<int64_t>(rows) * out_features_;
    const int64_t stride_x = static_cast<int64_t>(rows) * in_features_;
    const int64_t stride_w = static_cast<int64_t>(out_features_) * in_features_;
    gemm_nn_bf16(*ctx_, grad_input_->bf16_data(), true, grad_bf, weight_bf, rows, in_features_,
                 out_features_, 1, in_features_, out_features_, in_features_, stride_x, stride_g, stride_w,
                 1.f, 0.f);
    gemm_tn_bf16(*ctx_, grad_weight_.data(), false, grad_bf, cache_x_->bf16_data(), out_features_,
                 in_features_, rows, 1, in_features_, out_features_, in_features_, stride_w, stride_g,
                 stride_x, 1.f, 1.f);
    if (grad_bias_) {
      launch_bias_grad_bf16(*ctx_, grad_bf, grad_bias_->data(), rows, out_features_);
    }
    return *grad_input_;
  }
  const float one = 1.f;
  const float zero = 0.f;
  hipblasHandle_t handle = hipblas_of(*ctx_);
  check_hipblas(hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, in_features_, rows, out_features_,
                            &one, weight_.data(), HIP_R_32F, in_features_, grad_output.data(),
                            HIP_R_32F, out_features_, &zero, grad_input_->data(), HIP_R_32F,
                            in_features_, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
               "hipblasGemmEx");
  check_hipblas(hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_T, in_features_, out_features_, rows,
                            &one, cache_x_->data(), HIP_R_32F, in_features_, grad_output.data(),
                            HIP_R_32F, out_features_, &one, grad_weight_.data(), HIP_R_32F,
                            in_features_, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
               "hipblasGemmEx");
  if (grad_bias_) {
    launch_bias_grad(stream_of(*ctx_), grad_output.data(), grad_bias_->data(), rows, out_features_);
  }
  return *grad_input_;
}

}  // namespace nn
