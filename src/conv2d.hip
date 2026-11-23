#include "nn/layers/conv2d.hpp"

#include "activate.hpp"
#include "check.hpp"
#include "context_impl.hpp"
#include "ensure.hpp"
#include "launch.hpp"
#include "nn/config.hpp"
#include "nn/error.hpp"
#include "nn/init.hpp"

#include <algorithm>
#include <string>

namespace nn {

struct Conv2d::Impl {};

namespace {

constexpr size_t kMaxWorkspace = 64ull * 1024ull * 1024ull;

// col is image-major (local_n, k, spatial). That is the strided-batch GEMM view of (K, spatial).
__global__ void __launch_bounds__(256) im2col_kernel(const float* x, float* col, int n0, int nb, int ic,
                                                     int h, int w, int oh, int ow, int k, int stride,
                                                     int pad) {
  const int spatial = oh * ow;
  const int k_col = ic * k * k;
  const int64_t total = static_cast<int64_t>(nb) * k_col * spatial;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int s = static_cast<int>(idx % spatial);
  const int64_t tmp = idx / spatial;
  const int kc = static_cast<int>(tmp % k_col);
  const int local_n = static_cast<int>(tmp / k_col);
  const int xw = s % ow;
  const int xh = s / ow;
  const int kx = kc % k;
  const int ky = (kc / k) % k;
  const int channel = kc / (k * k);
  const int ih = xh * stride - pad + ky;
  const int iw = xw * stride - pad + kx;
  float value = 0.f;
  if (static_cast<unsigned>(ih) < static_cast<unsigned>(h) &&
      static_cast<unsigned>(iw) < static_cast<unsigned>(w)) {
    const int n = n0 + local_n;
    value = x[(((static_cast<int64_t>(n) * ic + channel) * h + ih) * w) + iw];
  }
  col[idx] = value;
}

__global__ void __launch_bounds__(256) col2im_kernel(const float* col, float* dx, int n0, int nb, int ic,
                                                     int h, int w, int oh, int ow, int k, int stride,
                                                     int pad) {
  const int64_t total = static_cast<int64_t>(nb) * ic * h * w;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int iw = static_cast<int>(idx % w);
  const int tmp1 = static_cast<int>(idx / w);
  const int ih = tmp1 % h;
  const int tmp2 = tmp1 / h;
  const int channel = tmp2 % ic;
  const int local_n = tmp2 / ic;
  const int spatial = oh * ow;
  const int k_col = ic * k * k;
  float sum = 0.f;
  for (int ky = 0; ky < k; ++ky) {
    const int h_nom = ih + pad - ky;
    if (h_nom < 0 || (h_nom % stride) != 0) continue;
    const int xh = h_nom / stride;
    if (xh >= oh) continue;
    for (int kx = 0; kx < k; ++kx) {
      const int w_nom = iw + pad - kx;
      if (w_nom < 0 || (w_nom % stride) != 0) continue;
      const int xw = w_nom / stride;
      if (xw >= ow) continue;
      const int kc = (channel * k + ky) * k + kx;
      const int s = xh * ow + xw;
      sum += col[((static_cast<int64_t>(local_n) * k_col + kc) * spatial) + s];
    }
  }
  const int n = n0 + local_n;
  dx[(((static_cast<int64_t>(n) * ic + channel) * h + ih) * w) + iw] = sum;
}

__global__ void __launch_bounds__(256) reduce_dw_kernel(float* grad, const float* partial, int nb, int count) {
  const int idx = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
  if (idx >= count) return;
  float sum = 0.f;
  for (int i = 0; i < nb; ++i) sum += partial[static_cast<int64_t>(i) * count + idx];
  grad[idx] += sum;
}

__global__ void __launch_bounds__(256) conv_bias_kernel(float* y, const float* bias, int n, int oc,
                                                        int spatial) {
  const int64_t total = static_cast<int64_t>(n) * oc * spatial;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int c = static_cast<int>((idx / spatial) % oc);
  y[idx] += bias[c];
}

__global__ void __launch_bounds__(256) conv_bias_grad_kernel(const float* gy, float* gb, int n, int oc,
                                                            int spatial) {
  const int c = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
  if (c >= oc) return;
  float sum = 0.f;
  for (int i = 0; i < n; ++i) {
    const float* plane = gy + (static_cast<int64_t>(i) * oc + c) * spatial;
    int s = 0;
    if ((spatial % 4) == 0) {
      const float4* v = reinterpret_cast<const float4*>(plane);
      for (; s < spatial; s += 4) {
        const float4 t = v[s / 4];
        sum += t.x + t.y + t.z + t.w;
      }
    } else {
      for (; s < spatial; ++s) sum += plane[s];
    }
  }
  gb[c] += sum;
}

void gemm_batched(hipblasHandle_t handle, hipblasOperation_t trans_a, hipblasOperation_t trans_b, int m,
                  int n, int k, const float* a, int lda, hipblasStride stride_a, const float* b, int ldb,
                  hipblasStride stride_b, float beta, float* c, int ldc, hipblasStride stride_c,
                  int batch) {
  const float one = 1.f;
  check_hipblas(hipblasGemmStridedBatchedEx(handle, trans_a, trans_b, m, n, k, &one, a, HIP_R_32F, lda,
                                            stride_a, b, HIP_R_32F, ldb, stride_b, &beta, c, HIP_R_32F, ldc,
                                            stride_c, batch, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
                "hipblasGemmStridedBatchedEx");
}

}  // namespace

Conv2d::Conv2d(Context& ctx, int in_channels, int out_channels, int kernel, int stride, int pad,
               bool use_bias, std::mt19937& rng)
    : ctx_(&ctx),
      in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_(kernel),
      stride_(stride),
      pad_(pad),
      use_bias_(use_bias),
      weight_(ctx, {out_channels, in_channels, kernel, kernel}),
      grad_weight_(ctx, {out_channels, in_channels, kernel, kernel}),
      impl_(std::make_unique<Impl>()) {
  if (in_channels < 1 || out_channels < 1) throw Error("Conv2d channels");
  conv_out_size(1 + kernel, kernel, stride, pad);
  kaiming_normal(weight_, in_channels * kernel * kernel, rng);
  grad_weight_.zero();
  if (use_bias_) {
    bias_.emplace(ctx, std::vector<int>{out_channels});
    grad_bias_.emplace(ctx, std::vector<int>{out_channels});
    bias_->zero();
    grad_bias_->zero();
  }
}

Conv2d::~Conv2d() {
  if (workspace_) {
    hipStreamSynchronize(stream_of(*ctx_));
    hipFree(workspace_);
  }
}

std::vector<NamedParam> Conv2d::parameters() const {
  std::vector<NamedParam> params;
  params.push_back(NamedParam{"weight", const_cast<Tensor*>(&weight_), const_cast<Tensor*>(&grad_weight_)});
  if (bias_) {
    params.push_back(NamedParam{"bias", const_cast<Tensor*>(&*bias_), const_cast<Tensor*>(&*grad_bias_)});
  }
  return params;
}

void Conv2d::zero_grad() {
  grad_weight_.zero();
  if (grad_bias_) grad_bias_->zero();
}

void Conv2d::prepare(int n, int h, int w) {
  const int ho = conv_out_size(h, kernel_, stride_, pad_);
  const int wo = conv_out_size(w, kernel_, stride_, pad_);
  if (ho > 0x7fffffff / wo) throw Error("Conv2d output dim mismatch");
  const int spatial = ho * wo;
  const int k_col = in_channels_ * kernel_ * kernel_;
  if (k_col <= 0 || spatial <= 0) throw Error("Conv2d output dim mismatch");
  const bool direct = kernel_ == 1 && stride_ == 1 && pad_ == 0;
  const size_t col_per = direct ? 0 : static_cast<size_t>(k_col) * static_cast<size_t>(spatial) * sizeof(float);
  const size_t dw_per = static_cast<size_t>(out_channels_) * static_cast<size_t>(k_col) * sizeof(float);
  const size_t per = col_per + dw_per;
  if (per > kMaxWorkspace) {
    throw Error("conv workspace " + std::to_string(per) + " exceeds 64 MiB budget");
  }
  int chunk = n;
  if (per > 0) chunk = static_cast<int>(std::min<size_t>(n, kMaxWorkspace / per));
  if (chunk < 1) chunk = 1;
  const size_t need = per * static_cast<size_t>(chunk);
  if (need > workspace_bytes_) {
    if (workspace_) {
      hipStreamSynchronize(stream_of(*ctx_));
      hipFree(workspace_);
      workspace_ = nullptr;
      workspace_bytes_ = 0;
    }
    if (need > 0) {
      check_hip(hipMalloc(&workspace_, need), "hipMalloc");
      workspace_bytes_ = need;
    }
  }
  ensure_activation(output_, *ctx_, {n, out_channels_, ho, wo});
  ensure_activation(grad_input_, *ctx_, {n, in_channels_, h, w});
}

const Tensor& Conv2d::forward(const Tensor& input) {
  if (input.rank() != 4 || input.shape()[1] != in_channels_) throw Error("Conv2d input shape");
  const int n = input.shape()[0];
  const int h = input.shape()[2];
  const int w = input.shape()[3];
  prepare(n, h, w);
  ensure_activation(cache_x_, *ctx_, input.shape());

  const int ho = output_->shape()[2];
  const int wo = output_->shape()[3];
  const int spatial = ho * wo;
  const int k_col = in_channels_ * kernel_ * kernel_;
  const bool direct = kernel_ == 1 && stride_ == 1 && pad_ == 0;
  const size_t col_per = direct ? 0 : static_cast<size_t>(k_col) * static_cast<size_t>(spatial) * sizeof(float);
  const size_t dw_per = static_cast<size_t>(out_channels_) * static_cast<size_t>(k_col) * sizeof(float);
  const size_t per = col_per + dw_per;
  int chunk = n;
  if (per > 0) chunk = static_cast<int>(std::min<size_t>(n, kMaxWorkspace / per));
  if (chunk < 1) chunk = 1;

  hipStream_t stream = stream_of(*ctx_);
  hipblasHandle_t handle = hipblas_of(*ctx_);
  float* col = static_cast<float*>(workspace_);
  float* x_base;
  float* y_base;
  if (output_->dtype() == DType::BFloat16) {
    arena_reset(*ctx_);
    x_base = stage_f32(*ctx_, input);
    y_base = arena_f32(*ctx_, output_->numel());
  } else {
    cache_x_->copy_from(input);
    x_base = cache_x_->data();
    y_base = output_->data();
  }
  for (int n0 = 0; n0 < n; n0 += chunk) {
    const int nb = std::min(chunk, n - n0);
    const float* a = x_base + static_cast<int64_t>(n0) * k_col * spatial;
    if (!direct) {
      im2col_kernel<<<blocks_for(static_cast<int64_t>(nb) * k_col * spatial), 256, 0, stream>>>(
          x_base, col, n0, nb, in_channels_, h, w, ho, wo, kernel_, stride_, pad_);
      check_hip(hipGetLastError(), "im2col");
      a = col;
    }
    // Y_cm(spatial, OC) = Col_cm(spatial, K) * W_cm(K, OC). W row-major (OC, K) is that W_cm.
    gemm_batched(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, spatial, out_channels_, k_col, a, spatial,
                 static_cast<hipblasStride>(k_col) * spatial, weight_.data(), k_col, 0, 0.f,
                 y_base + static_cast<int64_t>(n0) * out_channels_ * spatial, spatial,
                 static_cast<hipblasStride>(out_channels_) * spatial, nb);
  }
  if (bias_) {
    conv_bias_kernel<<<blocks_for(output_->numel()), 256, 0, stream>>>(y_base, bias_->data(), n,
                                                                        out_channels_, spatial);
    check_hip(hipGetLastError(), "conv_bias");
  }
  if (output_->dtype() == DType::BFloat16) {
    keep_f32(*ctx_, *cache_x_, x_base);
    keep_f32(*ctx_, *output_, y_base);
  }
  return *output_;
}

const Tensor& Conv2d::backward(const Tensor& grad_output) {
  if (!cache_x_ || !output_ || grad_output.shape() != output_->shape()) {
    throw Error("Conv2d backward shape");
  }
  const int n = cache_x_->shape()[0];
  const int h = cache_x_->shape()[2];
  const int w = cache_x_->shape()[3];
  const int ho = output_->shape()[2];
  const int wo = output_->shape()[3];
  const int spatial = ho * wo;
  const int k_col = in_channels_ * kernel_ * kernel_;
  const bool direct = kernel_ == 1 && stride_ == 1 && pad_ == 0;
  const size_t col_per = direct ? 0 : static_cast<size_t>(k_col) * static_cast<size_t>(spatial) * sizeof(float);
  const size_t dw_per = static_cast<size_t>(out_channels_) * static_cast<size_t>(k_col) * sizeof(float);
  const size_t per = col_per + dw_per;
  int chunk = n;
  if (per > 0) chunk = static_cast<int>(std::min<size_t>(n, kMaxWorkspace / per));
  if (chunk < 1) chunk = 1;

  hipStream_t stream = stream_of(*ctx_);
  hipblasHandle_t handle = hipblas_of(*ctx_);
  float* col = static_cast<float*>(workspace_);
  const int dw_count = out_channels_ * k_col;
  float* x_base;
  const float* dy_base;
  float* dx_base;
  if (grad_input_->dtype() == DType::BFloat16) {
    arena_reset(*ctx_);
    x_base = stage_f32(*ctx_, *cache_x_);
    dy_base = stage_f32(*ctx_, grad_output);
    dx_base = arena_f32(*ctx_, grad_input_->numel());
  } else {
    x_base = cache_x_->data();
    dy_base = grad_output.data();
    dx_base = grad_input_->data();
  }
  for (int n0 = 0; n0 < n; n0 += chunk) {
    const int nb = std::min(chunk, n - n0);
    float* partial = col + (direct ? 0 : static_cast<int64_t>(nb) * k_col * spatial);
    const float* a = x_base + static_cast<int64_t>(n0) * k_col * spatial;
    if (!direct) {
      im2col_kernel<<<blocks_for(static_cast<int64_t>(nb) * k_col * spatial), 256, 0, stream>>>(
          x_base, col, n0, nb, in_channels_, h, w, ho, wo, kernel_, stride_, pad_);
      check_hip(hipGetLastError(), "im2col");
      a = col;
    }
    // dW_cm(K, OC) = Col_cm(spatial, K)^T * dY_cm(spatial, OC), one matrix per image.
    gemm_batched(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, k_col, out_channels_, spatial, a, spatial,
                 static_cast<hipblasStride>(k_col) * spatial,
                 dy_base + static_cast<int64_t>(n0) * out_channels_ * spatial, spatial,
                 static_cast<hipblasStride>(out_channels_) * spatial, 0.f, partial, k_col,
                 static_cast<hipblasStride>(dw_count), nb);
    reduce_dw_kernel<<<blocks_for(dw_count), 256, 0, stream>>>(grad_weight_.data(), partial, nb, dw_count);
    check_hip(hipGetLastError(), "conv_dw");

    float* dx = direct ? dx_base + static_cast<int64_t>(n0) * k_col * spatial : col;
    // dCol_cm(spatial, K) = dY_cm(spatial, OC) * W_cm(K, OC)^T.
    gemm_batched(handle, HIPBLAS_OP_N, HIPBLAS_OP_T, spatial, k_col, out_channels_,
                 dy_base + static_cast<int64_t>(n0) * out_channels_ * spatial, spatial,
                 static_cast<hipblasStride>(out_channels_) * spatial, weight_.data(), k_col, 0, 0.f, dx,
                 spatial, static_cast<hipblasStride>(k_col) * spatial, nb);
    if (!direct) {
      col2im_kernel<<<blocks_for(static_cast<int64_t>(nb) * in_channels_ * h * w), 256, 0, stream>>>(
          col, dx_base, n0, nb, in_channels_, h, w, ho, wo, kernel_, stride_, pad_);
      check_hip(hipGetLastError(), "col2im");
    }
  }
  if (grad_bias_) {
    conv_bias_grad_kernel<<<blocks_for(out_channels_), 256, 0, stream>>>(dy_base, grad_bias_->data(), n,
                                                                         out_channels_, spatial);
    check_hip(hipGetLastError(), "conv_bias_grad");
  }
  if (grad_input_->dtype() == DType::BFloat16) keep_f32(*ctx_, *grad_input_, dx_base);
  return *grad_input_;
}

}  // namespace nn
