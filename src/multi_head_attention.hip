#include "nn/layers/multi_head_attention.hpp"

#include "activate.hpp"
#include "check.hpp"
#include "context_impl.hpp"
#include "ensure.hpp"
#include "kernels.hpp"
#include "launch.hpp"
#include "nn/error.hpp"
#include "reduce_device.hpp"

#include <hipblas/hipblas.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace nn {
namespace {

constexpr int kThreads = 256;
constexpr float kMask = -1.0e9f;

// (N,T,H,Dh) and (N,H,T,Dh) are not the same strides. reshape() would only copy.
__global__ void split_heads_kernel(float* dst, const float* src, int time, int heads, int dh,
                                   int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t d = i % dh;
  const int64_t h = (i / dh) % heads;
  const int64_t t = (i / (static_cast<int64_t>(dh) * heads)) % time;
  const int64_t batch = i / (static_cast<int64_t>(dh) * heads * time);
  const int64_t o = ((batch * heads + h) * time + t) * dh + d;
  dst[o] = src[i];
}

__global__ void split_heads4_kernel(float* dst, const float* src, int time, int heads, int dh,
                                    int64_t n4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  const int dh4 = dh / 4;
  const int64_t d = i % dh4;
  const int64_t h = (i / dh4) % heads;
  const int64_t t = (i / (static_cast<int64_t>(dh4) * heads)) % time;
  const int64_t batch = i / (static_cast<int64_t>(dh4) * heads * time);
  const int64_t o = ((batch * heads + h) * time + t) * dh4 + d;
  reinterpret_cast<float4*>(dst)[o] = reinterpret_cast<const float4*>(src)[i];
}

__global__ void merge_heads_kernel(float* dst, const float* src, int time, int heads, int dh,
                                   int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t d = i % dh;
  const int64_t h = (i / dh) % heads;
  const int64_t t = (i / (static_cast<int64_t>(dh) * heads)) % time;
  const int64_t batch = i / (static_cast<int64_t>(dh) * heads * time);
  const int64_t s = ((batch * heads + h) * time + t) * dh + d;
  dst[i] = src[s];
}

__global__ void merge_heads4_kernel(float* dst, const float* src, int time, int heads, int dh,
                                    int64_t n4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  const int dh4 = dh / 4;
  const int64_t d = i % dh4;
  const int64_t h = (i / dh4) % heads;
  const int64_t t = (i / (static_cast<int64_t>(dh4) * heads)) % time;
  const int64_t batch = i / (static_cast<int64_t>(dh4) * heads * time);
  const int64_t s = ((batch * heads + h) * time + t) * dh4 + d;
  reinterpret_cast<float4*>(dst)[i] = reinterpret_cast<const float4*>(src)[s];
}

// llama.cpp LLAMA_ROPE_TYPE_NORM: adjacent pairs inside the head, base 10000, full head dim.
// y0 = x0*cos - x1*sin, y1 = x0*sin + x1*cos. Inverse is the transpose of that rotation.
__global__ void rope_qk_kernel(float* q, float* k, int time, int dh, int pos0, int64_t npairs,
                               int inverse, float base_theta) {
  const int64_t pair_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_index >= npairs) return;
  const int pairs = dh / 2;
  const int pair = static_cast<int>(pair_index % pairs);
  const int64_t vec = pair_index / pairs;
  const int t = static_cast<int>(vec % time);
  const int64_t base = vec * dh + static_cast<int64_t>(pair) * 2;
  const float theta =
      static_cast<float>(pos0 + t) *
      expf(-logf(base_theta) * (2.f * static_cast<float>(pair) / static_cast<float>(dh)));
  const float c = cosf(theta);
  const float s = sinf(theta);
  const float q0 = q[base];
  const float q1 = q[base + 1];
  const float k0 = k[base];
  const float k1 = k[base + 1];
  if (inverse) {
    q[base] = q0 * c + q1 * s;
    q[base + 1] = -q0 * s + q1 * c;
    k[base] = k0 * c + k1 * s;
    k[base + 1] = -k0 * s + k1 * c;
  } else {
    q[base] = q0 * c - q1 * s;
    q[base + 1] = q0 * s + q1 * c;
    k[base] = k0 * c - k1 * s;
    k[base + 1] = k0 * s + k1 * c;
  }
}

// src is (N, H, 1, Dh). dst is (N, H, T_max, Dh). One token, so the copy is Dh-wide.
// dst is (N, Hkv*group, T, Dh). src is (N, Hkv, T, Dh). Each KV head is copied `group` times.
// The head axis is outside time, matching split_heads.
__global__ void repeat_kv_kernel(float* dst, const float* src, int time, int kv_heads, int group, int dh,
                                 int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int heads = kv_heads * group;
  const int64_t d = i % dh;
  const int64_t t = (i / dh) % time;
  const int64_t h = (i / (static_cast<int64_t>(dh) * time)) % heads;
  const int64_t batch = i / (static_cast<int64_t>(dh) * time * heads);
  const int64_t hkv = h / group;
  const int64_t s = ((batch * kv_heads + hkv) * time + t) * dh + d;
  dst[i] = src[s];
}

__global__ void repeat_kv4_kernel(float* dst, const float* src, int time, int kv_heads, int group, int dh,
                                  int64_t n4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  const int dh4 = dh / 4;
  const int heads = kv_heads * group;
  const int64_t d = i % dh4;
  const int64_t t = (i / dh4) % time;
  const int64_t h = (i / (static_cast<int64_t>(dh4) * time)) % heads;
  const int64_t batch = i / (static_cast<int64_t>(dh4) * time * heads);
  const int64_t hkv = h / group;
  const int64_t s = ((batch * kv_heads + hkv) * time + t) * dh4 + d;
  reinterpret_cast<float4*>(dst)[i] = reinterpret_cast<const float4*>(src)[s];
}

// dst is (N, Hkv, T, Dh). src is (N, Hkv*group, T, Dh). Gradients of repeated heads add.
__global__ void sum_kv_kernel(float* dst, const float* src, int time, int kv_heads, int group, int dh,
                              int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int heads = kv_heads * group;
  const int64_t d = i % dh;
  const int64_t t = (i / dh) % time;
  const int64_t hkv = (i / (static_cast<int64_t>(dh) * time)) % kv_heads;
  const int64_t batch = i / (static_cast<int64_t>(dh) * time * kv_heads);
  float sum = 0.f;
  for (int g = 0; g < group; ++g) {
    const int64_t h = hkv * group + g;
    const int64_t s = ((batch * heads + h) * time + t) * dh + d;
    sum += src[s];
  }
  dst[i] = sum;
}

__global__ void write_kv_kernel(float* dst, const float* src, int t, int t_max, int dh, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int d = static_cast<int>(i % dh);
  const int64_t head = i / dh;
  dst[(head * static_cast<int64_t>(t_max) + t) * dh + d] = src[i];
}

__global__ void write_kv_bf16_kernel(uint16_t* dst, const uint16_t* src, int t, int t_max, int dh,
                                     int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int d = static_cast<int>(i % dh);
  const int64_t head = i / dh;
  dst[(head * static_cast<int64_t>(t_max) + t) * dh + d] = src[i];
}

__device__ float masked_score(float value, int col, int query, int causal) {
  if (causal && col > query) return kMask;
  return value;
}

__global__ void softmax_rows_kernel(float* out, const float* in, int cols, int causal) {
  __shared__ float buffer[32];
  const int row = static_cast<int>(blockIdx.x);
  const int query = row % cols;
  const int64_t base = static_cast<int64_t>(row) * cols;
  const float* row_in = in + base;
  float* row_out = out + base;

  float max_v = -INFINITY;
  for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) {
    max_v = fmaxf(max_v, masked_score(row_in[col], col, query, causal));
  }
  max_v = block_max(max_v, buffer);

  float sum = 0.f;
  for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) {
    sum += expf(masked_score(row_in[col], col, query, causal) - max_v);
  }
  sum = block_sum(sum, buffer);
  const float inv = 1.f / sum;
  for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) {
    row_out[col] = expf(masked_score(row_in[col], col, query, causal) - max_v) * inv;
  }
}

__global__ void softmax_backward_kernel(float* ds, const float* probs, const float* dp, int cols,
                                        int causal) {
  __shared__ float buffer[32];
  const int row = static_cast<int>(blockIdx.x);
  const int query = row % cols;
  const int64_t base = static_cast<int64_t>(row) * cols;

  float dot = 0.f;
  for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) {
    dot += dp[base + col] * probs[base + col];
  }
  dot = block_sum(dot, buffer);

  for (int col = static_cast<int>(threadIdx.x); col < cols; col += kThreads) {
    float grad = probs[base + col] * (dp[base + col] - dot);
    // The mask overwrote the score with a constant, so that score has no gradient.
    if (causal && col > query) grad = 0.f;
    ds[base + col] = grad;
  }
}

int blocks_for(int64_t n) {
  return static_cast<int>((n + kThreads - 1) / kThreads);
}

void launch_split_heads(hipStream_t stream, float* dst, const float* src, int time, int heads,
                        int dh, int64_t n) {
  if ((dh % 4) == 0 && aligned16(dst) && aligned16(src)) {
    split_heads4_kernel<<<blocks_for(n / 4), kThreads, 0, stream>>>(dst, src, time, heads, dh, n / 4);
  } else {
    split_heads_kernel<<<blocks_for(n), kThreads, 0, stream>>>(dst, src, time, heads, dh, n);
  }
  check_hip(hipGetLastError(), "mha_split_heads");
}

void launch_merge_heads(hipStream_t stream, float* dst, const float* src, int time, int heads,
                        int dh, int64_t n) {
  if ((dh % 4) == 0 && aligned16(dst) && aligned16(src)) {
    merge_heads4_kernel<<<blocks_for(n / 4), kThreads, 0, stream>>>(dst, src, time, heads, dh, n / 4);
  } else {
    merge_heads_kernel<<<blocks_for(n), kThreads, 0, stream>>>(dst, src, time, heads, dh, n);
  }
  check_hip(hipGetLastError(), "mha_merge_heads");
}

void launch_write_kv(hipStream_t stream, float* dst, const float* src, int t, int t_max, int dh,
                     int64_t n) {
  write_kv_kernel<<<blocks_for(n), kThreads, 0, stream>>>(dst, src, t, t_max, dh, n);
  check_hip(hipGetLastError(), "mha_write_kv");
}

void launch_write_kv_bf16(hipStream_t stream, uint16_t* dst, const uint16_t* src, int t, int t_max, int dh,
                          int64_t n) {
  write_kv_bf16_kernel<<<blocks_for(n), kThreads, 0, stream>>>(dst, src, t, t_max, dh, n);
  check_hip(hipGetLastError(), "mha_write_kv");
}

void launch_rope(hipStream_t stream, float* q, float* k, int time, int dh, int pos0, int64_t n,
                 bool inverse, float base_theta) {
  const int64_t npairs = n / 2;
  rope_qk_kernel<<<blocks_for(npairs), kThreads, 0, stream>>>(q, k, time, dh, pos0, npairs,
                                                             inverse ? 1 : 0, base_theta);
  check_hip(hipGetLastError(), "mha_rope");
}

void launch_repeat_kv(hipStream_t stream, float* dst, const float* src, int time, int kv_heads,
                      int group, int dh, int64_t n) {
  if ((dh % 4) == 0 && aligned16(dst) && aligned16(src)) {
    repeat_kv4_kernel<<<blocks_for(n / 4), kThreads, 0, stream>>>(dst, src, time, kv_heads, group, dh,
                                                                 n / 4);
  } else {
    repeat_kv_kernel<<<blocks_for(n), kThreads, 0, stream>>>(dst, src, time, kv_heads, group, dh, n);
  }
  check_hip(hipGetLastError(), "mha_repeat_kv");
}

void launch_sum_kv(hipStream_t stream, float* dst, const float* src, int time, int kv_heads, int group,
                   int dh, int64_t n) {
  sum_kv_kernel<<<blocks_for(n), kThreads, 0, stream>>>(dst, src, time, kv_heads, group, dh, n);
  check_hip(hipGetLastError(), "mha_sum_kv");
}

// Row-major (T, Dh) occupies the same buffer as column-major (Dh, T).
// S = scale * Q K^T  <=>  S_cm = scale * K_cm^T Q_cm.
void gemm_scores(hipblasHandle_t handle, const float* q, const float* k, float* scores, int time,
                 int dh, int batch, float scale) {
  const float beta = 0.f;
  const long long stride_qk = static_cast<long long>(time) * dh;
  const long long stride_s = static_cast<long long>(time) * time;
  check_hipblas(hipblasGemmStridedBatchedEx(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, time, time, dh, &scale,
                                          k, HIP_R_32F, dh, stride_qk, q, HIP_R_32F, dh,
                                          stride_qk, &beta, scores, HIP_R_32F, time, stride_s,
                                          batch, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
               "hipblasGemmStridedBatchedEx");
}

// dQ = scale * dS K.  dQ_cm = scale * K_cm dS_cm.
void gemm_dq(hipblasHandle_t handle, const float* k, const float* ds, float* dq, int time, int dh,
             int batch, float scale) {
  const float beta = 0.f;
  const long long stride_qk = static_cast<long long>(time) * dh;
  const long long stride_s = static_cast<long long>(time) * time;
  check_hipblas(hipblasGemmStridedBatchedEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, dh, time, time, &scale,
                                          k, HIP_R_32F, dh, stride_qk, ds, HIP_R_32F, time,
                                          stride_s, &beta, dq, HIP_R_32F, dh, stride_qk, batch,
                                          HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
               "hipblasGemmStridedBatchedEx");
}

// dK = scale * dS^T Q.  dK_cm = scale * Q_cm dS_cm^T.
void gemm_dk(hipblasHandle_t handle, const float* q, const float* ds, float* dk, int time, int dh,
             int batch, float scale) {
  const float beta = 0.f;
  const long long stride_qk = static_cast<long long>(time) * dh;
  const long long stride_s = static_cast<long long>(time) * time;
  check_hipblas(hipblasGemmStridedBatchedEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_T, dh, time, time, &scale,
                                          q, HIP_R_32F, dh, stride_qk, ds, HIP_R_32F, time,
                                          stride_s, &beta, dk, HIP_R_32F, dh, stride_qk, batch,
                                          HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
               "hipblasGemmStridedBatchedEx");
}

void append_params(std::vector<NamedParam>& out, const Layer& layer, const char* prefix) {
  for (NamedParam param : layer.parameters()) {
    param.name = std::string(prefix) + param.name;
    out.push_back(param);
  }
}

Context* attention_context(Context& ctx, int d_model, int num_heads, int head_dim, int num_kv_heads,
                          bool use_rope, float rope_theta, float rms_eps) {
  if (d_model < 1 || num_heads < 1) throw Error("MultiHeadAttention size");
  const int dh = head_dim == 0 ? d_model / num_heads : head_dim;
  const int kv = num_kv_heads == 0 ? num_heads : num_kv_heads;
  if (head_dim == 0 && d_model % num_heads != 0) throw Error("MultiHeadAttention size");
  if (dh < 1 || kv < 1 || kv > num_heads || num_heads % kv != 0) throw Error("MultiHeadAttention size");
  if (dh > 0x7fffffff / num_heads || dh > 0x7fffffff / kv) throw Error("MultiHeadAttention size");
  if (use_rope && dh % 2 != 0) throw std::invalid_argument("use_rope");
  if (!(rope_theta > 0.f) || !std::isfinite(rope_theta)) throw std::invalid_argument("rope_theta");
  if (!(rms_eps > 0.f) || !std::isfinite(rms_eps)) throw std::invalid_argument("rms_eps");
  return &ctx;
}

}  // namespace

MultiHeadAttention::MultiHeadAttention(Context& ctx, int d_model, int num_heads, bool causal,
                                       bool use_bias, bool use_rope, std::mt19937& rng, int head_dim,
                                       int num_kv_heads, bool qk_norm, float rope_theta, float rms_eps)
    : ctx_(attention_context(ctx, d_model, num_heads, head_dim, num_kv_heads, use_rope, rope_theta,
                             rms_eps)),
      d_model_(d_model),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads == 0 ? num_heads : num_kv_heads),
      head_dim_(head_dim == 0 ? d_model / num_heads : head_dim),
      q_dim_(num_heads * (head_dim == 0 ? d_model / num_heads : head_dim)),
      kv_dim_((num_kv_heads == 0 ? num_heads : num_kv_heads) *
              (head_dim == 0 ? d_model / num_heads : head_dim)),
      causal_(causal),
      use_rope_(use_rope),
      qk_norm_(qk_norm),
      rope_theta_(rope_theta),
      q_(std::make_unique<Linear>(*ctx_, d_model, num_heads * (head_dim == 0 ? d_model / num_heads : head_dim),
                                  use_bias, rng)),
      k_(std::make_unique<Linear>(
          *ctx_, d_model,
          (num_kv_heads == 0 ? num_heads : num_kv_heads) * (head_dim == 0 ? d_model / num_heads : head_dim),
          use_bias, rng)),
      v_(std::make_unique<Linear>(
          *ctx_, d_model,
          (num_kv_heads == 0 ? num_heads : num_kv_heads) * (head_dim == 0 ? d_model / num_heads : head_dim),
          use_bias, rng)),
      out_(std::make_unique<Linear>(*ctx_, num_heads * (head_dim == 0 ? d_model / num_heads : head_dim), d_model,
                                    use_bias, rng)) {
  if (qk_norm) {
    q_norm_ = std::make_unique<RMSNorm>(*ctx_, head_dim_, rms_eps);
    k_norm_ = std::make_unique<RMSNorm>(*ctx_, head_dim_, rms_eps);
  }
}

std::vector<NamedParam> MultiHeadAttention::parameters() const {
  std::vector<NamedParam> params;
  append_params(params, *q_, "q.");
  append_params(params, *k_, "k.");
  append_params(params, *v_, "v.");
  append_params(params, *out_, "out.");
  if (q_norm_) append_params(params, *q_norm_, "q_norm.");
  if (k_norm_) append_params(params, *k_norm_, "k_norm.");
  return params;
}

void MultiHeadAttention::zero_grad() {
  q_->zero_grad();
  k_->zero_grad();
  v_->zero_grad();
  out_->zero_grad();
  if (q_norm_) q_norm_->zero_grad();
  if (k_norm_) k_norm_->zero_grad();
}

void MultiHeadAttention::lay_qkv(const Tensor& q, const Tensor& k, const Tensor& v, int time) {
  const int n = q.shape()[0];
  const int heads = num_heads_;
  const int kv = num_kv_heads_;
  const int group = heads / kv;
  const int dh = head_dim_;
  const int64_t q_n = static_cast<int64_t>(n) * time * q_dim_;
  const int64_t kv_n = static_cast<int64_t>(n) * time * kv_dim_;
  const int64_t full_n = static_cast<int64_t>(n) * heads * time * dh;
  hipStream_t stream = stream_of(*ctx_);
  if (q_heads_->dtype() == DType::BFloat16) {
    arena_reset(*ctx_);
    float* qf = stage_f32(*ctx_, q);
    float* kf = stage_f32(*ctx_, k);
    float* vf = stage_f32(*ctx_, v);
    float* qh = arena_f32(*ctx_, q_n);
    launch_split_heads(stream, qh, qf, time, heads, dh, q_n);
    if (group == 1) {
      float* kh = arena_f32(*ctx_, kv_n);
      float* vh = arena_f32(*ctx_, kv_n);
      launch_split_heads(stream, kh, kf, time, kv, dh, kv_n);
      launch_split_heads(stream, vh, vf, time, kv, dh, kv_n);
      keep_f32(*ctx_, *q_heads_, qh);
      keep_f32(*ctx_, *k_heads_, kh);
      keep_f32(*ctx_, *v_heads_, vh);
    } else {
      ensure_activation(k_kv_, *ctx_, {n, kv, time, dh});
      ensure_activation(v_kv_, *ctx_, {n, kv, time, dh});
      float* kkv = arena_f32(*ctx_, kv_n);
      float* vkv = arena_f32(*ctx_, kv_n);
      launch_split_heads(stream, kkv, kf, time, kv, dh, kv_n);
      launch_split_heads(stream, vkv, vf, time, kv, dh, kv_n);
      keep_f32(*ctx_, *q_heads_, qh);
      keep_f32(*ctx_, *k_kv_, kkv);
      keep_f32(*ctx_, *v_kv_, vkv);
    }
    if (q_norm_) q_heads_->copy_from(q_norm_->forward(*q_heads_));
    if (group == 1) {
      if (k_norm_) k_heads_->copy_from(k_norm_->forward(*k_heads_));
    } else {
      if (k_norm_) k_kv_->copy_from(k_norm_->forward(*k_kv_));
      arena_reset(*ctx_);
      float* ks = stage_f32(*ctx_, *k_kv_);
      float* vs = stage_f32(*ctx_, *v_kv_);
      float* kh = arena_f32(*ctx_, full_n);
      float* vh = arena_f32(*ctx_, full_n);
      launch_repeat_kv(stream, kh, ks, time, kv, group, dh, full_n);
      launch_repeat_kv(stream, vh, vs, time, kv, group, dh, full_n);
      keep_f32(*ctx_, *k_heads_, kh);
      keep_f32(*ctx_, *v_heads_, vh);
    }
    return;
  }
  launch_split_heads(stream, q_heads_->data(), q.data(), time, heads, dh, q_n);
  if (q_norm_) q_heads_->copy_from(q_norm_->forward(*q_heads_));
  if (group == 1) {
    launch_split_heads(stream, k_heads_->data(), k.data(), time, kv, dh, kv_n);
    launch_split_heads(stream, v_heads_->data(), v.data(), time, kv, dh, kv_n);
    if (k_norm_) k_heads_->copy_from(k_norm_->forward(*k_heads_));
  } else {
    ensure_activation(k_kv_, *ctx_, {n, kv, time, dh});
    ensure_activation(v_kv_, *ctx_, {n, kv, time, dh});
    launch_split_heads(stream, k_kv_->data(), k.data(), time, kv, dh, kv_n);
    launch_split_heads(stream, v_kv_->data(), v.data(), time, kv, dh, kv_n);
    if (k_norm_) k_kv_->copy_from(k_norm_->forward(*k_kv_));
    launch_repeat_kv(stream, k_heads_->data(), k_kv_->data(), time, kv, group, dh, full_n);
    launch_repeat_kv(stream, v_heads_->data(), v_kv_->data(), time, kv, group, dh, full_n);
  }
}

const Tensor& MultiHeadAttention::forward(const Tensor& input) {
  const std::vector<int>& shape = input.shape();
  if (shape.size() != 3 || shape[2] != d_model_) throw Error("MultiHeadAttention input shape");
  cached_forward_ = false;
  const int n = shape[0];
  const int time = shape[1];
  const int heads = num_heads_;
  const int dh = head_dim_;
  const int batch = n * heads;
  const std::vector<int> heads_shape = {n, heads, time, dh};
  const std::vector<int> score_shape = {n, heads, time, time};
  const std::vector<int> merged_shape = {n, time, q_dim_};
  const int64_t full_n = static_cast<int64_t>(n) * heads * time * dh;

  ensure_activation(q_heads_, *ctx_, heads_shape);
  ensure_activation(k_heads_, *ctx_, heads_shape);
  ensure_activation(v_heads_, *ctx_, heads_shape);
  ensure_activation(scores_, *ctx_, score_shape);
  ensure_activation(attn_, *ctx_, score_shape);
  ensure_activation(context_, *ctx_, heads_shape);
  ensure_activation(merged_, *ctx_, merged_shape);
  ensure_activation(output_, *ctx_, shape);

  if (output_->dtype() == DType::BFloat16) {
    hipStream_t stream = stream_of(*ctx_);
    lay_qkv(q_->forward(input), k_->forward(input), v_->forward(input), time);
    if (use_rope_) {
      arena_reset(*ctx_);
      float* qh = stage_f32(*ctx_, *q_heads_);
      float* kh = stage_f32(*ctx_, *k_heads_);
      launch_rope(stream, qh, kh, time, dh, 0, full_n, false, rope_theta_);
      keep_f32(*ctx_, *q_heads_, qh);
      keep_f32(*ctx_, *k_heads_, kh);
    }
    const float scale = 1.f / std::sqrt(static_cast<float>(dh));
    const int64_t stride_qk = static_cast<int64_t>(time) * dh;
    const int64_t stride_s = static_cast<int64_t>(time) * time;
    gemm_nt_bf16(*ctx_, scores_->bf16_data(), true, stage_bf16(*ctx_, *q_heads_),
                 stage_bf16(*ctx_, *k_heads_), time, time, dh, batch, time, dh, dh, stride_s, stride_qk,
                 stride_qk, scale, 0.f);
    arena_reset(*ctx_);
    float* scores_f = stage_f32(*ctx_, *scores_);
    float* attn_f = arena_f32(*ctx_, scores_->numel());
    softmax_rows_kernel<<<batch * time, kThreads, 0, stream>>>(attn_f, scores_f, time, causal_ ? 1 : 0);
    check_hip(hipGetLastError(), "mha_softmax");
    keep_f32(*ctx_, *attn_, attn_f);
    gemm_nn_bf16(*ctx_, context_->bf16_data(), true, stage_bf16(*ctx_, *attn_),
                 stage_bf16(*ctx_, *v_heads_), time, dh, time, batch, dh, time, dh, stride_qk, stride_s,
                 stride_qk, 1.f, 0.f);
    arena_reset(*ctx_);
    float* context_f = stage_f32(*ctx_, *context_);
    float* merged_f = arena_f32(*ctx_, merged_->numel());
    launch_merge_heads(stream, merged_f, context_f, time, heads, dh, full_n);
    keep_f32(*ctx_, *merged_, merged_f);
    output_->copy_from(out_->forward(*merged_));
    return *output_;
  }

  hipStream_t stream = stream_of(*ctx_);
  hipblasHandle_t handle = hipblas_of(*ctx_);
  lay_qkv(q_->forward(input), k_->forward(input), v_->forward(input), time);
  if (use_rope_) {
    launch_rope(stream, q_heads_->data(), k_heads_->data(), time, dh, 0, full_n, false, rope_theta_);
  }

  const float scale = 1.f / std::sqrt(static_cast<float>(dh));
  gemm_scores(handle, q_heads_->data(), k_heads_->data(), scores_->data(), time, dh, batch, scale);
  // -1e9, not -inf: a row of infinities makes max-subtract softmax NaN. Applied while scoring.
  softmax_rows_kernel<<<batch * time, kThreads, 0, stream>>>(attn_->data(), scores_->data(), time,
                                                            causal_ ? 1 : 0);
  check_hip(hipGetLastError(), "mha_softmax");
  // context = attn V, same row-major product as dQ = dS K with scale 1.
  gemm_dq(handle, v_heads_->data(), attn_->data(), context_->data(), time, dh, batch, 1.f);
  launch_merge_heads(stream, merged_->data(), context_->data(), time, heads, dh, full_n);
  output_->copy_from(out_->forward(*merged_));
  return *output_;
}

void MultiHeadAttention::reset_cache(Cache& cache, int batch, int t_max) const {
  if (batch < 1) throw std::invalid_argument("batch");
  if (t_max < 1) throw std::invalid_argument("t_max");
  const DType dtype = ctx_->activation_dtype();
  cache.k = Tensor(*ctx_, {batch, num_heads_, t_max, head_dim_}, dtype);
  cache.v = Tensor(*ctx_, {batch, num_heads_, t_max, head_dim_}, dtype);
  cache.k.zero();
  cache.v.zero();
  cache.t = 0;
}

const Tensor& MultiHeadAttention::forward_cached(const Tensor& x_t, Cache& cache) {
  if (&x_t.context() != ctx_ || x_t.rank() != 3 || x_t.shape()[1] != 1 || x_t.shape()[2] != d_model_) {
    throw std::invalid_argument("x_t");
  }
  const int n = x_t.shape()[0];
  const int heads = num_heads_;
  const int dh = head_dim_;
  const int64_t full_n = static_cast<int64_t>(n) * heads * dh;
  const std::vector<int>& k_shape = cache.k.shape();
  if (&cache.k.context() != ctx_ || &cache.v.context() != ctx_ || cache.k.rank() != 4 ||
      cache.v.shape() != k_shape || k_shape[0] != n || k_shape[1] != heads || k_shape[3] != dh ||
      cache.t < 0 || cache.t >= k_shape[2]) {
    throw std::invalid_argument("cache");
  }
  // Full-sequence backward must not run on this one-token scratch.
  cached_forward_ = true;

  const int t = cache.t;
  const int t_max = k_shape[2];
  const int tk = t + 1;
  constexpr int tq = 1;
  const int batch = n * heads;
  const std::vector<int> q_shape = {n, heads, tq, dh};
  const std::vector<int> score_shape = {n, heads, tq, tk};
  const std::vector<int> flat_shape = x_t.shape();
  const std::vector<int> merged_shape = {n, tq, q_dim_};

  ensure_activation(q_heads_, *ctx_, q_shape);
  ensure_activation(k_heads_, *ctx_, q_shape);
  ensure_activation(v_heads_, *ctx_, q_shape);
  ensure_activation(scores_, *ctx_, score_shape);
  ensure_activation(attn_, *ctx_, score_shape);
  ensure_activation(context_, *ctx_, q_shape);
  ensure_activation(merged_, *ctx_, merged_shape);
  ensure_activation(output_, *ctx_, flat_shape);

  if (output_->dtype() == DType::BFloat16) {
    hipStream_t stream = stream_of(*ctx_);
    lay_qkv(q_->forward(x_t), k_->forward(x_t), v_->forward(x_t), tq);
    if (use_rope_) {
      arena_reset(*ctx_);
      float* qh = stage_f32(*ctx_, *q_heads_);
      float* kh = stage_f32(*ctx_, *k_heads_);
      launch_rope(stream, qh, kh, tq, dh, t, full_n, false, rope_theta_);
      keep_f32(*ctx_, *q_heads_, qh);
      keep_f32(*ctx_, *k_heads_, kh);
    }
    const int64_t kv_n = static_cast<int64_t>(batch) * dh;
    launch_write_kv_bf16(stream, cache.k.bf16_data(), k_heads_->bf16_data(), t, t_max, dh, kv_n);
    launch_write_kv_bf16(stream, cache.v.bf16_data(), v_heads_->bf16_data(), t, t_max, dh, kv_n);
    const float scale = 1.f / std::sqrt(static_cast<float>(dh));
    gemm_nt_bf16(*ctx_, scores_->bf16_data(), true, stage_bf16(*ctx_, *q_heads_), stage_bf16(*ctx_, cache.k),
                 tq, tk, dh, batch, tk, dh, dh, static_cast<int64_t>(tq) * tk,
                 static_cast<int64_t>(tq) * dh, static_cast<int64_t>(t_max) * dh, scale, 0.f);
    arena_reset(*ctx_);
    float* scores_f = stage_f32(*ctx_, *scores_);
    float* attn_f = arena_f32(*ctx_, scores_->numel());
    softmax_rows_kernel<<<batch, kThreads, 0, stream>>>(attn_f, scores_f, tk, 0);
    check_hip(hipGetLastError(), "mha_softmax");
    keep_f32(*ctx_, *attn_, attn_f);
    gemm_nn_bf16(*ctx_, context_->bf16_data(), true, stage_bf16(*ctx_, *attn_), stage_bf16(*ctx_, cache.v),
                 tq, dh, tk, batch, dh, tk, dh, static_cast<int64_t>(tq) * dh,
                 static_cast<int64_t>(tq) * tk, static_cast<int64_t>(t_max) * dh, 1.f, 0.f);
    arena_reset(*ctx_);
    float* context_f = stage_f32(*ctx_, *context_);
    float* merged_f = arena_f32(*ctx_, merged_->numel());
    launch_merge_heads(stream, merged_f, context_f, tq, heads, dh, full_n);
    keep_f32(*ctx_, *merged_, merged_f);
    output_->copy_from(out_->forward(*merged_));
    cache.t = tk;
    return *output_;
  }

  hipStream_t stream = stream_of(*ctx_);
  hipblasHandle_t handle = hipblas_of(*ctx_);
  lay_qkv(q_->forward(x_t), k_->forward(x_t), v_->forward(x_t), tq);
  // Rotate this token at its absolute position before the cache stores K. V is not rotated.
  if (use_rope_) {
    launch_rope(stream, q_heads_->data(), k_heads_->data(), tq, dh, t, full_n, false, rope_theta_);
  }
  const int64_t kv_n = static_cast<int64_t>(batch) * dh;
  launch_write_kv(stream, cache.k.data(), k_heads_->data(), t, t_max, dh, kv_n);
  launch_write_kv(stream, cache.v.data(), v_heads_->data(), t, t_max, dh, kv_n);

  // S = scale * Q K^T over the written prefix only. K's stride stays T_max so the tail is unread.
  // Row-major (Tq, Tk) is column-major (Tk, Tq): S_cm = scale * K_cm^T Q_cm.
  const float scale = 1.f / std::sqrt(static_cast<float>(dh));
  const float beta = 0.f;
  const long long stride_k = static_cast<long long>(t_max) * dh;
  const long long stride_q = static_cast<long long>(tq) * dh;
  const long long stride_s = static_cast<long long>(tq) * tk;
  check_hipblas(hipblasGemmStridedBatchedEx(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, tk, tq, dh, &scale,
                                          cache.k.data(), HIP_R_32F, dh, stride_k, q_heads_->data(),
                                          HIP_R_32F, dh, stride_q, &beta, scores_->data(), HIP_R_32F,
                                          tk, stride_s, batch, HIPBLAS_COMPUTE_32F,
                                          HIPBLAS_GEMM_DEFAULT),
               "hipblasGemmStridedBatchedEx");
  softmax_rows_kernel<<<batch, kThreads, 0, stream>>>(attn_->data(), scores_->data(), tk, 0);
  check_hip(hipGetLastError(), "mha_softmax");
  // context = attn V. context_cm = V_cm attn_cm, same product as dQ = dS K with scale 1.
  const float one = 1.f;
  check_hipblas(hipblasGemmStridedBatchedEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, dh, tq, tk, &one,
                                          cache.v.data(), HIP_R_32F, dh, stride_k, attn_->data(),
                                          HIP_R_32F, tk, stride_s, &beta, context_->data(),
                                          HIP_R_32F, dh, stride_q, batch, HIPBLAS_COMPUTE_32F,
                                          HIPBLAS_GEMM_DEFAULT),
               "hipblasGemmStridedBatchedEx");
  launch_merge_heads(stream, merged_->data(), context_->data(), tq, heads, dh, full_n);
  output_->copy_from(out_->forward(*merged_));
  cache.t = tk;
  return *output_;
}

const Tensor& MultiHeadAttention::backward(const Tensor& grad_output) {
  if (cached_forward_) throw std::invalid_argument("forward_cached");
  if (!output_ || !attn_ || !q_heads_ || !k_heads_ || !v_heads_ ||
      grad_output.shape() != output_->shape()) {
    throw Error("MultiHeadAttention backward shape");
  }
  const int n = grad_output.shape()[0];
  const int time = grad_output.shape()[1];
  const int heads = num_heads_;
  const int kv = num_kv_heads_;
  const int group = heads / kv;
  const int dh = head_dim_;
  const int batch = n * heads;
  const std::vector<int> heads_shape = {n, heads, time, dh};
  const std::vector<int> score_shape = {n, heads, time, time};
  const std::vector<int>& flat_shape = grad_output.shape();
  const int64_t full_n = static_cast<int64_t>(n) * heads * time * dh;
  const int64_t q_n = static_cast<int64_t>(n) * time * q_dim_;
  const int64_t kv_n = static_cast<int64_t>(n) * time * kv_dim_;

  ensure_activation(d_context_, *ctx_, heads_shape);
  ensure_activation(d_attn_, *ctx_, score_shape);
  ensure_activation(d_scores_, *ctx_, score_shape);
  ensure_activation(d_q_, *ctx_, heads_shape);
  ensure_activation(d_k_, *ctx_, heads_shape);
  ensure_activation(d_v_, *ctx_, heads_shape);
  ensure_activation(q_flat_grad_, *ctx_, {n, time, q_dim_});
  ensure_activation(kv_flat_grad_, *ctx_, {n, time, kv_dim_});
  ensure_activation(grad_pair_, *ctx_, flat_shape);
  ensure_activation(grad_input_, *ctx_, flat_shape);

  const int64_t hidden = grad_output.numel();
  const float scale = 1.f / std::sqrt(static_cast<float>(dh));
  if (grad_input_->dtype() == DType::BFloat16) {
    const Tensor& d_merged = out_->backward(grad_output);
    hipStream_t stream = stream_of(*ctx_);
    const int64_t stride_qk = static_cast<int64_t>(time) * dh;
    const int64_t stride_s = static_cast<int64_t>(time) * time;
    arena_reset(*ctx_);
    float* merged_f = stage_f32(*ctx_, d_merged);
    float* d_context_f = arena_f32(*ctx_, d_context_->numel());
    launch_split_heads(stream, d_context_f, merged_f, time, heads, dh, q_n);
    keep_f32(*ctx_, *d_context_, d_context_f);
    gemm_nt_bf16(*ctx_, d_attn_->bf16_data(), true, stage_bf16(*ctx_, *d_context_),
                 stage_bf16(*ctx_, *v_heads_), time, time, dh, batch, time, dh, dh, stride_s, stride_qk,
                 stride_qk, 1.f, 0.f);
    gemm_tn_bf16(*ctx_, d_v_->bf16_data(), true, stage_bf16(*ctx_, *attn_),
                 stage_bf16(*ctx_, *d_context_), time, dh, time, batch, dh, time, dh, stride_qk, stride_s,
                 stride_qk, 1.f, 0.f);
    arena_reset(*ctx_);
    float* probs = stage_f32(*ctx_, *attn_);
    float* dp = stage_f32(*ctx_, *d_attn_);
    float* ds = arena_f32(*ctx_, d_scores_->numel());
    softmax_backward_kernel<<<batch * time, kThreads, 0, stream>>>(ds, probs, dp, time, causal_ ? 1 : 0);
    check_hip(hipGetLastError(), "mha_softmax_backward");
    keep_f32(*ctx_, *d_scores_, ds);
    gemm_nn_bf16(*ctx_, d_q_->bf16_data(), true, stage_bf16(*ctx_, *d_scores_),
                 stage_bf16(*ctx_, *k_heads_), time, dh, time, batch, dh, time, dh, stride_qk, stride_s,
                 stride_qk, scale, 0.f);
    gemm_tn_bf16(*ctx_, d_k_->bf16_data(), true, stage_bf16(*ctx_, *d_scores_),
                 stage_bf16(*ctx_, *q_heads_), time, dh, time, batch, dh, time, dh, stride_qk, stride_s,
                 stride_qk, scale, 0.f);
    if (use_rope_) {
      arena_reset(*ctx_);
      float* dq = stage_f32(*ctx_, *d_q_);
      float* dk = stage_f32(*ctx_, *d_k_);
      launch_rope(stream, dq, dk, time, dh, 0, full_n, true, rope_theta_);
      keep_f32(*ctx_, *d_q_, dq);
      keep_f32(*ctx_, *d_k_, dk);
    }

    const Tensor* dq_src = &*d_q_;
    if (q_norm_) dq_src = &q_norm_->backward(*d_q_);
    arena_reset(*ctx_);
    float* dq_f = stage_f32(*ctx_, *dq_src);
    float* q_flat = arena_f32(*ctx_, q_flat_grad_->numel());
    launch_merge_heads(stream, q_flat, dq_f, time, heads, dh, q_n);
    keep_f32(*ctx_, *q_flat_grad_, q_flat);
    const Tensor& gq = q_->backward(*q_flat_grad_);

    const Tensor* dk_src = &*d_k_;
    if (group > 1) {
      ensure_activation(k_kv_, *ctx_, {n, kv, time, dh});
      const int64_t kv_heads_n = static_cast<int64_t>(n) * kv * time * dh;
      arena_reset(*ctx_);
      float* src = stage_f32(*ctx_, *d_k_);
      float* summed = arena_f32(*ctx_, kv_heads_n);
      launch_sum_kv(stream, summed, src, time, kv, group, dh, kv_heads_n);
      keep_f32(*ctx_, *k_kv_, summed);
      dk_src = &*k_kv_;
      if (k_norm_) dk_src = &k_norm_->backward(*k_kv_);
    } else if (k_norm_) {
      dk_src = &k_norm_->backward(*d_k_);
    }
    arena_reset(*ctx_);
    float* dk_f = stage_f32(*ctx_, *dk_src);
    float* kv_flat = arena_f32(*ctx_, kv_flat_grad_->numel());
    launch_merge_heads(stream, kv_flat, dk_f, time, kv, dh, kv_n);
    keep_f32(*ctx_, *kv_flat_grad_, kv_flat);
    const Tensor& gk = k_->backward(*kv_flat_grad_);

    const Tensor* dv_src = &*d_v_;
    if (group > 1) {
      ensure_activation(v_kv_, *ctx_, {n, kv, time, dh});
      const int64_t kv_heads_n = static_cast<int64_t>(n) * kv * time * dh;
      arena_reset(*ctx_);
      float* src = stage_f32(*ctx_, *d_v_);
      float* summed = arena_f32(*ctx_, kv_heads_n);
      launch_sum_kv(stream, summed, src, time, kv, group, dh, kv_heads_n);
      keep_f32(*ctx_, *v_kv_, summed);
      dv_src = &*v_kv_;
    }
    arena_reset(*ctx_);
    float* dv_f = stage_f32(*ctx_, *dv_src);
    float* v_flat = arena_f32(*ctx_, kv_flat_grad_->numel());
    launch_merge_heads(stream, v_flat, dv_f, time, kv, dh, kv_n);
    keep_f32(*ctx_, *kv_flat_grad_, v_flat);
    const Tensor& gv = v_->backward(*kv_flat_grad_);

    arena_reset(*ctx_);
    float* gq_f = stage_f32(*ctx_, gq);
    float* gk_f = stage_f32(*ctx_, gk);
    float* gv_f = stage_f32(*ctx_, gv);
    float* pair = arena_f32(*ctx_, hidden);
    float* dx = arena_f32(*ctx_, hidden);
    launch_add2(stream, pair, gq_f, gk_f, hidden);
    launch_add2(stream, dx, pair, gv_f, hidden);
    keep_f32(*ctx_, *grad_pair_, pair);
    keep_f32(*ctx_, *grad_input_, dx);
    return *grad_input_;
  }

  hipStream_t stream = stream_of(*ctx_);
  hipblasHandle_t handle = hipblas_of(*ctx_);

  const Tensor& d_merged = out_->backward(grad_output);
  launch_split_heads(stream, d_context_->data(), d_merged.data(), time, heads, dh, q_n);
  // dAttn = dContext V^T, the same product as scores = Q K^T.
  gemm_scores(handle, d_context_->data(), v_heads_->data(), d_attn_->data(), time, dh, batch, 1.f);
  // dV = attn^T dContext, the same product as dK = dS^T Q.
  gemm_dk(handle, d_context_->data(), attn_->data(), d_v_->data(), time, dh, batch, 1.f);
  softmax_backward_kernel<<<batch * time, kThreads, 0, stream>>>(
      d_scores_->data(), attn_->data(), d_attn_->data(), time, causal_ ? 1 : 0);
  check_hip(hipGetLastError(), "mha_softmax_backward");
  gemm_dq(handle, k_heads_->data(), d_scores_->data(), d_q_->data(), time, dh, batch, scale);
  gemm_dk(handle, q_heads_->data(), d_scores_->data(), d_k_->data(), time, dh, batch, scale);
  // dQ and dK are gradients of the rotated vectors. Undo the rotation before the norms.
  if (use_rope_) {
    launch_rope(stream, d_q_->data(), d_k_->data(), time, dh, 0, full_n, true, rope_theta_);
  }

  const float* dq = d_q_->data();
  if (q_norm_) dq = q_norm_->backward(*d_q_).data();
  launch_merge_heads(stream, q_flat_grad_->data(), dq, time, heads, dh, q_n);
  const Tensor& gq = q_->backward(*q_flat_grad_);

  const float* dk = d_k_->data();
  if (group > 1) {
    ensure_activation(k_kv_, *ctx_, {n, kv, time, dh});
    const int64_t kv_heads_n = static_cast<int64_t>(n) * kv * time * dh;
    launch_sum_kv(stream, k_kv_->data(), d_k_->data(), time, kv, group, dh, kv_heads_n);
    dk = k_kv_->data();
    if (k_norm_) dk = k_norm_->backward(*k_kv_).data();
  } else if (k_norm_) {
    dk = k_norm_->backward(*d_k_).data();
  }
  launch_merge_heads(stream, kv_flat_grad_->data(), dk, time, kv, dh, kv_n);
  const Tensor& gk = k_->backward(*kv_flat_grad_);

  const float* dv = d_v_->data();
  if (group > 1) {
    ensure_activation(v_kv_, *ctx_, {n, kv, time, dh});
    const int64_t kv_heads_n = static_cast<int64_t>(n) * kv * time * dh;
    launch_sum_kv(stream, v_kv_->data(), d_v_->data(), time, kv, group, dh, kv_heads_n);
    dv = v_kv_->data();
  }
  launch_merge_heads(stream, kv_flat_grad_->data(), dv, time, kv, dh, kv_n);
  const Tensor& gv = v_->backward(*kv_flat_grad_);

  launch_add2(stream, grad_pair_->data(), gq.data(), gk.data(), hidden);
  launch_add2(stream, grad_input_->data(), grad_pair_->data(), gv.data(), hidden);
  return *grad_input_;
}

}  // namespace nn
