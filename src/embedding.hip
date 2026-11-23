#include "nn/layers/embedding.hpp"

#include "activate.hpp"
#include "check.hpp"
#include "context_impl.hpp"
#include "ensure.hpp"
#include "launch.hpp"
#include "nn/error.hpp"
#include "nn/init.hpp"

#include <cstdint>
#include <vector>

namespace nn {
namespace {

std::vector<int> embedding_weight_shape(int num_embeddings, int embedding_dim) {
  if (num_embeddings < 1 || embedding_dim < 1) throw Error("Embedding size");
  return {num_embeddings, embedding_dim};
}

__global__ void embedding_check_kernel(const int64_t* indices, int64_t rows, int64_t vocab, int* bad) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows) return;
  const int64_t id = indices[i];
  if (id < 0 || id >= vocab) atomicExch(bad, 1);
}

__global__ void embedding_gather_kernel(float* out, const float* weight, const int64_t* indices,
                                        int64_t rows, int dim) {
  const int64_t n = rows * static_cast<int64_t>(dim);
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t row = i / dim;
  const int col = static_cast<int>(i - row * dim);
  const int64_t id = indices[row];
  out[i] = weight[id * static_cast<int64_t>(dim) + col];
}

__global__ void embedding_gather4_kernel(float* out, const float* weight, const int64_t* indices,
                                         int64_t rows, int dim) {
  const int dim4 = dim / 4;
  const int64_t n4 = rows * static_cast<int64_t>(dim4);
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  const int64_t row = i / dim4;
  const int col = static_cast<int>(i - row * dim4);
  const int64_t id = indices[row];
  reinterpret_cast<float4*>(out)[i] =
      reinterpret_cast<const float4*>(weight)[id * static_cast<int64_t>(dim4) + col];
}

__global__ void embedding_scatter_kernel(float* grad_weight, const float* grad_output,
                                         const int64_t* indices, int64_t rows, int dim) {
  const int64_t n = rows * static_cast<int64_t>(dim);
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t row = i / dim;
  const int col = static_cast<int>(i - row * dim);
  const int64_t id = indices[row];
  // One batch can name the same row twice; those rows must add, not overwrite.
  atomicAdd(grad_weight + id * static_cast<int64_t>(dim) + col, grad_output[i]);
}

void copy_indices(hipStream_t stream, IndexTensor& dst, const IndexTensor& src) {
  check_hip(hipMemcpyAsync(const_cast<int64_t*>(dst.data()), src.data(),
                           static_cast<size_t>(src.numel()) * sizeof(int64_t), hipMemcpyDeviceToDevice,
                           stream),
            "hipMemcpyAsync");
}

void require_indices(hipStream_t stream, const int64_t* indices, int64_t rows, int vocab) {
  int* bad = nullptr;
  check_hip(hipMalloc(&bad, sizeof(int)), "hipMalloc");
  try {
    check_hip(hipMemsetAsync(bad, 0, sizeof(int), stream), "hipMemsetAsync");
    embedding_check_kernel<<<blocks_for(rows), 256, 0, stream>>>(indices, rows, vocab, bad);
    check_hip(hipGetLastError(), "embedding_check");
    check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize");
    int bad_host = 0;
    check_hip(hipMemcpy(&bad_host, bad, sizeof(int), hipMemcpyDeviceToHost), "hipMemcpy");
    hipFree(bad);
    bad = nullptr;
    if (bad_host) throw Error("index out of range");
  } catch (...) {
    if (bad) hipFree(bad);
    throw;
  }
}

}  // namespace

Embedding::Embedding(Context& ctx, int num_embeddings, int embedding_dim, std::mt19937& rng)
    : ctx_(&ctx),
      num_embeddings_(num_embeddings),
      embedding_dim_(embedding_dim),
      weight_(ctx, embedding_weight_shape(num_embeddings, embedding_dim)),
      grad_weight_(ctx, {num_embeddings, embedding_dim}) {
  normal_(weight_, 0.f, 0.02f, rng);
  grad_weight_.zero();
}

std::vector<NamedParam> Embedding::parameters() const {
  return {NamedParam{"weight", const_cast<Tensor*>(&weight_), const_cast<Tensor*>(&grad_weight_)}};
}

void Embedding::zero_grad() { grad_weight_.zero(); }

const Tensor& Embedding::forward(const Tensor&) { throw Error("Embedding requires indices"); }

const Tensor& Embedding::forward_indices(const IndexTensor& indices) {
  if (&indices.context() != ctx_) throw Error("Embedding context");
  hipStream_t stream = stream_of(*ctx_);
  // Check before the cache is replaced. A rejected index keeps the last successful forward.
  require_indices(stream, indices.data(), indices.numel(), num_embeddings_);
  if (!cache_indices_ || cache_indices_->shape() != indices.shape()) {
    cache_indices_.emplace(*ctx_, indices.shape());
  }
  copy_indices(stream, *cache_indices_, indices);

  std::vector<int> out_shape = indices.shape();
  out_shape.push_back(embedding_dim_);
  ensure_activation(output_, *ctx_, out_shape);
  const int64_t rows = cache_indices_->numel();
  const int64_t n = rows * static_cast<int64_t>(embedding_dim_);
  if (output_->dtype() == DType::BFloat16) {
    arena_reset(*ctx_);
    float* y = arena_f32(*ctx_, n);
    if ((embedding_dim_ % 4) == 0 && aligned16(y) && aligned16(weight_.data())) {
      embedding_gather4_kernel<<<blocks_for(n / 4), 256, 0, stream>>>(
          y, weight_.data(), cache_indices_->data(), rows, embedding_dim_);
    } else {
      embedding_gather_kernel<<<blocks_for(n), 256, 0, stream>>>(y, weight_.data(), cache_indices_->data(),
                                                                 rows, embedding_dim_);
    }
    check_hip(hipGetLastError(), "embedding_gather");
    keep_f32(*ctx_, *output_, y);
    return *output_;
  }
  if ((embedding_dim_ % 4) == 0 && aligned16(output_->data()) && aligned16(weight_.data())) {
    embedding_gather4_kernel<<<blocks_for(n / 4), 256, 0, stream>>>(
        output_->data(), weight_.data(), cache_indices_->data(), rows, embedding_dim_);
  } else {
    embedding_gather_kernel<<<blocks_for(n), 256, 0, stream>>>(
        output_->data(), weight_.data(), cache_indices_->data(), rows, embedding_dim_);
  }
  check_hip(hipGetLastError(), "embedding_gather");
  return *output_;
}

const Tensor& Embedding::backward(const Tensor& grad_output) {
  if (!cache_indices_ || !output_) throw Error("Embedding backward without forward");
  if (grad_output.shape() != output_->shape()) throw Error("Embedding backward shape");
  const int64_t rows = cache_indices_->numel();
  const int64_t n = rows * static_cast<int64_t>(embedding_dim_);
  if (grad_output.dtype() == DType::BFloat16) {
    arena_reset(*ctx_);
    float* dy = stage_f32(*ctx_, grad_output);
    embedding_scatter_kernel<<<blocks_for(n), 256, 0, stream_of(*ctx_)>>>(
        grad_weight_.data(), dy, cache_indices_->data(), rows, embedding_dim_);
  } else {
    embedding_scatter_kernel<<<blocks_for(n), 256, 0, stream_of(*ctx_)>>>(
        grad_weight_.data(), grad_output.data(), cache_indices_->data(), rows, embedding_dim_);
  }
  check_hip(hipGetLastError(), "embedding_scatter");
  // Ids are discrete. Own a zero tensor so the returned reference follows the layer rule.
  ensure_activation(grad_input_, *ctx_, grad_output.shape());
  grad_input_->zero();
  return *grad_input_;
}

}  // namespace nn
