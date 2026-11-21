#include "nn/loss.hpp"

#include "check.hpp"
#include "context_impl.hpp"
#include "kernels.hpp"
#include "nn/error.hpp"
#include "reduce_device.hpp"

#include <utility>

namespace nn {
namespace {

__global__ void softmax_ce_kernel(const float* logits, const int64_t* labels, float* grad,
                                  float* per_sample, int n, int classes, int* bad) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= n) return;
  const float* row_logits = logits + static_cast<int64_t>(row) * classes;
  float* row_grad = grad + static_cast<int64_t>(row) * classes;

  __shared__ float scratch[32];
  float thread_max = -1e30f;
  for (int c = static_cast<int>(threadIdx.x); c < classes; c += static_cast<int>(blockDim.x)) {
    thread_max = fmaxf(thread_max, row_logits[c]);
  }
  const float max_logit = block_max(thread_max, scratch);

  float thread_sum = 0.f;
  for (int c = static_cast<int>(threadIdx.x); c < classes; c += static_cast<int>(blockDim.x)) {
    thread_sum += expf(row_logits[c] - max_logit);
  }
  const float sum = block_sum(thread_sum, scratch);
  const int64_t label = labels[row];
  if (label < 0 || label >= classes) {
    if (threadIdx.x == 0) atomicExch(bad, 1);
    return;
  }
  const float inv_sum = 1.f / sum;
  const float inv_n = 1.f / static_cast<float>(n);
  for (int c = static_cast<int>(threadIdx.x); c < classes; c += static_cast<int>(blockDim.x)) {
    const float softmax = expf(row_logits[c] - max_logit) * inv_sum;
    const float indicator = (static_cast<int64_t>(c) == label) ? 1.f : 0.f;
    row_grad[c] = (softmax - indicator) * inv_n;
  }
  if (threadIdx.x == 0) per_sample[row] = -(row_logits[label] - (max_logit + logf(sum)));
}

// ignore_index is allowed to be negative or >= V. Classification rejects every label < 0.
__global__ void sequence_label_count_kernel(const int64_t* labels, int rows, int classes,
                                            int64_t ignore_index, int* n_valid, int* bad) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const int64_t label = labels[row];
  if (label == ignore_index) return;
  if (label < 0 || label >= static_cast<int64_t>(classes)) {
    atomicExch(bad, 1);
    return;
  }
  atomicAdd(n_valid, 1);
}

__global__ void sequence_ce_kernel(const float* logits, const int64_t* labels, float* grad,
                                   float* per_sample, int rows, int classes, int64_t ignore_index,
                                   int n_valid, int* bad) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) return;
  const float* row_logits = logits + static_cast<int64_t>(row) * classes;
  float* row_grad = grad + static_cast<int64_t>(row) * classes;
  const int64_t label = labels[row];

  if (label == ignore_index) {
    for (int c = static_cast<int>(threadIdx.x); c < classes; c += static_cast<int>(blockDim.x)) {
      row_grad[c] = 0.f;
    }
    if (threadIdx.x == 0) per_sample[row] = 0.f;
    return;
  }
  if (label < 0 || label >= static_cast<int64_t>(classes)) {
    for (int c = static_cast<int>(threadIdx.x); c < classes; c += static_cast<int>(blockDim.x)) {
      row_grad[c] = 0.f;
    }
    if (threadIdx.x == 0) {
      atomicExch(bad, 1);
      per_sample[row] = 0.f;
    }
    return;
  }

  __shared__ float scratch[32];
  float thread_max = -1e30f;
  for (int c = static_cast<int>(threadIdx.x); c < classes; c += static_cast<int>(blockDim.x)) {
    thread_max = fmaxf(thread_max, row_logits[c]);
  }
  const float max_logit = block_max(thread_max, scratch);

  float thread_sum = 0.f;
  for (int c = static_cast<int>(threadIdx.x); c < classes; c += static_cast<int>(blockDim.x)) {
    thread_sum += expf(row_logits[c] - max_logit);
  }
  const float sum = block_sum(thread_sum, scratch);
  const float inv_sum = 1.f / sum;
  const float inv_n = 1.f / static_cast<float>(n_valid);
  for (int c = static_cast<int>(threadIdx.x); c < classes; c += static_cast<int>(blockDim.x)) {
    const float softmax = expf(row_logits[c] - max_logit) * inv_sum;
    const float indicator = (static_cast<int64_t>(c) == label) ? 1.f : 0.f;
    row_grad[c] = (softmax - indicator) * inv_n;
  }
  if (threadIdx.x == 0) per_sample[row] = -(row_logits[label] - (max_logit + logf(sum)));
}

}  // namespace

DeviceIndices::DeviceIndices(Context& ctx, int64_t n) : ctx_(&ctx), n_(n) {
  if (n <= 0) throw Error("empty labels");
  check_hip(hipMalloc(&data_, static_cast<size_t>(n) * sizeof(int64_t)), "hipMalloc");
}

DeviceIndices::~DeviceIndices() {
  if (!data_) return;
  if (ctx_) hipStreamSynchronize(stream_of(*ctx_));
  hipFree(data_);
  data_ = nullptr;
}

DeviceIndices::DeviceIndices(DeviceIndices&& other) noexcept
    : ctx_(other.ctx_), data_(other.data_), n_(other.n_) {
  other.ctx_ = nullptr;
  other.data_ = nullptr;
  other.n_ = 0;
}

DeviceIndices& DeviceIndices::operator=(DeviceIndices&& other) noexcept {
  if (this == &other) return *this;
  if (data_) {
    if (ctx_) hipStreamSynchronize(stream_of(*ctx_));
    hipFree(data_);
  }
  ctx_ = other.ctx_;
  data_ = other.data_;
  n_ = other.n_;
  other.ctx_ = nullptr;
  other.data_ = nullptr;
  other.n_ = 0;
  return *this;
}

void DeviceIndices::copy_from_host(const int64_t* src, int64_t n) {
  if (n != n_) throw Error("label count mismatch");
  check_hip(hipMemcpyAsync(data_, src, static_cast<size_t>(n) * sizeof(int64_t),
                             hipMemcpyHostToDevice, stream_of(*ctx_)),
             "hipMemcpyAsync");
}

LossResult softmax_cross_entropy(Context& ctx, const Tensor& logits, const DeviceIndices& labels) {
  if (logits.rank() != 2) throw Error("logits rank");
  const int n = logits.shape()[0];
  const int classes = logits.shape()[1];
  if (n <= 0 || classes <= 1) throw Error("logits shape");
  if (labels.size() != n) throw Error("label count mismatch");

  LossResult result{0.f, Tensor(ctx, {n, classes})};
  float* per_sample = nullptr;
  int* bad = nullptr;
  check_hip(hipMalloc(&per_sample, static_cast<size_t>(n) * sizeof(float)), "hipMalloc");
  check_hip(hipMalloc(&bad, sizeof(int)), "hipMalloc");
  hipStream_t stream = stream_of(ctx);
  check_hip(hipMemsetAsync(bad, 0, sizeof(int), stream), "hipMemsetAsync");
  softmax_ce_kernel<<<n, 256, 0, stream>>>(logits.data(), labels.data(), result.grad_logits.data(),
                                           per_sample, n, classes, bad);
  check_hip(hipGetLastError(), "softmax_cross_entropy");

  float sum = 0.f;
  try {
    sum = launch_sum(stream, per_sample, n);
  } catch (...) {
    hipFree(per_sample);
    hipFree(bad);
    throw;
  }
  ctx.synchronize();
  int bad_host = 0;
  check_hip(hipMemcpy(&bad_host, bad, sizeof(int), hipMemcpyDeviceToHost), "hipMemcpy");
  hipFree(per_sample);
  hipFree(bad);
  if (bad_host) throw Error("label out of range");
  result.loss = sum / static_cast<float>(n);
  return result;
}

LossResult sequence_cross_entropy(Context& ctx, const Tensor& logits, const IndexTensor& labels,
                                  int64_t ignore_index) {
  int classes = 0;
  int rows = 0;
  if (logits.rank() == 2) {
    rows = logits.shape()[0];
    classes = logits.shape()[1];
    if (labels.shape().size() != 1 || labels.shape()[0] != rows) throw Error("label shape");
  } else if (logits.rank() == 3) {
    if (labels.shape().size() != 2 || labels.shape()[0] != logits.shape()[0] ||
        labels.shape()[1] != logits.shape()[1]) {
      throw Error("label shape");
    }
    classes = logits.shape()[2];
    rows = static_cast<int>(static_cast<int64_t>(logits.shape()[0]) * logits.shape()[1]);
  } else {
    throw Error("logits rank");
  }
  if (classes <= 1) throw Error("logits shape");

  // Scale inside the kernel by 1/n_valid, matching softmax_ce_kernel's 1/N.
  // An empty kept set must return before that division.
  LossResult result{0.f, Tensor(ctx, logits.shape())};
  int* counts = nullptr;
  float* per_sample = nullptr;
  check_hip(hipMalloc(&counts, 2 * sizeof(int)), "hipMalloc");
  int* bad = counts;
  int* n_valid_dev = counts + 1;
  hipStream_t stream = stream_of(ctx);
  const hipError_t set = hipMemsetAsync(counts, 0, 2 * sizeof(int), stream);
  if (set != hipSuccess) {
    hipFree(counts);
    check_hip(set, "hipMemsetAsync");
  }
  constexpr int kCountThreads = 256;
  const int count_blocks =
      static_cast<int>((static_cast<int64_t>(rows) + kCountThreads - 1) / kCountThreads);
  sequence_label_count_kernel<<<count_blocks, kCountThreads, 0, stream>>>(
      labels.data(), rows, classes, ignore_index, n_valid_dev, bad);
  const hipError_t count_launch = hipGetLastError();
  if (count_launch != hipSuccess) {
    hipFree(counts);
    check_hip(count_launch, "sequence_cross_entropy");
  }
  // synchronize() throws through check_hip. per_sample is still null here.
  const hipError_t count_sync = hipStreamSynchronize(stream);
  if (count_sync != hipSuccess) {
    hipFree(counts);
    check_hip(count_sync, "hipStreamSynchronize");
  }
  int bad_host = 0;
  int n_valid = 0;
  const hipError_t copy_bad = hipMemcpy(&bad_host, bad, sizeof(int), hipMemcpyDeviceToHost);
  if (copy_bad != hipSuccess) {
    hipFree(counts);
    check_hip(copy_bad, "hipMemcpy");
  }
  const hipError_t copy_n = hipMemcpy(&n_valid, n_valid_dev, sizeof(int), hipMemcpyDeviceToHost);
  if (copy_n != hipSuccess) {
    hipFree(counts);
    check_hip(copy_n, "hipMemcpy");
  }
  if (bad_host) {
    hipFree(counts);
    throw Error("label out of range");
  }
  if (n_valid == 0) {
    hipFree(counts);
    result.grad_logits.zero();
    ctx.synchronize();
    return result;
  }

  const hipError_t alloc = hipMalloc(&per_sample, static_cast<size_t>(rows) * sizeof(float));
  if (alloc != hipSuccess) {
    hipFree(counts);
    check_hip(alloc, "hipMalloc");
  }
  sequence_ce_kernel<<<rows, 256, 0, stream>>>(logits.data(), labels.data(), result.grad_logits.data(),
                                               per_sample, rows, classes, ignore_index, n_valid, bad);
  const hipError_t launch = hipGetLastError();
  if (launch != hipSuccess) {
    hipFree(per_sample);
    hipFree(counts);
    check_hip(launch, "sequence_cross_entropy");
  }

  float sum = 0.f;
  try {
    sum = launch_sum(stream, per_sample, rows);
  } catch (...) {
    hipFree(per_sample);
    hipFree(counts);
    throw;
  }
  const hipError_t loss_sync = hipStreamSynchronize(stream);
  if (loss_sync != hipSuccess) {
    hipFree(per_sample);
    hipFree(counts);
    check_hip(loss_sync, "hipStreamSynchronize");
  }
  const hipError_t copy = hipMemcpy(&bad_host, bad, sizeof(int), hipMemcpyDeviceToHost);
  hipFree(per_sample);
  hipFree(counts);
  check_hip(copy, "hipMemcpy");
  if (bad_host) throw Error("label out of range");
  result.loss = sum / static_cast<float>(n_valid);
  return result;
}

}  // namespace nn
