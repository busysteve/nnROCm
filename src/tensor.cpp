#include "nn/tensor.hpp"

#include "activate.hpp"
#include "check.hpp"
#include "context_impl.hpp"
#include "kernels.hpp"

#include <utility>
#include <vector>

namespace nn {
namespace {

int64_t checked_numel(const std::vector<int>& shape) {
  if (shape.empty() || shape.size() > 4) throw Error("tensor rank out of range");
  int64_t n = 1;
  for (int dim : shape) {
    if (dim < 1) throw Error("tensor dimension out of range");
    if (n > static_cast<int64_t>(INT32_MAX) / dim) throw Error("tensor numel out of range");
    n *= dim;
  }
  return n;
}

}  // namespace

Tensor::Tensor(Context& ctx, std::vector<int> shape) : Tensor(ctx, std::move(shape), DType::Float32) {}

Tensor::Tensor(Context& ctx, std::vector<int> shape, DType dtype)
    : ctx_(&ctx), shape_(std::move(shape)), dtype_(dtype) {
  if (dtype_ != DType::Float32 && dtype_ != DType::BFloat16) throw Error("dtype");
  numel_ = checked_numel(shape_);
  bytes_ = static_cast<size_t>(numel_) * static_cast<size_t>(dtype_size(dtype_));
  check_hip(hipMalloc(&storage_, bytes_), "hipMalloc");
}

void Tensor::allocate() {}

void Tensor::release() noexcept {
  if (!storage_) return;
  if (ctx_) hipStreamSynchronize(stream_of(*ctx_));
  hipFree(storage_);
  storage_ = nullptr;
  bytes_ = 0;
  numel_ = 0;
}

Tensor::~Tensor() { release(); }

Tensor::Tensor(Tensor&& other) noexcept
    : ctx_(other.ctx_),
      shape_(std::move(other.shape_)),
      numel_(other.numel_),
      dtype_(other.dtype_),
      storage_(other.storage_),
      bytes_(other.bytes_) {
  other.ctx_ = nullptr;
  other.numel_ = 0;
  other.storage_ = nullptr;
  other.bytes_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this == &other) return *this;
  release();
  ctx_ = other.ctx_;
  shape_ = std::move(other.shape_);
  numel_ = other.numel_;
  dtype_ = other.dtype_;
  storage_ = other.storage_;
  bytes_ = other.bytes_;
  other.ctx_ = nullptr;
  other.numel_ = 0;
  other.storage_ = nullptr;
  other.bytes_ = 0;
  return *this;
}

float* Tensor::data() {
  if (dtype_ != DType::Float32) throw Error("tensor dtype");
  return static_cast<float*>(storage_);
}

const float* Tensor::data() const {
  if (dtype_ != DType::Float32) throw Error("tensor dtype");
  return static_cast<const float*>(storage_);
}

uint16_t* Tensor::bf16_data() {
  if (dtype_ != DType::BFloat16) throw Error("tensor dtype");
  return static_cast<uint16_t*>(storage_);
}

const uint16_t* Tensor::bf16_data() const {
  if (dtype_ != DType::BFloat16) throw Error("tensor dtype");
  return static_cast<const uint16_t*>(storage_);
}

void Tensor::fill(float value) {
  if (dtype_ == DType::Float32) {
    launch_fill(stream_of(*ctx_), data(), numel_, value);
    return;
  }
  if (value == 0.f) {
    zero();
    return;
  }
  std::vector<uint16_t> host(static_cast<size_t>(numel_), f32_to_bf16(value));
  check_hip(hipMemcpyAsync(storage_, host.data(), bytes_, hipMemcpyHostToDevice, stream_of(*ctx_)),
            "hipMemcpyAsync");
  ctx_->synchronize();
}

void Tensor::zero() {
  check_hip(hipMemsetAsync(storage_, 0, bytes_, stream_of(*ctx_)), "hipMemsetAsync");
}

void Tensor::copy_from(const Tensor& src) {
  if (numel_ != src.numel_) throw Error("tensor copy numel mismatch");
  if (storage_ == src.storage_) return;
  if (&src.context() != ctx_) {
    const std::vector<float> host = src.to_host();
    copy_from_host(host);
    return;
  }
  if (dtype_ == src.dtype_) {
    check_hip(hipMemcpyAsync(storage_, src.storage_, bytes_, hipMemcpyDeviceToDevice, stream_of(*ctx_)),
              "hipMemcpyAsync");
    return;
  }
  if (dtype_ == DType::BFloat16 && src.dtype_ == DType::Float32) {
    cast_f32_to_bf16(*ctx_, bf16_data(), src.data(), numel_);
    return;
  }
  if (dtype_ == DType::Float32 && src.dtype_ == DType::BFloat16) {
    cast_bf16_to_f32(*ctx_, data(), src.bf16_data(), numel_);
    return;
  }
  throw Error("tensor dtype");
}

void Tensor::copy_from_host(const float* src, int64_t n) {
  if (n != numel_) throw Error("tensor host copy numel mismatch");
  if (dtype_ == DType::Float32) {
    check_hip(hipMemcpyAsync(storage_, src, bytes_, hipMemcpyHostToDevice, stream_of(*ctx_)),
              "hipMemcpyAsync");
    return;
  }
  std::vector<uint16_t> host(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) host[static_cast<size_t>(i)] = f32_to_bf16(src[i]);
  check_hip(hipMemcpyAsync(storage_, host.data(), bytes_, hipMemcpyHostToDevice, stream_of(*ctx_)),
            "hipMemcpyAsync");
  ctx_->synchronize();
}

void Tensor::copy_from_host(const std::vector<float>& src) {
  copy_from_host(src.data(), static_cast<int64_t>(src.size()));
}

std::vector<float> Tensor::to_host() const {
  ctx_->synchronize();
  if (dtype_ == DType::Float32) {
    std::vector<float> host(static_cast<size_t>(numel_));
    check_hip(hipMemcpy(host.data(), storage_, bytes_, hipMemcpyDeviceToHost), "hipMemcpy");
    return host;
  }
  std::vector<uint16_t> bits(static_cast<size_t>(numel_));
  check_hip(hipMemcpy(bits.data(), storage_, bytes_, hipMemcpyDeviceToHost), "hipMemcpy");
  std::vector<float> host(bits.size());
  for (size_t i = 0; i < bits.size(); ++i) host[i] = bf16_to_f32(bits[i]);
  return host;
}

Tensor Tensor::reshape(const std::vector<int>& shape) const {
  if (checked_numel(shape) != numel_) throw Error("tensor reshape numel mismatch");
  Tensor out(*ctx_, shape, dtype_);
  out.copy_from(*this);
  return out;
}

}  // namespace nn
