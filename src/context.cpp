#include "context_impl.hpp"
#include "check.hpp"

#include <cctype>
#include <string>
#include <vector>

extern "C" void rocblas_initialize();

namespace nn {
namespace {

std::string trimmed(std::string text) {
  auto space = [](unsigned char c) { return std::isspace(c) != 0; };
  while (!text.empty() && space(static_cast<unsigned char>(text.front()))) text.erase(text.begin());
  while (!text.empty() && space(static_cast<unsigned char>(text.back()))) text.pop_back();
  return text;
}

bool is_auto(const std::string& text) {
  if (text.size() != 4) return false;
  return std::tolower(static_cast<unsigned char>(text[0])) == 'a' &&
         std::tolower(static_cast<unsigned char>(text[1])) == 'u' &&
         std::tolower(static_cast<unsigned char>(text[2])) == 't' &&
         std::tolower(static_cast<unsigned char>(text[3])) == 'o';
}

bool parse_index(const std::string& text, int& index) {
  if (text.empty()) return false;
  int value = 0;
  for (char ch : text) {
    if (ch < '0' || ch > '9') return false;
    const int digit = ch - '0';
    if (value > (2147483647 - digit) / 10) return false;
    value = value * 10 + digit;
  }
  index = value;
  return true;
}

}  // namespace

void Context::open_handles() {
  check_hip(hipStreamCreate(&impl_->stream), "hipStreamCreate");
  check_hipblas(hipblasCreate(&impl_->blas), "hipblasCreate");
  check_hipblas(hipblasSetStream(impl_->blas, impl_->stream), "hipblasSetStream");
  check_hipblas(hipblasSetPointerMode(impl_->blas, HIPBLAS_POINTER_MODE_HOST), "hipblasSetPointerMode");
  // Split-K atomics make two identical GEMMs disagree. The checkpoint test compares them.
  check_hipblas(hipblasSetAtomicsMode(impl_->blas, HIPBLAS_ATOMICS_NOT_ALLOWED), "hipblasSetAtomicsMode");
  // Tensile loads gfx1152 kernels on first use. Do that once, off the training step.
  rocblas_initialize();
}

int device_count() {
  int count = 0;
  check_hip(hipGetDeviceCount(&count), "hipGetDeviceCount");
  return count;
}

std::string device_name(int device) {
  if (device < 0 || device >= device_count()) throw Error("device");
  hipDeviceProp_t prop{};
  check_hip(hipGetDeviceProperties(&prop, device), "hipGetDeviceProperties");
  return prop.name;
}

std::vector<DeviceInfo> detect_devices() {
  const int count = device_count();
  std::vector<DeviceInfo> found;
  found.reserve(static_cast<size_t>(count));
  for (int device = 0; device < count; ++device) {
    hipDeviceProp_t prop{};
    check_hip(hipGetDeviceProperties(&prop, device), "hipGetDeviceProperties");
    found.push_back(DeviceInfo{device, prop.name});
  }
  return found;
}

int select_device(const std::string& choice) {
  const std::vector<DeviceInfo> found = detect_devices();
  if (found.empty()) throw Error("device");
  const std::string text = trimmed(choice);
  if (is_auto(text)) return found.front().index;
  int index = 0;
  if (!parse_index(text, index) || index >= static_cast<int>(found.size())) throw Error("device");
  return index;
}

Context::Context() : impl_(std::make_unique<Impl>()) { open_handles(); }

Context::Context(int device) : impl_(std::make_unique<Impl>()) {
  if (device < 0 || device >= device_count()) throw Error("device");
  check_hip(hipSetDevice(device), "hipSetDevice");
  open_handles();
}

Context::Context(const std::string& choice) : impl_(std::make_unique<Impl>()) {
  check_hip(hipSetDevice(select_device(choice)), "hipSetDevice");
  open_handles();
}

Context::~Context() {
  if (!impl_) return;
  if (impl_->blas) hipblasDestroy(impl_->blas);
  if (impl_->stream) hipStreamDestroy(impl_->stream);
}

void Context::synchronize() const {
  check_hip(hipStreamSynchronize(impl_->stream), "hipStreamSynchronize");
}

}  // namespace nn
