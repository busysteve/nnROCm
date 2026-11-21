#include "check.hpp"

#include <string>

namespace nn {

void check_hip(hipError_t status, const char* call) {
  if (status == hipSuccess) return;
  throw Error(std::string(call) + ": " + hipGetErrorName(status));
}

void check_hipblas(hipblasStatus_t status, const char* call) {
  if (status == HIPBLAS_STATUS_SUCCESS) return;
  const char* name = hipblasStatusToString(status);
  throw Error(std::string(call) + ": " + (name ? name : std::to_string(static_cast<int>(status))));
}

}  // namespace nn
