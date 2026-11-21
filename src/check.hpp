#pragma once

#include "nn/error.hpp"

#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>

namespace nn {

void check_hip(hipError_t status, const char* call);
void check_hipblas(hipblasStatus_t status, const char* call);

}  // namespace nn
