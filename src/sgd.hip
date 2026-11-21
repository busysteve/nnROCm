#include "kernels.hpp"

#include "check.hpp"
#include "launch.hpp"

namespace nn {
namespace {

__device__ void sgd_one(float& param, float& velocity, float grad, float lr, float momentum) {
  const float updated = momentum * velocity + grad;
  velocity = updated;
  param = param - lr * updated;
}

__global__ void __launch_bounds__(256) sgd_kernel(float* param, float* velocity, const float* grad,
                                                  int64_t n, float lr, float momentum) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  sgd_one(param[i], velocity[i], grad[i], lr, momentum);
}

__global__ void __launch_bounds__(256) sgd4_kernel(float* param, float* velocity, const float* grad,
                                                   int64_t n4, float lr, float momentum) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  float4 p = reinterpret_cast<float4*>(param)[i];
  float4 vel = reinterpret_cast<float4*>(velocity)[i];
  const float4 g = reinterpret_cast<const float4*>(grad)[i];
  sgd_one(p.x, vel.x, g.x, lr, momentum);
  sgd_one(p.y, vel.y, g.y, lr, momentum);
  sgd_one(p.z, vel.z, g.z, lr, momentum);
  sgd_one(p.w, vel.w, g.w, lr, momentum);
  reinterpret_cast<float4*>(param)[i] = p;
  reinterpret_cast<float4*>(velocity)[i] = vel;
}

}  // namespace

void launch_sgd(hipStream_t stream, float* param, float* velocity, const float* grad, int64_t n,
                float learning_rate, float momentum) {
  if (n <= 0) return;
  if ((n % 4) == 0 && aligned16(param) && aligned16(velocity) && aligned16(grad)) {
    sgd4_kernel<<<blocks_for(n / 4), 256, 0, stream>>>(param, velocity, grad, n / 4, learning_rate,
                                                       momentum);
  } else {
    sgd_kernel<<<blocks_for(n), 256, 0, stream>>>(param, velocity, grad, n, learning_rate, momentum);
  }
  check_hip(hipGetLastError(), "sgd");
}

}  // namespace nn
