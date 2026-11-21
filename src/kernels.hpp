#pragma once

#include <hip/hip_runtime_api.h>

#include <cstdint>

namespace nn {

void launch_add_bias(hipStream_t stream, float* y, const float* bias, int rows, int cols);
void launch_bias_grad(hipStream_t stream, const float* grad_y, float* grad_bias, int rows, int cols);
void launch_add2(hipStream_t stream, float* dst, const float* a, const float* b, int64_t n);
void launch_mul(hipStream_t stream, float* y, const float* a, const float* b, int64_t n);
void launch_gelu(hipStream_t stream, float* y, const float* x, int64_t n);
void launch_gelu_backward(hipStream_t stream, float* grad_x, const float* x, const float* grad_y,
                          int64_t n);
void launch_swiglu(hipStream_t stream, float* y, const float* x, int rows, int64_t inner);
void launch_swiglu_backward(hipStream_t stream, float* grad_x, const float* x, const float* grad_y,
                            int rows, int64_t inner);
void launch_tanh(hipStream_t stream, float* y, const float* x, int64_t n);
void launch_tanh_backward(hipStream_t stream, float* grad_x, const float* y, const float* grad_y,
                          int64_t n);
void launch_gather_time(hipStream_t stream, float* dst, const float* src, int batch, int time,
                        int features, int index);
void launch_scatter_time(hipStream_t stream, float* dst, const float* src, int batch, int time,
                         int features, int index);
void launch_sgd(hipStream_t stream, float* param, float* velocity, const float* grad, int64_t n,
                float learning_rate, float momentum);
void launch_adamw(hipStream_t stream, float* param, float* m, float* v, const float* grad, int64_t n,
                  float lr, float beta1, float beta2, float eps, float weight_decay, float bc1,
                  float bc2);
double launch_sum_squares(hipStream_t stream, const float* x, int64_t n);
float launch_sum(hipStream_t stream, const float* x, int64_t n);
void launch_scale(hipStream_t stream, float* x, int64_t n, float scale);
void launch_fill(hipStream_t stream, float* x, int64_t n, float value);

}  // namespace nn
