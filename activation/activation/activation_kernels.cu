// Required for Microsoft math constants and must be defined before including <cmath>
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/math-constants?view=msvc-170
#define _USE_MATH_DEFINES

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

// The shim's stream accessor is guarded by USE_CUDA, so declare it here.
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
extern "C" AOTITorchError aoti_torch_get_current_cuda_stream(
    int32_t device_index, void** ret_stream);

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "cuda_compat.h"
#include "dispatch_utils.h"

using torch::stable::Tensor;

namespace vllm {

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y) {
  return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}
// Activation and gating kernel template.

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d, const int64_t in_row_stride) {
  const int64_t token_idx = blockIdx.x;
  const int64_t in_base = token_idx * in_row_stride;
  const int64_t out_base = token_idx * static_cast<int64_t>(d);
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[in_base + idx]);
    const scalar_t y =
        VLLM_LDG(&input[in_base + static_cast<int64_t>(d) + idx]);
    out[out_base + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
  }
}

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

template <typename T>
__device__ __forceinline__ T gelu_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  const float f = (float)x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T)(f * 0.5f * (1.0f + erf(f * ALPHA)));
}

template <typename T>
__device__ __forceinline__ T gelu_tanh_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  const float f = (float)x;
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;
  float x_cube = f * f * f;
  float inner = BETA * (f + KAPPA * x_cube);
  return (T)(0.5f * f * (1.0f + ::tanhf(inner)));
}

}  // namespace vllm

// Row stride, in elements, of a tensor viewed as (num_rows, size(-1)).
//
// Contiguous inputs keep the packed stride and never call this. The supported
// non-contiguous layout is a contiguous last dimension whose leading
// dimensions collapse to one row stride, such as chunk() on the last dim.
// A negative stride, or a gap that is not a single row stride, is rejected.
static int64_t activation_row_stride(const Tensor& input) {
  const int64_t nd = input.dim();
  STD_TORCH_CHECK(nd >= 1, "activation input must have at least 1 dimension");
  const int64_t cols = input.size(nd - 1);
  if (cols > 1) {
    STD_TORCH_CHECK(input.stride(nd - 1) == 1,
                    "activation input last dimension must be contiguous, got "
                    "stride ",
                    input.stride(nd - 1));
  }
  if (nd == 1) {
    return cols;
  }
  int64_t row_stride = 0;
  int64_t inner = 1;
  bool found = false;
  for (int64_t i = nd - 2; i >= 0; --i) {
    const int64_t sz = input.size(i);
    if (sz == 1) {
      continue;
    }
    const int64_t stride_i = input.stride(i);
    STD_TORCH_CHECK(
        stride_i >= 0,
        "activation input last dimension must be contiguous and row strides "
        "must be non-negative");
    if (!found) {
      row_stride = stride_i;
      inner = sz;
      found = true;
      continue;
    }
    STD_TORCH_CHECK(
        inner > 0 && row_stride <= std::numeric_limits<int64_t>::max() / inner &&
            stride_i == row_stride * inner,
        "activation input leading dimensions must form a single row stride");
    STD_TORCH_CHECK(sz <= std::numeric_limits<int64_t>::max() / inner,
                    "activation input row stride overflow");
    inner *= sz;
  }
  return found ? row_stride : cols;
}

// Launch activation and gating kernel.
// Use ACT_FIRST (bool) indicating whether to apply the activation function
// first.
#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, ACT_FIRST)                       \
  int d = input.size(-1) / 2;                                                  \
  int64_t num_tokens = input.numel() / input.size(-1);                        \
  const int64_t in_row_stride =                                                \
      input.is_contiguous() ? static_cast<int64_t>(d) * 2                      \
                            : activation_row_stride(input);                    \
  dim3 grid(num_tokens);                                                       \
  dim3 block(std::min(d, 1024));                                               \
  if (num_tokens == 0) {                                                       \
    return;                                                                    \
  }                                                                            \
  const torch::stable::accelerator::DeviceGuard device_guard(                  \
      input.get_device_index());                                              \
  void* stream_ptr = nullptr;                                                  \
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(                   \
      input.get_device_index(), &stream_ptr));                                 \
  const cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);           \
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(                                                \
      input.scalar_type(), "act_and_mul_kernel", [&] {                        \
        vllm::act_and_mul_kernel<scalar_t, KERNEL<scalar_t>, ACT_FIRST>       \
            <<<grid, block, 0, stream>>>(                                      \
                static_cast<scalar_t*>(out.data_ptr()),                        \
                static_cast<const scalar_t*>(input.data_ptr()), d,            \
                in_row_stride);                                                \
      });

void silu_and_mul(Tensor& out,          // [..., d]
                  Tensor const& input)  // [..., 2 * d]
{
  STD_TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, true);
}

void mul_and_silu(Tensor& out,          // [..., d]
                  Tensor const& input)  // [..., 2 * d]
{
  // The difference between mul_and_silu and silu_and_mul is that mul_and_silu
  // applies the silu to the latter half of the input.
  STD_TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, false);
}

void gelu_and_mul(Tensor& out,          // [..., d]
                  Tensor const& input)  // [..., 2 * d]
{
  STD_TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_kernel, true);
}

void gelu_tanh_and_mul(Tensor& out,          // [..., d]
                       Tensor const& input)  // [..., 2 * d]
{
  STD_TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_tanh_kernel, true);
}

namespace vllm {

template <typename T>
__device__ __forceinline__ T fatrelu_kernel(const T& x, const float threshold) {
  const float f = (float)x;
  return (T)(f > threshold ? f : 0.0f);
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&, const float)>
__global__ void act_and_mul_kernel_with_param(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input, const int d,
    const float param, const int64_t in_row_stride) {
  const int64_t token_idx = blockIdx.x;
  const int64_t in_base = token_idx * in_row_stride;
  const int64_t out_base = token_idx * static_cast<int64_t>(d);
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[in_base + idx]);
    const scalar_t y =
        VLLM_LDG(&input[in_base + static_cast<int64_t>(d) + idx]);
    out[out_base + idx] = ACT_FN(x, param) * y;
  }
}

}  // namespace vllm

#define LAUNCH_ACTIVATION_GATE_KERNEL_WITH_PARAM(KERNEL, PARAM)               \
  int d = input.size(-1) / 2;                                                 \
  int64_t num_tokens = input.numel() / input.size(-1);                       \
  const int64_t in_row_stride =                                               \
      input.is_contiguous() ? static_cast<int64_t>(d) * 2                     \
                            : activation_row_stride(input);                   \
  dim3 grid(num_tokens);                                                      \
  dim3 block(std::min(d, 1024));                                              \
  if (num_tokens == 0) {                                                      \
    return;                                                                   \
  }                                                                           \
  const torch::stable::accelerator::DeviceGuard device_guard(                 \
      input.get_device_index());                                             \
  void* stream_ptr = nullptr;                                                 \
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(                  \
      input.get_device_index(), &stream_ptr));                                \
  const cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);          \
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(                                               \
      input.scalar_type(), "act_and_mul_kernel_with_param", [&] {            \
        vllm::act_and_mul_kernel_with_param<scalar_t, KERNEL<scalar_t>>      \
            <<<grid, block, 0, stream>>>(                                     \
                static_cast<scalar_t*>(out.data_ptr()),                       \
                static_cast<const scalar_t*>(input.data_ptr()), d, PARAM,    \
                in_row_stride);                                               \
      });

void fatrelu_and_mul(Tensor& out,          // [..., d],
                     Tensor const& input,  // [..., 2 * d]
                     double threshold) {
  STD_TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  LAUNCH_ACTIVATION_GATE_KERNEL_WITH_PARAM(vllm::fatrelu_kernel, threshold);
}
namespace vllm {

// Element-wise activation kernel template.
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
__global__ void activation_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., d]
    const int d, const int64_t in_row_stride) {
  const int64_t token_idx = blockIdx.x;
  const int64_t in_base = token_idx * in_row_stride;
  const int64_t out_base = token_idx * static_cast<int64_t>(d);
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[in_base + idx]);
    out[out_base + idx] = ACT_FN(x);
  }
}

}  // namespace vllm

// Launch element-wise activation kernel.
#define LAUNCH_ACTIVATION_KERNEL(KERNEL)                                     \
  int d = input.size(-1);                                                    \
  int64_t num_tokens = input.numel() / d;                                    \
  const int64_t in_row_stride =                                              \
      input.is_contiguous() ? static_cast<int64_t>(d)                        \
                            : activation_row_stride(input);                  \
  dim3 grid(num_tokens);                                                     \
  dim3 block(std::min(d, 1024));                                             \
  if (num_tokens == 0) {                                                     \
    return;                                                                  \
  }                                                                          \
  const torch::stable::accelerator::DeviceGuard device_guard(               \
      input.get_device_index());                                            \
  void* stream_ptr = nullptr;                                                \
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(                 \
      input.get_device_index(), &stream_ptr));                               \
  const cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);         \
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(                                              \
      input.scalar_type(), "activation_kernel", [&] {                       \
        vllm::activation_kernel<scalar_t, KERNEL<scalar_t>>                 \
            <<<grid, block, 0, stream>>>(                                    \
                static_cast<scalar_t*>(out.data_ptr()),                      \
                static_cast<const scalar_t*>(input.data_ptr()), d,          \
                in_row_stride);                                              \
      });

namespace vllm {


template <typename T>
__device__ __forceinline__ T gelu_new_kernel(const T& x) {
  const float x3 = (float)(x * x * x);
  const T t = (T)tanhf((T)(0.79788456f * (float)(x + (T)(0.044715f * x3))));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__device__ __forceinline__ T gelu_fast_kernel(const T& x) {
  const float f = (float)x;
  const T t =
      (T)tanhf(((T)(f * 0.79788456f)) * (((T)1.0) + (T)(0.044715f * f) * x));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__device__ __forceinline__ T gelu_quick_kernel(const T& x) {
  // x * sigmoid(1.702 * x)
  return (T)(((float)x) / (1.0f + expf(-1.702f * (float)x)));
}

}  // namespace vllm

void gelu_new(Tensor& out,          // [..., d]
              Tensor const& input)  // [..., d]
{
  STD_TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_new_kernel);
}

void gelu_fast(Tensor& out,          // [..., d]
               Tensor const& input)  // [..., d]
{
  STD_TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_fast_kernel);
}

void gelu_quick(Tensor& out,          // [..., d]
                Tensor const& input)  // [..., d]
{
  STD_TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_quick_kernel);
}

void gelu(Tensor& out,          // [..., d]
          Tensor const& input)  // [..., d]
{
  STD_TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_kernel);
}

void gelu_tanh(Tensor& out,          // [..., d]
               Tensor const& input)  // [..., d]
{
  STD_TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_tanh_kernel);
}

void silu(Tensor& out,          // [..., d]
          Tensor const& input)  // [..., d]
{
  STD_TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  LAUNCH_ACTIVATION_KERNEL(vllm::silu_kernel);
}
