#include <torch/torch.h>

#include <limits>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Include the auto-generated header with embedded metallib
#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#else
#error "EMBEDDED_METALLIB_HEADER not defined"
#endif

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor &tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

struct LaunchLayout {
  uint32_t row_stride;
  uint32_t vec_ok;
};

// Same acceptance rule as activation_row_stride in activation_kernels.cu:
// contiguous last dimension, leading dimensions collapsed to one row stride.
static int64_t activationRowStride(const torch::Tensor &input) {
  const int64_t nd = input.dim();
  TORCH_CHECK(nd >= 1, "activation input must have at least 1 dimension");
  const int64_t cols = input.size(nd - 1);
  if (cols > 1) {
    TORCH_CHECK(input.stride(nd - 1) == 1,
                "activation input last dimension must be contiguous, got stride ",
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
    TORCH_CHECK(stride_i >= 0,
                "activation input last dimension must be contiguous and row "
                "strides must be non-negative");
    if (!found) {
      row_stride = stride_i;
      inner = sz;
      found = true;
      continue;
    }
    TORCH_CHECK(inner > 0 &&
                    row_stride <= std::numeric_limits<int64_t>::max() / inner &&
                    stride_i == row_stride * inner,
                "activation input leading dimensions must form a single row "
                "stride");
    TORCH_CHECK(sz <= std::numeric_limits<int64_t>::max() / inner,
                "activation input row stride overflow");
    inner *= sz;
  }
  return found ? row_stride : cols;
}

static LaunchLayout launchLayout(const torch::Tensor &input, uint32_t d,
                                 bool gated) {
  const int64_t packed =
      gated ? static_cast<int64_t>(d) * 2 : static_cast<int64_t>(d);
  const int64_t row_stride =
      input.is_contiguous() ? packed : activationRowStride(input);
  TORCH_CHECK(row_stride >= 0 &&
                  row_stride <= std::numeric_limits<uint32_t>::max(),
              "activation input row stride does not fit in a 32-bit index");
  LaunchLayout layout;
  layout.row_stride = static_cast<uint32_t>(row_stride);
  const bool vec = input.storage_offset() % 4 == 0 && row_stride % 4 == 0 &&
                   static_cast<int64_t>(d) % 4 == 0;
  layout.vec_ok = vec ? 1u : 0u;
  return layout;
}

static void dispatchGatedKernel(const std::string &kernelName,
                                torch::Tensor &out,
                                torch::Tensor const &input,
                                LaunchLayout layout) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    const uint32_t d = static_cast<uint32_t>(input.size(-1) / 2);
    const uint32_t numTokens =
        static_cast<uint32_t>(input.numel() / input.size(-1));

    if (numTokens == 0 || d == 0) return;

    NSError *error = nil;
    id<MTLLibrary> library =
        EMBEDDED_METALLIB_NAMESPACE::createLibrary(device, &error);
    TORCH_CHECK(library, "Failed to create Metal library: ",
                error.localizedDescription.UTF8String);

    std::string kernel_name =
        kernelName + (input.scalar_type() == torch::kFloat ? "_f32" : "_f16");
    id<MTLFunction> function = [library
        newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
    TORCH_CHECK(function, "Failed to create function: ", kernel_name.c_str());

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:function error:&error];
    TORCH_CHECK(pso, "Failed to create pipeline state: ",
                error.localizedDescription.UTF8String);

    id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
    TORCH_CHECK(commandBuffer, "Failed to get command buffer");

    dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

    dispatch_sync(serialQueue, ^() {
      id<MTLComputeCommandEncoder> encoder =
          [commandBuffer computeCommandEncoder];
      TORCH_CHECK(encoder, "Failed to create compute encoder");

      [encoder setComputePipelineState:pso];
      [encoder setBuffer:getMTLBufferStorage(out)
                  offset:out.storage_offset() * out.element_size()
                 atIndex:0];
      [encoder setBuffer:getMTLBufferStorage(input)
                  offset:input.storage_offset() * input.element_size()
                 atIndex:1];
      [encoder setBytes:&d length:sizeof(uint32_t) atIndex:2];
      [encoder setBytes:&layout.row_stride length:sizeof(uint32_t) atIndex:3];
      [encoder setBytes:&layout.vec_ok length:sizeof(uint32_t) atIndex:4];

      uint32_t numChunks = (d + 7) / 8;
      MTLSize gridSize = MTLSizeMake(numChunks, numTokens, 1);
      NSUInteger threadGroupWidth =
          std::min<NSUInteger>(numChunks, pso.maxTotalThreadsPerThreadgroup);
      MTLSize threadGroupSize = MTLSizeMake(threadGroupWidth, 1, 1);

      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
      [encoder endEncoding];

      torch::mps::commit();
    });
  }
}

static void dispatchGatedKernelWithThreshold(const std::string &kernelName,
                                             torch::Tensor &out,
                                             torch::Tensor const &input,
                                             float threshold,
                                             LaunchLayout layout) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    const uint32_t d = static_cast<uint32_t>(input.size(-1) / 2);
    const uint32_t numTokens =
        static_cast<uint32_t>(input.numel() / input.size(-1));

    if (numTokens == 0 || d == 0) return;

    NSError *error = nil;
    id<MTLLibrary> library =
        EMBEDDED_METALLIB_NAMESPACE::createLibrary(device, &error);
    TORCH_CHECK(library, "Failed to create Metal library: ",
                error.localizedDescription.UTF8String);

    std::string kernel_name =
        kernelName + (input.scalar_type() == torch::kFloat ? "_f32" : "_f16");
    id<MTLFunction> function = [library
        newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
    TORCH_CHECK(function, "Failed to create function: ", kernel_name.c_str());

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:function error:&error];
    TORCH_CHECK(pso, "Failed to create pipeline state: ",
                error.localizedDescription.UTF8String);

    id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
    TORCH_CHECK(commandBuffer, "Failed to get command buffer");

    dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

    dispatch_sync(serialQueue, ^() {
      id<MTLComputeCommandEncoder> encoder =
          [commandBuffer computeCommandEncoder];
      TORCH_CHECK(encoder, "Failed to create compute encoder");

      [encoder setComputePipelineState:pso];
      [encoder setBuffer:getMTLBufferStorage(out)
                  offset:out.storage_offset() * out.element_size()
                 atIndex:0];
      [encoder setBuffer:getMTLBufferStorage(input)
                  offset:input.storage_offset() * input.element_size()
                 atIndex:1];
      [encoder setBytes:&d length:sizeof(uint32_t) atIndex:2];
      [encoder setBytes:&layout.row_stride length:sizeof(uint32_t) atIndex:3];
      [encoder setBytes:&layout.vec_ok length:sizeof(uint32_t) atIndex:4];
      [encoder setBytes:&threshold length:sizeof(float) atIndex:5];

      uint32_t numChunks = (d + 7) / 8;
      MTLSize gridSize = MTLSizeMake(numChunks, numTokens, 1);
      NSUInteger threadGroupWidth =
          std::min<NSUInteger>(numChunks, pso.maxTotalThreadsPerThreadgroup);
      MTLSize threadGroupSize = MTLSizeMake(threadGroupWidth, 1, 1);

      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
      [encoder endEncoding];

      torch::mps::commit();
    });
  }
}

static void dispatchElementwiseKernel(const std::string &kernelName,
                                      torch::Tensor &out,
                                      torch::Tensor const &input,
                                      LaunchLayout layout) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    const uint32_t d = static_cast<uint32_t>(input.size(-1));
    const uint32_t numTokens =
        static_cast<uint32_t>(input.numel() / input.size(-1));

    if (numTokens == 0 || d == 0) return;

    NSError *error = nil;
    id<MTLLibrary> library =
        EMBEDDED_METALLIB_NAMESPACE::createLibrary(device, &error);
    TORCH_CHECK(library, "Failed to create Metal library: ",
                error.localizedDescription.UTF8String);

    std::string kernel_name =
        kernelName + (input.scalar_type() == torch::kFloat ? "_f32" : "_f16");
    id<MTLFunction> function = [library
        newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
    TORCH_CHECK(function, "Failed to create function: ", kernel_name.c_str());

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:function error:&error];
    TORCH_CHECK(pso, "Failed to create pipeline state: ",
                error.localizedDescription.UTF8String);

    id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
    TORCH_CHECK(commandBuffer, "Failed to get command buffer");

    dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

    dispatch_sync(serialQueue, ^() {
      id<MTLComputeCommandEncoder> encoder =
          [commandBuffer computeCommandEncoder];
      TORCH_CHECK(encoder, "Failed to create compute encoder");

      [encoder setComputePipelineState:pso];
      [encoder setBuffer:getMTLBufferStorage(out)
                  offset:out.storage_offset() * out.element_size()
                 atIndex:0];
      [encoder setBuffer:getMTLBufferStorage(input)
                  offset:input.storage_offset() * input.element_size()
                 atIndex:1];
      [encoder setBytes:&d length:sizeof(uint32_t) atIndex:2];
      [encoder setBytes:&layout.row_stride length:sizeof(uint32_t) atIndex:3];
      [encoder setBytes:&layout.vec_ok length:sizeof(uint32_t) atIndex:4];

      uint32_t numChunks = (d + 7) / 8;
      MTLSize gridSize = MTLSizeMake(numChunks, numTokens, 1);
      NSUInteger threadGroupWidth =
          std::min<NSUInteger>(numChunks, pso.maxTotalThreadsPerThreadgroup);
      MTLSize threadGroupSize = MTLSizeMake(threadGroupWidth, 1, 1);

      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
      [encoder endEncoding];

      torch::mps::commit();
    });
  }
}

static LaunchLayout checkInputs(torch::Tensor &out, torch::Tensor const &input,
                                bool gated) {
  TORCH_CHECK(input.device().is_mps(), "input must be a MPS tensor");
  TORCH_CHECK(out.device().is_mps(), "output must be a MPS tensor");
  TORCH_CHECK(out.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kFloat ||
                  input.scalar_type() == torch::kHalf,
              "Unsupported data type: ", input.scalar_type());
  TORCH_CHECK(input.scalar_type() == out.scalar_type(),
              "Input and output must have the same dtype");
  const uint32_t d = static_cast<uint32_t>(
      gated ? input.size(-1) / 2 : input.size(-1));
  return launchLayout(input, d, gated);
}

void silu_and_mul(torch::Tensor &out, torch::Tensor &input) {
  LaunchLayout layout = checkInputs(out, input, true);
  dispatchGatedKernel("silu_and_mul", out, input, layout);
}

void mul_and_silu(torch::Tensor &out, torch::Tensor &input) {
  LaunchLayout layout = checkInputs(out, input, true);
  dispatchGatedKernel("mul_and_silu", out, input, layout);
}

void gelu_and_mul(torch::Tensor &out, torch::Tensor &input) {
  LaunchLayout layout = checkInputs(out, input, true);
  dispatchGatedKernel("gelu_and_mul", out, input, layout);
}

void gelu_tanh_and_mul(torch::Tensor &out, torch::Tensor &input) {
  LaunchLayout layout = checkInputs(out, input, true);
  dispatchGatedKernel("gelu_tanh_and_mul", out, input, layout);
}

void fatrelu_and_mul(torch::Tensor &out, torch::Tensor &input,
                     double threshold) {
  LaunchLayout layout = checkInputs(out, input, true);
  dispatchGatedKernelWithThreshold("fatrelu_and_mul", out, input,
                                   static_cast<float>(threshold), layout);
}

void silu(torch::Tensor &out, torch::Tensor &input) {
  LaunchLayout layout = checkInputs(out, input, false);
  dispatchElementwiseKernel("silu", out, input, layout);
}

void gelu(torch::Tensor &out, torch::Tensor &input) {
  LaunchLayout layout = checkInputs(out, input, false);
  dispatchElementwiseKernel("gelu", out, input, layout);
}

void gelu_tanh(torch::Tensor &out, torch::Tensor &input) {
  LaunchLayout layout = checkInputs(out, input, false);
  dispatchElementwiseKernel("gelu_tanh", out, input, layout);
}

void gelu_new(torch::Tensor &out, torch::Tensor &input) {
  LaunchLayout layout = checkInputs(out, input, false);
  dispatchElementwiseKernel("gelu_new", out, input, layout);
}

void gelu_fast(torch::Tensor &out, torch::Tensor &input) {
  LaunchLayout layout = checkInputs(out, input, false);
  dispatchElementwiseKernel("gelu_fast", out, input, layout);
}

void gelu_quick(torch::Tensor &out, torch::Tensor &input) {
  LaunchLayout layout = checkInputs(out, input, false);
  dispatchElementwiseKernel("gelu_quick", out, input, layout);
}
