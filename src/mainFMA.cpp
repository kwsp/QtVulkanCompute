#include "vcm/vcm.hpp"
#include <cmath>
#include <fmt/core.h>
#include <vector>
#include <vulkan/vulkan.hpp>

// Question: uniform buffer vs push constant
struct PushConstantData {
  int width;
  int height;
};

bool verifyOutput(const std::vector<float> &outputData, float expectedValue) {
  for (size_t i = 0; i < outputData.size(); ++i) {
    if (outputData[i] != expectedValue) {
      fmt::println("Mismatch at index {}; expected {}, but got {}", i,
                   expectedValue, outputData[i]);
      return false;
    }
  }
  return true;
}

int main() {
  const uint32_t WIDTH = 512 * 2;
  const uint32_t HEIGHT = 512 * 2;

  vcm::VulkanComputeManager cm;

  /*
  Create buffers
  */
  vk::DeviceSize bufferSize = WIDTH * HEIGHT * sizeof(float);

  vcm::VulkanBuffer inputBuf1Staging = cm.createStagingBufferSrc(bufferSize);
  vcm::VulkanBuffer inputBuf1 = cm.createDeviceBufferDst(bufferSize);

  vcm::VulkanBuffer inputBuf2Staging = cm.createStagingBufferSrc(bufferSize);
  vcm::VulkanBuffer inputBuf2 = cm.createDeviceBufferDst(bufferSize);

  vcm::VulkanBuffer inputBuf3Staging = cm.createStagingBufferSrc(bufferSize);
  vcm::VulkanBuffer inputBuf3 = cm.createDeviceBufferDst(bufferSize);

  vcm::ComputeShaderBuffers<3> buffers{};
  auto outputBufStaging = cm.createStagingBufferDst(bufferSize);
  auto outputBuf = cm.createDeviceBufferSrc(bufferSize);
  buffers.inWidth = WIDTH;
  buffers.inHeight = HEIGHT;
  buffers.outWidth = WIDTH;
  buffers.outHeight = HEIGHT;

  buffers.in = {{{inputBuf1.ref(), inputBuf1Staging.ref()},
                 {inputBuf2.ref(), inputBuf2Staging.ref()},
                 {inputBuf3.ref(), inputBuf3Staging.ref()}}};
  buffers.out = {outputBuf.ref(), outputBufStaging.ref()};

  // Copy data to staging buffers
  std::vector<float> input1(WIDTH * HEIGHT, 3);
  std::vector<float> input2(WIDTH * HEIGHT, 2);
  std::vector<float> input3(WIDTH * HEIGHT, 1);

  cm.copyToStagingBuffer<float>(input1, buffers.in[0].staging);
  cm.copyToStagingBuffer<float>(input2, buffers.in[1].staging);
  cm.copyToStagingBuffer<float>(input3, buffers.in[2].staging);

  /* Create shader */
  vcm::ShaderExecutor<3, PushConstantData> shader("shaders/fma2D.spv", cm,
                                                  buffers, {16, 16, 1});

  auto &commandBuffer = cm.commandBuffer;

  PushConstantData pushConstant{WIDTH, HEIGHT};
  shader.recordCommandBuffer(cm, commandBuffer, buffers, pushConstant);

  // Submit the command buffer to the compute queue
  {
    vcm::TimeIt<true> timeit("Wait queue");

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    cm.queue.submit(submitInfo);

    cm.queue.waitIdle();
  }

  std::vector<float> outputData(WIDTH * HEIGHT);
  cm.copyFromStagingBuffer<float>(buffers.out.staging, outputData);

  {
    bool isCorrect = verifyOutput(outputData, 7.0F);
    if (isCorrect) {
      fmt::println("The output is correct!");
    } else {
      fmt::println("The output is incorrect.");
    }
  }

  std::vector<float> outputCPU(WIDTH * HEIGHT);
  {
    vcm::TimeIt<true> timeit("CPU version");
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
      outputCPU[i] = std::fmaf(input1[i], input2[i], input3[i]);
    }
  }

  // Verify results
  {
    bool isCorrect = verifyOutput(outputCPU, 7.0F);
    if (isCorrect) {
      fmt::println("The output is correct!");
    } else {
      fmt::println("The output is incorrect.");
    }
  }

  return 0;
}
