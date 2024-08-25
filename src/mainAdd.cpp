#include "vcm/VulkanComputeManager.hpp"
#include "vcm/vcm.hpp"
#include "vulkan/vulkan_enums.hpp"
#include "vulkan/vulkan_structs.hpp"
#include <cstddef>
#include <fmt/core.h>
#include <iostream>
#include <span>
#include <stdexcept>
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
      std::cerr << "Mismatch at index " << i << ": expected " << expectedValue
                << ", but got " << outputData[i] << "\n";
      return false;
    }
  }
  return true;
}

namespace fs = std::filesystem;

int main() {
  const uint32_t WIDTH = 512;
  const uint32_t HEIGHT = 512;

  vcm::VulkanComputeManager cm;

  /*
  Create buffers
  */
  vk::DeviceSize bufferSize = WIDTH * HEIGHT * sizeof(float);

  vcm::VulkanBuffer inputBuf1Staging = cm.createStagingBufferSrc(bufferSize);
  vcm::VulkanBuffer inputBuf1 = cm.createDeviceBufferDst(bufferSize);

  vcm::VulkanBuffer inputBuf2Staging = cm.createStagingBufferSrc(bufferSize);
  vcm::VulkanBuffer inputBuf2 = cm.createDeviceBufferDst(bufferSize);

  vcm::ComputeShaderBuffers<2> buffers{};
  auto outputBufStaging = cm.createStagingBufferDst(bufferSize);
  auto outputBuf = cm.createDeviceBufferSrc(bufferSize);
  buffers.inWidth = WIDTH;
  buffers.inHeight = HEIGHT;

  buffers.in = {{{inputBuf1.ref(), inputBuf1Staging.ref()},
                 {inputBuf2.ref(), inputBuf2Staging.ref()}}};
  buffers.out = {outputBuf.ref(), outputBufStaging.ref()};

  // Copy data to staging buffers
  std::vector<float> input1(WIDTH * HEIGHT, 1);
  std::vector<float> input2(WIDTH * HEIGHT, 2);

  cm.copyToStagingBuffer<float>(input1, buffers.in[0].staging);
  cm.copyToStagingBuffer<float>(input2, buffers.in[1].staging);

  /* Create shader */
  vcm::ShaderExecutor<2, PushConstantData> shader("shaders/add2D.spv", cm,
                                                  buffers);

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

  // Verify that each element in the output matrix is 3.0f
  {
    bool isCorrect = verifyOutput(outputData, 3.0F);
    if (isCorrect) {
      std::cout << "The output is correct!\n";
    } else {
      std::cout << "The output is incorrect.\n";
    }
  }

  std::vector<float> outputCPU(WIDTH * HEIGHT);
  {
    vcm::TimeIt<true> timeit("CPU version");
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
      outputCPU[i] = input1[i] + input2[i];
    }
  }

  // Verify results
  {
    bool isCorrect = verifyOutput(outputCPU, 3.0F);
    if (isCorrect) {
      std::cout << "The output is correct!\n";
    } else {
      std::cout << "The output is incorrect.\n";
    }
  }

  return 0;
}
