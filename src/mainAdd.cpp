#include "timeit.hpp"
#include "vcm/VulkanComputeManager.hpp"
#include "vulkan/vulkan_enums.hpp"
#include "vulkan/vulkan_structs.hpp"
#include <cstddef>
#include <fmt/core.h>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.hpp>

// Descriptor set and pipeline required to setup the compute shader
// Trying to make this general to all compute shaders
struct ComputeShaderResources {
  vk::UniqueDescriptorSetLayout
      descriptorSetLayout;                 // Compute shader binding layout
  vk::UniqueDescriptorSet descriptorSet;   // Compute shader bindings
  vk::UniquePipelineLayout pipelineLayout; // Layout of the compute pipeline
  vk::UniquePipeline pipeline;             // Compute pipeline (can define more)
};

// Shader specific buffer
struct ComputeShaderBuffers {
  // Question: uniform buffer vs push constant
  vcm::VulkanBuffer stagingBuffer1;
  vcm::VulkanBuffer stagingBuffer2;
  vcm::VulkanBuffer stagingBuffer3;

  vcm::VulkanBuffer buffer1;
  vcm::VulkanBuffer buffer2;
  vcm::VulkanBuffer buffer3;
};
struct PushConstantData {
  int width;
  int height;
};

void createComputePipeline(vcm::VulkanComputeManager &cm,
                           ComputeShaderResources &resources) {
  // Load the SPIR-V binary
  vk::UniqueShaderModule shaderModule = cm.loadShader("shaders/add2D.spv");

  vk::PushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eCompute;
  pushConstantRange.offset = 0;
  pushConstantRange.size = sizeof(PushConstantData);

  vk::PipelineShaderStageCreateInfo shaderStageCreateInfo{};
  shaderStageCreateInfo.stage = vk::ShaderStageFlagBits::eCompute;
  shaderStageCreateInfo.module = *shaderModule;
  shaderStageCreateInfo.pName = "main";

  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
  pipelineLayoutCreateInfo.setLayoutCount = 1;
  pipelineLayoutCreateInfo.pSetLayouts = &resources.descriptorSetLayout.get();
  pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
  pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

  resources.pipelineLayout =
      cm.device->createPipelineLayoutUnique(pipelineLayoutCreateInfo);

  vk::ComputePipelineCreateInfo pipelineCreateInfo{};
  pipelineCreateInfo.stage = shaderStageCreateInfo;
  pipelineCreateInfo.layout = *resources.pipelineLayout;

  auto pipelineResult =
      cm.device->createComputePipelineUnique(nullptr, pipelineCreateInfo);

  if (pipelineResult.result != vk::Result::eSuccess) {
    throw std::runtime_error("Failed to create compute pipeline!");
  }

  resources.pipeline = std::move(pipelineResult.value);
}

void createDescriptorPoolAndSet(vcm::VulkanComputeManager &cm,
                                ComputeShaderResources &resources,
                                ComputeShaderBuffers &buffers) {

  // Create descriptor set
  {
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = cm.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &*resources.descriptorSetLayout;

    auto descriptorSets = cm.device->allocateDescriptorSetsUnique(allocInfo);
    resources.descriptorSet = std::move(descriptorSets.front());
  }

  // Bind device buffers to the descriptor set
  {
    vk::DescriptorBufferInfo bufferInfo1{};
    bufferInfo1.buffer = buffers.buffer1.buffer.get();
    bufferInfo1.offset = 0;
    bufferInfo1.range = VK_WHOLE_SIZE;

    vk::DescriptorBufferInfo bufferInfo2{};
    bufferInfo2.buffer = buffers.buffer2.buffer.get();
    bufferInfo2.offset = 0;
    bufferInfo2.range = VK_WHOLE_SIZE;

    vk::DescriptorBufferInfo outputBufferInfo{};
    outputBufferInfo.buffer = buffers.buffer3.buffer.get();
    outputBufferInfo.offset = 0;
    outputBufferInfo.range = VK_WHOLE_SIZE;

    std::array<vk::WriteDescriptorSet, 3> descriptorWrites{};

    descriptorWrites[0].dstSet = resources.descriptorSet.get();
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = vk::DescriptorType::eStorageBuffer;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &bufferInfo1;

    descriptorWrites[1].dstSet = resources.descriptorSet.get();
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType = vk::DescriptorType::eStorageBuffer;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &bufferInfo2;

    descriptorWrites[2].dstSet = resources.descriptorSet.get();
    descriptorWrites[2].dstBinding = 2;
    descriptorWrites[2].dstArrayElement = 0;
    descriptorWrites[2].descriptorType = vk::DescriptorType::eStorageBuffer;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pBufferInfo = &outputBufferInfo;

    cm.device->updateDescriptorSets(descriptorWrites, {});
  }
}

// Bind command buffer to pipeline
// Bind command buffer to descriptor set
// Dispatch command buffer
void dispatchComputeShader(vcm::VulkanComputeManager &cm,
                           ComputeShaderResources &resources, int inputWidth,
                           int inputHeight) {
  cm.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                resources.pipeline.get());

  cm.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                      resources.pipelineLayout.get(), 0,
                                      resources.descriptorSet.get(), {});

  PushConstantData pushConstantData{inputWidth, inputHeight};

  cm.commandBuffer.pushConstants(resources.pipelineLayout.get(),
                                 vk::ShaderStageFlagBits::eCompute, 0,
                                 sizeof(PushConstantData), &pushConstantData);

  cm.commandBuffer.dispatch((inputWidth + 15) / 16, (inputHeight + 15) / 16, 1);
}

void recordCommandBuffer(vcm::VulkanComputeManager &cm,
                         vk::CommandBuffer commandBuffer,
                         ComputeShaderResources &resources,
                         ComputeShaderBuffers &buffers,
                         vk::DeviceSize bufferSize, int width, int height) {
  uspam::TimeIt<true> timeit("Recording command buffer");

  // Record the command buffer
  {
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    commandBuffer.begin(beginInfo);
  }

  // // Copy data from staging to host buffers
  // {
  //   // Synchronous copy
  //   uspam::TimeIt<true> timeit("Copy buffer");
  //   cm.copyBuffer(buffers.stagingBuffer1.buffer, buffers.buffer1.buffer,
  //                 bufferSize);
  //   cm.copyBuffer(buffers.stagingBuffer2.buffer, buffers.buffer2.buffer,
  //                 bufferSize);

  //   std::array<vcm::VulkanComputeManager::CopyBufferT, 2> buffersToCopy = {
  //       {{buffers.stagingBuffer1.buffer, buffers.buffer1.buffer,
  //       bufferSize},
  //        {buffers.stagingBuffer2.buffer, buffers.buffer2.buffer,
  //         bufferSize}}};

  //   cm.copyBuffers(buffersToCopy);
  // }

  {
    // Async copy with barrier
    cm.copyBuffer(buffers.stagingBuffer1.buffer.get(),
                  buffers.buffer1.buffer.get(), bufferSize, commandBuffer);
    cm.copyBuffer(buffers.stagingBuffer2.buffer.get(),
                  buffers.buffer2.buffer.get(), bufferSize, commandBuffer);

    vk::MemoryBarrier memoryBarrier{};
    memoryBarrier.srcAccessMask =
        vk::AccessFlagBits::eTransferWrite; // After copying
    memoryBarrier.dstAccessMask =
        vk::AccessFlagBits::eShaderRead; // Before compute shader reads

    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,      // src: after the transfer op
        vk::PipelineStageFlagBits::eComputeShader, // dst: before the compute
                                                   // shader
        {}, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
  }

  dispatchComputeShader(cm, resources, width, height);

  {
    // (Optional) Step 4: Insert another pipeline barrier if needed
    vk::MemoryBarrier memoryBarrier{};
    memoryBarrier.srcAccessMask =
        vk::AccessFlagBits::eShaderWrite; // After compute shader writes
    memoryBarrier.dstAccessMask =
        vk::AccessFlagBits::eTransferRead; // Before transfer reads

    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader, // src: after compute
        vk::PipelineStageFlagBits::eTransfer,      // dst: before next transfer
        {}, 1, &memoryBarrier, 0, nullptr, 0, nullptr);

    // Copy result back to staging
    cm.copyBuffer(buffers.buffer3.buffer.get(),
                  buffers.stagingBuffer3.buffer.get(), bufferSize,
                  commandBuffer);
  }

  commandBuffer.end();
}

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

int main() {
  const uint32_t WIDTH = 512;
  const uint32_t HEIGHT = 512;

  vcm::VulkanComputeManager cm;
  ComputeShaderResources resources{};
  ComputeShaderBuffers buffers{};

  /*
  Create buffers
  */
  vk::DeviceSize bufferSize = WIDTH * HEIGHT * sizeof(float);

  {
    using vk::BufferUsageFlagBits::eStorageBuffer;
    using vk::BufferUsageFlagBits::eTransferDst;
    using vk::BufferUsageFlagBits::eTransferSrc;
    using vk::MemoryPropertyFlagBits::eDeviceLocal;
    using vk::MemoryPropertyFlagBits::eHostCoherent;
    using vk::MemoryPropertyFlagBits::eHostVisible;

    buffers.stagingBuffer1 =
        cm.createBuffer(bufferSize, eTransferSrc, eHostVisible | eHostCoherent);
    buffers.buffer1 = cm.createBuffer(bufferSize, eStorageBuffer | eTransferDst,
                                      eDeviceLocal);

    buffers.stagingBuffer2 =
        cm.createBuffer(bufferSize, eTransferSrc, eHostVisible | eHostCoherent);
    buffers.buffer2 = cm.createBuffer(bufferSize, eStorageBuffer | eTransferDst,
                                      eDeviceLocal);

    buffers.stagingBuffer3 =
        cm.createBuffer(bufferSize, eTransferDst, eHostVisible | eHostCoherent);
    buffers.buffer3 = cm.createBuffer(bufferSize, eStorageBuffer | eTransferSrc,
                                      eDeviceLocal);
  }

  // descriptorset layout bindings
  {
    std::array<vk::DescriptorSetLayoutBinding, 3> descriptorSetLayoutBindings{};

    descriptorSetLayoutBindings[0].binding = 0;
    descriptorSetLayoutBindings[0].descriptorType =
        vk::DescriptorType::eStorageBuffer;
    descriptorSetLayoutBindings[0].descriptorCount = 1;

    descriptorSetLayoutBindings[0].stageFlags =
        vk::ShaderStageFlagBits::eCompute;

    descriptorSetLayoutBindings[1].binding = 1;
    descriptorSetLayoutBindings[1].descriptorType =
        vk::DescriptorType::eStorageBuffer;
    descriptorSetLayoutBindings[1].descriptorCount = 1;
    descriptorSetLayoutBindings[1].stageFlags =
        vk::ShaderStageFlagBits::eCompute;

    descriptorSetLayoutBindings[2].binding = 2;
    descriptorSetLayoutBindings[2].descriptorType =
        vk::DescriptorType::eStorageBuffer;
    descriptorSetLayoutBindings[2].descriptorCount = 1;
    descriptorSetLayoutBindings[2].stageFlags =
        vk::ShaderStageFlagBits::eCompute;

    // Descriptor set layout
    vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{};
    descriptorSetLayoutCreateInfo.bindingCount =
        descriptorSetLayoutBindings.size();
    descriptorSetLayoutCreateInfo.pBindings =
        descriptorSetLayoutBindings.data();

    resources.descriptorSetLayout = cm.device->createDescriptorSetLayoutUnique(
        descriptorSetLayoutCreateInfo);
  }

  createDescriptorPoolAndSet(cm, resources, buffers);

  createComputePipeline(cm, resources);

  // Copy data to staging buffers
  std::vector<float> input1(WIDTH * HEIGHT, 1);
  std::vector<float> input2(WIDTH * HEIGHT, 2);

  cm.copyToStagingBuffer<float>(input1, buffers.stagingBuffer1);
  cm.copyToStagingBuffer<float>(input2, buffers.stagingBuffer2);

  auto &commandBuffer = cm.commandBuffer;
  recordCommandBuffer(cm, cm.commandBuffer, resources, buffers, bufferSize,
                      WIDTH, HEIGHT);

  // Submit the command buffer to the compute queue
  {
    uspam::TimeIt<true> timeit("Wait queue");

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    cm.queue.submit(submitInfo);

    cm.queue.waitIdle();
  }

  std::vector<float> outputData(WIDTH * HEIGHT);
  cm.copyFromStagingBuffer<float>(buffers.stagingBuffer3, outputData);

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
    uspam::TimeIt<true> timeit("CPU version");
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
