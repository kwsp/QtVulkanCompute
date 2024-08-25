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
template <int NInputBuffers> struct ComputeShaderBuffers {
  struct BufferPair {
    vcm::VulkanBufferRef buffer;
    vcm::VulkanBufferRef staging;
  };

  std::array<BufferPair, NInputBuffers> in;
  BufferPair out;
};

// Question: uniform buffer vs push constant
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
                                ComputeShaderBuffers<2> &buffers) {

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
    bufferInfo1.buffer = buffers.in[0].buffer.buffer;
    bufferInfo1.offset = 0;
    bufferInfo1.range = VK_WHOLE_SIZE;

    vk::DescriptorBufferInfo bufferInfo2{};
    bufferInfo2.buffer = buffers.in[1].buffer.buffer;
    bufferInfo2.offset = 0;
    bufferInfo2.range = VK_WHOLE_SIZE;

    vk::DescriptorBufferInfo outputBufferInfo{};
    outputBufferInfo.buffer = buffers.out.buffer.buffer;
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
                         ComputeShaderBuffers<2> &buffers,
                         vk::DeviceSize bufferSize, int width, int height) {
  uspam::TimeIt<true> timeit("Recording command buffer");

  // Record the command buffer
  {
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    commandBuffer.begin(beginInfo);
  }

  // // Copy data from staging to host buffers
  {
    // Async copy with barrier
    cm.copyBuffer(buffers.in[0].staging.buffer, buffers.in[0].buffer.buffer,
                  bufferSize, commandBuffer);
    cm.copyBuffer(buffers.in[1].staging.buffer, buffers.in[1].buffer.buffer,
                  bufferSize, commandBuffer);

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
    cm.copyBuffer(buffers.out.buffer.buffer, buffers.out.staging.buffer,
                  bufferSize, commandBuffer);
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

template <int NInputBuf> class ShaderExecutor {
public:
  ShaderExecutor(vcm::VulkanComputeManager &cm,
                 ComputeShaderBuffers<NInputBuf> buffers) {

    createDescriptorSet(cm);
    createDescriptorPoolAndSet(cm, resources, buffers);
  }

  // private:
  ComputeShaderResources resources{};

private:
  void createDescriptorSet(vcm::VulkanComputeManager &cm) {
    // descriptorset layout bindings
    const auto makeComputeDescriptorSetLayoutBinding = [](int binding) {
      vk::DescriptorSetLayoutBinding descriptorSetLayoutBinding;
      descriptorSetLayoutBinding.binding = binding;
      descriptorSetLayoutBinding.descriptorType =
          vk::DescriptorType::eStorageBuffer;
      descriptorSetLayoutBinding.descriptorCount = 1;
      descriptorSetLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eCompute;
      return descriptorSetLayoutBinding;
    };

    std::array<vk::DescriptorSetLayoutBinding, 3> descriptorSetLayoutBindings{{
        makeComputeDescriptorSetLayoutBinding(0),
        makeComputeDescriptorSetLayoutBinding(1),
        makeComputeDescriptorSetLayoutBinding(2),
    }};

    // Descriptor set layout
    vk::DescriptorSetLayoutCreateInfo createInfo{};
    createInfo.bindingCount = descriptorSetLayoutBindings.size();
    createInfo.pBindings = descriptorSetLayoutBindings.data();

    resources.descriptorSetLayout =
        cm.device->createDescriptorSetLayoutUnique(createInfo);
  }
};

int main() {
  const uint32_t WIDTH = 512;
  const uint32_t HEIGHT = 512;

  vcm::VulkanComputeManager cm;

  /*
  Create buffers
  */
  vk::DeviceSize bufferSize = WIDTH * HEIGHT * sizeof(float);
  ComputeShaderBuffers<2> buffers{};

  vcm::VulkanBuffer inputBuf1Staging = cm.createStagingBufferSrc(bufferSize);
  vcm::VulkanBuffer inputBuf1 = cm.createDeviceBufferDst(bufferSize);

  vcm::VulkanBuffer inputBuf2Staging = cm.createStagingBufferSrc(bufferSize);
  vcm::VulkanBuffer inputBuf2 = cm.createDeviceBufferDst(bufferSize);

  auto outputBufStaging = cm.createStagingBufferDst(bufferSize);
  auto outputBuf = cm.createDeviceBufferSrc(bufferSize);

  buffers.in = {{{inputBuf1.ref(), inputBuf1Staging.ref()},
                 {inputBuf2.ref(), inputBuf2Staging.ref()}}};
  buffers.out = {outputBuf.ref(), outputBufStaging.ref()};

  ShaderExecutor<2> shader(cm, buffers);

  createComputePipeline(cm, shader.resources);

  // Copy data to staging buffers
  std::vector<float> input1(WIDTH * HEIGHT, 1);
  std::vector<float> input2(WIDTH * HEIGHT, 2);

  cm.copyToStagingBuffer<float>(input1, buffers.in[0].staging);
  cm.copyToStagingBuffer<float>(input2, buffers.in[1].staging);

  auto &commandBuffer = cm.commandBuffer;
  recordCommandBuffer(cm, cm.commandBuffer, shader.resources, buffers,
                      bufferSize, WIDTH, HEIGHT);

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
