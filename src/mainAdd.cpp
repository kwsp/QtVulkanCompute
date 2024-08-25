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

  int inWidth;
  int inHeight;

  vk::DeviceSize bufferSize() const {
    return inWidth * inHeight * sizeof(float);
  }
};

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
template <int NInputBuf, typename PushConstantT = PushConstantData>
class ShaderExecutor {
public:
  ShaderExecutor(fs::path shaderFile, vcm::VulkanComputeManager &cm,
                 const ComputeShaderBuffers<NInputBuf> &buffers)
      : shaderFilename(std::move(shaderFile)) {

    createDescriptorSet(cm);
    createDescriptorPoolAndSet(cm, buffers);
    createComputePipeline(cm);
  }

  // private:
  ComputeShaderResources resources{};
  fs::path shaderFilename;

  void recordCommandBuffer(vcm::VulkanComputeManager &cm,
                           vk::CommandBuffer commandBuffer,
                           const ComputeShaderBuffers<NInputBuf> &buffers,
                           const PushConstantT &pushConstant) {

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
      for (int i = 0; i < NInputBuf; ++i) {
        cm.copyBuffer(buffers.in[i].staging.buffer, buffers.in[i].buffer.buffer,
                      buffers.bufferSize(), commandBuffer);
      }

      vk::MemoryBarrier memoryBarrier{};
      memoryBarrier.srcAccessMask =
          vk::AccessFlagBits::eTransferWrite; // After copying
      memoryBarrier.dstAccessMask =
          vk::AccessFlagBits::eShaderRead; // Before compute shader reads

      commandBuffer.pipelineBarrier(
          vk::PipelineStageFlagBits::eTransfer, // src: after the transfer op
          vk::PipelineStageFlagBits::eComputeShader, // dst: before the compute
                                                     // shader
          {}, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
    }

    dispatchComputeShader(cm, buffers.inWidth, buffers.inHeight, pushConstant);

    {
      // (Optional) Step 4: Insert another pipeline barrier if needed
      vk::MemoryBarrier memoryBarrier{};
      memoryBarrier.srcAccessMask =
          vk::AccessFlagBits::eShaderWrite; // After compute shader writes
      memoryBarrier.dstAccessMask =
          vk::AccessFlagBits::eTransferRead; // Before transfer reads

      commandBuffer.pipelineBarrier(
          vk::PipelineStageFlagBits::eComputeShader, // src: after compute
          vk::PipelineStageFlagBits::eTransfer, // dst: before next transfer
          {}, 1, &memoryBarrier, 0, nullptr, 0, nullptr);

      // Copy result back to staging
      cm.copyBuffer(buffers.out.buffer.buffer, buffers.out.staging.buffer,
                    buffers.bufferSize(), commandBuffer);
    }

    commandBuffer.end();
  }

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

    constexpr int TotalBuffers = NInputBuf + 1;

    std::array<vk::DescriptorSetLayoutBinding, TotalBuffers>
        descriptorSetLayoutBindings{};

    for (int i = 0; i < TotalBuffers; ++i) {
      // NOLINTNEXTLINE(*-constant-array-index)
      descriptorSetLayoutBindings[i] = makeComputeDescriptorSetLayoutBinding(i);
    }

    // Descriptor set layout
    vk::DescriptorSetLayoutCreateInfo createInfo{};
    createInfo.bindingCount = descriptorSetLayoutBindings.size();
    createInfo.pBindings = descriptorSetLayoutBindings.data();

    resources.descriptorSetLayout =
        cm.device->createDescriptorSetLayoutUnique(createInfo);
  }

  void createDescriptorPoolAndSet(vcm::VulkanComputeManager &cm,
                                  const ComputeShaderBuffers<2> &buffers) {

    // Create descriptor set
    {
      vk::DescriptorSetAllocateInfo allocInfo{};
      allocInfo.descriptorPool = cm.descriptorPool;
      allocInfo.descriptorSetCount = 1;
      allocInfo.pSetLayouts = &resources.descriptorSetLayout.get();

      auto descriptorSets = cm.device->allocateDescriptorSetsUnique(allocInfo);
      resources.descriptorSet = std::move(descriptorSets.front());
    }

    // Bind device buffers to the descriptor set
    {
      constexpr int TotalBuffers = NInputBuf + 1;
      std::array<vk::DescriptorBufferInfo, TotalBuffers> bufferInfo{};
      std::array<vk::WriteDescriptorSet, 3> descriptorWrites{};

      // NOLINTBEGIN(*-constant-array-index)
      for (int i = 0; i < NInputBuf; ++i) {
        bufferInfo[i].buffer = buffers.in[i].buffer.buffer;
        bufferInfo[i].offset = 0;
        bufferInfo[i].range = VK_WHOLE_SIZE;

        descriptorWrites[i].dstSet = resources.descriptorSet.get();
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = vk::DescriptorType::eStorageBuffer;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfo[i];
      }

      int i = NInputBuf;
      bufferInfo[i].buffer = buffers.out.buffer.buffer;
      bufferInfo[i].offset = 0;
      bufferInfo[i].range = VK_WHOLE_SIZE;

      descriptorWrites[i].dstSet = resources.descriptorSet.get();
      descriptorWrites[i].dstBinding = i;
      descriptorWrites[i].dstArrayElement = 0;
      descriptorWrites[i].descriptorType = vk::DescriptorType::eStorageBuffer;
      descriptorWrites[i].descriptorCount = 1;
      descriptorWrites[i].pBufferInfo = &bufferInfo[i];

      // NOLINTEND(*-constant-array-index)

      cm.device->updateDescriptorSets(descriptorWrites, {});
    }
  }

  void createComputePipeline(vcm::VulkanComputeManager &cm) {
    // Load the SPIR-V binary
    vk::UniqueShaderModule shaderModule = cm.loadShader(shaderFilename);

    vk::PushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eCompute;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PushConstantT);

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

  // Bind command buffer to pipeline
  // Bind command buffer to descriptor set
  // Dispatch command buffer
  void dispatchComputeShader(vcm::VulkanComputeManager &cm, int inputWidth,
                             int inputHeight,
                             const PushConstantT &pushConstant) {
    cm.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                                  resources.pipeline.get());

    cm.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                        resources.pipelineLayout.get(), 0,
                                        resources.descriptorSet.get(), {});

    cm.commandBuffer.pushConstants(resources.pipelineLayout.get(),
                                   vk::ShaderStageFlagBits::eCompute, 0,
                                   sizeof(PushConstantData), &pushConstant);

    cm.commandBuffer.dispatch((inputWidth + 15) / 16, (inputHeight + 15) / 16,
                              1);
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

  vcm::VulkanBuffer inputBuf1Staging = cm.createStagingBufferSrc(bufferSize);
  vcm::VulkanBuffer inputBuf1 = cm.createDeviceBufferDst(bufferSize);

  vcm::VulkanBuffer inputBuf2Staging = cm.createStagingBufferSrc(bufferSize);
  vcm::VulkanBuffer inputBuf2 = cm.createDeviceBufferDst(bufferSize);

  ComputeShaderBuffers<2> buffers{};
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
  ShaderExecutor<2> shader("shaders/add2D.spv", cm, buffers);

  auto &commandBuffer = cm.commandBuffer;

  PushConstantData pushConstant{WIDTH, HEIGHT};
  shader.recordCommandBuffer(cm, commandBuffer, buffers, pushConstant);

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
