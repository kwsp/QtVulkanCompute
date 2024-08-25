#pragma once

#include "Common.hpp"
#include "TimeIt.hpp"
#include "VulkanComputeManager.hpp"

namespace vcm {

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

  [[nodiscard]] vk::DeviceSize bufferSize() const {
    return inWidth * inHeight * sizeof(float);
  }
};

struct WorkGroupSize {
  int x{16}, y{16}, z{0};
};

template <int NInputBuf, typename PushConstantT> class ShaderExecutor {
public:
  ShaderExecutor(fs::path shaderFile, vcm::VulkanComputeManager &cm,
                 const ComputeShaderBuffers<NInputBuf> &buffers)
      : shaderFilename(std::move(shaderFile)) {

    createDescriptorSetLayout(cm);
    createDescriptorSet(cm, buffers);
    createComputePipeline(cm);
  }

  void recordCommandBuffer(vcm::VulkanComputeManager &cm,
                           vk::CommandBuffer commandBuffer,
                           const ComputeShaderBuffers<NInputBuf> &buffers,
                           const PushConstantT &pushConstant) {

    TimeIt<true> timeit("Recording command buffer");

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
        // NOLINTNEXTLINE(*-constant-array-index)
        cm.copyBuffer(buffers.in[i].staging.buffer, buffers.in[i].buffer.buffer,
                      buffers.bufferSize(), commandBuffer);
      }

      vcm::memoryBarrierTransferThenCompute(commandBuffer);
    }

    dispatchComputeShader(cm, commandBuffer, buffers.inWidth, buffers.inHeight,
                          pushConstant);

    {
      // (Optional) Step 4: Insert another pipeline barrier if needed
      vcm::memoryBarrierComputeThenTransfer(commandBuffer);

      // Copy result back to staging
      cm.copyBuffer(buffers.out.buffer.buffer, buffers.out.staging.buffer,
                    buffers.bufferSize(), commandBuffer);
    }

    commandBuffer.end();
  }

private:
  ComputeShaderResources resources{};
  WorkGroupSize wgSize;
  fs::path shaderFilename;

  void createDescriptorSetLayout(vcm::VulkanComputeManager &cm) {
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

  void createDescriptorSet(vcm::VulkanComputeManager &cm,
                           const ComputeShaderBuffers<NInputBuf> &buffers) {

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
      std::array<vk::WriteDescriptorSet, TotalBuffers> descriptorWrites{};

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
  void dispatchComputeShader(vcm::VulkanComputeManager &cm,
                             vk::CommandBuffer &commandBuffer, int inputWidth,
                             int inputHeight,
                             const PushConstantT &pushConstant) {
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                               resources.pipeline.get());

    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                     resources.pipelineLayout.get(), 0,
                                     resources.descriptorSet.get(), {});

    commandBuffer.pushConstants(resources.pipelineLayout.get(),
                                vk::ShaderStageFlagBits::eCompute, 0,
                                sizeof(PushConstantT), &pushConstant);

    commandBuffer.dispatch((inputWidth + wgSize.x - 1) / wgSize.x,
                           (inputHeight + wgSize.y - 1) / wgSize.y, 1);
  }
};

} // namespace vcm