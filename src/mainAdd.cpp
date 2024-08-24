#include "vcm/VulkanComputeManager.hpp"
#include "vulkan/vulkan_core.h"
#include <fmt/core.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>

struct PushConstantData {
  int width;
  int height;
};

// Descriptor set and pipeline required to setup the compute shader
// Trying to make this general to all compute shaders
struct ComputeShaderResources {
  VkDescriptorSetLayout descriptorSetLayout; // Compute shader binding layout
  VkDescriptorSet descriptorSet;             // Compute shader bindings
  VkPipelineLayout pipelineLayout;           // Layout of the compute pipeline
  VkPipeline pipeline; // Compute pipeline (can define more)
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

void createComputePipeline(vcm::VulkanComputeManager &cm,
                           ComputeShaderResources &resources) {
  // Load the SPIR-V binary
  VkShaderModule shaderModule = cm.loadShader("shaders/add2D.spv");

  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags =
      VK_SHADER_STAGE_COMPUTE_BIT; // Set this to VK_SHADER_STAGE_COMPUTE_BIT
  pushConstantRange.offset = 0;
  pushConstantRange.size = sizeof(PushConstantData);

  VkPipelineShaderStageCreateInfo shaderStageCreateInfo{};
  shaderStageCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  shaderStageCreateInfo.module = shaderModule;
  shaderStageCreateInfo.pName = "main";

  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
  pipelineLayoutCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCreateInfo.setLayoutCount = 1;
  pipelineLayoutCreateInfo.pSetLayouts = &resources.descriptorSetLayout;
  pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
  pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

  if (vkCreatePipelineLayout(cm.device, &pipelineLayoutCreateInfo, nullptr,
                             &resources.pipelineLayout) != VK_SUCCESS) {
    throw std::runtime_error("Failed to created compute pipeline layout!");
  }

  VkComputePipelineCreateInfo pipelineCreateInfo{};
  pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineCreateInfo.stage = shaderStageCreateInfo;
  pipelineCreateInfo.layout = resources.pipelineLayout;

  if (vkCreateComputePipelines(cm.device, VK_NULL_HANDLE, 1,
                               &pipelineCreateInfo, nullptr,
                               &resources.pipeline) != VK_SUCCESS) {
    throw std::runtime_error("Failed to created compute pipeline layout!");
  }

  vkDestroyShaderModule(cm.device, shaderModule, nullptr);
}

void createDescriptorPoolAndSet(vcm::VulkanComputeManager &cm,
                                ComputeShaderResources &resources,
                                ComputeShaderBuffers &buffers) {

  // Create descriptor set
  {
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = cm.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &resources.descriptorSetLayout;

    if (const auto ret = vkAllocateDescriptorSets(cm.device, &allocInfo,
                                                  &resources.descriptorSet);
        ret != VK_SUCCESS) {
      std::cout << "ret: " << ret << "\n";
      throw std::runtime_error("Failed to allocate descriptor set");
    }
  }

  // Bind device buffers to the descriptor set

  {
    VkDescriptorBufferInfo bufferInfo1{};
    bufferInfo1.buffer = buffers.buffer1.buffer;
    bufferInfo1.offset = 0;
    bufferInfo1.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo bufferInfo2{};
    bufferInfo2.buffer = buffers.buffer2.buffer;
    bufferInfo2.offset = 0;
    bufferInfo2.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo outputBufferInfo{};
    outputBufferInfo.buffer = buffers.buffer3.buffer;
    outputBufferInfo.offset = 0;
    outputBufferInfo.range = VK_WHOLE_SIZE;

    std::array<VkWriteDescriptorSet, 3> descriptorWrites{};

    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = resources.descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &bufferInfo1;

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = resources.descriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &bufferInfo2;

    descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[2].dstSet = resources.descriptorSet;
    descriptorWrites[2].dstBinding = 2;
    descriptorWrites[2].dstArrayElement = 0;
    descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pBufferInfo = &outputBufferInfo;

    vkUpdateDescriptorSets(cm.device,
                           static_cast<uint32_t>(descriptorWrites.size()),
                           descriptorWrites.data(), 0, nullptr);
  }
}

// Bind command buffer to pipeline
// Bind command buffer to descriptor set
// Dispatch command buffer
void dispatchComputeShader(vcm::VulkanComputeManager &cm,
                           ComputeShaderResources &resources, int inputWidth,
                           int inputHeight) {
  vkCmdBindPipeline(cm.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    resources.pipeline);

  vkCmdBindDescriptorSets(cm.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          resources.pipelineLayout, 0, 1,
                          &resources.descriptorSet, 0, nullptr);

  PushConstantData pushConstantData{inputWidth, inputHeight};
  vkCmdPushConstants(cm.commandBuffer, resources.pipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantData),
                     &pushConstantData);

  vkCmdDispatch(cm.commandBuffer, (inputWidth + 15) / 16,
                (inputHeight + 15) / 16, 1);
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

std::vector<float> retrieveOutput(VkDevice device, VkDeviceMemory outputMemory,
                                  VkDeviceSize size) {
  void *mappedMemory{};
  if (vkMapMemory(device, outputMemory, 0, size, 0, &mappedMemory) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to map output memory");
  }

  // Copy the data from GPU memory to a local vector
  std::vector<float> outputData(size / sizeof(float));
  memcpy(outputData.data(), mappedMemory, static_cast<size_t>(size));

  // Unmap the memory after retrieving the data
  vkUnmapMemory(device, outputMemory);

  return outputData;
}

int main() {
  const uint32_t WIDTH = 512;
  const uint32_t HEIGHT = 512;

  vcm::VulkanComputeManager cm;
  ComputeShaderResources resources;
  ComputeShaderBuffers buffers;

  /*
  Create buffers
  */
  VkDeviceSize bufferSize = WIDTH * HEIGHT * sizeof(float);

  buffers.stagingBuffer1 =
      cm.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  buffers.buffer1 = cm.createBuffer(bufferSize,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  {
    void *data;
    vkMapMemory(cm.device, buffers.stagingBuffer1.memory, 0, bufferSize, 0,
                &data);
    // memcpy(data, vertices.data(), (size_t)bufferSize);
    vkUnmapMemory(cm.device, buffers.stagingBuffer1.memory);
  }

  buffers.stagingBuffer2 =
      cm.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  buffers.stagingBuffer3 =
      cm.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  buffers.buffer2 = cm.createBuffer(bufferSize,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  buffers.buffer3 = cm.createBuffer(bufferSize,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // descriptorset layout bindings
  {
    std::array<VkDescriptorSetLayoutBinding, 3> descriptorSetLayoutBindings{};

    descriptorSetLayoutBindings[0].binding = 0;
    descriptorSetLayoutBindings[0].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBindings[0].descriptorCount = 1;
    descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptorSetLayoutBindings[1].binding = 1;
    descriptorSetLayoutBindings[1].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBindings[1].descriptorCount = 1;
    descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptorSetLayoutBindings[2].binding = 2;
    descriptorSetLayoutBindings[2].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBindings[2].descriptorCount = 1;
    descriptorSetLayoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Descriptor set layout
    {
      VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{};
      descriptorSetLayoutCreateInfo.sType =
          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      descriptorSetLayoutCreateInfo.bindingCount =
          descriptorSetLayoutBindings.size();
      descriptorSetLayoutCreateInfo.pBindings =
          descriptorSetLayoutBindings.data();
      if (vkCreateDescriptorSetLayout(
              cm.device, &descriptorSetLayoutCreateInfo, nullptr,
              &resources.descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error(
            "Failed to create compute descriptor set layout!");
      }
    }
  }

  createDescriptorPoolAndSet(cm, resources, buffers);

  createComputePipeline(cm, resources);

  // Record the command buffer
  VkCommandBufferBeginInfo commandBufferBeginInfo{};
  commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  if (vkBeginCommandBuffer(cm.commandBuffer, &commandBufferBeginInfo) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to begin command buffer!");
  }
  dispatchComputeShader(cm, resources, WIDTH, HEIGHT);
  vkEndCommandBuffer(cm.commandBuffer);

  // Submit the command buffer to the compute queue
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cm.commandBuffer;
  vkQueueSubmit(cm.queue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(cm.queue);

  std::vector<float> outputData =
      retrieveOutput(cm.device, buffers.stagingBuffer3.memory, bufferSize);

  // Verify that each element in the output matrix is 3.0f
  bool isCorrect = verifyOutput(outputData, 3.0f);
  if (isCorrect) {
    std::cout << "The output is correct!\n";
  } else {
    std::cout << "The output is incorrect.\n";
  }

  // Cleanup (omitting details for brevity)
  // ...
  cm.destroyBuffer(buffers.buffer1);
  cm.destroyBuffer(buffers.buffer2);
  cm.destroyBuffer(buffers.buffer3);
  cm.destroyBuffer(buffers.stagingBuffer1);
  cm.destroyBuffer(buffers.stagingBuffer2);
  cm.destroyBuffer(buffers.stagingBuffer3);

  vkDestroyPipelineLayout(cm.device, resources.pipelineLayout, nullptr);
  vkDestroyPipeline(cm.device, resources.pipeline, nullptr);
  vkDestroyDescriptorSetLayout(cm.device, resources.descriptorSetLayout,
                               nullptr);
  vkFreeDescriptorSets(cm.device, cm.descriptorPool, 1,
                       &resources.descriptorSet);

  return 0;
}
