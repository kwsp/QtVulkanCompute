#pragma once

#include "vulkan/vulkan_core.h"
#include <fmt/format.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>

namespace vcm {

struct VulkanBuffer {
  VkBuffer buffer;
  VkDeviceMemory memory;
};

class VulkanComputeManager {
public:
  VulkanComputeManager();

  VulkanComputeManager(const VulkanComputeManager &) = default;
  VulkanComputeManager(VulkanComputeManager &&) = delete;
  VulkanComputeManager &operator=(const VulkanComputeManager &) = delete;
  VulkanComputeManager &operator=(VulkanComputeManager &&) = delete;

  ~VulkanComputeManager() { cleanup(); }

  static auto queryInstanceExtensionSupport() {
    // check for extension support

    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
                                           extensions.data());
    return extensions;
  }

  static void printInstanceExtensionSupport() {
    const auto extensions = queryInstanceExtensionSupport();

    fmt::print("Available Vulkan extensions:\n");
    for (const auto &extension : extensions) {
      fmt::print("\t{}\n", extension.extensionName);
    }
  }

  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties, VkBuffer &buffer,
                    VkDeviceMemory &bufferMemory);
  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties, VulkanBuffer &buf) {
    createBuffer(size, usage, properties, buf.buffer, buf.memory);
  }
  VulkanBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                            VkMemoryPropertyFlags properties) {
    VulkanBuffer buf{};
    createBuffer(size, usage, properties, buf.buffer, buf.memory);
    return buf;
  }

  void destroyBuffer(VkBuffer &buffer, VkDeviceMemory &memory) const {
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, memory, nullptr);
  }
  void destroyBuffer(VulkanBuffer &buffer) const {
    destroyBuffer(buffer.buffer, buffer.memory);
  }

  void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer,
                  VkDeviceSize size) const {
    // Memory transfer ops are executed using command buffers.
    // We must first allocate a temporary command buffer.
    // We can create a short-lived command pool for this because the
    // implementation may be able to apply memory allocation optimizations.
    // (should set VK_COMMAND_POOL_CREATE_TRANSIENT_BIT) flag during command
    // pool creation

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    // Immediately start recording the command buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0; // Optional
    copyRegion.dstOffset = 0; // Optional
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
  }

  // private:
  // QVulkanInstance vulkanInstance;
  VkInstance instance{};

  // Physical device
  VkPhysicalDevice physicalDevice{};
  std::string physicalDeviceName;

  // Logical device
  VkDevice device{};
  // Compute queue
  VkQueue queue{};

  // Command pool
  // Manage the memory that is used to store the buffers and command buffers are
  // allocated from them
  VkCommandPool commandPool;

  // Command buffer
  // Command buffers will be automatically freed when the command pool is
  // destroyed, so we don't need to explicitly cleanup.
  VkCommandBuffer commandBuffer;

  // Descriptor pool
  // Discriptor sets for buffers are allocated from this
  VkDescriptorPool descriptorPool;

  const std::vector<const char *> validationLayers = {
      "VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
  const bool enableValidationLayers = false;
#else
  const bool enableValidationLayers = true;
#endif

  /* Create Instance */
  void createInstance();
  bool checkValidationLayerSupport();

  /* Find physical device */
  // Select a graphics card in the system that supports the features we need.
  // Stick to the first suitable card we find.
  void pickPhysicalDevice();
  static int rateDeviceSuitability(VkPhysicalDevice device);
  struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> computeFamily;
  };
  static QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

  /* Create logical device */
  void createLogicalDevice();

  /* Create command pool */
  void createCommandPool();

  /* Create command buffer */
  void createCommandBuffer();
  void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);

  /* Create descriptor pool */
  void createDescriptorPool();

  /* Create buffers */
  [[nodiscard]] uint32_t findMemoryType(uint32_t typeFilter,
                                        VkMemoryPropertyFlags properties) const;

  /* Load shaders */
  VkShaderModule loadShader(const char *filename) const;
  [[nodiscard]] VkShaderModule
  createShaderModule(const std::vector<char> &computeShaderCode) const;
  void destroyShaderModule(VkShaderModule module) const;

  /* Cleanup */
  void cleanup();
};

} // namespace vcm