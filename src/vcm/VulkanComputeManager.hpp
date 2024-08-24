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

  static auto queryInstanceExtensionSupport()
      -> std::vector<VkExtensionProperties>;
  static void printInstanceExtensionSupport();

  // Create buffer helpers
  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties, VkBuffer &buffer,
                    VkDeviceMemory &bufferMemory) const;
  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties, VulkanBuffer &buf) const {
    createBuffer(size, usage, properties, buf.buffer, buf.memory);
  }
  [[nodiscard]] VulkanBuffer
  createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
               VkMemoryPropertyFlags properties) const {
    VulkanBuffer buf{};
    createBuffer(size, usage, properties, buf.buffer, buf.memory);
    return buf;
  }

  // Destroy buffer helpers
  void destroyBuffer(VkBuffer &buffer, VkDeviceMemory &memory) const {
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, memory, nullptr);
  }
  void destroyBuffer(VulkanBuffer &buffer) const {
    destroyBuffer(buffer.buffer, buffer.memory);
  }

  // If commandBuffer is provided, use it and only record
  // else allocate a temporary command buffer
  void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size,
                  VkCommandBuffer commandBuffer = VK_NULL_HANDLE) const;

  struct CopyBufferT {
    VkBuffer src;
    VkBuffer dst;
    VkDeviceSize size;
  };
  void copyBuffers(std::span<CopyBufferT> buffersToCopy) const;

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