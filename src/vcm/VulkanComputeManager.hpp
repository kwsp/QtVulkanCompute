#pragma once

#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace vcm {

struct VulkanBuffer {
  vk::UniqueBuffer buffer;
  vk::UniqueDeviceMemory memory;
};

class VulkanComputeManager {
public:
  VulkanComputeManager();

  VulkanComputeManager(const VulkanComputeManager &) = delete;
  VulkanComputeManager(VulkanComputeManager &&) = delete;
  VulkanComputeManager &operator=(const VulkanComputeManager &) = delete;
  VulkanComputeManager &operator=(VulkanComputeManager &&) = delete;

  ~VulkanComputeManager() = default;

  static void printInstanceExtensionSupport();

  // Create buffer helpers
  void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                    vk::MemoryPropertyFlags properties,
                    vk::UniqueBuffer &buffer,
                    vk::UniqueDeviceMemory &bufferMemory) const;
  void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                    vk::MemoryPropertyFlags properties,
                    VulkanBuffer &buf) const {
    createBuffer(size, usage, properties, buf.buffer, buf.memory);
  }
  [[nodiscard]] VulkanBuffer
  createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
               vk::MemoryPropertyFlags properties) const {
    VulkanBuffer buf{};
    createBuffer(size, usage, properties, buf.buffer, buf.memory);
    return buf;
  }

  // If commandBuffer is provided, use it and only record
  // else allocate a temporary command buffer
  void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer,
                  vk::DeviceSize size,
                  vk::CommandBuffer commandBuffer = nullptr) const;

  struct CopyBufferT {
    vk::Buffer src;
    vk::Buffer dst;
    vk::DeviceSize size;
  };
  void copyBuffers(std::span<CopyBufferT> buffersToCopy) const;

  // private:
  // QVulkanInstance vulkanInstance;
  vk::UniqueInstance instance;

  // Physical device
  vk::PhysicalDevice physicalDevice;
  std::string physicalDeviceName;

  // Logical device
  vk::UniqueDevice device;
  // Compute queue
  vk::Queue queue;

  // Command pool
  // Manage the memory that is used to store the buffers and command buffers are
  // allocated from them
  // Command pool should be thread local
  vk::UniqueCommandPool commandPool;

  // Command buffer
  vk::CommandBuffer commandBuffer;

  // Descriptor pool
  // Discriptor sets for buffers are allocated from this
  vk::DescriptorPool descriptorPool;

  static constexpr std::array<const char *, 1> validationLayers = {
      {"VK_LAYER_KHRONOS_validation"}};

#ifdef NDEBUG
  static constexpr bool enableValidationLayers = false;
#else
  static constexpr bool enableValidationLayers = true;
#endif

  /* Create Instance */
  void createInstance();
  static bool checkValidationLayerSupport();

  /* Find physical device */
  // Select a graphics card in the system that supports the features we need.
  // Stick to the first suitable card we find.
  void pickPhysicalDevice();
  static int rateDeviceSuitability(vk::PhysicalDevice device);
  struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> computeFamily;
  };
  static QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device);

  /* Create logical device */
  void createLogicalDevice();

  /* Create command pool */
  void createCommandPool();

  /* Create command buffer */
  void createCommandBuffer();
  void recordCommandBuffer(vk::CommandBuffer commandBuffer,
                           uint32_t imageIndex);

  /* Create descriptor pool */
  void createDescriptorPool();

  /* Create buffers */
  [[nodiscard]] uint32_t
  findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const;

  /* Load shaders */
  vk::UniqueShaderModule loadShader(const char *filename) const;
  [[nodiscard]] vk::UniqueShaderModule
  createShaderModule(const std::vector<char> &computeShaderCode) const;

  /* Cleanup */
  void cleanup();
};

} // namespace vcm