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

struct VulkanImage {
  vk::UniqueImage image;
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
  [[nodiscard]] VulkanBuffer
  createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
               vk::MemoryPropertyFlags properties) const;

  [[nodiscard]] VulkanImage createImage2D(uint32_t width, uint32_t height,
                                          vk::Format format,
                                          vk::ImageUsageFlags usage) const;

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

  void copyBufferToImage(vk::Buffer buffer, vk::Image image,
                         uint32_t imageWidth, uint32_t imageHeight,
                         vk::CommandBuffer commandBuffer = nullptr) const;

  void copyImageToBuffer(vk::Image image, vk::Buffer buffer, uint32_t width,
                         uint32_t height,
                         vk::CommandBuffer commandBuffer) const;

  // Transfer data from a user host buffer to a Vulkan staging buffer
  template <typename T>
  void copyToStagingBuffer(std::span<const T> data,
                           vk::DeviceMemory memory) const {
    vk::DeviceSize size = data.size() * sizeof(T);
    // Step 1: Map the memory associated with the staging buffer
    void *mappedMemory = device->mapMemory(memory, 0, size);
    // Step 2: Copy data from the vector to the mapped memory
    memcpy(mappedMemory, data.data(), static_cast<size_t>(size));
    // Step 3: Unmap the memory so the GPU can access it
    device->unmapMemory(memory);
  }
  template <typename T>
  void copyToStagingBuffer(std::span<const T> data,
                           VulkanBuffer &stagingBuffer) const {
    copyToStagingBuffer(data, stagingBuffer.memory.get());
  }

  template <typename T>
  void copyFromStagingBuffer(vk::DeviceMemory memory, std::span<T> data) const {
    vk::DeviceSize size = data.size() * sizeof(T);
    void *mappedMemory = device->mapMemory(memory, 0, size);
    // Copy the data from GPU memory to a local buffer
    memcpy(data.data(), mappedMemory, static_cast<size_t>(size));
    // Unmap the memory after retrieving the data
    device->unmapMemory(memory);
  }
  template <typename T>
  void copyFromStagingBuffer(const VulkanBuffer &stagingBuffer,
                             std::span<T> data) const {
    copyFromStagingBuffer<T>(stagingBuffer.memory.get(), data);
  }

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

  [[nodiscard]] vk::CommandBuffer beginTempOneTimeCommandBuffer() const {
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandPool = *commandPool;
    allocInfo.commandBufferCount = 1;

    auto commandBuffer = device->allocateCommandBuffers(allocInfo)[0];

    // Immediately start recording the command buffer
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

    commandBuffer.begin(beginInfo);
    return commandBuffer;
  }

  void endOneTimeCommandBuffer(vk::CommandBuffer commandBuffer) const {
    commandBuffer.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    queue.submit(submitInfo);
    queue.waitIdle();
  }

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