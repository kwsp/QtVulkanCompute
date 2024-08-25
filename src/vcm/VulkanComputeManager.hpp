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
                         vk::CommandBuffer commandBuffer = nullptr) const {

    bool useTempCommandBuffer = !commandBuffer;
    if (useTempCommandBuffer) {
      commandBuffer = beginTempOneTimeCommandBuffer();
    }

    // Transition the image to the correct layout before the copy operation
    vk::ImageMemoryBarrier barrier{};
    barrier.oldLayout = vk::ImageLayout::eUndefined;
    barrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = vk::AccessFlagBits::eNone;
    barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                  vk::PipelineStageFlagBits::eTransfer,
                                  vk::DependencyFlags(), nullptr, nullptr,
                                  barrier);

    // Copy data from the staging buffer to the image
    vk::BufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageOffset = vk::Offset3D{0, 0, 0};
    copyRegion.imageExtent = vk::Extent3D{imageWidth, imageHeight, 1};

    commandBuffer.copyBufferToImage(
        buffer, image, vk::ImageLayout::eTransferDstOptimal, copyRegion);

    // Transition the image to the desired layout for shader access
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                  vk::PipelineStageFlagBits::eFragmentShader,
                                  vk::DependencyFlags(), nullptr, nullptr,
                                  barrier);

    if (useTempCommandBuffer) {
      endOneTimeCommandBuffer(commandBuffer);
    }
  }

  void copyImageToBuffer(vk::Image image, vk::Buffer buffer, uint32_t width,
                         uint32_t height, vk::CommandBuffer commandBuffer) {
    bool useTempCommandBuffer = !commandBuffer;
    if (useTempCommandBuffer) {
      commandBuffer = beginTempOneTimeCommandBuffer();
    }

    // Transition the image layout to eTransferSrcOptimal
    vk::ImageMemoryBarrier barrier{};
    barrier.oldLayout =
        vk::ImageLayout::eShaderReadOnlyOptimal; // Assuming the image is in a
                                                 // shader read-only layout
    barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = vk::AccessFlagBits::eShaderRead;
    barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,
                                  vk::PipelineStageFlagBits::eTransfer,
                                  vk::DependencyFlags(), nullptr, nullptr,
                                  barrier);

    // 3. Copy the image to the buffer
    vk::BufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;   // Tightly packed
    copyRegion.bufferImageHeight = 0; // Tightly packed
    copyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageOffset = vk::Offset3D{0, 0, 0};
    copyRegion.imageExtent = vk::Extent3D{width, height, 1};

    commandBuffer.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal,
                                    buffer, copyRegion);

    // Transition the image back to its original layout if needed
    barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
    barrier.newLayout =
        vk::ImageLayout::eShaderReadOnlyOptimal; // Or whatever the original
                                                 // layout was
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                  vk::PipelineStageFlagBits::eFragmentShader,
                                  vk::DependencyFlags(), nullptr, nullptr,
                                  barrier);

    if (useTempCommandBuffer) {
      endOneTimeCommandBuffer(commandBuffer);
    }
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