#pragma once

#include <filesystem>
#include <fmt/format.h>
#include <vulkan/vulkan.hpp>

namespace vcm {

namespace fs = std::filesystem;

struct VulkanBufferRef {
  vk::Buffer buffer;
  vk::DeviceMemory memory;
};

struct VulkanBuffer {
  vk::UniqueBuffer buffer;
  vk::UniqueDeviceMemory memory;

  [[nodiscard]] VulkanBufferRef ref() const {
    return {buffer.get(), memory.get()};
  }
};

struct VulkanImage {
  vk::UniqueImage image;
  vk::UniqueDeviceMemory memory;
};

} // namespace vcm