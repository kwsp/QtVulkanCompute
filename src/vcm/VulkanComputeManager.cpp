#include "VulkanComputeManager.hpp"
#include "vulkan/vulkan_enums.hpp"
#include "vulkan/vulkan_structs.hpp"
#include <armadillo>
#include <array>
#include <fmt/format.h>
#include <fstream>
#include <ios>
#include <map>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>

// NOLINTBEGIN(*-reinterpret-cast)

namespace vcm {

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

VulkanComputeManager::VulkanComputeManager() {
  // Init Vulkan instance
  createInstance();

  pickPhysicalDevice();

  createLogicalDevice();

  createCommandPool();

  createCommandBuffer();

  createDescriptorPool();

  // loadComputeShader("shaders/warpPolarCompute.spv");
}

void VulkanComputeManager::createCommandPool() {
  QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

  vk::CommandPoolCreateInfo poolInfo{};
  poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
  poolInfo.queueFamilyIndex = queueFamilyIndices.computeFamily.value();

  // There are 2 possible flags for command pools
  // - VK_COMMAND_POOL_CREATE_TRANSIENT_BIT: hint that command buffers are
  // rerecorded with new commands often.
  // - VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: allow command buffers to
  // be rerecorded individually, without this flag they all have to be reset
  // together.

  // We will record a command buffer every frame, so we want to be able to reset
  // and rerecord over it. Thus we use
  // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT

  // Command buffers are executed by submitting them on one of the device queues
  // Each command pool can only allocate command buffers that are submitted on a
  // single type of queue.
  commandPool = device->createCommandPoolUnique(poolInfo);
}

void VulkanComputeManager::createCommandBuffer() {
  vk::CommandBufferAllocateInfo allocInfo{};
  allocInfo.commandPool = *commandPool;
  // VK_COMMAND_BUFFER_LEVEL_PRIMARY: can be submitted to a queue for execution,
  // but cannot be called from other command buffers.
  // VK_COMMAND_BUFFER_LEVEL_SECONDARY: cannot be submitted directly, but can be
  // called from primary command buffers.
  allocInfo.level = vk::CommandBufferLevel::ePrimary;
  allocInfo.commandBufferCount = 1;

  commandBuffer = device->allocateCommandBuffers(allocInfo)[0];
}

void VulkanComputeManager::recordCommandBuffer(vk::CommandBuffer commandBuffer,
                                               uint32_t imageIndex) {
  vk::CommandBufferBeginInfo beginInfo{};
  // VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: rerecorded right after
  // executing it once. VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT:
  // secondary command buffer that will be entirely within a single render pass
  // VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT: can be resubmitted while it
  // is also already pending execution.
  // Non are applicable right now.
  beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
  commandBuffer.begin(beginInfo);
}

void VulkanComputeManager::createDescriptorPool() {
  // Create descriptor pool
  // describe which descriptor types our descriptor sets are going to contain
  // and how many
  std::array<vk::DescriptorPoolSize, 2> poolSize{{
      {vk::DescriptorType::eStorageBuffer, 20},
      {vk::DescriptorType::eUniformBuffer, 10},
  }};

  vk::DescriptorPoolCreateInfo poolInfo{};
  poolInfo.maxSets = 20;
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSize.size());
  poolInfo.pPoolSizes = poolSize.data();
  poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;

  descriptorPool = device->createDescriptorPool(poolInfo);
}

void VulkanComputeManager::createBuffer(
    vk::DeviceSize size, vk::BufferUsageFlags usage,
    vk::MemoryPropertyFlags properties, vk::UniqueBuffer &buffer,
    vk::UniqueDeviceMemory &bufferMemory) const {
  vk::BufferCreateInfo bufferInfo{};
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = vk::SharingMode::eExclusive;

  buffer = device->createBufferUnique(bufferInfo);

  // To allocate memory for a buffer we need to first query its memory
  // requirements using vkGetBufferMemoryRequirements
  vk::MemoryRequirements memRequirements =
      device->getBufferMemoryRequirements(*buffer);

  vk::MemoryAllocateInfo allocInfo{};
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex =
      findMemoryType(memRequirements.memoryTypeBits, properties);

  bufferMemory = device->allocateMemoryUnique(allocInfo);

  device->bindBufferMemory(*buffer, *bufferMemory, 0);
}

void VulkanComputeManager::copyBuffer(vk::Buffer srcBuffer,
                                      vk::Buffer dstBuffer, vk::DeviceSize size,
                                      vk::CommandBuffer commandBuffer) const {
  // Memory transfer ops are executed using command buffers.
  // We must first allocate a temporary command buffer.
  // We can create a short-lived command pool for this because the
  // implementation may be able to apply memory allocation optimizations.
  // (should set VK_COMMAND_POOL_CREATE_TRANSIENT_BIT) flag during command
  // pool creation

  const bool allocTempCommandBuffer = !commandBuffer;

  if (allocTempCommandBuffer) {
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandPool = *commandPool;
    allocInfo.commandBufferCount = 1;

    commandBuffer = device->allocateCommandBuffers(allocInfo)[0];

    // Immediately start recording the command buffer
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

    commandBuffer.begin(beginInfo);
  }

  vk::BufferCopy copyRegion{};
  copyRegion.size = size;
  commandBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);

  if (allocTempCommandBuffer) {
    commandBuffer.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    queue.submit(submitInfo);
    queue.waitIdle();
  }
}

void VulkanComputeManager::copyBuffers(
    std::span<CopyBufferT> buffersToCopy) const {
  // Memory transfer ops are executed using command buffers.
  // We must first allocate a temporary command buffer.
  // We can create a short-lived command pool for this because the
  // implementation may be able to apply memory allocation optimizations.
  // (should set VK_COMMAND_POOL_CREATE_TRANSIENT_BIT) flag during command
  // pool creation

  vk::CommandBufferAllocateInfo allocInfo{};
  allocInfo.level = vk::CommandBufferLevel::ePrimary;
  allocInfo.commandPool = *commandPool;
  allocInfo.commandBufferCount = 1;

  vk::CommandBuffer commandBuffer =
      device->allocateCommandBuffers(allocInfo)[0];

  // Immediately start recording the command buffer
  vk::CommandBufferBeginInfo beginInfo{};
  beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

  commandBuffer.begin(beginInfo);

  for (const auto &buffers : buffersToCopy) {
    vk::BufferCopy copyRegion{};
    copyRegion.srcOffset = 0; // Optional
    copyRegion.dstOffset = 0; // Optional
    copyRegion.size = buffers.size;
    commandBuffer.copyBuffer(buffers.src, buffers.dst, copyRegion);
  }

  vkEndCommandBuffer(commandBuffer);

  vk::SubmitInfo submitInfo{};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  queue.submit(submitInfo);
  queue.waitIdle();
}

uint32_t
VulkanComputeManager::findMemoryType(uint32_t typeFilter,
                                     vk::MemoryPropertyFlags properties) const {

  // First query info about available types of memory using
  vk::PhysicalDeviceMemoryProperties memProperties =
      physicalDevice.getMemoryProperties();

  // VkPhysicalDeviceMemoryProperties has 2 arrays: memoryTypes and
  // memoryHeaps Memory heaps are distinct memory resources like dedicated
  // VRAM and swap space in RAM for when VRAM runs out.
  //
  // Right now we're only concerned with the type of memory and not the heap
  // it comes from right now
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
    // NOLINTNEXTLINE(*-implicit-bool-conversion)
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }
  throw std::runtime_error("Failed to find suitable memory type!");
}

void VulkanComputeManager::createInstance() {
  vk::ApplicationInfo appInfo{};
  appInfo.pApplicationName = "QtVulkanCompute";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_1;

  vk::InstanceCreateInfo createInfo{};
  createInfo.pApplicationInfo = &appInfo;

  // Enable validation layers in debug
  if (enableValidationLayers && !checkValidationLayerSupport()) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();
  } else {
    createInfo.enabledLayerCount = 0;
  }

  std::vector<const char *> instanceExtensions;

// Enable VK_KHR_portability_enumeration for molten-vk on M-series mac
#ifdef __APPLE__
  // Tested for M series mac
  fmt::println("On macOS with molten-vk, enable "
               "VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME");
  instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
  createInfo.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif

  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(instanceExtensions.size());
  createInfo.ppEnabledExtensionNames = instanceExtensions.data();

  instance = vk::createInstanceUnique(createInfo);
  fmt::println("Created Vulkan instance.");
}

bool VulkanComputeManager::checkValidationLayerSupport() {
  auto availableLayers = vk::enumerateInstanceLayerProperties();

  for (const char *layerName : validationLayers) {
    bool layerFound = false;
    for (const auto &layerProperties : availableLayers) {
      if (std::strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }
    if (!layerFound) {
      return false;
    }
  }

  return true;
}

int VulkanComputeManager::rateDeviceSuitability(vk::PhysicalDevice device) {
  auto deviceProperties = device.getProperties();
  auto deviceFeatures = device.getFeatures();

  int score = 0;

  // Discrete GPUs have a significant performance advantage
  if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
    score += 1000; // NOLINT
  }

  // Maximum possible size of textures affects graphics quality
  score += deviceProperties.limits.maxImageDimension2D; // NOLINT

  // Need compute bit
  const auto indices = findQueueFamilies(device);
  if (!indices.computeFamily.has_value()) {
    return 0;
  }

  return score;
}

void VulkanComputeManager::pickPhysicalDevice() {
  physicalDevice = VK_NULL_HANDLE;
  physicalDeviceName = {};

  std::vector<vk::PhysicalDevice> devices =
      instance->enumeratePhysicalDevices();

  if (devices.empty()) {
    throw std::runtime_error("Failed to find GPUs with Vulkan support");
  }

  // A sorted associative container to rank candidates by increasing score
  std::multimap<int, VkPhysicalDevice> candidates;
  for (const auto &device : devices) {
    const int score = rateDeviceSuitability(device);
    candidates.insert({score, device});
  }

  if (candidates.rbegin()->first > 0) {
    physicalDevice = candidates.rbegin()->second;

    auto deviceProperties = physicalDevice.getProperties();
    physicalDeviceName = std::string{deviceProperties.deviceName};

  } else {
    throw std::runtime_error("Failed to find a suitable GPU");
  }

  fmt::println("Picked physical device '{}'.", physicalDeviceName);
}

VulkanComputeManager::QueueFamilyIndices
VulkanComputeManager::findQueueFamilies(vk::PhysicalDevice device) {
  QueueFamilyIndices indices{};

  std::vector<vk::QueueFamilyProperties> queueFamilies =
      device.getQueueFamilyProperties();

  int i = 0;
  for (const auto &queueFamily : queueFamilies) {
    if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
      indices.graphicsFamily = i;
    }
    if (queueFamily.queueFlags & vk::QueueFlagBits::eCompute) {
      indices.computeFamily = i;
    }
    i++;
  }

  return indices;
}

void VulkanComputeManager::createLogicalDevice() {
  // Specify the queues to be created
  const auto indices = findQueueFamilies(physicalDevice);

  vk::DeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.queueFamilyIndex = indices.computeFamily.value();
  queueCreateInfo.queueCount = 1;

  float queuePriority = 1.0F;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  vk::PhysicalDeviceFeatures deviceFeatures{};

  std::vector<const char *> deviceExtensions;
#ifdef __APPLE__
  deviceExtensions.push_back("VK_KHR_portability_subset");
#endif

  vk::DeviceCreateInfo createInfo{};
  createInfo.pQueueCreateInfos = &queueCreateInfo;
  createInfo.queueCreateInfoCount = 1;
  createInfo.pEnabledFeatures = &deviceFeatures;
  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(deviceExtensions.size());
  createInfo.ppEnabledExtensionNames = deviceExtensions.data();

  device = physicalDevice.createDeviceUnique(createInfo);
  queue = device->getQueue(indices.computeFamily.value(), 0);

  fmt::println("Created Vulkan logical device and compute queue.");
}

auto readFile(const char *filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open shader file");
  }

  const size_t fileSize = file.tellg();
  std::vector<char> buffer(fileSize);
  file.seekg(0);
  file.read(buffer.data(), static_cast<std::streamsize>(fileSize));

  file.close();
  return buffer;
}

vk::UniqueShaderModule VulkanComputeManager::createShaderModule(
    const std::vector<char> &shaderCode) const {
  vk::ShaderModuleCreateInfo createInfo{};
  createInfo.codeSize = shaderCode.size();
  createInfo.pCode = reinterpret_cast<const uint32_t *>(shaderCode.data());

  return device->createShaderModuleUnique(createInfo);
}

vk::UniqueShaderModule
VulkanComputeManager::loadShader(const char *filename) const {
  auto computeShaderModule = createShaderModule(readFile(filename));
  fmt::print("Successfully loaded shader {}.\n", filename);
  return computeShaderModule;

  /*
  To execute a compute shader we need to:

  1. Create a descriptor set that has two VkDescriptorBufferInfo’s for each of
our buffers (one for each binding in the compute shader).
  2. Update the descriptor set to set the bindings of both of the VkBuffer’s
we created earlier.
  3. Create a command pool with our queue family index.
  4. Allocate a command buffer from the command pool (we’re using
VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT as we aren’t resubmitting the
buffer in our sample).
  5. Begin the command buffer.
  6. Bind our compute pipeline.
  7. Bind our descriptor set at the VK_PIPELINE_BIND_POINT_COMPUTE.
  8. Dispatch a compute shader for each element of our buffer.
  9. End the command buffer.
  10. And submit it to the queue!

  */
}

void VulkanComputeManager::printInstanceExtensionSupport() {
  const auto extensions = vk::enumerateInstanceExtensionProperties();

  fmt::print("Available Vulkan extensions:\n");
  for (const auto &extension : extensions) {
    fmt::print("\t{}\n", static_cast<const char *>(extension.extensionName));
  }
}
} // namespace vcm

// NOLINTEND(*-reinterpret-cast)