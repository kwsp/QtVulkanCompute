#include "VulkanComputeManager.hpp"
#include "vulkan/vulkan_core.h"
#include <armadillo>
#include <array>
#include <fmt/format.h>
#include <fstream>
#include <map>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>

#define BAIL_ON_BAD_RESULT(result)                                             \
  if (VK_SUCCESS != (result)) {                                                \
    fprintf(stderr, "Failure at %u %s\n", __LINE__, __FILE__);                 \
    exit(-1);                                                                  \
  }

namespace vcm {

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

VulkanComputeManager::VulkanComputeManager() {
  // Init Vulkan instance
  createInstance();

  pickPhysicalDevice();

  createLogicalDevice();

  createCommandPool();

  createCommandBuffer();

#ifndef NDEBUG
  // Check device memory property
  {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    // Enumerate memory types
    std::cout << "Memory Type Count: " << memProperties.memoryTypeCount
              << std::endl;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      const auto propFlags = memProperties.memoryTypes[i].propertyFlags;
      std::cout << "-- Memory Type " << i << ": Property Flags = " << propFlags
                << std::endl;

      if (!(propFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
        std::cout << "--   Warning: Memory Type " << i
                  << " does not have "
                     "VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT"
                  << std::endl;
      }
    }
  }
#endif

  // Create descriptor pool
  createDescriptorPool();

  // loadComputeShader("shaders/warpPolarCompute.spv");
}

void VulkanComputeManager::createCommandPool() {
  QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
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

  if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create command pool!");
  }
}

void VulkanComputeManager::createCommandBuffer() {
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = commandPool;
  // VK_COMMAND_BUFFER_LEVEL_PRIMARY: can be submitted to a queue for execution,
  // but cannot be called from other command buffers.
  // VK_COMMAND_BUFFER_LEVEL_SECONDARY: cannot be submitted directly, but can be
  // called from primary command buffers.
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;

  if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate command buffers!");
  }
}

void VulkanComputeManager::recordCommandBuffer(VkCommandBuffer commandBuffer,
                                               uint32_t imageIndex) {
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  // VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: rerecorded right after
  // executing it once. VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT:
  // secondary command buffer that will be entirely within a single render pass
  // VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT: can be resubmitted while it
  // is also already pending execution.
  // Non are applicable right now.
  beginInfo.flags = 0;
  beginInfo.pInheritanceInfo = nullptr;

  if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
    throw std::runtime_error("Failed to begin recording command buffer!");
  }
}

void VulkanComputeManager::createDescriptorPool() {
  // Create descriptor pool
  // describe which descriptor types our descriptor sets are going to contain
  // and how many
  std::array<VkDescriptorPoolSize, 2> poolSize;
  poolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize[0].descriptorCount = 20;
  poolSize[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSize[1].descriptorCount = 10;

  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.maxSets = 20;
  poolInfo.poolSizeCount = poolSize.size();
  poolInfo.pPoolSizes = poolSize.data();
  poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

  if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create descriptor pool!");
  }
}

void VulkanComputeManager::createBuffer(VkDeviceSize size,
                                        VkBufferUsageFlags usage,
                                        VkMemoryPropertyFlags properties,
                                        VkBuffer &buffer,
                                        VkDeviceMemory &bufferMemory) {
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
    throw std::runtime_error("failed to create buffer!");
  }

  // To allocate memory for a buffer we need to first query its memory
  // requirements using vkGetBufferMemoryRequirements
  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex =
      findMemoryType(memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to allocate buffer memory!");
  }

  vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

uint32_t
VulkanComputeManager::findMemoryType(uint32_t typeFilter,
                                     VkMemoryPropertyFlags properties) const {

  // First query info about available types of memory using
  // vkGetPhysicalDeviceMemoryProperties
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

  // VkPhysicalDeviceMemoryProperties has 2 arrays: memoryTypes and
  // memoryHeaps Memory heaps are distinct memory resources like dedicated
  // VRAM and swap space in RAM for when VRAM runs out.
  //
  // Right now we're only concerned with the type of memory and not the heap
  // it comes from right now
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }
  throw std::runtime_error("Failed to find suitable memory type!");
}

void VulkanComputeManager::createInstance() {
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "QtVulkanCompute";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
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
  createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(instanceExtensions.size());
  createInfo.ppEnabledExtensionNames = instanceExtensions.data();

  const auto ret = vkCreateInstance(&createInfo, nullptr, &instance);
  if (ret != VK_SUCCESS) {
    fmt::println("Failed to create Vulkan instance! Code: {}",
                 static_cast<int32_t>(ret));
    throw std::runtime_error("Failed to create Vulkan instance!");
  }
  fmt::println("Created Vulkan instance.");
}

bool VulkanComputeManager::checkValidationLayerSupport() {
  uint32_t layerCount{};
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

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

  return false;
}

int VulkanComputeManager::rateDeviceSuitability(VkPhysicalDevice device) {
  VkPhysicalDeviceProperties deviceProperties;
  VkPhysicalDeviceFeatures deviceFeatures;
  vkGetPhysicalDeviceProperties(device, &deviceProperties);
  vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

  int score = 0;

  // Discrete GPUs have a significant performance advantage
  if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
    score += 1000;
  }

  // Maximum possible size of textures affects graphics quality
  score += deviceProperties.limits.maxImageDimension2D;

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

  uint32_t deviceCount{};
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

  if (deviceCount == 0) {
    throw std::runtime_error("Failed to find GPUs with Vulkan support");
  }

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

  // A sorted associative container to rank candidates by increasing score
  std::multimap<int, VkPhysicalDevice> candidates;
  for (const auto &device : devices) {
    const int score = rateDeviceSuitability(device);
    candidates.insert({score, device});
  }

  if (candidates.rbegin()->first > 0) {
    physicalDevice = candidates.rbegin()->second;

    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    physicalDeviceName = std::string{deviceProperties.deviceName};

  } else {
    throw std::runtime_error("Failed to find a suitable GPU");
  }

  fmt::println("Picked physical device '{}'.", physicalDeviceName);
}

VulkanComputeManager::QueueFamilyIndices
VulkanComputeManager::findQueueFamilies(VkPhysicalDevice device) {
  QueueFamilyIndices indices{};

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                           queueFamilies.data());

  int i = 0;
  for (const auto &queueFamily : queueFamilies) {
    if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      indices.graphicsFamily = i;
    }
    if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
      indices.computeFamily = i;
    }
    i++;
  }

  return indices;
}

void VulkanComputeManager::createLogicalDevice() {
  // Specify the queues to be created
  const auto indices = findQueueFamilies(physicalDevice);

  VkDeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = indices.computeFamily.value();
  queueCreateInfo.queueCount = 1;

  float queuePriority = 1.0F;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  // Specifiy used device features
  VkPhysicalDeviceFeatures deviceFeatures{};

  // Create the logical device
  VkDeviceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

  // Add pointers to the queue creation info and device features structs
  createInfo.pQueueCreateInfos = &queueCreateInfo;
  createInfo.queueCreateInfoCount = 1;

  createInfo.pEnabledFeatures = &deviceFeatures;

  // Device extensions used
  std::vector<const char *> deviceExtensions;
#ifdef __APPLE__
  {
    // VK_KHR_portability_subset
    deviceExtensions.push_back("VK_KHR_portability_subset");
  }
#endif
  createInfo.enabledExtensionCount = deviceExtensions.size();
  createInfo.ppEnabledExtensionNames = deviceExtensions.data();

  if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create logical device!");
  }

  vkGetDeviceQueue(device, indices.computeFamily.value(), 0, &queue);

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
  file.read(buffer.data(), fileSize);
  file.close();
  return buffer;
}

VkShaderModule VulkanComputeManager::createShaderModule(
    const std::vector<char> &shaderCode) const {
  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = shaderCode.size();
  createInfo.pCode = reinterpret_cast<const uint32_t *>(shaderCode.data());

  VkShaderModule shaderModule{};
  if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create shader module!");
  }

  return shaderModule;
}

VkShaderModule VulkanComputeManager::loadShader(const char *filename) const {
  /*
   * Create compute pipeline
   */

  VkShaderModule computeShaderModule = createShaderModule(readFile(filename));
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

void VulkanComputeManager::destroyShaderModule(VkShaderModule module) const {
  vkDestroyShaderModule(device, module, nullptr);
}

void VulkanComputeManager::cleanup() {
  // destroy descriptor pool
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);

  vkDestroyCommandPool(device, commandPool, nullptr);

  vkDestroyDevice(device, nullptr);

  vkDestroyInstance(instance, nullptr);
}

} // namespace vcm