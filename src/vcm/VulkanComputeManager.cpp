#include "VulkanComputeManager.hpp"
#include "vulkan/vulkan_core.h"
#include <array>
#include <fmt/format.h>
#include <fstream>
#include <map>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>

namespace vcm {

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

VulkanComputeManager::VulkanComputeManager() {
  // Init Vulkan instance
  createInstance();

  pickPhysicalDevice();

  createLogicalDevice();

  loadComputeShader("shaders/warpPolarCompute.spv");
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

  const auto ret = vkCreateInstance(&createInfo, nullptr, &vulkanInstance);
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
  vkEnumeratePhysicalDevices(vulkanInstance, &deviceCount, nullptr);

  if (deviceCount == 0) {
    throw std::runtime_error("Failed to find GPUs with Vulkan support");
  }

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(vulkanInstance, &deviceCount, devices.data());

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

  vkGetDeviceQueue(device, indices.computeFamily.value(), 0, &computeQueue);

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

VkShaderModule
VulkanComputeManager::createShaderModule(const std::vector<char> &shaderCode) {

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

void VulkanComputeManager::loadComputeShader(const char *filename) {
  const auto buffer = readFile(filename);

  auto computeShaderModule = createShaderModule(buffer);

  VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
  computeShaderStageInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  computeShaderStageInfo.module = computeShaderModule;
  computeShaderStageInfo.pName = "main";

  fmt::print("Successfully loaded shader {}.\n", filename);

  vkDestroyShaderModule(device, computeShaderModule, nullptr);
}

void VulkanComputeManager::cleanup() {
  if (device != nullptr) {
    vkDestroyDevice(device, nullptr);
  }

  if (vulkanInstance != nullptr) {
    vkDestroyInstance(vulkanInstance, nullptr);
  }
}

} // namespace vcm
