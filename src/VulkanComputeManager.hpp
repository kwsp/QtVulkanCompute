#pragma once

#include <fmt/core.h>
#include <optional>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

class VulkanComputeManager {
public:
  VulkanComputeManager();

  VulkanComputeManager(const VulkanComputeManager &) = default;
  VulkanComputeManager(VulkanComputeManager &&) = delete;
  VulkanComputeManager &operator=(const VulkanComputeManager &) = delete;
  VulkanComputeManager &operator=(VulkanComputeManager &&) = delete;

  ~VulkanComputeManager() { cleanup(); }

private:
  // QVulkanInstance vulkanInstance;
  VkInstance vulkanInstance{};
  VkPhysicalDevice physicalDevice{};
  std::string physicalDeviceName;

  VkPipeline computePipelinne{};
  VkShaderModule computeShaderModule{};

  const std::vector<const char *> validationLayers = {
      "VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
  const bool enableValidationLayers = false;
#else
  const bool enableValidationLayers = true;
#endif

  void createInstance();
  bool checkValidationLayerSupport();

  // Select a graphics card in the system that supports the features we need.
  // Stick to the first suitable card we find.
  void pickPhysicalDevice();
  static int rateDeviceSuitability(VkPhysicalDevice device);
  struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> computeFamily;
  };
  static QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

  void loadShader(const char *filename);

  void cleanup();
};
