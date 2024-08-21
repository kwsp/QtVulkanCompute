#pragma once

#include <fmt/format.h>
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

private:
  // QVulkanInstance vulkanInstance;
  VkInstance vulkanInstance{};

  VkPhysicalDevice physicalDevice{};
  std::string physicalDeviceName;

  VkDevice device{};
  VkQueue computeQueue{};

  VkPipeline computePipeline{};

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

  /* Load shaders */
  void loadComputeShader(const char *filename);
  VkShaderModule createShaderModule(const std::vector<char> &computeShaderCode);

  /* Cleanup */
  void cleanup();
};
