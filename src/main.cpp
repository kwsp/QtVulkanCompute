#include <fmt/core.h>

#include <fstream>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

class VulkanComputeManager {
public:
  VulkanComputeManager() {
    // Init Vulkan instance
    createInstance();

    // loadShader("shaders/warpPolarCompute.spv");
  }

  ~VulkanComputeManager() { cleanup(); }

private:
  // QVulkanInstance vulkanInstance;
  VkInstance vulkanInstance;

  VkPipeline computePipelinne;
  VkShaderModule computeShaderModule;

  const std::vector<const char *> validationLayers = {
      "VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
  const bool enableValidationLayers = false;
#else
  const bool enableValidationLayers = true;
#endif

  void createInstance() {
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

    auto ret = vkCreateInstance(&createInfo, nullptr, &vulkanInstance);
    if (ret != VK_SUCCESS) {
      throw std::runtime_error("Failed to create Vulkan instance!");
    }
    fmt::println("Created Vulkan instance.");

    // check for extension support
    {
      uint32_t extensionCount = 0;
      vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

      std::vector<VkExtensionProperties> extensions(extensionCount);
      vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
                                             extensions.data());
      fmt::print("Available Vulkan extensions:\n");
      for (const auto &extension : extensions) {
        fmt::print("\t{}\n", extension.extensionName);
      }
    }
  }

  bool checkValidationLayerSupport() {
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

  // void loadShader(const char *filename) {
  //   std::ifstream file(filename, std::ios::binary | std::ios::ate);
  //   if (!file.is_open()) {
  //     throw std::runtime_error("Failed to open shader file");
  //   }

  //   const size_t fileSize = file.tellg();
  //   std::vector<char> buffer(fileSize);
  //   file.seekg(0);
  //   file.read(buffer.data(), fileSize);
  //   file.close();

  //   VkShaderModuleCreateInfo createInfo{};
  //   createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  //   createInfo.codeSize = buffer.size();
  //   createInfo.pCode = reinterpret_cast<const uint32_t *>(buffer.data());

  //   if (vkCreateShaderModule(device, &createInfo, nullptr,
  //                            &computeShaderModule) != VK_SUCCESS) {
  //     throw std::runtime_error("Failed to create shader module!");
  //   }
  // }

  void cleanup() {
    if (vulkanInstance != nullptr) {
      vkDestroyInstance(vulkanInstance, nullptr);
    }
  }
};

int main(int argc, char *argv[]) {
  fmt::print("Hello, world!\n");

  VulkanComputeManager manager;
  return 0;
}
//