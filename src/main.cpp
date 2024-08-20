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
    fmt::println("Created Vulkan instance.");
  }

private:
  // QVulkanInstance vulkanInstance;
  VkInstance vulkanInstance;

  VkPipeline computePipelinne;
  VkShaderModule computeShaderModule;

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

    createInfo.enabledLayerCount = 0;
    if (vkCreateInstance(&createInfo, nullptr, &vulkanInstance) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create Vulkan instance!");
    }
  }
};

void loadShader() {
  const char *filename = "shaders/warpPolarCompute.spv";
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open shader file");
  }

  const size_t fileSize = file.tellg();
  std::vector<char> buffer(fileSize);
  file.seekg(0);
  file.read(buffer.data(), fileSize);
  file.close();

  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = buffer.size();
  createInfo.pCode = reinterpret_cast<const uint32_t *>(buffer.data());

  // if (vkCreateShaderModule(device, &createInfo, nullptr,
  //                          &computeShaderModule) != VK_SUCCESS) {
  //   throw std::runtime_error("Failed to create shader module!");
  // }
}

int main(int argc, char *argv[]) {
  fmt::print("Hello, world!\n");

  VulkanComputeManager manager;
  return 0;
}
