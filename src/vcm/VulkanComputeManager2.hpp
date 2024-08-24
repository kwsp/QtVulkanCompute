#pragma once

#include <vector>
#include <vulkan/vulkan.h>

void run() {
  VkInstance instance;

  uint32_t physicalDeviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, 0);

  std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
  vkEnumeratePhysicalDevices(instance, physicalDeviceCount,
                             physicalDevices.data());
}