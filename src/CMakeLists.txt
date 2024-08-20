set(EXE_NAME main)

find_package(fmt CONFIG REQUIRED)
find_package(Qt6 COMPONENTS Core Gui Widgets REQUIRED)

add_subdirectory(shaders)

find_package(Vulkan REQUIRED)

set(SOURCES
  main.cpp
  VulkanComputeManager.hpp
  VulkanComputeManager.cpp
)


set(ENV{VK_ADD_LAYER_PATH} "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/bin")

add_executable(${EXE_NAME}
  ${SOURCES}
)

target_link_libraries(${EXE_NAME} PRIVATE
  Vulkan::Vulkan
  Qt6::Core
  Qt6::Gui
  Qt6::Widgets
)

add_dependencies(${EXE_NAME} compile_shaders)

set_target_properties(${EXE_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
)

target_link_libraries(${EXE_NAME} PRIVATE
  fmt::fmt
)