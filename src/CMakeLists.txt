set(EXE_NAME main)

find_package(fmt CONFIG REQUIRED)
find_package(Qt6 COMPONENTS Core Gui Widgets REQUIRED)
find_package(Armadillo CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(shaders)

find_package(Vulkan REQUIRED)

set(SOURCES
  main.cpp
  vcm/VulkanComputeManager.hpp
  vcm/VulkanComputeManager.cpp
)


set(ENV{VK_ADD_LAYER_PATH} "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/bin")

add_executable(${EXE_NAME}
  ${SOURCES}
)

target_link_libraries(${EXE_NAME} PRIVATE
  fmt::fmt
  armadillo
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


add_executable(add
  mainAdd.cpp
  vcm/VulkanComputeManager.hpp
  vcm/VulkanComputeManager.cpp
  vcm/ComputeShaderExecutor.hpp
)
target_link_libraries(add PRIVATE
  fmt::fmt
  armadillo
  Vulkan::Vulkan
)
add_dependencies(add compile_shaders)
set_target_properties(add PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
)


add_executable(fma
  mainFMA.cpp
  vcm/VulkanComputeManager.hpp
  vcm/VulkanComputeManager.cpp
  vcm/ComputeShaderExecutor.hpp
)
target_link_libraries(fma PRIVATE
  fmt::fmt
  armadillo
  Vulkan::Vulkan
)
add_dependencies(fma compile_shaders)
set_target_properties(fma PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
)

set(NAME addImage2D)
add_executable(${NAME}
  mainAddImage2d.cpp
  vcm/VulkanComputeManager.hpp
  vcm/VulkanComputeManager.cpp
)
target_link_libraries(${NAME} PRIVATE
  fmt::fmt
  armadillo
  Vulkan::Vulkan
)
add_dependencies(${NAME} compile_shaders)
set_target_properties(${NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
)

set(NAME medianBlur)
add_executable(${NAME}
  mainMedianBlur2D.cpp
  vcm/VulkanComputeManager.hpp
  vcm/VulkanComputeManager.cpp
)
target_link_libraries(${NAME} PRIVATE
  fmt::fmt
  armadillo
  Vulkan::Vulkan
  opencv_world
)
add_dependencies(${NAME} compile_shaders)
set_target_properties(${NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
)