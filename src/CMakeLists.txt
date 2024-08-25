set(EXE_NAME main)

find_package(fmt CONFIG REQUIRED)
find_package(Qt6 COMPONENTS Core Gui Widgets REQUIRED)
find_package(Armadillo CONFIG REQUIRED)

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


function(create_test_executable NAME)
  add_executable(${NAME}
    ${ARGN}
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
endfunction()

create_test_executable(add
  mainAdd.cpp
  vcm/VulkanComputeManager.hpp
  vcm/VulkanComputeManager.cpp
)


create_test_executable(addImage2D
  mainAddImage2D.cpp
  vcm/VulkanComputeManager.hpp
  vcm/VulkanComputeManager.cpp
)