set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# find glslangValidator
find_package(glslang CONFIG REQUIRED)
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)


set(SHADERS
    warpPolarCompute.comp.glsl
)

# Compile the shaders once
foreach(SHADER ${SHADERS})
    get_filename_component(FILE_NAME ${SHADER} NAME_WE)
    set(SPIRV ${SHADER_BINARY_DIR}/${FILE_NAME}.spv)

    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${GLSLANG_VALIDATOR} -V ${SHADER_SOURCE_DIR}/${SHADER} -o ${SPIRV}
        DEPENDS ${SHADER_SOURCE_DIR}/${SHADER}
        COMMENT "Compiling ${SHADER} to SPIR-V"
    )

    list(APPEND SPIRV_BINARIES ${SPIRV})
endforeach()

set(TMP_PATH ${CMAKE_CURRENT_BINARY_DIR})
cmake_path(GET TMP_PATH PARENT_PATH SRC_BINARY_DIR)

# Copy the compiled SPIR-V files to the build configuration directories
foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
    set(CONFIG_BINARY_DIR ${SRC_BINARY_DIR}/${CONFIG}/shaders)

    foreach(SPRIV ${SPIRV_BINARIES})
        set(CONFIG_SPIRV ${CONFIG_BINARY_DIR}/${FILENAME})

        add_custom_command(
            OUTPUT ${CONFIG_SPIRV}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CONFIG_BINARY_DIR}
            COMMAND ${CMAKE_COMMAND} -E copy ${SPIRV} ${CONFIG_SPIRV}
            DEPENDS ${SPIRV}
            COMMENT "Copying ${SPIRV} to ${CONFIG} configuration"
        )

        list(APPEND CONFIG_SPIRV_BINARIES ${CONFIG_SPIRV})
    endforeach()
endforeach()

# Add a custom target for compiling shaders
add_custom_target(compile_shaders DEPENDS ${CONFIG_SPIRV_BINARIES})