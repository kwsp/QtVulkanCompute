set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Find glslangValidator
find_package(glslang CONFIG REQUIRED)
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)

set(SHADERS
    warpPolarCompute.comp
    add2D.comp
    addImage2D.comp
    fma2D.comp
)

cmake_path(GET SHADER_BINARY_DIR PARENT_PATH SRC_BINARY_DIR)

# Custom target to hold all shader compilation targets
add_custom_target(compile_shaders)

# Compile each shader and generate a separate custom command
foreach(SHADER ${SHADERS})
    get_filename_component(FILE_NAME ${SHADER} NAME_WE)
    set(SPIRV ${SHADER_BINARY_DIR}/${FILE_NAME}.spv)

    # Add custom command to compile each shader
    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${GLSLANG_VALIDATOR} -V ${SHADER_SOURCE_DIR}/${SHADER} -o ${SPIRV}
        DEPENDS ${SHADER_SOURCE_DIR}/${SHADER}
        COMMENT "Compiling ${SHADER} to SPIR-V"
        VERBATIM
    )

    # Create a target that depends on the SPIR-V file being generated
    add_custom_target(${FILE_NAME}_shader DEPENDS ${SPIRV})

    add_dependencies(compile_shaders ${FILE_NAME}_shader)

    # Copy the compiled SPIR-V files to the build configuration directories
    foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
        set(CONFIG_BINARY_DIR ${SRC_BINARY_DIR}/${CONFIG}/shaders)
        set(CONFIG_SPIRV ${CONFIG_BINARY_DIR}/${FILE_NAME}.spv)

        add_custom_command(
            OUTPUT ${CONFIG_SPIRV}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CONFIG_BINARY_DIR}
            COMMAND ${CMAKE_COMMAND} -E copy ${SPIRV} ${CONFIG_SPIRV}
            DEPENDS ${SPIRV}
            COMMENT "Copying ${SPIRV} to ${CONFIG} configuration"
        )

        # Create a target for copying the SPIR-V file
        add_custom_target(${FILE_NAME}_${CONFIG}_shader DEPENDS ${CONFIG_SPIRV})
        add_dependencies(compile_shaders ${FILE_NAME}_${CONFIG}_shader)
    endforeach()

endforeach()
