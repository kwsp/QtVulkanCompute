set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# find glslangValidator
find_package(glslang CONFIG REQUIRED)
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)


set (SHADERS
    warpPolarCompute.comp.glsl
)

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

# Add a custom target for compiling shaders
add_custom_target(compile_shaders DEPENDS ${SPIRV_BINARIES})