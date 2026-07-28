# Set output directory for compiled shaders
set(VULKAN_SHADER_OUTPUT_DIR ${CMAKE_BINARY_DIR}/vulkan_shaders)

set(VULKAN_SHADERS
Default.vert
    Default.frag
    Default.vert
    DepthOnly.frag
    DepthOnlyPoint.frag
    GroundGrid.vert
    GroundGrid.frag
    Point.vert
    PointFlat.frag
    PointNormal.frag
    PointShaded.frag
    Surface.vert
    SurfaceAlbedoTextureFlat.frag
    SurfaceAlbedoTextureShaded.frag
    SurfacePerVertexFlat.frag
    SurfacePerVertexNormal.frag
    SurfacePerVertexShaded.frag
)

# Compile shaders with glslc
foreach(SHADER ${VULKAN_SHADERS})
    get_filename_component(FILE_NAME ${SHADER} NAME)
    set(SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER})
    set(SPIRV_PATH ${VULKAN_SHADER_OUTPUT_DIR}/${FILE_NAME}.spv)

    add_custom_command(
        OUTPUT ${SPIRV_PATH}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -o ${SPIRV_PATH} ${SRC_PATH}
        DEPENDS ${SRC_PATH}
        COMMENT "Compiling shader: ${FILE_NAME}"
        VERBATIM
    )

    list(APPEND COMPILED_SHADERS ${SPIRV_PATH})
    list(APPEND SHADER_SOURCE_LIST ${SRC_PATH})
endforeach()

# Group compiled shaders under a custom target
add_custom_target(compiled_vulkan_shaders ALL DEPENDS ${COMPILED_SHADERS})

# Add to target sources for visibility
target_sources(${PROJECT_NAME} PRIVATE ${SHADER_SOURCE_LIST})