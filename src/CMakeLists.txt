# Collect all source files
set(SOURCES
    application.cpp
    application.hpp
    camera.cpp
    camera.hpp
    layer.cpp
    layer.hpp
    main.cpp
    material.cpp
    material.hpp
    mesh.cpp
    mesh.hpp
    object.cpp
    object.hpp
    raster_view.cpp
    raster_view.hpp
    raytrace_view.cpp
    raytrace_view.hpp
    scene.cpp
    scene.hpp
    transform.cpp
    transform.hpp
)

# Create the main executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Add subdirectories
add_subdirectory(vulkan_engine)
add_subdirectory(vulkan_shaders)

# Link dependencies from the top level
target_link_libraries(${PROJECT_NAME}
PRIVATE
    plog::plog
    glm::glm
    Vulkan::Vulkan
    SDL3::SDL3
    $<IF:$<TARGET_EXISTS:SDL3_image::SDL3_image-shared>,SDL3_image::SDL3_image-shared,SDL3_image::SDL3_image-static>
    tinyobjloader::tinyobjloader
    imgui
    implot::implot
)

# Include external headers
target_include_directories(${PROJECT_NAME}
PRIVATE
    ${CMAKE_SOURCE_DIR}/external/imgui
    ${CMAKE_SOURCE_DIR}/external/optix_utils
)
