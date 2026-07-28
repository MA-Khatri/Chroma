# Collect all source files
set(SOURCES
    application.cpp
    application.hpp
    camera.cpp
    camera.hpp
    controller.cpp
    controller.hpp
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
    scene.cpp
    scene.hpp
    texture.hpp
    transform.cpp
    transform.hpp
)

# Create the main executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Add subdirectories
add_subdirectory(vulkan_engine)
add_subdirectory(sharing)

# Link dependencies from the top level
target_link_libraries(${PROJECT_NAME}
PRIVATE
    plog::plog
    glm::glm
    Vulkan::Vulkan
    SDL3::SDL3
    $<IF:$<TARGET_EXISTS:SDL3_image::SDL3_image-shared>,SDL3_image::SDL3_image-shared,SDL3_image::SDL3_image-static>
    tinyobjloader::tinyobjloader
    tinyply
    imgui::imgui
    implot::implot
    nlohmann_json::nlohmann_json
    cppzmq
    cppzmq-static
)