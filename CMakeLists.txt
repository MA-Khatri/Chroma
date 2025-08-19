cmake_minimum_required(VERSION 3.28.3)

project(Chroma VERSION 0.1)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)

# For VSCode Intellisense (/Clangd)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add source files
set(SOURCES
    # external/optix_utils/vector_math.h

    # src/vulkan_engine/image.hpp
    # src/vulkan_engine/image.cpp
    # src/vulkan_engine/shader.hpp
    # src/vulkan_engine/shader.cpp
    src/vulkan_engine/vulkan_utils.hpp
    src/vulkan_engine/vulkan_utils.cpp

    src/vulkan_engine/shaders/Flat.frag
    src/vulkan_engine/shaders/Flat.vert
    src/vulkan_engine/shaders/FlatWithSpecular.frag
    src/vulkan_engine/shaders/HelloTriangle.frag
    src/vulkan_engine/shaders/HelloTriangle.vert
    src/vulkan_engine/shaders/Lines.frag
    src/vulkan_engine/shaders/Lines.vert
    src/vulkan_engine/shaders/Normal.frag
    src/vulkan_engine/shaders/Solid.frag
    src/vulkan_engine/shaders/Solid.vert

    src/application.hpp
    src/application.cpp
    src/layer.cpp
    src/layer.hpp
    src/main.cpp
    src/material.hpp
    src/material.cpp
    src/mesh.hpp
    src/mesh.cpp
    src/object.hpp
    src/object.cpp
    src/raster_view.cpp
    src/raster_view.hpp
    src/raytrace_view.cpp
    src/raytrace_view.hpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

# External Linked Packages
find_package(plog REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE plog::plog)

find_package(glm CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)

find_package(Vulkan REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan)

find_package(SDL3 CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE SDL3::SDL3)

find_package(SDL3_image CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE $<IF:$<TARGET_EXISTS:SDL3_image::SDL3_image-shared>,SDL3_image::SDL3_image-shared,SDL3_image::SDL3_image-static>)

# TODO: Switch to better mesh loader (assimp?)
find_package(tinyobjloader CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE tinyobjloader::tinyobjloader)

### Start ImGui setup
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui)
set(IMGUI_BACKENDS_DIR ${IMGUI_DIR}/backends)
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_BACKENDS_DIR}/imgui_impl_sdl3.cpp
    ${IMGUI_BACKENDS_DIR}/imgui_impl_vulkan.cpp
)

add_library(imgui STATIC ${IMGUI_SOURCES})

target_include_directories(imgui PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_BACKENDS_DIR}
)

target_link_libraries(imgui 
PUBLIC 
    SDL3::SDL3 
    Vulkan::Vulkan
)

target_link_libraries(${PROJECT_NAME} PRIVATE imgui)

# Not part of base ImGui but included here to keep ImGui stuff together
find_package(implot CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE implot::implot)

# Copy fonts folder to build dir
file(COPY ${IMGUI_DIR}/misc/fonts DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

### End ImGui setup

target_include_directories(${PROJECT_NAME}
PRIVATE
    ${IMGUI_DIR}
    external/optix_utils
)