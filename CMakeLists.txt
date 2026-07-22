cmake_minimum_required(VERSION 3.28.3)

project(Chroma VERSION 0.1)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# External libraries
find_package(plog REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(SDL3_image CONFIG REQUIRED)
find_package(implot CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(tinyply CONFIG REQUIRED)

if(NOT Vulkan_GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found. Make sure the Vulkan SDK is properly installed and sourced.")
endif()

# ImGui setup
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
target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_BACKENDS_DIR})
target_link_libraries(imgui PUBLIC SDL3::SDL3 Vulkan::Vulkan)

# Export fonts for ImGui
file(COPY ${IMGUI_DIR}/misc/fonts DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Ensure executables are output to the top-level build directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Add main source subdirectory 
add_subdirectory(src)

# Preprocessor definitions for the entire project
target_compile_definitions(${PROJECT_NAME} PRIVATE
    ROOT_DIR="${CMAKE_SOURCE_DIR}/"
    SRC_DIR="${CMAKE_SOURCE_DIR}/src/"
    RES_DIR="${CMAKE_SOURCE_DIR}/res/"
)