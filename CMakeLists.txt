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
find_package(imgui CONFIG REQUIRED)
find_package(implot CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(tinyply CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(cppzmq CONFIG REQUIRED)

if(NOT Vulkan_GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found. Make sure the Vulkan SDK is properly installed and sourced.")
endif()

# Export fonts for ImGui
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/external/fonts DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

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