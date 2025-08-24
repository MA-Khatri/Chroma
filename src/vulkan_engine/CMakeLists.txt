
set(VULKAN_ENGINE_SRC
    vulkan_utils.cpp
    vulkan_utils.hpp
)

# Add files to the global SOURCES variable
target_sources(${PROJECT_NAME} PRIVATE ${VULKAN_ENGINE_SRC})