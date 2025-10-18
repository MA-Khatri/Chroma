
set(VULKAN_ENGINE_SRC
    vulkan_engine.cpp
    vulkan_engine.hpp
    vulkan_utils.cpp
    vulkan_utils.hpp
)

# Add files to the global SOURCES variable
target_sources(${PROJECT_NAME} PRIVATE ${VULKAN_ENGINE_SRC})