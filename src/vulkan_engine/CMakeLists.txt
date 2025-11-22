
set(VULKAN_ENGINE_SRC
    vulkan_engine.cpp
    vulkan_engine.hpp
    vulkan_utils.cpp
    vulkan_utils.hpp

    vk_material.hpp
    vk_material.cpp
    vk_object.hpp
    vk_object.cpp
    vk_scene.hpp
    vk_scene.cpp
)

# Add files to the global SOURCES variable
target_sources(${PROJECT_NAME} PRIVATE ${VULKAN_ENGINE_SRC})

add_subdirectory(vulkan_shaders)