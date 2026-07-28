
set(SHARING_SRC
    cross_platform_shared_memory.cpp
    cross_platform_shared_memory.hpp
    network.cpp
    network.hpp
    shared_memory_live_view_layout.hpp
    shared_memory_model_layout.hpp
)

# Add files to the global SOURCES variable
target_sources(${PROJECT_NAME} PRIVATE ${SHARING_SRC})
