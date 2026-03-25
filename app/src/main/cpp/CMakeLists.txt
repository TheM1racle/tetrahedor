cmake_minimum_required(VERSION 3.18.1)

project("vulkanhello")

# Fetch GLM (Header-only math library)
include(FetchContent)
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 0.9.9.8
)
FetchContent_MakeAvailable(glm)

add_library(${CMAKE_PROJECT_NAME} SHARED
    main.cpp
)

target_link_libraries(${CMAKE_PROJECT_NAME}
    android
    log
    vulkan
    GLM::GLM
)

# Standard NDK settings
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE 
    -fvisibility=hidden 
    -fexceptions 
    -frtti
)