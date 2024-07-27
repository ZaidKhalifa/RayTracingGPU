cmake_minimum_required(VERSION 3.17.0)

# set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin/nvcc.exe")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "Building in Debug mode")
endif()

project(RayTracingGPU LANGUAGES CXX CUDA)

message("Project name: " ${CMAKE_PROJECT_NAME})

if (MSVC)
    message(STATUS "Building for Windows with MSVC")
    set(_CRT_SECURE_NO_WARNINGS 1)
    set(CURL_STATICLIB 1)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
endif()

# find_package( OpenCV REQUIRED)

## target_include_directories(${CMAKE_PROJECT_NAME} ${OpenCV_INCLUDE_DIRS})

add_executable(${CMAKE_PROJECT_NAME} )

target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    RayTracingGPU/main.cu
    RayTracingGPU/stb_image_write.h
    RayTracingGPU/vec3.cuh
    RayTracingGPU/color.cuh
    RayTracingGPU/ray.cuh
    RayTracingGPU/hitbox.cuh
    RayTracingGPU/sphere.cuh
    RayTracingGPU/hitbox_list.cuh
    RayTracingGPU/rtutil.cuh
    RayTracingGPU/camera.cuh
    RayTracingGPU/material.cuh
)

# target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS} )

target_compile_options(${CMAKE_PROJECT_NAME} 
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>: >
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr> # -g -G --ptxas-options=–allow-expensive-optimizations=true>
)

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 61 89)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE _DEBUG)
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

