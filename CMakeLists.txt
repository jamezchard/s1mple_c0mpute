cmake_minimum_required(VERSION 3.18)
project(s1mple_c0mpute LANGUAGES CXX CUDA)

set(CUDAToolkit "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

list(APPEND CMAKE_PREFIX_PATH "D:/JBYO/deps/opencv-4.10.0-install/x64/vc17/lib")

add_subdirectory(mat_mul)
add_subdirectory(vector_add)
add_subdirectory(bgr2gray)
add_subdirectory(img_blur)
add_subdirectory(query_device_properties)
add_subdirectory(convolution)

