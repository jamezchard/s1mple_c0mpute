cmake_minimum_required(VERSION 3.18)
project(s1mple_c0mpute LANGUAGES CXX CUDA)

set(CUDAToolkit "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

add_subdirectory(cuda_basic)
add_subdirectory(vector_add)
