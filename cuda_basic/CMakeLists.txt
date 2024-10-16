cmake_minimum_required(VERSION 3.18)
project(cuda_basic LANGUAGES CXX CUDA)

set(CUDAToolkit "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

set(SOURCES main.cu)

add_executable(cuda_basic ${SOURCES})
target_link_libraries(cuda_basic ${CUDA_LIBRARIES})
