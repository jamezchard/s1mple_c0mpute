
set(SOURCES main.cu)
add_executable(cuda_basic ${SOURCES})
target_link_libraries(cuda_basic ${CUDA_LIBRARIES})