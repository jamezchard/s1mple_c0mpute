
set(SOURCES vector_add.cu)
add_executable(vector_add ${SOURCES})
target_link_libraries(vector_add ${CUDA_LIBRARIES})