set(exec_name mat_mul)
add_executable(${exec_name} ${exec_name}.cu)
target_link_libraries(${exec_name} ${CUDA_LIBRARIES})
