set(exec_name query_device_properties)
add_executable(${exec_name} ${exec_name}.cu)
target_link_libraries(${exec_name} ${CUDA_LIBRARIES})
