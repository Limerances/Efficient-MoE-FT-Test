cmake_minimum_required(VERSION 3.18)
project(egm_test CUDA CXX)

# 需要 CUDA
find_package(CUDAToolkit REQUIRED)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 优化选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# EGM 测试程序
add_executable(egm_test egm_test.cu)

# 链接 CUDA Driver API
target_link_libraries(egm_test
    CUDA::cuda_driver
    CUDA::cudart
)

# 打印配置信息
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Include: ${CUDAToolkit_INCLUDE_DIRS}")
