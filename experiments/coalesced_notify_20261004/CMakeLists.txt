# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
cmake_minimum_required(VERSION 3.26)
project(coalesced_seal_notifications LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit REQUIRED)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
include_directories("${REPO_ROOT}/csrc/libtorch_stable")
add_library(seal_native_harness SHARED native_harness.cpp)
add_library(seal_execution_observer SHARED observer.cpp)
target_include_directories(seal_execution_observer PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_link_libraries(seal_execution_observer PRIVATE CUDA::cudart)
# Use only the async target: the inherited tree's core kernels stay untouched.
find_package(Torch REQUIRED)
add_library(glm53_async SHARED
  "${REPO_ROOT}/csrc/libtorch_stable/async_batch_memcpy.cu"
  "${REPO_ROOT}/csrc/libtorch_stable/async_batch_bindings.cpp")
target_include_directories(glm53_async PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(glm53_async PRIVATE ${TORCH_LIBRARIES} CUDA::cuda_driver)
set_target_properties(glm53_async PROPERTIES PREFIX "_" CUDA_ARCHITECTURES "100")
