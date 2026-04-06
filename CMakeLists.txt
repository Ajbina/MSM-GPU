cmake_minimum_required(VERSION 3.18)
project(msm_bn254_mgpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(msm_bn254_mgpu
  src/main.cu
  src/kernels.cu
)

target_include_directories(msm_bn254_mgpu PRIVATE include)

# Helpful flags
target_compile_options(msm_bn254_mgpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>
)

# If you want to force a specific arch, e.g. Ada (sm_89), uncomment:
# set_property(TARGET msm_bn254_mgpu PROPERTY CUDA_ARCHITECTURES 89)
