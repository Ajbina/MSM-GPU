cmake_minimum_required(VERSION 3.18)
project(msm_bn254_mgpu LANGUAGES CXX CUDA)

# ============================================================================
# MSM GPU: Multi-Scalar Multiplication on GPU (BN254)
# ============================================================================

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type defaults to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================================
# CUDA Configuration
# ============================================================================

# Auto-detect GPU architecture (Ada = sm_89, Ampere = sm_80, Turing = sm_75)
# Comment out to use default; uncomment to force specific arch
# set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4000 Ada

# ============================================================================
# Main Executable
# ============================================================================

add_executable(msm_bn254_mgpu
  src/main.cu
  src/kernels.cu
)

# Include directories
target_include_directories(msm_bn254_mgpu PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

# ============================================================================
# Compiler Flags
# ============================================================================

# C++ compiler flags
target_compile_options(msm_bn254_mgpu PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-std=c++17 -Wall -Wextra>
)

# CUDA compiler flags
target_compile_options(msm_bn254_mgpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    -lineinfo
    -Xcompiler "-std=c++17 -Wall"
  >
)

# Release vs Debug flags
if(CMAKE_BUILD_TYPE MATCHES Release)
  target_compile_options(msm_bn254_mgpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--optimize=3>
  )
endif()

# ============================================================================
# Build Output
# ============================================================================

# Place binary in build directory root
set_target_properties(msm_bn254_mgpu PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "======================================")
message(STATUS "MSM GPU Build Configuration")
message(STATUS "======================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Output: ${CMAKE_BINARY_DIR}/msm_bn254_mgpu")
message(STATUS "======================================")
