#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

#define CUDA_CALL(x) do {                                \
  cudaError_t err = (x);                                 \
  if (err != cudaSuccess) {                              \
    throw std::runtime_error(cudaGetErrorString(err));    \
  }                                                      \
} while(0)

inline void print_device_info() {
  int G=0; CUDA_CALL(cudaGetDeviceCount(&G));
  std::cout << "Found " << G << " GPU(s)\n";
  for (int g=0; g<G; g++) {
    cudaDeviceProp p; CUDA_CALL(cudaGetDeviceProperties(&p, g));
    std::cout << "GPU " << g << " : " << p.name << "\n";
    std::cout << "  SMs: " << p.multiProcessorCount << "\n";
    std::cout << "  sharedMemPerBlock: " << p.sharedMemPerBlock << " bytes\n";
    std::cout << "  sharedMemPerMultiprocessor: " << p.sharedMemPerMultiprocessor << " bytes\n";
    std::cout << "  maxThreadsPerBlock: " << p.maxThreadsPerBlock << "\n";
  }
}
