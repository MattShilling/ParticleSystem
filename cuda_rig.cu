#include "cuda_rig.h"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_cuda.h"
#include "helper_functions.h"

void CudaRig::Init() {
  // Initialize test memory.
  test_init_(mem_);
}

int CudaRig::InitAndCopy(void **dev_mem, void *host_mem, size_t sz) {
  cudaError_t status;
  status = cudaMalloc(dev_mem, sz);
  checkCudaErrors(status);
  // Copy host memory to the GPU.
  status =
      cudaMemcpy(*dev_mem, host_mem, sz, cudaMemcpyHostToDevice);
  checkCudaErrors(status);

  return status;
}

void CudaRig::StartCudaTimer(CudaTimer *t) {
  cudaError_t status;
  // Create and start timer.
  cudaDeviceSynchronize();

  // Allocate CUDA events that we'll use for timing.
  status = cudaEventCreate(&(t->start));
  checkCudaErrors(status);
  status = cudaEventCreate(&(t->stop));
  checkCudaErrors(status);

  // Record the start event.
  status = cudaEventRecord(t->start, NULL);
  checkCudaErrors(status);
}

void CudaRig::StopCudaTimer(CudaTimer *t){
  cudaError_t status;
  // Record the stop event.
  status = cudaEventRecord(t->stop, NULL);
  checkCudaErrors(status);

  // Wait for the stop event to complete.
  status = cudaEventSynchronize(t->stop);
  checkCudaErrors(status);
}