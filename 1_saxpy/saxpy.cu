#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

extern float toBW(int bytes, float sec);

#define DEBUG
#ifdef DEBUG
 
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
        if (abort) exit(code);    
    }
}
#else
#define cudaCheckError(ans) ans

#endif

__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
       result[index] = alpha * x[index] + y[index];
}

void
saxpyCuda(int N, float alpha, float* xarray, float* yarray, float* resultarray) {
    int totalBytes = sizeof(float) * 3 * N;

    // compute number of blocks and threads per block
    const int threadsPerBlock = 512;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* device_x;
    float* device_y;
    float* device_result;

    int size = N *sizeof(float);
    //
    // TODO: allocate device memory buffers on the GPU using
    // cudaMalloc.  The started code issues warnings on build because
    // these buffers are used in the call to saxpy_kernel below
    // without being initialized.
    //

    cudaMalloc(&device_x, size);
    cudaMalloc(&device_y, size);
    cudaMalloc(&device_result, size);

    // start timing after allocation of device memory.
    double startTime = CycleTimer::currentSeconds();

    //
    // TODO: copy input arrays to the GPU using cudaMemcpy
    //
    cudaMemcpy(device_x, xarray, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, yarray, size, cudaMemcpyHostToDevice);

    //
    // TODO: insert time here to begin timing only the kernel
    //
    double kernel_startTime = CycleTimer::currentSeconds();

    // run saxpy_kernel on the GPU
    saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);

    //
    // TODO: insert timer here to time only the kernel.  Since the
    // kernel will run asynchronously with the calling CPU thread, you
    // need to call cudaThreadSynchronize() before your timer to
    // ensure the kernel running on the GPU has completed.  (Otherwise
    // you will incorrectly observe that almost no time elapses!)
    //
    cudaThreadSynchronize();
    double kernel_endTime = CycleTimer::currentSeconds();

    //
    // TODO: copy result from GPU using cudaMemcpy
    //
    cudaMemcpy(resultarray, device_result, size, cudaMemcpyDeviceToHost);
    
    // end timing after result has been copied back into host memory.
    // The time elapsed between startTime and endTime is the total
    // time to copy data to the GPU, run the kernel, and copy the
    // result back to the CPU
    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    double kernelDuration = kernel_endTime - kernel_startTime;
    printf("Overall time: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));
    printf("Kernel time: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * kernelDuration, toBW(totalBytes, kernelDuration));
    //
    // TODO free memory buffers on the GPU
    //
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);

    // below code calculates the time that sequential CPU-based SAXPY used
    // float* resultarray_CPU = new float[N];
    // double startTime_CPU = CycleTimer::currentSeconds();
    // for (int i=0; i<N; i++) {
    //     resultarray_CPU[i] = xarray[i]*alpha+yarray[i];
    // }
    // double endTime_CPU = CycleTimer::currentSeconds();
    // double CPUDuration = endTime_CPU - startTime_CPU;
    // printf("CPU time: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * CPUDuration, toBW(totalBytes, CPUDuration));
    // delete [] resultarray_CPU;
}
void
printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
