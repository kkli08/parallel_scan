#include "implementation.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

void printSubmissionInfo()
{
    // This will be published in the leaderboard on piazza
    // Please modify this field with something interesting
    char nick_name[] = "Kenji-Fujima";

    // Please fill in your information (for marking purposes only)
    char student_first_name[] = "Damian";
    char student_last_name[] = "Li";
    char student_student_number[] = "1005842554";

    // Printing out team information
    printf("*******************************************************************************************************\n");
    printf("Submission Information:\n");
    printf("\tnick_name: %s\n", nick_name);
    printf("\tstudent_first_name: %s\n", student_first_name);
    printf("\tstudent_last_name: %s\n", student_last_name);
    printf("\tstudent_student_number: %s\n", student_student_number);
}

// Define maximum threads per block
#define MAX_THREADS_PER_BLOCK 1024

// Struct to hold timing information
typedef struct {
    float time_small_kernel;
    float time_block_kernel;
    float time_adjust_kernel;
    float time_recursive_call;
    float time_alloc;
} TimingInfo;

// Kernel to perform per-block inclusive scan
__global__ void block_inclusive_scan_kernel(const int32_t *d_input, int32_t *d_output, int32_t *d_block_sums, size_t n)
{
    extern __shared__ int32_t s_data[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    s_data[tid] = (gid < n) ? d_input[gid] : 0;

    __syncthreads();

    // Inclusive scan within the block using Kogge-Stone algorithm
    for (int offset = 1; offset < blockDim.x; offset <<= 1)
    {
        int temp = 0;
        if (tid >= offset)
            temp = s_data[tid - offset];
        __syncthreads();
        s_data[tid] += temp;
        __syncthreads();
    }

    // Write the scanned data to global memory
    if (gid < n)
    {
        d_output[gid] = s_data[tid];
    }

    // Write the total sum of this block to d_block_sums
    if (tid == blockDim.x - 1)
    {
        d_block_sums[blockIdx.x] = s_data[tid];
    }
}

// Kernel to adjust the scanned data with the scanned block sums
__global__ void adjust_with_block_sums_kernel(int32_t *d_output, const int32_t *d_scanned_block_sums, size_t n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n && blockIdx.x > 0)
    {
        d_output[gid] += d_scanned_block_sums[blockIdx.x - 1];
    }
}

// Kernel to perform an inclusive scan on small arrays (fits within one block)
__global__ void small_inclusive_scan_kernel(int32_t *d_input, int32_t *d_output, size_t n)
{
    extern __shared__ int32_t s_data[];

    int tid = threadIdx.x;

    // Load data into shared memory
    s_data[tid] = (tid < n) ? d_input[tid] : 0;

    __syncthreads();

    // Inclusive scan within the block using Kogge-Stone algorithm
    for (int offset = 1; offset < n; offset <<= 1)
    {
        int temp = 0;
        if (tid >= offset)
            temp = s_data[tid - offset];
        __syncthreads();
        s_data[tid] += temp;
        __syncthreads();
    }

    // Write the scanned data back to global memory
    if (tid < n)
    {
        d_output[tid] = s_data[tid];
    }
}

// Recursive function to perform inclusive scan on the device
void device_inclusive_scan(int32_t *d_input, int32_t *d_output, size_t n, TimingInfo *timing)
{
    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    if (numBlocks <= 1)
    {
        // The array fits in a single block
        size_t sharedMemSize = threadsPerBlock * sizeof(int32_t);

        cudaEvent_t startKernel, stopKernel;
        cudaEventCreate(&startKernel);
        cudaEventCreate(&stopKernel);
        cudaEventRecord(startKernel, 0);

        small_inclusive_scan_kernel<<<1, threadsPerBlock, sharedMemSize>>>(d_input, d_output, n);
        cudaDeviceSynchronize();

        cudaEventRecord(stopKernel, 0);
        cudaEventSynchronize(stopKernel);
        float timeKernel = 0;
        cudaEventElapsedTime(&timeKernel, startKernel, stopKernel);
        timing->time_small_kernel += timeKernel;

        cudaEventDestroy(startKernel);
        cudaEventDestroy(stopKernel);
    }
    else
    {
        cudaEvent_t startAlloc, stopAlloc;
        cudaEventCreate(&startAlloc);
        cudaEventCreate(&stopAlloc);
        cudaEventRecord(startAlloc, 0);

        // Allocate device memory for block sums
        int32_t *d_block_sums;
        cudaMalloc(&d_block_sums, numBlocks * sizeof(int32_t));

        cudaEventRecord(stopAlloc, 0);
        cudaEventSynchronize(stopAlloc);
        float timeAlloc = 0;
        cudaEventElapsedTime(&timeAlloc, startAlloc, stopAlloc);
        timing->time_alloc += timeAlloc;

        size_t sharedMemSize = threadsPerBlock * sizeof(int32_t);

        cudaEvent_t startKernel, stopKernel;
        cudaEventCreate(&startKernel);
        cudaEventCreate(&stopKernel);
        cudaEventRecord(startKernel, 0);

        // Perform per-block inclusive scan
        block_inclusive_scan_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_input, d_output, d_block_sums, n);
        cudaDeviceSynchronize();

        cudaEventRecord(stopKernel, 0);
        cudaEventSynchronize(stopKernel);
        float timeKernel = 0;
        cudaEventElapsedTime(&timeKernel, startKernel, stopKernel);
        timing->time_block_kernel += timeKernel;

        cudaEventDestroy(startKernel);
        cudaEventDestroy(stopKernel);

        // Allocate device memory for scanned block sums
        cudaEventRecord(startAlloc, 0);
        int32_t *d_scanned_block_sums;
        cudaMalloc(&d_scanned_block_sums, numBlocks * sizeof(int32_t));
        cudaEventRecord(stopAlloc, 0);
        cudaEventSynchronize(stopAlloc);
        cudaEventElapsedTime(&timeAlloc, startAlloc, stopAlloc);
        timing->time_alloc += timeAlloc;

        cudaEventDestroy(startAlloc);
        cudaEventDestroy(stopAlloc);

        cudaEvent_t startRecursive, stopRecursive;
        cudaEventCreate(&startRecursive);
        cudaEventCreate(&stopRecursive);
        cudaEventRecord(startRecursive, 0);

        // Recursively scan the block sums
        device_inclusive_scan(d_block_sums, d_scanned_block_sums, numBlocks, timing);

        cudaEventRecord(stopRecursive, 0);
        cudaEventSynchronize(stopRecursive);
        float timeRecursive = 0;
        cudaEventElapsedTime(&timeRecursive, startRecursive, stopRecursive);
        timing->time_recursive_call += timeRecursive;

        cudaEventDestroy(startRecursive);
        cudaEventDestroy(stopRecursive);

        cudaEvent_t startAdjust, stopAdjust;
        cudaEventCreate(&startAdjust);
        cudaEventCreate(&stopAdjust);
        cudaEventRecord(startAdjust, 0);

        // Adjust the output with scanned block sums
        adjust_with_block_sums_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_scanned_block_sums, n);
        cudaDeviceSynchronize();

        cudaEventRecord(stopAdjust, 0);
        cudaEventSynchronize(stopAdjust);
        float timeAdjust = 0;
        cudaEventElapsedTime(&timeAdjust, startAdjust, stopAdjust);
        timing->time_adjust_kernel += timeAdjust;

        cudaEventDestroy(startAdjust);
        cudaEventDestroy(stopAdjust);

        // Free device memory
        cudaFree(d_block_sums);
        cudaFree(d_scanned_block_sums);
    }
}

/**
 * Implement your CUDA inclusive scan here. Feel free to add helper functions, kernels or allocate temporary memory.
 * However, you must not modify other files. CAUTION: make sure you synchronize your kernels properly and free all
 * allocated memory.
 *
 * @param d_input: input array on device
 * @param d_output: output array on device
 * @param size: number of elements in the input array
 */
void implementation(const int32_t *d_input, int32_t *d_output, size_t size)
{
    TimingInfo timing = {0};

    cudaEvent_t startTotal, stopTotal;
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventRecord(startTotal, 0);

    // Since d_input is const, we need to create a mutable copy for the recursive function
    int32_t *d_input_copy;
    cudaMalloc(&d_input_copy, size * sizeof(int32_t));

    cudaEvent_t startCopy, stopCopy;
    cudaEventCreate(&startCopy);
    cudaEventCreate(&stopCopy);
    cudaEventRecord(startCopy, 0);

    cudaMemcpy(d_input_copy, d_input, size * sizeof(int32_t), cudaMemcpyDeviceToDevice);

    cudaEventRecord(stopCopy, 0);
    cudaEventSynchronize(stopCopy);
    float timeCopy = 0;
    cudaEventElapsedTime(&timeCopy, startCopy, stopCopy);
    printf("Time to copy input data: %f ms\n", timeCopy);

    cudaEventDestroy(startCopy);
    cudaEventDestroy(stopCopy);

    // Perform inclusive scan
    device_inclusive_scan(d_input_copy, d_output, size, &timing);

    // Free temporary device memory
    cudaFree(d_input_copy);

    cudaEventRecord(stopTotal, 0);
    cudaEventSynchronize(stopTotal);
    float timeTotal = 0;
    cudaEventElapsedTime(&timeTotal, startTotal, stopTotal);

    // Print timing information
    printf("*******************************************************************************************************\n");
    printf("Total time: %f ms\n", timeTotal);
    printf("Timing breakdown:\n");
    printf("\tTime to copy input data: %f ms\n", timeCopy);
    printf("\tTime in small_inclusive_scan_kernel: %f ms\n", timing.time_small_kernel);
    printf("\tTime in block_inclusive_scan_kernel: %f ms\n", timing.time_block_kernel);
    printf("\tTime in adjust_with_block_sums_kernel: %f ms\n", timing.time_adjust_kernel);
    printf("\tTime in recursive calls: %f ms\n", timing.time_recursive_call);
    printf("\tTime in memory allocations: %f ms\n", timing.time_alloc);
    printf("*******************************************************************************************************\n");

    cudaEventDestroy(startTotal);
    cudaEventDestroy(stopTotal);
}
