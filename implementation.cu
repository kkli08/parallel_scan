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

////////////////////////////////////////////////////////////////////////////////
// CUDA kernels for the Brent-Kung inclusive scan

// Up-Sweep Kernel (Reduction Phase)
__global__ void upsweep_kernel(int32_t *d_data, int stride, size_t n)
{
    // Calculate thread index
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int index = (idx + 1) * stride * 2 - 1;

    if (index < n)
    {
        d_data[index] += d_data[index - stride];
    }
}

// Down-Sweep Kernel (Distribution Phase)
__global__ void downsweep_kernel(int32_t *d_data, int stride, size_t n)
{
    // Calculate thread index
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int index = (idx + 1) * stride * 2 - 1;

    if (index < n)
    {
        int temp = d_data[index - stride];
        d_data[index - stride] = d_data[index];
        d_data[index] += temp;
    }
}

// Copy Input to Temporary Array Kernel
__global__ void copy_input_kernel(const int32_t *d_input, int32_t *d_data, size_t n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
    {
        d_data[idx] = d_input[idx];
    }
    else if (idx < n * 2) // For padding if necessary
    {
        d_data[idx] = 0;
    }
}


// Adjusting the scan to be inclusive
__global__ void inclusive_adjust_kernel(int32_t *d_data, const int32_t *d_input, int32_t *d_output, size_t n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
    {
        d_output[idx] = d_data[idx] + d_input[idx];
    }
}

///////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>

// CPU Sequential Inclusive Scan
void cpu_inclusive_scan(const int32_t *h_input, int32_t *h_cpu_output, size_t n)
{
    if (n == 0) return;
    h_cpu_output[0] = h_input[0];
    for (size_t i = 1; i < n; ++i)
    {
        h_cpu_output[i] = h_cpu_output[i - 1] + h_input[i];
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
    // Allocate host memory for debugging
    int32_t *h_input = (int32_t *)malloc(size * sizeof(int32_t));
    int32_t *h_gpu_output = (int32_t *)malloc(size * sizeof(int32_t));
    int32_t *h_cpu_output = (int32_t *)malloc(size * sizeof(int32_t));

    // Copy input data from device to host for printing
    cudaMemcpy(h_input, d_input, size * sizeof(int32_t), cudaMemcpyDeviceToHost);


    // Allocate temporary device memory
    int32_t *d_data;
    size_t n = size;

    // Next power of two for array size
    size_t padded_size = 1;
    while (padded_size < n)
        padded_size <<= 1;

    cudaMalloc(&d_data, padded_size * sizeof(int32_t));

    // Copy input data to temporary array
    int threadsPerBlock = 256;
    int blocksPerGrid = ((padded_size) + threadsPerBlock - 1) / threadsPerBlock;
    copy_input_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_data, n);
    cudaDeviceSynchronize();

    // Up-Sweep (Reduction Phase)
    int stride;
    for (stride = 1; stride < padded_size; stride *= 2)
    {
        int numThreads = padded_size / (stride * 2);
        if (numThreads > 0)
        {
            blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
            upsweep_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, stride, padded_size);
            cudaDeviceSynchronize();
        }
    }

    // Set last element to zero
    if (padded_size > 1)
    {
        cudaMemset(d_data + padded_size - 1, 0, sizeof(int32_t));
        cudaDeviceSynchronize();
    }

    // Down-Sweep (Distribution Phase)
    for (stride = padded_size / 2; stride >= 1; stride /= 2)
    {
        int numThreads = padded_size / (stride * 2);
        if (numThreads > 0)
        {
            blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
            downsweep_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, stride, padded_size);
            cudaDeviceSynchronize();
        }
    }
    
    // Launch the kernel to adjust for inclusive scan
    blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    inclusive_adjust_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_input, d_output, n);
    cudaDeviceSynchronize();

    // Copy GPU output back to host for debugging
    cudaMemcpy(h_gpu_output, d_output, n * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Free temporary memory
    cudaFree(d_data);

    // Free host memory
    free(h_input);
    free(h_cpu_output);
    free(h_gpu_output);
}

