// #include "implementation.h"
// #include <cuda_runtime.h>
// #include <stdio.h>
// #include <stdint.h>

// void printSubmissionInfo()
// {
//     // This will be published in the leaderboard on piazza
//     // Please modify this field with something interesting
//     char nick_name[] = "Kenji-Fujima";

//     // Please fill in your information (for marking purposes only)
//     char student_first_name[] = "Damian";
//     char student_last_name[] = "Li";
//     char student_student_number[] = "1005842554";

//     // Printing out team information
//     printf("*******************************************************************************************************\n");
//     printf("Submission Information:\n");
//     printf("\tnick_name: %s\n", nick_name);
//     printf("\tstudent_first_name: %s\n", student_first_name);
//     printf("\tstudent_last_name: %s\n", student_last_name);
//     printf("\tstudent_student_number: %s\n", student_student_number);
// }

// #define MAX_THREADS_PER_BLOCK 1024

// // Kernel to perform per-block inclusive scan
// __global__ void block_inclusive_scan_kernel(const int32_t *d_input, int32_t *d_output, int32_t *d_block_sums, size_t n)
// {
//     extern __shared__ int32_t s_data[];

//     int tid = threadIdx.x;
//     int gid = blockIdx.x * blockDim.x + tid;

//     // Load data into shared memory
//     s_data[tid] = (gid < n) ? d_input[gid] : 0;

//     __syncthreads();

//     // Inclusive scan within the block using Kogge-Stone algorithm
//     int offset;
//     for (offset = 1; offset < blockDim.x; offset *= 2)
//     {
//         int temp = 0;
//         if (tid >= offset)
//         {
//             temp = s_data[tid - offset];
//         }
//         __syncthreads();
//         s_data[tid] += temp;
//         __syncthreads();
//     }

//     // Write the scanned data to global memory
//     if (gid < n)
//     {
//         d_output[gid] = s_data[tid];
//     }

//     // Write the total sum of this block to d_block_sums
//     if (tid == blockDim.x - 1)
//     {
//         d_block_sums[blockIdx.x] = s_data[tid];
//     }
// }

// // Kernel to perform an inclusive scan on small arrays (fits within one block)
// __global__ void small_inclusive_scan_kernel(int32_t *d_input, int32_t *d_output, size_t n)
// {
//     extern __shared__ int32_t s_data[];

//     int tid = threadIdx.x;

//     // Load data into shared memory
//     s_data[tid] = (tid < n) ? d_input[tid] : 0;

//     __syncthreads();

//     // Inclusive scan within the block using Kogge-Stone algorithm
//     int offset;
//     for (offset = 1; offset < n; offset *= 2)
//     {
//         int temp = 0;
//         if (tid >= offset)
//         {
//             temp = s_data[tid - offset];
//         }
//         __syncthreads();
//         s_data[tid] += temp;
//         __syncthreads();
//     }

//     // Write the scanned data back to global memory
//     if (tid < n)
//     {
//         d_output[tid] = s_data[tid];
//     }
// }

// // Kernel to add the scanned block sums to each block's output
// __global__ void add_scanned_block_sums_kernel(int32_t *d_output, const int32_t *d_scanned_block_sums, size_t n)
// {
//     int gid = blockIdx.x * blockDim.x + threadIdx.x;

//     if (gid < n && blockIdx.x > 0)
//     {
//         d_output[gid] += d_scanned_block_sums[blockIdx.x - 1];
//     }
// }

// /**
//  * Implement your CUDA inclusive scan here. Feel free to add helper functions, kernels or allocate temporary memory.
//  * However, you must not modify other files. CAUTION: make sure you synchronize your kernels properly and free all
//  * allocated memory.
//  *
//  * @param d_input: input array on device
//  * @param d_output: output array on device
//  * @param size: number of elements in the input array
//  */
// void implementation(const int32_t *d_input, int32_t *d_output, size_t size)
// {
//     // Determine block and grid sizes
//     int threadsPerBlock = 512; // You can adjust this value based on your GPU's capabilities
//     int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

//     // Allocate device memory for block sums
//     int32_t *d_block_sums;
//     cudaMalloc(&d_block_sums, numBlocks * sizeof(int32_t));

//     // Perform per-block inclusive scan
//     size_t sharedMemSize = threadsPerBlock * sizeof(int32_t);
//     block_inclusive_scan_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_input, d_output, d_block_sums, size);
//     cudaDeviceSynchronize();

//     // If there are multiple blocks, perform scan on block sums
//     if (numBlocks > 1)
//     {
//         // Allocate device memory for scanned block sums
//         int32_t *d_scanned_block_sums;
//         cudaMalloc(&d_scanned_block_sums, numBlocks * sizeof(int32_t));

//         // Perform inclusive scan on block sums
//         int threads = numBlocks;
//         int blocks = 1;
//         size_t sharedMemSizeBlockSums = ((threads + 31) / 32) * 32 * sizeof(int32_t); // Ensure multiple of warp size

//         small_inclusive_scan_kernel<<<blocks, threads, sharedMemSizeBlockSums>>>(d_block_sums, d_scanned_block_sums, numBlocks);
//         cudaDeviceSynchronize();

//         // Adjust the output with scanned block sums
//         add_scanned_block_sums_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_scanned_block_sums, size);
//         cudaDeviceSynchronize();

//         // Free device memory for scanned block sums
//         cudaFree(d_scanned_block_sums);
//     }

//     // Free device memory for block sums
//     cudaFree(d_block_sums);
// }


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

#define MAX_THREADS_PER_BLOCK 1024

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
void device_inclusive_scan(int32_t *d_input, int32_t *d_output, size_t n)
{
    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    if (numBlocks <= 1)
    {
        // The array fits in a single block
        size_t sharedMemSize = threadsPerBlock * sizeof(int32_t);
        small_inclusive_scan_kernel<<<1, threadsPerBlock, sharedMemSize>>>(d_input, d_output, n);
        cudaDeviceSynchronize();
    }
    else
    {
        // Allocate device memory for block sums
        int32_t *d_block_sums;
        cudaMalloc(&d_block_sums, numBlocks * sizeof(int32_t));

        // Perform per-block inclusive scan
        size_t sharedMemSize = threadsPerBlock * sizeof(int32_t);
        block_inclusive_scan_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_input, d_output, d_block_sums, n);
        cudaDeviceSynchronize();

        // Allocate device memory for scanned block sums
        int32_t *d_scanned_block_sums;
        cudaMalloc(&d_scanned_block_sums, numBlocks * sizeof(int32_t));

        // Recursively scan the block sums
        device_inclusive_scan(d_block_sums, d_scanned_block_sums, numBlocks);

        // Adjust the output with scanned block sums
        adjust_with_block_sums_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_scanned_block_sums, n);
        cudaDeviceSynchronize();

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
    // Since d_input is const, we need to create a mutable copy for the recursive function
    int32_t *d_input_copy;
    cudaMalloc(&d_input_copy, size * sizeof(int32_t));
    cudaMemcpy(d_input_copy, d_input, size * sizeof(int32_t), cudaMemcpyDeviceToDevice);

    // Perform inclusive scan
    device_inclusive_scan(d_input_copy, d_output, size);

    // Free temporary device memory
    cudaFree(d_input_copy);
}
