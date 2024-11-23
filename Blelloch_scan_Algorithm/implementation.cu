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

// Shared memory size per block
#define MAX_BLOCK_SIZE 512

// Kernel for block-wise Blelloch exclusive scan
__global__ void blelloch_scan_kernel(const int32_t *d_input, int32_t *d_output, int32_t *d_block_sums, int n)
{
    extern __shared__ int32_t temp[]; // Shared memory for scan

    int thid = threadIdx.x;
    int offset = 1;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load data into shared memory
    int input_idx = idx;
    int input_idx_plus_block = idx + blockDim.x;

    temp[2 * thid] = (input_idx < n) ? d_input[input_idx] : 0;
    temp[2 * thid + 1] = (input_idx_plus_block < n) ? d_input[input_idx_plus_block] : 0;

    // Up-sweep (reduce) phase
    for (int d = blockDim.x; d > 0; d >>= 1)
    {
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // Save the total sum of this block to block sums
    if (thid == 0)
    {
        d_block_sums[blockIdx.x] = temp[2 * blockDim.x - 1];
        temp[2 * blockDim.x - 1] = 0; // Set last element to zero for exclusive scan
    }

    // Down-sweep phase
    for (int d = 1; d <= blockDim.x; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Write results to global memory
    if (input_idx < n)
        d_output[input_idx] = temp[2 * thid];
    if (input_idx_plus_block < n)
        d_output[input_idx_plus_block] = temp[2 * thid + 1];
}

// Kernel to add scanned block sums to each block's output
__global__ void add_block_sums_kernel(int32_t *d_output, const int32_t *d_block_sums, int n)
{
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    if (blockIdx.x > 0)
    {
        int32_t add_value = d_block_sums[blockIdx.x - 1];

        if (idx < n)
            d_output[idx] += add_value;
        if (idx + blockDim.x < n)
            d_output[idx + blockDim.x] += add_value;
    }
}

// Kernel to adjust exclusive scan to inclusive scan
__global__ void inclusive_adjust_kernel(const int32_t *d_input, int32_t *d_output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        d_output[idx] += d_input[idx];
    }
}

// Recursive function to perform the scan
void blelloch_scan_recursive(const int32_t *d_input, int32_t *d_output, size_t n)
{
    int threadsPerBlock = MAX_BLOCK_SIZE;
    int elementsPerBlock = threadsPerBlock * 2;
    int numBlocks = (n + elementsPerBlock - 1) / elementsPerBlock;

    // Allocate device memory for block sums
    int32_t *d_block_sums;
    cudaMalloc(&d_block_sums, numBlocks * sizeof(int32_t));

    size_t sharedMemSize = elementsPerBlock * sizeof(int32_t);

    // First pass: Scan within blocks
    blelloch_scan_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_input, d_output, d_block_sums, n);
    cudaDeviceSynchronize();

    // If there is more than one block, we need to scan the block sums
    if (numBlocks > 1)
    {
        // Allocate device memory for scanned block sums
        int32_t *d_scanned_block_sums;
        cudaMalloc(&d_scanned_block_sums, numBlocks * sizeof(int32_t));

        // Recursively call blelloch_scan_recursive on block sums
        blelloch_scan_recursive(d_block_sums, d_scanned_block_sums, numBlocks);

        // Second pass: Add scanned block sums to each block's output
        add_block_sums_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_scanned_block_sums, n);
        cudaDeviceSynchronize();

        // Free temporary device memory
        cudaFree(d_scanned_block_sums);
    }

    // Free device memory
    cudaFree(d_block_sums);
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
    // Copy input to output array
    cudaMemcpy(d_output, d_input, size * sizeof(int32_t), cudaMemcpyDeviceToDevice);

    // Perform the exclusive scan recursively
    blelloch_scan_recursive(d_output, d_output, size);

    // Adjust for inclusive scan by adding the original input
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    inclusive_adjust_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);
    cudaDeviceSynchronize();
}
