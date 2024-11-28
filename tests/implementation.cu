// #include "implementation.h"
// #include <cuda_runtime.h>
// #include <stdio.h>
// #include <stdint.h>

// void printSubmissionInfo()
// {
//     char nick_name[] = "Kenji-Fujima";
//     char student_first_name[] = "Damian";
//     char student_last_name[] = "Li";
//     char student_student_number[] = "1005842554";

//     printf("*******************************************************************************************************\n");
//     printf("Submission Information:\n");
//     printf("\tnick_name: %s\n", nick_name);
//     printf("\tstudent_first_name: %s\n", student_first_name);
//     printf("\tstudent_last_name: %s\n", student_last_name);
//     printf("\tstudent_student_number: %s\n", student_student_number);
// }

// #define MAX_THREADS_PER_BLOCK 1024

// // Kernel 1: Per-block inclusive scan
// __global__ void block_inclusive_scan_kernel(const int32_t *d_input, int32_t *d_output, int32_t *d_block_sums, size_t n)
// {
//     extern __shared__ int32_t s_data[];

//     int tid = threadIdx.x;
//     int gid = blockIdx.x * blockDim.x + tid;

//     // Load data into shared memory
//     s_data[tid] = (gid < n) ? d_input[gid] : 0;
//     __syncthreads();

//     // Inclusive scan within the block using Kogge-Stone algorithm
//     for (int offset = 1; offset < blockDim.x; offset <<= 1)
//     {
//         int temp = 0;
//         if (tid >= offset)
//             temp = s_data[tid - offset];
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

// // CPU function to perform sequential prefix addition (if needed)
// void cpu_prefix_add(int32_t *array, size_t n)
// {
//     for (size_t i = 1; i < n; i++)
//     {
//         array[i] += array[i - 1];
//     }
// }

// // Kernel 2: Adjust scanned data with block sums
// __global__ void adjust_with_block_sums_kernel(int32_t *d_output, const int32_t *d_scanned_block_sums, size_t n)
// {
//     int gid = blockIdx.x * blockDim.x + threadIdx.x;

//     if (gid < n && blockIdx.x > 0)
//     {
//         d_output[gid] += d_scanned_block_sums[blockIdx.x - 1];
//     }
// }

// // Main inclusive scan implementation
// void implementation(const int32_t *d_input, int32_t *d_output, size_t size)
// {
//     int threadsPerBlock = 512;
//     int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

//     // Allocate memory for block sums
//     int32_t *d_block_sums;
//     cudaMalloc(&d_block_sums, numBlocks * sizeof(int32_t));

//     // Kernel 1: Per-block inclusive scan
//     size_t sharedMemSize = threadsPerBlock * sizeof(int32_t);
//     block_inclusive_scan_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_input, d_output, d_block_sums, size);
//     cudaDeviceSynchronize();

//     // Kernel 2: Inclusive scan on block sums
//     block_inclusive_scan_kernel<<<1, threadsPerBlock, numBlocks * sizeof(int32_t)>>>(d_block_sums, d_block_sums, NULL, numBlocks);
//     cudaDeviceSynchronize();

//     // Copy block sums to host memory for sequential addition
//     int32_t *h_block_sums = (int32_t *)malloc(numBlocks * sizeof(int32_t));
//     cudaMemcpy(h_block_sums, d_block_sums, numBlocks * sizeof(int32_t), cudaMemcpyDeviceToHost);

//     // CPU sequential addition (only necessary if final adjustments are needed)
//     cpu_prefix_add(h_block_sums, numBlocks);

//     // Copy updated block sums back to device
//     cudaMemcpy(d_block_sums, h_block_sums, numBlocks * sizeof(int32_t), cudaMemcpyHostToDevice);

//     // Kernel 3: Adjust the block sums with the fully scanned block sums
//     adjust_with_block_sums_kernel<<<1, threadsPerBlock>>>(d_block_sums, d_block_sums, numBlocks);
//     cudaDeviceSynchronize();

//     // Kernel 4: Adjust final output with scanned block sums
//     adjust_with_block_sums_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_block_sums, size);
//     cudaDeviceSynchronize();

//     // Free allocated memory
//     free(h_block_sums);
//     cudaFree(d_block_sums);
// }







// #include "implementation.h"
// #include <cuda_runtime.h>
// #include <stdio.h>
// #include <stdint.h>
// #include <omp.h>

// void printSubmissionInfo()
// {
//     char nick_name[] = "Kenji-Fujima";
//     char student_first_name[] = "Damian";
//     char student_last_name[] = "Li";
//     char student_student_number[] = "1005842554";

//     printf("*******************************************************************************************************\n");
//     printf("Submission Information:\n");
//     printf("\tnick_name: %s\n", nick_name);
//     printf("\tstudent_first_name: %s\n", student_first_name);
//     printf("\tstudent_last_name: %s\n", student_last_name);
//     printf("\tstudent_student_number: %s\n", student_student_number);
// }

// #define MAX_THREADS_PER_BLOCK 1024

// // Kernel 1: Per-block inclusive scan
// __global__ void block_inclusive_scan_kernel(const int32_t *d_input, int32_t *d_output, int32_t *d_block_sums, size_t n)
// {
//     extern __shared__ int32_t s_data[];

//     int tid = threadIdx.x;
//     int gid = blockIdx.x * blockDim.x + tid;

//     // Load data into shared memory
//     s_data[tid] = (gid < n) ? d_input[gid] : 0;
//     __syncthreads();

//     // Inclusive scan within the block using Kogge-Stone algorithm
//     for (int offset = 1; offset < blockDim.x; offset <<= 1)
//     {
//         int temp = 0;
//         if (tid >= offset)
//             temp = s_data[tid - offset];
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

// // Multi-threaded CPU function to perform prefix scan using OpenMP
// void parallel_prefix_scan(const int32_t *input, int32_t *output, size_t n)
// {
//     int levels = 0;
//     while ((1 << levels) < n)
//         levels++;

//     // Allocate temporary arrays for each level
//     int32_t *temp1 = (int32_t *)malloc(n * sizeof(int32_t));
//     int32_t *temp2 = (int32_t *)malloc(n * sizeof(int32_t));
//     memcpy(temp1, input, n * sizeof(int32_t));

//     for (int level = 0; level < levels; ++level)
//     {
//         int offset = 1 << level;

// #pragma omp parallel for
//         for (size_t i = 0; i < n; ++i)
//         {
//             if (i >= offset)
//                 temp2[i] = temp1[i] + temp1[i - offset];
//             else
//                 temp2[i] = temp1[i];
//         }

//         // Swap arrays
//         int32_t *tmp = temp1;
//         temp1 = temp2;
//         temp2 = tmp;
//     }

//     memcpy(output, temp1, n * sizeof(int32_t));
//     free(temp1);
//     free(temp2);
// }

// // Kernel 2: Adjust scanned data with block sums
// __global__ void adjust_with_block_sums_kernel(int32_t *d_output, const int32_t *d_scanned_block_sums, size_t n)
// {
//     int gid = blockIdx.x * blockDim.x + threadIdx.x;

//     if (gid < n && blockIdx.x > 0)
//     {
//         d_output[gid] += d_scanned_block_sums[blockIdx.x - 1];
//     }
// }

// // Main inclusive scan implementation
// void implementation(const int32_t *d_input, int32_t *d_output, size_t size)
// {
//     int threadsPerBlock = 512;
//     int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

//     // Allocate memory for block sums
//     int32_t *d_block_sums;
//     cudaMalloc(&d_block_sums, numBlocks * sizeof(int32_t));

//     // Kernel 1: Per-block inclusive scan
//     size_t sharedMemSize = threadsPerBlock * sizeof(int32_t);
//     block_inclusive_scan_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_input, d_output, d_block_sums, size);
//     cudaDeviceSynchronize();

//     // Kernel 2: Inclusive scan on block sums
//     block_inclusive_scan_kernel<<<1, threadsPerBlock, numBlocks * sizeof(int32_t)>>>(d_block_sums, d_block_sums, NULL, numBlocks);
//     cudaDeviceSynchronize();

//     // Copy block sums to host memory for parallel prefix scan
//     int32_t *h_block_sums = (int32_t *)malloc(numBlocks * sizeof(int32_t));
//     cudaMemcpy(h_block_sums, d_block_sums, numBlocks * sizeof(int32_t), cudaMemcpyDeviceToHost);

//     // Multi-threaded prefix scan on block sums using OpenMP
//     int32_t *h_scanned_block_sums = (int32_t *)malloc(numBlocks * sizeof(int32_t));
//     parallel_prefix_scan(h_block_sums, h_scanned_block_sums, numBlocks);

//     // Copy updated block sums back to device
//     cudaMemcpy(d_block_sums, h_scanned_block_sums, numBlocks * sizeof(int32_t), cudaMemcpyHostToDevice);

//     // Kernel 3: Adjust the block sums with the fully scanned block sums
//     adjust_with_block_sums_kernel<<<1, threadsPerBlock>>>(d_block_sums, d_block_sums, numBlocks);
//     cudaDeviceSynchronize();

//     // Kernel 4: Adjust final output with scanned block sums
//     adjust_with_block_sums_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_block_sums, size);
//     cudaDeviceSynchronize();

//     // Free allocated memory
//     free(h_block_sums);
//     free(h_scanned_block_sums);
//     cudaFree(d_block_sums);
// }

