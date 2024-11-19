// test.cpp

#include <iostream>
#include <cuda_runtime.h>
#include "implementation.h"

int main() {
    // Define a small input size
    size_t size = 8; // You can change this to any small number
    int32_t h_input[] = {1, 2, 3, 4, 5, 6, 7, 8};

    // Allocate host memory for output
    int32_t* h_output = new int32_t[size];

    // Allocate device memory
    int32_t* d_input;
    int32_t* d_output;
    cudaMalloc(&d_input, size * sizeof(int32_t));
    cudaMalloc(&d_output, size * sizeof(int32_t));

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, size * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Call the implementation function
    implementation(d_input, d_output, size);

    // Copy output data from device to host
    cudaMemcpy(h_output, d_output, size * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Print the input and output arrays
    std::cout << "Input Array: ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Output Array (Inclusive Prefix Sum): ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Verify correctness by computing CPU inclusive scan
    int32_t* h_cpu_output = new int32_t[size];
    h_cpu_output[0] = h_input[0];
    for (size_t i = 1; i < size; ++i) {
        h_cpu_output[i] = h_cpu_output[i - 1] + h_input[i];
    }

    // Compare GPU output with CPU output
    bool correct = true;
    for (size_t i = 0; i < size; ++i) {
        if (h_output[i] != h_cpu_output[i]) {
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "The GPU inclusive scan is correct!" << std::endl;
    } else {
        std::cout << "The GPU inclusive scan is incorrect." << std::endl;
    }

    // Free memory
    delete[] h_output;
    delete[] h_cpu_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
