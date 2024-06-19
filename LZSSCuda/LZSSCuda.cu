#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

// Definirea funcției `min` pentru utilizare în kernel-uri CUDA
__device__ int Min(int a, int b) {
    return (a < b) ? a : b;
}
// Kernel pentru compresie LZSS
__device__ int findLongestMatch(const unsigned char* window, int windowSize, const unsigned char* lookahead, int lookaheadSize, int& matchDistance) {
    int maxLength = 0;
    for (int i = 0; i < windowSize; ++i) {
        int length = 0;
        while (length < lookaheadSize && window[i + length] == lookahead[length]) {
            ++length;
        }
        if (length > maxLength) {
            maxLength = length;
            matchDistance = windowSize - i;
        }
    }
    return maxLength;
}

__global__ void CompressKernel(const unsigned char* input, int inputLength, unsigned char* output, int* outputLength) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = tid * 32;
    if (offset >= inputLength) return;

    int windowSize = 4096;
    int lookaheadSize = 18;
    int minMatchLength = 3;
    int compressedIndex = offset;

    for (int i = offset; i < inputLength && i < offset + 32; ) {
        int matchLength = 0;
        int matchDistance = 0;

        if (i >= windowSize) {
            matchLength = findLongestMatch(input + i - windowSize, windowSize, input + i, Min(lookaheadSize, inputLength - i), matchDistance);
        }

        if (matchLength >= minMatchLength) {
            output[compressedIndex++] = 1; // Match flag
            output[compressedIndex++] = matchDistance >> 8;
            output[compressedIndex++] = matchDistance & 0xFF;
            output[compressedIndex++] = matchLength;
            i += matchLength;
        }
        else {
            output[compressedIndex++] = 0; // Literal flag
            output[compressedIndex++] = input[i++];
        }
    }

    outputLength[tid] = compressedIndex - offset;
}

__global__ void DecompressKernel(const unsigned char* input, int inputLength, unsigned char* output, int* outputLength) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = tid * 32;
    if (offset >= inputLength) return;

    int decompressedIndex = offset;

    for (int i = offset; i < inputLength && i < offset + 32; ) {
        unsigned char flag = input[i++];
        if (flag == 0) {
            output[decompressedIndex++] = input[i++];
        }
        else {
            int matchDistance = (input[i] << 8) | input[i + 1];
            int matchLength = input[i + 2];
            i += 3;
            int start = decompressedIndex - matchDistance;
            for (int j = 0; j < matchLength; ++j) {
                output[decompressedIndex++] = output[start + j];
            }
        }
    }

    outputLength[tid] = decompressedIndex - offset;
}

extern "C" {
    void CompressCuda(const unsigned char* input, int inputLength, unsigned char* output, int* outputLength) {
        unsigned char* d_input;
        unsigned char* d_output;
        int* d_outputLength;

        int numBlocks = (inputLength + 31) / 32;

        cudaMalloc((void**)&d_input, inputLength);
        cudaMalloc((void**)&d_output, inputLength * 2); // Dimensiunea maximă estimată pentru output
        cudaMalloc((void**)&d_outputLength, numBlocks * sizeof(int));

        cudaMemcpy(d_input, input, inputLength, cudaMemcpyHostToDevice);

        CompressKernel << <numBlocks, 32 >> > (d_input, inputLength, d_output, d_outputLength);

        std::vector<int> h_outputLength(numBlocks);
        cudaMemcpy(h_outputLength.data(), d_outputLength, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

        int totalLength = 0;
        for (int len : h_outputLength) {
            totalLength += len;
        }

        cudaMemcpy(output, d_output, totalLength, cudaMemcpyDeviceToHost);
        *outputLength = totalLength;

        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_outputLength);
    }

    void DecompressCuda(const unsigned char* input, int inputLength, unsigned char* output, int* outputLength) {
        unsigned char* d_input;
        unsigned char* d_output;
        int* d_outputLength;

        int numBlocks = (inputLength + 31) / 32;

        cudaMalloc((void**)&d_input, inputLength);
        cudaMalloc((void**)&d_output, inputLength * 2); // Dimensiunea maximă estimată pentru output
        cudaMalloc((void**)&d_outputLength, numBlocks * sizeof(int));

        cudaMemcpy(d_input, input, inputLength, cudaMemcpyHostToDevice);

        DecompressKernel << <numBlocks, 32 >> > (d_input, inputLength, d_output, d_outputLength);

        std::vector<int> h_outputLength(numBlocks);
        cudaMemcpy(h_outputLength.data(), d_outputLength, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

        int totalLength = 0;
        for (int len : h_outputLength) {
            totalLength += len;
        }

        cudaMemcpy(output, d_output, totalLength, cudaMemcpyDeviceToHost);
        *outputLength = totalLength;

        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_outputLength);
    }
}
