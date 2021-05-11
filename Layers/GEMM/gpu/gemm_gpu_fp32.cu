#include <algorithm>
#include <vector>
#include <random>
#include <functional>
#include <iostream>
#include <stdio.h>

#include <cuda_runtime.h>

#define TILE_WIDTH 16

#define GPU_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Basic cuda kernel for GEMM, no tileing or tensorcore
// This is a generic kernel for full and half float
__global__ void gemmBasicCudaKernelFp32(const float* A, const float* B, float* C, int m, int n, int k) {
    // each thread will calculate 
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (rowIdx < m && colIdx < n) {
        float val = 0.0f;
        for(int kdx = 0; kdx < k; kdx++) {
            val += A[rowIdx * k + kdx] * B[kdx * n + colIdx];
        }
        C[rowIdx * n + colIdx] = val;
    }
} 

__global__ void gemmTiledCudaKernelFp32(const float* A, const float* B, float* C, int m, int n, int k) {
    __shared__ float aTile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float bTile[TILE_WIDTH][TILE_WIDTH];

    // Each thread will calculate a value for C[rowTileIdx * Tile_Width + rowThreadIdx, colTileIdx * TILE_WIDTH + colThreadIdx]
    int rowTileIdx = blockIdx.y;
    int colTileIdx = blockIdx.x;
    int rowThreadIdx = threadIdx.y;
    int colThreadIdx = threadIdx.x;

    int rowIdx = rowTileIdx * TILE_WIDTH + rowThreadIdx;
    int colIdx = colTileIdx * TILE_WIDTH + colThreadIdx;
    float value = 0.0f;

    // Loop through tiles. 
    // Step 1: Load tile data into shared memory 
    // Step 2: Accumulate the result into value
    for(int iter = 0; iter < k / TILE_WIDTH; iter++) {
        aTile[rowThreadIdx][colThreadIdx] = A[rowIdx * k + iter * TILE_WIDTH + colThreadIdx];
        bTile[rowThreadIdx][colThreadIdx] = B[(iter * TILE_WIDTH + rowThreadIdx) * n + colIdx];
        __syncthreads();

        #pragma unroll
        for(int kdx = 0; kdx < TILE_WIDTH; kdx++) {
            value += aTile[rowThreadIdx][kdx] * bTile[kdx][colThreadIdx];
        }
        __syncthreads();
    }

    C[rowIdx * n + colIdx] = value;
}

void runExperimentFp32(int m, int n, int k) {
    std::vector<float> aVec(m * k);
    std::vector<float> bVec(n * k);
    std::vector<float> cVec(m * n, 0.0f);

    // initialize matrix with random values
    std::uniform_real_distribution<float> distribution(0.0f, 5.0f);
    std::mt19937 engine;
    std::generate(aVec.begin(), aVec.end(), std::bind(distribution, engine));
    std::generate(bVec.begin(), bVec.end(), std::bind(distribution, engine));

    // malloc device memory and initialize them
    float *deviceA, *deviceB, *deviceC;
    GPU_ERROR(cudaMalloc(&deviceA, m * k * sizeof(float)));
    GPU_ERROR(cudaMalloc(&deviceB, k * n * sizeof(float)));
    GPU_ERROR(cudaMalloc(&deviceC, m * n * sizeof(float)));

    GPU_ERROR(cudaMemcpy(deviceA, aVec.data(), m * k * sizeof(float), cudaMemcpyHostToDevice));
    GPU_ERROR(cudaMemcpy(deviceB, bVec.data(), k * n * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 gridDim(n / TILE_WIDTH, m / TILE_WIDTH, 1);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    gemmBasicCudaKernelFp32<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, m, n, k);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float basicGemmFp32Millis = 0.0f;
    cudaEventElapsedTime(&basicGemmFp32Millis, start, stop);
    std::cout << "case " << k <<" basic GEMM gpu take " << basicGemmFp32Millis << " to complete." << std::endl;

    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess) {
        fprintf(stderr,"ERROR basic : %s\n", cudaGetErrorString(error) );
        exit(-1);
    }

    cudaEventRecord(start, 0);
    gemmTiledCudaKernelFp32<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, m, n, k);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float tiledGemmFp32Millis;
    cudaEventElapsedTime(&tiledGemmFp32Millis, start, stop);

    error = cudaGetLastError();
    if(error!=cudaSuccess) {
        fprintf(stderr,"ERROR tiled: %s\n", cudaGetErrorString(error) );
        exit(-1);
    }
    std::cout << "case " << k <<" Tiled GEMM gpu take " << tiledGemmFp32Millis << " to complete." << std::endl;
}

int main(int argc, char** argv) {
    runExperimentFp32(64, 64, 64);
    runExperimentFp32(128, 128, 128);
    runExperimentFp32(256, 256, 256);
    runExperimentFp32(512, 512, 512);
    runExperimentFp32(1024, 1024, 1024);
}