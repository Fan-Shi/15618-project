#include <algorithm>
#include <vector>
#include <random>
#include <functional>
#include <iostream>
#include <stdio.h>

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

#define TILE_WIDTH 16
#define WARP_SIZE 32

#define GPU_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using namespace nvcuda;

__global__ void gemmBasicKernelFp16(const half *A, const half *B, half *C, int M, int N, int K) {
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (rowIdx < M && colIdx < N) {
        half val = __float2half(0.0f);
        for(int kdx = 0; kdx < K; kdx++) {
            val = __hfma(A[rowIdx * K + kdx], B[kdx * N + colIdx], val); // Notice matrix B is transposed
        }
        C[rowIdx * N + colIdx] = val;
    }
}

// Different from normal tiled gemm kernel. This kernel will use w whole warp to compute a 16 x 16 output tile. 
// We use same setting as tensor core kernel. Each block will contain 4 warps (128 threads)
// The assignment for loading ofr input tile and computation within output tile is 
// 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
// 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
// 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
// 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
// 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
// 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
// ...............................................
// In this way. Each thread will need to loop on row index of tiles. This way will best resolve bank conflict
// This kernel will simulate the behavior of tensor core (warp level MMA). However, tensorcore paradigm will 
// load data directly into register which is faster than this dem0
__global__ void gemmWarpTilingKernelFp16(const half *A, const half *B, half *C, int M, int N, int K) {
    __shared__ half aTile[2 * TILE_WIDTH][2 * TILE_WIDTH];
    __shared__ half bTile[2 * TILE_WIDTH][2 * TILE_WIDTH];

    int tileAIndex = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int tileBIndex = (blockIdx.y * blockDim.y + threadIdx.y);

    int rowAStartIndex = tileAIndex * TILE_WIDTH;
    int rowBStartIndex = tileBIndex * TILE_WIDTH;

    int sharedMemOffsetX = (threadIdx.x / WARP_SIZE) * TILE_WIDTH;
    int sharedMemOffsetY = threadIdx.y * TILE_WIDTH;

    int tId = threadIdx.x % TILE_WIDTH;
    int tgroup = (threadIdx.x % WARP_SIZE) / TILE_WIDTH;
    half outputVal[TILE_WIDTH * TILE_WIDTH / WARP_SIZE];  // Each thread will be responsible a series of output value

    #pragma unroll
    for(int idx = 0; idx < (TILE_WIDTH * TILE_WIDTH / WARP_SIZE); idx++) {
        outputVal[idx] = __float2half(0.0f);
    }

    for(int kdx = 0; kdx < K; kdx += TILE_WIDTH) {
        const half* tiledInputA = A + rowAStartIndex * K + kdx;
        const half* tiledInputB = B + rowBStartIndex * K + kdx;
        // Load data into shared memory. Notice the shared memory is of shape 32 * 32 and each warp will only take one of 16 x 16 block
        #pragma unroll 
        for (int i = 0; i < TILE_WIDTH; i+=2) {
            aTile[sharedMemOffsetX + i + tgroup][sharedMemOffsetY + tId] = tiledInputA[(i + tgroup) * K + tId];
            bTile[sharedMemOffsetX + i + tgroup][sharedMemOffsetY + tId] = tiledInputB[(i + tgroup) * K + tId];
        }
        
        #pragma unroll
        for(int iter = 0; iter < TILE_WIDTH; iter++) {
            half bElem = bTile[sharedMemOffsetX + tId][sharedMemOffsetY + iter];
            #pragma unroll
            for(int i = 0; i < (TILE_WIDTH * TILE_WIDTH / WARP_SIZE); i++) {
                outputVal[i] = __hfma(aTile[sharedMemOffsetX + i * 2 + tgroup][sharedMemOffsetY + iter], bElem, outputVal[i]);
            }
        }
    }

    // Store all the results back to global memory 
    #pragma unroll
    for(int i = 0; i < (TILE_WIDTH * TILE_WIDTH / WARP_SIZE); i++) {
        C[(rowAStartIndex + i * 2 + tgroup) * N + rowBStartIndex + tId] = outputVal[i];
    }
}

__global__ void gemmTensorCoreKernelFp16(const half *A, const half *B, half *C, int M, int N, int K) {
    // Each block contains 4 warps, with 2 x 2 latout. 
    // Each Warp contain 32 threads, and will be responsible for compute 16 x 16 output tile using tensor core
    // Then, each block will output (2 x 16) x (2 x 16) = 32 x 32 output tiles 
    int tileAIndex = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int tileBIndex = (blockIdx.y * blockDim.y + threadIdx.y);

    // Matrix A is stored in row major layout of shape M * K. 
    // Matrix B is stored in col major layout of shale N * K to simplify programming  
    // Each output tile has shape 16 * 16. initialzie to 0
    wmma::fragment<wmma::matrix_a, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH, half, wmma::col_major> bFrag;
    wmma::fragment<wmma::accumulator, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH, half> valFrag;
    wmma::fill_fragment(valFrag, __float2half(0.0f));

    int rowAStartIndex = tileAIndex * TILE_WIDTH;
    int rowBStartIndex = tileBIndex * TILE_WIDTH;

    for(int kdx = 0; kdx < K; kdx += TILE_WIDTH) {
        // Assume the matrix has perfect dimension. We don't check bound here
        wmma::load_matrix_sync(aFrag, A + rowAStartIndex * K + kdx, K /*leading index*/); 
        wmma::load_matrix_sync(aFrag, B + rowBStartIndex * K + kdx, K /*leading index*/); 

        // Multiplication perfromed on ATile * BTile. Notice B is loaded as col major layout
        wmma::mma_sync(valFrag, aFrag, bFrag, valFrag);
    }

    // The upper left position of output matrix is [rowAStartIndex, rowBStartIndex] 
    wmma::store_matrix_sync(C + rowBStartIndex + rowAStartIndex * N, valFrag, N, wmma::mem_row_major);
}

void runExperimentFp16(int m, int n, int k) {
    std::vector<half> aVec(m * k);
    std::vector<half> bVec(n * k);
    std::vector<half> cVec(m * n, __float2half(0.0f));

    // initialize matrix with random values
    std::uniform_real_distribution<float> distribution(0.0f, 5.0f);
    std::mt19937 engine;
    std::generate(aVec.begin(), aVec.end(), [&distribution, &engine](){
        return __float2half(distribution(engine));
    });
    std::generate(bVec.begin(), bVec.end(), [&distribution, &engine](){
        return __float2half(distribution(engine));
    });

    // malloc device memory and initialize them
    half *deviceA, *deviceB, *deviceC;
    GPU_ERROR(cudaMalloc(&deviceA, m * k * sizeof(half)));
    GPU_ERROR(cudaMalloc(&deviceB, k * n * sizeof(half)));
    GPU_ERROR(cudaMalloc(&deviceC, m * n * sizeof(half)));

    GPU_ERROR(cudaMemcpy(deviceA, aVec.data(), m * k * sizeof(half), cudaMemcpyHostToDevice));
    GPU_ERROR(cudaMemcpy(deviceB, bVec.data(), k * n * sizeof(half), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 gridDimBasic(m / TILE_WIDTH, n / TILE_WIDTH, 1);
    dim3 blockDimBasic(TILE_WIDTH, TILE_WIDTH, 1);
    cudaEventRecord(start, 0);
    gemmBasicKernelFp16<<<gridDimBasic, blockDimBasic>>>(deviceA, deviceB, deviceC, m, n, k);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float basicGemmFp16Millis = 0.0f;
    cudaEventElapsedTime(&basicGemmFp16Millis, start, stop);
    std::cout << "case " << k <<" BASIC GEMM gpu take " << basicGemmFp16Millis << " to complete." << std::endl;

    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess) {
        fprintf(stderr,"ERROR Basic : %s\n", cudaGetErrorString(error) );
        exit(-1);
    }

    dim3 gridDim(m / (2 * TILE_WIDTH), n / (2 * TILE_WIDTH), 1);
    dim3 blockDim(2 * WARP_SIZE, 2, 1);

    cudaEventRecord(start, 0);
    gemmWarpTilingKernelFp16<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, m, n, k);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float warpTiledGemmFp16Millis = 0.0f;
    cudaEventElapsedTime(&warpTiledGemmFp16Millis, start, stop);
    std::cout << "case " << k <<" Tensor Core GEMM gpu take " << warpTiledGemmFp16Millis << " to complete." << std::endl;

    error = cudaGetLastError();
    if(error!=cudaSuccess) {
        fprintf(stderr,"ERROR Warp : %s\n", cudaGetErrorString(error) );
        exit(-1);
    }

    cudaEventRecord(start, 0);
    gemmTensorCoreKernelFp16<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, m, n, k);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float tensorCoreGemmFp16Millis = 0.0f;
    cudaEventElapsedTime(&tensorCoreGemmFp16Millis, start, stop);
    std::cout << "case " << k <<" Tensor Core GEMM gpu take " << tensorCoreGemmFp16Millis << " to complete." << std::endl;

    error = cudaGetLastError();
    if(error!=cudaSuccess) {
        fprintf(stderr,"ERROR WMMA : %s\n", cudaGetErrorString(error) );
        exit(-1);
    }
}

int main(int argc, char** argv) {
    runExperimentFp16(64, 64, 64);
    runExperimentFp16(128, 128, 128);
    runExperimentFp16(256, 256, 256);
    runExperimentFp16(512, 512, 512);
    runExperimentFp16(1024, 1024, 1024);
}

