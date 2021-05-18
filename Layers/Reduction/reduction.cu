#include <algorithm>
#include <vector>
#include <random>
#include <functional>
#include <iostream>
#include <stdio.h>

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

#define MAX_SEGMENTS 256
#define TILE_WIDTH 16
#define WARP_SIZE 32
#define TOTAL_WARP 48

#define GPU_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using namespace nvcuda;

__global__ void computeWmmaReduction(const half* values, float* output, int numSegmentsPerWarp) {
    __shared__ half partialResult[MAX_SEGMENTS];
    __shared__ half tempValues[TILE_WIDTH * TILE_WIDTH];
    __shared__ half bMatrixVal[TILE_WIDTH * TILE_WIDTH];

    int warpIndex = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for(int idx = laneId; idx < TILE_WIDTH * TILE_WIDTH; idx += WARP_SIZE) {
        int col = idx % TILE_WIDTH;
        bMatrixVal[idx] = col == 0 ? __float2half(1.0f) : __float2half(0.0f);
    }
    __syncwarp();
    wmma::fragment<wmma::matrix_a, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH, half> tempValFrag;

    // load b matrix from shared memory to fragment
    wmma::load_matrix_sync(bFrag, bMatrixVal, TILE_WIDTH);  // Set leading dimension to 16 to load Consecutive values
    const half* startPtr = values + (warpIndex * TILE_WIDTH * TILE_WIDTH * numSegmentsPerWarp);
    for(int i = 0; i < numSegmentsPerWarp; i++) {
        // zeroed result fragment 
        wmma::fill_fragment(tempValFrag, __float2half(0.0f));
        wmma::load_matrix_sync(aFrag, startPtr + i * TILE_WIDTH * TILE_WIDTH, TILE_WIDTH);
        wmma::mma_sync(tempValFrag, aFrag, bFrag, tempValFrag);

        // Store tempoary values back to shared memory using column layout so output values
        // will be moved to first row for future operation
        wmma::store_matrix_sync(tempValues, tempValFrag, TILE_WIDTH, wmma::mem_col_major);
        wmma::fill_fragment(tempValFrag, __float2half(0.0f));
        wmma::load_matrix_sync(aFrag, tempValues, TILE_WIDTH);
        wmma::mma_sync(tempValFrag, aFrag, bFrag, tempValFrag);
        // Here, summed value will appear at the top left element. 
        if(laneId == 0) {
            partialResult[i] = tempValFrag.x[0];
        }
    }

    // Now we have partial values stored into partial results. We again perform one iteration of reduction on
    // the partial values to get all results. 
    wmma::fragment<wmma::accumulator, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH, float> finalFrag;
    wmma::fill_fragment(finalFrag, 0.0f);
    wmma::fill_fragment(tempValFrag, __float2half(0.0f));
    wmma::load_matrix_sync(aFrag, partialResult, TILE_WIDTH);
    wmma::mma_sync(tempValFrag, aFrag, bFrag, tempValFrag);

    // Store tempoary values back to shared memory using column layout so output values
    // will be moved to first row for future operation
    wmma::store_matrix_sync(tempValues, tempValFrag, TILE_WIDTH, wmma::mem_col_major);
    wmma::fill_fragment(tempValFrag, __float2half(0.0f));
    wmma::load_matrix_sync(aFrag, tempValues, TILE_WIDTH);
    wmma::mma_sync(finalFrag, aFrag, bFrag, finalFrag);

    // put output into global memory 
    if(laneId == 0) {
        output[warpIndex] = finalFrag.x[0];
    }
}

void runExperimentMalloc(int numSegmentsPerWarp) {
    std::size_t totalInput = TILE_WIDTH * TILE_WIDTH * numSegmentsPerWarp * TOTAL_WARP;
    std::vector<half> input(totalInput);
    std::vector<float> output(TOTAL_WARP);

    half *deviceIn;
    float *deviceOut;
    GPU_ERROR(cudaMalloc(&deviceIn, totalInput * sizeof(half)));
    GPU_ERROR(cudaMalloc(&deviceOut, TOTAL_WARP * sizeof(float)));

    for(int idx = 0; idx < totalInput; idx++) {
        input[idx] = __float2half(1.0f);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    GPU_ERROR(cudaMemcpy(deviceIn, input.data(), totalInput * sizeof(half), cudaMemcpyHostToDevice));

    dim3 gridDim(TOTAL_WARP, 1, 1);
    dim3 blockDim(WARP_SIZE, 1, 1);

    computeWmmaReduction<<<gridDim, blockDim>>>(deviceIn, deviceOut, numSegmentsPerWarp);
    GPU_ERROR(cudaMemcpy(output.data(), deviceOut, TOTAL_WARP * sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float reductionGemmMillis = 0.0f;
    cudaEventElapsedTime(&reductionGemmMillis, start, stop);

    std::cout <<" WMMA reduction gpu take " << reductionGemmMillis << " to complete." << std::endl;
}

void runExperimentMallocManaged(int numSegmentsPerWarp) {
    std::size_t totalInput = TILE_WIDTH * TILE_WIDTH * numSegmentsPerWarp * TOTAL_WARP;

    half *deviceIn;
    float *deviceOut;
    GPU_ERROR(cudaMallocManaged(&deviceIn, totalInput * sizeof(half)));
    GPU_ERROR(cudaMallocManaged(&deviceOut, TOTAL_WARP * sizeof(float)));

    for(int idx = 0; idx < totalInput; idx++) {
        deviceIn[idx] = __float2half(1.0f);
    }

    dim3 gridDim(TOTAL_WARP, 1, 1);
    dim3 blockDim(WARP_SIZE, 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    computeWmmaReduction<<<gridDim, blockDim>>>(deviceIn, deviceOut, numSegmentsPerWarp);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float reductionGemmMillis = 0.0f;
    cudaEventElapsedTime(&reductionGemmMillis, start, stop);

    std::cout <<" WMMA reduction gpu take " << reductionGemmMillis << " to complete." << std::endl;
}
int main() {
    runExperimentMalloc(256);
    runExperimentMallocManaged(256);
}
