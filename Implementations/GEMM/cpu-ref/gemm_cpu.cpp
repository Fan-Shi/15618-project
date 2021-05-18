
#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <functional>
#include <string>
#include <omp.h>

// basic cpu kernel function for gemm. 
// No omp, no transpose
void gemmBasic(const float* A, const float* B, float* C, int m, int n, int k) {
    for (int idx = 0; idx < m ; idx++) {
        for (int jdx = 0; jdx < n; jdx++) {
            float val = 0.0f;
            for (int kdx = 0; kdx < k; kdx++) {
                val += A[idx * k + kdx] * B[kdx * n + jdx]; // A[idx, kdx] * B[kdx, jdx]
            }
            C[idx * n + jdx] = val;
        }
    }
}

// cpu kernel function for gemm with openmp
void gemmBasicOpenmp(const float* A, const float* B, float* C, int m, int n, int k) {
    #pragma omp parallel for
    for (int idx = 0; idx < m ; idx++) {
        for (int jdx = 0; jdx < n; jdx++) {
            float val = 0.0f;
            for (int kdx = 0; kdx < k; kdx++) {
                val += A[idx * k + kdx] * B[kdx * n + jdx]; // A[idx, kdx] * B[kdx, jdx]
            }
            C[idx * n + jdx] = val;
        }
    }
}

// transpose function for matrix
void transposeMatrix(float* A, int m, int k) {
    #pragma omp parallel for
    for (int idx = 0; idx < m; idx++) {
        for (int jdx = 0; jdx < k; jdx++) {
            std::swap(A[idx * k + jdx], A[jdx * k + idx]);
        }
    }
}

// improved gemm with transposed B matrix
void gemmTransposedOpenmp(const float* A, const float* B, float* C, int m, int n, int k) {
    #pragma omp parallel for
    for (int idx = 0; idx < m ; idx++) {
        for (int jdx = 0; jdx < n; jdx++) {
            float val = 0.0f;
            for (int kdx = 0; kdx < k; kdx++) {
                val += A[idx * k + kdx] * B[jdx * k + kdx]; // A[idx, kdx] * B[jdx, kdx]. 
            }
            C[idx * n + jdx] = val;
        }
    }
}

void runExperiment(int m, int n, int k) {
    // Allocate memory buffer for data
    std::vector<float> aVec(m * k);
    std::vector<float> bVec(n * k);
    std::vector<float> cVec(m * n, 0.0f);

    // initialize matrix with random values
    std::uniform_real_distribution<float> distribution(0.0f, 5.0f);
    std::mt19937 engine;
    std::generate(aVec.begin(), aVec.end(), std::bind(distribution, engine));
    std::generate(bVec.begin(), bVec.end(), std::bind(distribution, engine));

    // Perform experiment on each settings
    double dTime;
    dTime = omp_get_wtime();
    gemmBasic(aVec.data(), bVec.data(), cVec.data(), m, n, k);
    double basicGemmTime = omp_get_wtime() - dTime;

    std::fill(cVec.begin(), cVec.end(), 0.0f);
    dTime = omp_get_wtime();
    gemmBasicOpenmp(aVec.data(), bVec.data(), cVec.data(), m, n, k);
    double openmpGemmTime = omp_get_wtime() - dTime;

    std::fill(cVec.begin(), cVec.end(), 0.0f);
    dTime = omp_get_wtime();
    transposeMatrix(bVec.data(), k, n);
    gemmTransposedOpenmp(aVec.data(), bVec.data(), cVec.data(), m, n, k);
    double openmpTransposedGemmTime = omp_get_wtime() - dTime;

    // Display result of experiments
    std::cout << "case " << k <<" Basic GEMM cpu take " << basicGemmTime << " to complete." << std::endl;
    std::cout << "case " << k <<" Basic openmp GEMM cpu take " << openmpGemmTime << " to complete." << std::endl;
    std::cout << "case " << k <<" Transposed  openmp GEMM cpu take " << openmpTransposedGemmTime << " to complete." << std::endl;
}
// main function for cpu reference. conduct experiment for 64 x 64 x 64, 128 x 128 x 128, 256 x 256 x 256, 512 x 512 x 512
int main(int argc, char** argv) {
    runExperiment(64, 64, 64);
    runExperiment(128, 128, 128);
    runExperiment(256, 256, 256);
    runExperiment(512, 512, 512);
}