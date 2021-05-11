CC=g++
NVCC=nvcc
CXXFLAGS= -Wextra -std=c++11
CUDAFLAGS= --gpu-architecture compute_72
LDFLAGS=-L/usr/local/cuda/lib64 -lcudart
INCDIRS=-I/usr/local/cuda/include

all: gemm_gpu_fp32 gemm_gpu_fp16

gemm_gpu_fp32: gemm_gpu_fp32.o
	$(CC) $(CXXFLAGS) gemm_gpu_fp32.o -o gemm_gpu_fp32 $(LDFLAGS)

gemm_gpu_fp32.o: gemm_gpu_fp32.cu
	$(NVCC) $(CUDAFLAGS) -c gemm_gpu_fp32.cu 

gemm_gpu_fp16: gemm_gpu_fp16.o
	$(CC) $(CXXFLAGS) gemm_gpu_fp16.o -o gemm_gpu_fp16 $(LDFLAGS)

gemm_gpu_fp16.o: gemm_gpu_fp16.cu
	$(NVCC) $(CUDAFLAGS) -c gemm_gpu_fp16.cu 