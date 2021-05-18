CC=g++
NVCC=nvcc
CXXFLAGS= -Wextra -std=c++11
CUDAFLAGS= --gpu-architecture compute_72
LDFLAGS=-L/usr/local/cuda/lib64 -lcudart
INCDIRS=-I/usr/local/cuda/include

CCFLAGS = -fopenmp
CCLDFLAGS = -fopenmp 

all: reduction reduction_cpu

reduction: reduction.o
	$(CC) $(CXXFLAGS) reduction.o -o reduction $(LDFLAGS)

reduction.o: reduction.cu
	$(NVCC) $(CUDAFLAGS) -c reduction.cu 

reduction_cpu: reduction_cpu.o
	$(CC) $(CCLDFLAGS) -o reduction_cpu reduction_cpu.o 

reduction_cpu.o: reduction_cpu.cpp 
	$(CC) $(CCFLAGS) -c reduction_cpu.cpp 