CC = g++
CFLAGS = -fopenmp
LDFLAGS = -fopenmp

all: test1

test1: gemm_cpu.o
	$(CC) $(LDFLAGS) -o gemm_cpu gemm_cpu.o 

.cpp.o:
	$(CC) $(CFLAGS) -c $<
