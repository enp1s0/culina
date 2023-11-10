NVCC=nvcc
NVCCFLAGS=-std=c++17 -Xcompiler="-Wall -fopenmp"
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80
NVCCFLAGS+=-gencode arch=compute_60,code=sm_60
NVCCFLAGS+=-lcublas -I../../include

TARGETS=gemm.test

all: $(TARGETS)

%.test:%.cu
		$(NVCC) $< -o $@ $(NVCCFLAGS)
		  
clean:
		rm -f $(TARGETS)
