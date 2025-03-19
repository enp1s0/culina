NVCC=nvcc
NVCCFLAGS=-std=c++20 -Xcompiler="-Wall -fopenmp"
NVCCFLAGS+=-gencode arch=compute_90,code=sm_90
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80
INCDIR=../../include
NVCCFLAGS+=-lcublas -I$(INCDIR)
HEADERS=$(shell find $(INCDIR) -name '*.cuh' -o -name '*.hpp' -o -name '*.h')

TARGETS=gemm.test gemv.test dot.test

all: $(TARGETS)

%.test:%.cu $(HEADERS)
		$(NVCC) $< -o $@ $(NVCCFLAGS)
		  
clean:
		rm -f $(TARGETS)
