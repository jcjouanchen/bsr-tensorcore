CUDA_TOOLKIT := $(abspath $(shell dirname $$(command -v nvcc))/..)
ifeq ($(shell uname -m), aarch64)
ifeq ($(shell uname -s), Linux)
    OS_ARCH_NVRTC := "sbsa-linux"
endif
endif
ifeq ($(shell uname -m), x86_64)
ifeq ($(shell uname -s), Linux)
    OS_ARCH_NVRTC := "x86_64-linux"
endif
endif
# INCS := -I$(CUDA_TOOLKIT)/include #-I${CUSPARSELT_PATH}/include
# LIBS := -lcublas -lcudart -lcusparse -ldl

NVRTC_SHARED := /opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/lib64/libnvrtc.so # ${CUDA_TOOLKIT}/targets/${OS_ARCH_NVRTC}/lib/libnvrtc.so
NVCUSPARSE_SHARED := /opt/nvidia/hpc_sdk/Linux_x86_64/22.7/math_libs/11.7/targets/x86_64-linux/lib/libcusparse.so
INCS         := -I$(CUDA_TOOLKIT)/include -I/global/homes/x/xshen5/MPGNN_ws/CUDALibrarySamples/cuSPARSELt/libcusparse_lt-linux-x86_64-0.4.0.7-archive/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/math_libs/11.7/targets/x86_64-linux/include/
LIBS         := -lcublas -lcudart -lcusparse -ldl ${NVCUSPARSE_SHARED} ${NVRTC_SHARED}

CC = nvcc
FLAGS = -arch=sm_80 -O3 -std=c++11 -w -L/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/math_libs/11.7/targets/x86_64-linux/lib/



spmm: backend/spmm.cu backend/utility.cu 
	$(CC) $(FLAGS) ${INCS} -L/global/homes/x/xshen5/MPGNN_ws/CUDALibrarySamples/cuSPARSELt/libcusparse_lt-linux-x86_64-0.4.0.7-archive/lib -lcusparseLt ${LIBS} main.cu -o spmm

clean:
	rm -f spmm


# CC = nvcc
# FLAGS = -arch=sm_80 -O3 -std=c++11 -w -L/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/math_libs/11.7/targets/x86_64-linux/lib/ -I/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/math_libs/11.7/targets/x86_64-linux/include/
# LINK = -lcublas -lcurand -lcusparse -lcudart

# all: bsrwmma

# bsrwmma: backend/spmm_wmma.cu backend/utility.cu 
# 		$(CC) $(FLAGS) $(LINK) bsr_wmma.cu -o bsrwmma

# bsrwmmatest: backend/spmm_wmma.cu backend/utility.cu 
# 		$(CC) $(FLAGS) ${INCS} -L/global/homes/x/xshen5/MPGNN_ws/CUDALibrarySamples/cuSPARSELt/libcusparse_lt-linux-x86_64-0.4.0.7-archive/lib -lcusparseLt ${LIBS} bsr_wmma.cu -o bsrwmmatest

# TCGemm: TCGemm.cu
# 		$(CC) $(FLAGS) $(LINK)  TCGemm.cu -o TCGemm 