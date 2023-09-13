// cublas
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

// cusparselt
// #include <iostream>
// #include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
// #include <cusparseLt.h>       // cusparseLt header
// #include <cstdio>             // printf
// #include <cstdlib>            // std::rand
// #include "utility.cu"

#include <mma.h>
using namespace nvcuda;

// #define TEST_TIMES 10

// #if TEST_TIMES > 1
//     float alpha = 1.0, beta = 1.0;
// #else
//     float alpha = 1.0, beta = 0.0;
// #endif

__global__ void convertFp32ToFp16 (__half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// reference: https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/tensor-cores/simpleTensorCoreGEMM.cu
__global__ void bsr_wmma(half *a, half *b, float *c, int M, int N, int K, float alpha, float beta,
                                         const int *__restrict__ rowptr, const int *__restrict__ colind)
{
   // Leading dimensions. Packed with no transpositions.
   int lda = M;
   int ldb = K;
   int ldc = M;

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
 
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

   wmma::fill_fragment(acc_frag, 0.0f);

   // infer aRow and bCol from warp
   int aRow = warpM * WMMA_M;
   int bCol = warpN * WMMA_N;

   // loop over colind
   for (int i = rowptr[warpM]; i < rowptr[warpM+1]; i += 1) {

      int aCol = colind[i] * WMMA_K;
      int bRow = colind[i] * WMMA_K;

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {

         // Load the inputs
         wmma::load_matrix_sync(a_frag, a + i * WMMA_M * WMMA_K, WMMA_M);
         wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

         // Perform the matrix multiplication
         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < M && cCol < N) {
      wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);

#pragma unroll
      for(int i=0; i < c_frag.num_elements; i++) {
         c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }

      // Store the output
      wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
   }
}

__global__ void bsr_wmma_half_half_half(__half *a, __half *b, __half *c, int M, int N, int K, __half alpha, __half beta,
                                       const int *__restrict__ rowptr, const int *__restrict__ colind)
{
   // Leading dimensions. Packed with no transpositions.
   int lda = M;
   int ldb = K;
   int ldc = M;

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
 
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

   wmma::fill_fragment(acc_frag, static_cast<__half>(0.0f));

   // infer aRow and bCol from warp
   int aRow = warpM * WMMA_M;
   int bCol = warpN * WMMA_N;

   // loop over colind
   for (int i = rowptr[warpM]; i < rowptr[warpM+1]; i += 1) {

      int aCol = colind[i] * WMMA_K;
      int bRow = colind[i] * WMMA_K;

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {

         // Load the inputs
         wmma::load_matrix_sync(a_frag, a + i * WMMA_M * WMMA_K, WMMA_M);
         wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

         // Perform the matrix multiplication
         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < M && cCol < N) {
      wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);

#pragma unroll
      for(int i=0; i < c_frag.num_elements; i++) {
         c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }

      // Store the output
      wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
   }
}

// double evalCuBLASHGemm(__half *hA, __half *hB, __half *hC, 
//                        int M, int N, int K)
// {
//    // Because CUBLAS uses column major, C^T = B^T * A^T.
//    bool trans_A = false;
//    bool trans_B = false;
//    cublasOperation_t cublas_trans_A = trans_A?CUBLAS_OP_T:CUBLAS_OP_N;
//    cublasOperation_t cublas_trans_B = trans_B?CUBLAS_OP_T:CUBLAS_OP_N;

//    __half* hfA = NULL; 
//    cudaMalloc(&hfA, M*K*sizeof(__half));
//    cudaMemcpy(hfA, hA, M*K*sizeof(__half), cudaMemcpyHostToDevice);

//    __half* hfB = NULL;
//    cudaMalloc(&hfB, K*N*sizeof(__half));
//    cudaMemcpy(hfB, hB, K*N*sizeof(__half), cudaMemcpyHostToDevice);

//    __half* hfC = NULL;
//    cudaMalloc(&hfC, M*N*sizeof(__half));
//    cudaMemcpy(hfC, hC, M*N*sizeof(__half), cudaMemcpyHostToDevice);

//    cublasHandle_t handle;
//    cublasCreate(&handle);

//    // convert alpha, beta to half
//    __half hf_alpha = __float2half(alpha);
//    __half hf_beta = __float2half(beta);

//    //----------------------- 
//    // warm up
//    cublasHgemm(handle, cublas_trans_B, cublas_trans_A, N, M, K,
//    &hf_alpha, hfB, N, hfA, K, &hf_beta, hfC, N);

//    GpuTimer cublas_timer;
//    cublas_timer.Start();
//    for (int i=0; i<TEST_TIMES; i++)
//    {
//       cublasHgemm(handle, cublas_trans_B, cublas_trans_A, N, M, K,
//       &hf_alpha, hfB, N, hfA, K, &hf_beta, hfC, N);
//    }
//    cublas_timer.Stop();
//    double cublas_time = cublas_timer.ElapsedMillis()/TEST_TIMES;
//    //----------------------- 
//    cudaMemcpy(hC, hfC, M*N*sizeof(__half), cudaMemcpyDeviceToHost);

//    cudaFree(hfA);
//    cudaFree(hfB);
//    cudaFree(hfC);

//    return cublas_time;
// }

// double evalCuSPARSELtMatmul(__half *hA, __half *hB, __half *hC, int M, int N, int K)
// {
//    // Host problem definition, row-major order
//    // bigger sizes may require dynamic allocations
//    int m            = M;
//    int n            = N;
//    int k            = K;
//    auto          order        = CUSPARSE_ORDER_ROW;
//    auto          opA          = CUSPARSE_OPERATION_NON_TRANSPOSE;
//    auto          opB          = CUSPARSE_OPERATION_NON_TRANSPOSE;
//    auto          type         = CUDA_R_16F;
//    auto          compute_type = CUSPARSE_COMPUTE_16F;

//    bool     is_rowmajor    = (order == CUSPARSE_ORDER_ROW);
//    bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
//    bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
//    auto     num_A_rows     = (isA_transposed) ? k : m;
//    auto     num_A_cols     = (isA_transposed) ? m : k;
//    auto     num_B_rows     = (isB_transposed) ? n : k;
//    auto     num_B_cols     = (isB_transposed) ? k : n;
//    auto     num_C_rows     = m;
//    auto     num_C_cols     = n;
//    unsigned alignment      = 16;
//    auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows;
//    auto     ldb            = (is_rowmajor) ? num_B_cols : num_B_rows;
//    auto     ldc            = (is_rowmajor) ? num_C_cols : num_C_rows;
//    auto     A_height       = (is_rowmajor) ? num_A_rows : num_A_cols;
//    auto     B_height       = (is_rowmajor) ? num_B_rows : num_B_cols;
//    auto     C_height       = (is_rowmajor) ? num_C_rows : num_C_cols;
//    auto     A_size         = A_height * lda * sizeof(__half);
//    auto     B_size         = B_height * ldb * sizeof(__half);
//    auto     C_size         = C_height * ldc * sizeof(__half);

//    //--------------------------------------------------------------------------
//    // Device memory management
//    __half *dA, *dB, *dC, *dD, *dA_compressed;
//    int    *d_valid;
//    CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
//    CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
//    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )
//    CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(int)) )
//    dD = dC;

//    CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) )
//    CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) )
//    CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )
//    //--------------------------------------------------------------------------
//    cusparseLtHandle_t             handle;
//    cusparseLtMatDescriptor_t      matA, matB, matC;
//    cusparseLtMatmulDescriptor_t   matmul;
//    cusparseLtMatmulAlgSelection_t alg_sel;
//    cusparseLtMatmulPlan_t         plan;
//    cudaStream_t                   stream = nullptr;
//    CHECK_CUSPARSE( cusparseLtInit(&handle) )

//    // matrix descriptor initialization
//    CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(
//                                           &handle, &matA, num_A_rows,
//                                           num_A_cols, lda, alignment,
//                                           type, order,
//                                           CUSPARSELT_SPARSITY_50_PERCENT) )
//    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
//                                           &handle, &matB, num_B_rows,
//                                           num_B_cols, ldb, alignment,
//                                           type, order) )
//    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
//                                           &handle, &matC, num_C_rows,
//                                           num_C_cols, ldc, alignment,
//                                           type, order) )
//    // matmul, algorithm selection, and plan initialization
//    CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(
//                                           &handle, &matmul, opA, opB,
//                                           &matA, &matB, &matC, &matC,
//                                           compute_type) )
//    CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit(
//                                           &handle, &alg_sel, &matmul,
//                                           CUSPARSELT_MATMUL_ALG_DEFAULT) )
//    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel))

//    //--------------------------------------------------------------------------
//    // Prune the A matrix (in-place) and check the correctness
//    CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dA, dA,
//                                        CUSPARSELT_PRUNE_SPMMA_TILE, stream) )
//    CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul, dA,
//                                              d_valid, stream) )
//    int is_valid;
//    CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(int),
//                               cudaMemcpyDeviceToHost, stream) )
//    CHECK_CUDA( cudaStreamSynchronize(stream) )
//    if (is_valid != 0) {
//       std::printf("!!!! The matrix has been pruned in a wrong way. "
//                   "cusparseLtMatmul will not provide correct results\n");
//       return EXIT_FAILURE;
//    }

//    //--------------------------------------------------------------------------
//    // Compress the A matrix
//    size_t compressed_size, compressed_buffer_size;
//    void*  dA_compressedBuffer;
//    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan,
//                                                 &compressed_size,
//                                                 &compressed_buffer_size) )
//    CHECK_CUDA( cudaMalloc((void**) &dA_compressed, compressed_size) )
//    CHECK_CUDA( cudaMalloc((void**) &dA_compressedBuffer,
//                         compressed_buffer_size) )

//    CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed,
//                                           dA_compressedBuffer,stream) )
//    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//    // Search the best kernel
//    int           num_streams = 0;
//    cudaStream_t* streams     = nullptr;
//    CHECK_CUSPARSE( cusparseLtMatmulSearch(&handle, &plan, &alpha,
//                                           dA_compressed, dB, &beta,
//                                           dC, dD, nullptr,
//                                           streams, num_streams) )
                                          
//    // otherwise, it is possible to set it directly:
//    int alg = 0;
//    CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel,
//                                                    CUSPARSELT_MATMUL_ALG_CONFIG_ID,
//                                                    &alg, sizeof(alg)))
//    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//    size_t workspace_size;
//    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel))

//    CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &plan,
//                                                 &workspace_size))
//    void* d_workspace;
//    CHECK_CUDA( cudaMalloc((void**) &d_workspace, workspace_size) )


//    // ===========================================================
//    // warm up
//    cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
//                   &beta, dC, dD, d_workspace, streams,
//                   num_streams);


//    float milliseconds = 0;
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start);
//    for (int i=0; i<TEST_TIMES; i++)
//    {
//       // Perform the matrix multiplication
//       cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
//                      &beta, dC, dD, d_workspace, streams,
//                      num_streams);
//    }
//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    milliseconds = 0;
//    cudaEventElapsedTime(&milliseconds,start,stop);
//    double cusparselt_time = (milliseconds)/double(TEST_TIMES);
//    // ===========================================================

//    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//    // destroy plan and handle
//    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matA) )
//    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matB) )
//    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matC) )
//    CHECK_CUSPARSE( cusparseLtMatmulPlanDestroy(&plan) )
//    CHECK_CUSPARSE( cusparseLtDestroy(&handle) )

//    //--------------------------------------------------------------------------
//    // device memory deallocation
//    CHECK_CUDA( cudaFree(dA_compressed) )
//    CHECK_CUDA( cudaFree(dA) )
//    CHECK_CUDA( cudaFree(dB) )
//    CHECK_CUDA( cudaFree(dC) )
//    CHECK_CUDA( cudaFree(d_valid) )
//    CHECK_CUDA( cudaFree(d_workspace) )
//    CHECK_CUDA( cudaFree(dA_compressedBuffer) )

//    return cusparselt_time;
// }


// double evalCuSPARSESpMMBlockedell(int *ell_columns, __half *ell_values, int ell_width,
//                                  __half *hB, __half *hC,
//                                  int M, int N, int K, int block_dim=16)
// {
//    cusparseSpMMAlg_t alg = CUSPARSE_SPMM_BLOCKED_ELL_ALG1;//CUSPARSE_SPMM_ALG_DEFAULT;

//    // Host problem definition
//    int   A_num_rows      = M;
//    int   A_num_cols      = K;
//    int   A_ell_blocksize = block_dim;
//    int   A_ell_cols      = ell_width;
//    int   A_num_blocks    = A_ell_cols * A_num_rows /
//                         (A_ell_blocksize * A_ell_blocksize);
//    int   B_num_rows      = A_num_cols;
//    int   B_num_cols      = N;
//    int   ldb             = B_num_rows;
//    int   ldc             = A_num_rows;
//    int   B_size          = ldb * B_num_cols;
//    int   C_size          = ldc * B_num_cols;
//    // int   *hA_columns     = h_ell_columns;
//    // __half *hA_values     = h_ell_values;

//    // Device memory management
//    int    *dA_columns;
//    __half *dA_values, *dB, *dC;
//    dA_columns = ell_columns;
//    dA_values = ell_values;
//    CHECK_CUDA( cudaMalloc((void**) &dB, B_size * sizeof(__half)) )
//    CHECK_CUDA( cudaMalloc((void**) &dC, C_size * sizeof(__half)) )
//    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(__half),
//                            cudaMemcpyHostToDevice) )
//    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(__half),
//                            cudaMemcpyHostToDevice) )

//    //--------------------------------------------------------------------------
//    // CUSPARSE APIs
//    cusparseHandle_t     bhandle = NULL;
//    cusparseSpMatDescr_t bmatA;
//    cusparseDnMatDescr_t bmatB, bmatC;
//    void*                bdBuffer    = NULL;
//    size_t               bbufferSize = 0;
//    CHECK_CUSPARSE( cusparseCreate(&bhandle) )

//    // Create sparse matrix A in blocked ELL format
//    CHECK_CUSPARSE( cusparseCreateBlockedEll(&bmatA,
//                                           A_num_rows, A_num_cols, A_ell_blocksize,
//                                           A_ell_cols, dA_columns, dA_values,
//                                           CUSPARSE_INDEX_32I,
//                                           CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F) )
//    // Create dense matrix B
//    CHECK_CUSPARSE( cusparseCreateDnMat(&bmatB, A_num_cols, B_num_cols, ldb, dB,
//                                        CUDA_R_16F, CUSPARSE_ORDER_COL) )
//    // Create dense matrix C
//    CHECK_CUSPARSE( cusparseCreateDnMat(&bmatC, A_num_rows, B_num_cols, ldc, dC,
//                                        CUDA_R_16F, CUSPARSE_ORDER_COL) )
//    // allocate an external buffer if needed
//    CHECK_CUSPARSE( cusparseSpMM_bufferSize(bhandle,
//                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
//                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
//                                              &alpha, bmatA, bmatB, &beta, bmatC, CUDA_R_16F,
//                                              alg, &bbufferSize) )
//    CHECK_CUDA( cudaMalloc(&bdBuffer, bbufferSize) )

//    // execute SpMM
//    // warm-up
//    cusparseSpMM(bhandle,
//                CUSPARSE_OPERATION_NON_TRANSPOSE,
//                CUSPARSE_OPERATION_NON_TRANSPOSE,
//                &alpha, bmatA, bmatB, &beta, bmatC, CUDA_R_16F,
//                alg, bdBuffer);

//    GpuTimer cusparse_timer;
//    cusparse_timer.Start();
//    for (int i=0; i<TEST_TIMES; i++)
//    {
//       cusparseSpMM(bhandle,
//                   CUSPARSE_OPERATION_NON_TRANSPOSE,
//                   CUSPARSE_OPERATION_NON_TRANSPOSE,
//                   &alpha, bmatA, bmatB, &beta, bmatC, CUDA_R_16F,
//                   alg, bdBuffer);
//    }
//    cusparse_timer.Stop();
//    double cusparse_time = cusparse_timer.ElapsedMillis()/TEST_TIMES;

//    // destroy matrix/vector descriptors
//    CHECK_CUSPARSE( cusparseDestroySpMat(bmatA) )
//    CHECK_CUSPARSE( cusparseDestroyDnMat(bmatB) )
//    CHECK_CUSPARSE( cusparseDestroyDnMat(bmatC) )
//    CHECK_CUSPARSE( cusparseDestroy(bhandle) )

//    //--------------------------------------------------------------------------
//    // device result check
//    CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(__half),
//                         cudaMemcpyDeviceToHost) )

//    // device memory deallocation
//    CHECK_CUDA( cudaFree(bdBuffer) )
//    CHECK_CUDA( cudaFree(dA_columns) )
//    CHECK_CUDA( cudaFree(dA_values) )
//    CHECK_CUDA( cudaFree(dB) )
//    CHECK_CUDA( cudaFree(dC) )

//    return cusparse_time;
// }

// double evalCustomBsrwmma(int *bsrRowPtr, int *bsrColInd, __half *hbsrVal,
//                         __half *hB, __half *hC,
//                         int M, int N, int K, int block_dim=16)
// {
//    //  // init C (result storage)
//    //  cudaMalloc(&fC, nrows * nBcols * sizeof(float));
//    //  cudaMemset(fC, 0, nrows * nBcols * sizeof(float));

//    //  // // define thread blocks & thread
//    //  // int tiles_per_tb = (1024 / nBcols);
//    //  // dim3 BLOCKS = dim3((nblockrows+tiles_per_tb-1)/tiles_per_tb);
//    //  // int THREADS = 1024;
//    // half *tA_fp16;
//    // half *fB_fp16;
//    // cudaMalloc((void**)&tA_fp16, nblocks*tiledim*tiledim * sizeof(half));
//    // cudaMalloc((void**)&fB_fp16, ncols * nBcols * sizeof(half));
//    // convertFp32ToFp16 <<< (nblocks*tiledim*tiledim + 255) / 256, 256 >>> (tA_fp16, tA, nblocks*tiledim*tiledim);
//    // convertFp32ToFp16 <<< (ncols * nBcols + 255) / 256, 256 >>> (fB_fp16, fB, ncols * nBcols);
//    // __half hf_alpha = __float2half(alpha);
//    // __half hf_beta = __float2half(beta);

//    // blockDim.x must be a multple of warpSize
//    // 128x4 means we have 16 (4x4) warps and a block computes a 64x64 output tile
//    dim3 gridDim;
//    dim3 blockDim;

//    blockDim.x = 128;
//    blockDim.y = 4;
//    gridDim.x = (M + (16 * blockDim.x / 32 - 1)) / (16 * blockDim.x / 32);
//    gridDim.y = (N + 16 * blockDim.y - 1) / (16 * blockDim.y);

//    // ------
//    // warm up
//    bsr_wmma<<<gridDim, blockDim>>>(hbsrVal, hB, hC, 
//                                    M, N, K, 
//                                    alpha, beta,
//                                    bsrRowPtr, bsrColInd);

//    GpuTimer bsrwmma_timer;
//    bsrwmma_timer.Start();
//    for (int i = 0; i < TEST_TIMES; i++)
//    {
//       bsr_wmma<<<gridDim, blockDim>>>(hbsrVal, hB, hC, 
//                                        M, N, K, 
//                                        alpha, beta,
//                                        bsrRowPtr, bsrColInd);
//    }
//    bsrwmma_timer.Stop();
//    double bsrwmma_time = bsrwmma_timer.ElapsedMillis() / double(TEST_TIMES);
//    // ------

//    // cudaFree(tA_fp16);
//    // cudaFree(fB_fp16);

//    // result_bsrwmmafloat = (float *)malloc(nrows * nBcols * sizeof(float));
//    // cudaMemcpy(result_bsrwmmafloat, fC, nrows * nBcols * sizeof(float), cudaMemcpyDeviceToHost);
//    // cudaFree(fC);

//    return bsrwmma_time;
// }

// double evalCustomBsrwmmabackup(int *bsrRowPtr, int *bsrColInd, __half *hbsrVal,
//                               __half *hB, __half *hC,
//                               int M, int N, int K, int block_dim=16)
// {
//     // init C (result storage)
//     float *fC;
//     cudaMalloc(&fC, M * N * sizeof(float));
//     cudaMemset(fC, 0, M * N * sizeof(float));

//    //  // // define thread blocks & thread
//    //  // int tiles_per_tb = (1024 / nBcols);
//    //  // dim3 BLOCKS = dim3((nblockrows+tiles_per_tb-1)/tiles_per_tb);
//    //  // int THREADS = 1024;
//    // half *tA_fp16;
//    // half *fB_fp16;
//    // cudaMalloc((void**)&tA_fp16, nblocks*tiledim*tiledim * sizeof(half));
//    // cudaMalloc((void**)&fB_fp16, ncols * nBcols * sizeof(half));
//    // convertFp32ToFp16 <<< (nblocks*tiledim*tiledim + 255) / 256, 256 >>> (tA_fp16, bsrVal, nblocks*tiledim*tiledim);
//    // convertFp32ToFp16 <<< (ncols * nBcols + 255) / 256, 256 >>> (fB_fp16, fB, ncols * nBcols);

//    // blockDim.x must be a multple of warpSize
//    // 128x4 means we have 16 (4x4) warps and a block computes a 64x64 output tile
//    dim3 gridDim;
//    dim3 blockDim;

//    blockDim.x = 128;
//    blockDim.y = 4;
//    gridDim.x = (M + (16 * blockDim.x / 32 - 1)) / (16 * blockDim.x / 32);
//    gridDim.y = (N + 16 * blockDim.y - 1) / (16 * blockDim.y);

//    // ------
//    // warm up
//    bsr_wmma_half_half_float<<<gridDim, blockDim>>>(hbsrVal, hB, fC, 
//                                    M, N, K, 
//                                    alpha, beta,
//                                    bsrRowPtr, bsrColInd);

//    GpuTimer bsrwmma_timer;
//    bsrwmma_timer.Start();
//    for (int i = 0; i < TEST_TIMES; i++)
//    {
//       bsr_wmma_half_half_float<<<gridDim, blockDim>>>(hbsrVal, hB, fC, 
//                                        M, N, K, 
//                                        alpha, beta,
//                                        bsrRowPtr, bsrColInd);
//    }
//    bsrwmma_timer.Stop();
//    double bsrwmma_time = bsrwmma_timer.ElapsedMillis() / double(TEST_TIMES);
//    // ------

//    // cudaFree(tA_fp16);
//    // cudaFree(fB_fp16);

//    // result_bsrwmmafloat = (float *)malloc(nrows * nBcols * sizeof(float));
//    // cudaMemcpy(result_bsrwmmafloat, fC, nrows * nBcols * sizeof(float), cudaMemcpyDeviceToHost);
//    cudaFree(fC);

//    return bsrwmma_time;
// }