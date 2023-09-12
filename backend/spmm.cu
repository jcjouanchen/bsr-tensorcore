#include "utility.cu"

// cublas
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

// cusparselt
#include <iostream>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparseLt.h>       // cusparseLt header
#include <cstdio>             // printf
#include <cstdlib>            // std::rand


#include <mma.h>
using namespace nvcuda;

#define TEST_TIMES 10

#if TEST_TIMES > 1
    float alpha = 1.0, beta = 1.0;
#else
    float alpha = 1.0, beta = 0.0;
#endif

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

// m16n16k16
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

// m16n8k16
__global__ void bsr_mmasp_half_half_half(__half *a, __half *b, __half *c, int M, int N, int K, __half alpha, __half beta,
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

         // create register for each thread
         // half frag_A[4]; // two b32 registrs, 4 half non-zero elements, 16 dense 
         // half frag_B[4]; // two f16x2 registers, 8 half dense elements
         // half frag_D[4]; //result(fp32) 4 f32 registers
         // load E
         uint32_t frag_E[1]; // A .b32 register containing 16 2-bit vectors to for indexing non-zero of A
         for(int i=0; i < 1 ; i ++){
            frag_E[i] = meteE[i];
         }

         // create pointer to frags
         uint32_t const *A = reinterpret_cast<uint32_t const *>(&a_frag[0]);
         uint32_t const *B = reinterpret_cast<uint32_t const *>(&b_frag[0]);
         uint32_t *C = reinterpret_cast<uint32_t *>(&acc_frag[0]);
         uint32_t *D = reinterpret_cast<uint32_t *>(&c_frag[0]);; 
         uint32_t const *E = reinterpret_cast<uint32_t const*>(&frag_E[0]); ;

         // Perform the matrix multiplication
         // wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

         // D = A*B+C
         asm volatile("mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, "
         "{%2,%3}, {%4,%5}, {%6,%7}, %8, 0x0;\n"
         : "=r"(D[0]), "=r"(D[1]) //, "=f"(D[2]), "=f"(D[3])
         : "r"(A[0]), "r"(A[1]), 
            "r"(B[0]), "r"(B[1]),
            "r"(C[0]), "r"(C[1]), //"f"(C[2]), "f"(C[3]),
            "r"(E[0]));
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

double evalCuBLASHGemm(__half *hA, __half *hB, __half *hC, 
                       int M, int N, int K)
{
   // Because CUBLAS uses column major, C^T = B^T * A^T.
   bool trans_A = false;
   bool trans_B = true;
   cublasOperation_t cublas_trans_A = trans_A?CUBLAS_OP_T:CUBLAS_OP_N;
   cublasOperation_t cublas_trans_B = trans_B?CUBLAS_OP_T:CUBLAS_OP_N;

   __half* hfA = NULL; 
   cudaMalloc(&hfA, M*K*sizeof(__half));
   cudaMemcpy(hfA, hA, M*K*sizeof(__half), cudaMemcpyHostToDevice);

   __half* hfB = NULL;
   cudaMalloc(&hfB, K*N*sizeof(__half));
   cudaMemcpy(hfB, hB, K*N*sizeof(__half), cudaMemcpyHostToDevice);

   __half* hfC = NULL;
   cudaMalloc(&hfC, M*N*sizeof(__half));
   cudaMemcpy(hfC, hC, M*N*sizeof(__half), cudaMemcpyHostToDevice);

   cublasHandle_t handle;
   cublasCreate(&handle);

   // convert alpha, beta to half
   __half hf_alpha = __float2half(alpha);
   __half hf_beta = __float2half(beta);

   //----------------------- 
   // warm up
   cublasHgemm(handle, cublas_trans_B, cublas_trans_A, N, M, K,
   &hf_alpha, hfB, N, hfA, K, &hf_beta, hfC, N);
   // cublasHgemm(handle, cublas_trans_A, cublas_trans_B, M, N, K,
   // &hf_alpha, hfA, M, hfB, K, &hf_beta, hfC, M);

   GpuTimer cublas_timer;
   cublas_timer.Start();
   for (int i=0; i<TEST_TIMES; i++)
   {
      cublasHgemm(handle, cublas_trans_B, cublas_trans_A, N, M, K,
                  &hf_alpha, hfB, N, hfA, K, &hf_beta, hfC, N);
      // cublasHgemm(handle, cublas_trans_A, cublas_trans_B, M, N, K,
      // &hf_alpha, hfA, M, hfB, K, &hf_beta, hfC, M);
   }
   cublas_timer.Stop();
   double cublas_time = cublas_timer.ElapsedMillis()/TEST_TIMES;
   //----------------------- 
   cudaMemcpy(hC, hfC, M*N*sizeof(__half), cudaMemcpyDeviceToHost);
   __half *hC_trans = (__half*)malloc(M * N * sizeof(__half));
   transpose(hC_trans, hC, M, N);
   cudaMemcpy(hC, hC_trans, M*N*sizeof(__half), cudaMemcpyHostToHost);
   free(hC_trans);

   // cudaFree(hfA);
   // cudaFree(hfB);
   // cudaFree(hfC);

   return cublas_time;
}

double evaluCuBLASGemmex(__half *hA, __half *hB, __half *hC, 
                        int M, int N, int K)
{
   cublasHandle_t cublasH = NULL;
   // cudaStream_t stream = NULL;

   const int m = M;
   const int n = N;
   const int k = K;
   const int lda = m;
   const int ldb = k;
   const int ldc = m;
   /*
   *   A = | 1.0 | 2.0 |
   *       | 3.0 | 4.0 |
   *
   *   B = | 5.0 | 6.0 |
   *       | 7.0 | 8.0 |
   */

   // const std::vector<data_type> A = {1.0, 3.0, 2.0, 4.0};
   // const std::vector<data_type> B = {5.0, 7.0, 6.0, 8.0};
   // std::vector<data_type> C(m * n);
   // const data_type alpha = 1.0;
   // const data_type beta = 0.0;

   __half *d_A = nullptr;
   __half *d_B = nullptr;
   __half *d_C = nullptr;

   cublasOperation_t transa = CUBLAS_OP_N;
   cublasOperation_t transb = CUBLAS_OP_N;

   // printf("A\n");
   // print_matrix(m, k, A.data(), lda);
   // printf("=====\n");

   // printf("B\n");
   // print_matrix(k, n, B.data(), ldb);
   // printf("=====\n");

   __half hf_alpha = __float2half(alpha);
   __half hf_beta = __float2half(beta);

   /* step 1: create cublas handle, bind a stream */
   cublasCreate(&cublasH);

   // CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
   // CUBLAS_CHECK(cublasSetStream(cublasH, stream));

   /* step 2: copy data to device */
   CHECK_CUDA( cudaMalloc((void **)&d_A, sizeof(__half) * M * K) )
   CHECK_CUDA( cudaMalloc((void **)&d_B, sizeof(__half) * K * N) )
   CHECK_CUDA( cudaMalloc((void **)&d_C, sizeof(__half) * M * N) )

   __half *hA_trans = (__half*)malloc(M * K * sizeof(__half));
   transpose(hA_trans, hA, M, K);

   CHECK_CUDA( cudaMemcpy(d_A, hA_trans, sizeof(__half) * M * K, cudaMemcpyHostToDevice) )
   CHECK_CUDA( cudaMemcpy(d_B, hB, sizeof(__half) * K * N, cudaMemcpyHostToDevice) )

   /* step 3: compute */
   cublasGemmEx(cublasH, transa, transb, m, n, k, &hf_alpha, d_A, CUDA_R_16F, lda, d_B,
   CUDA_R_16F, ldb, &hf_beta, d_C, CUDA_R_16F, ldc,
   CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

   GpuTimer cublas_timer;
   cublas_timer.Start();
   for (int i=0; i<TEST_TIMES; i++)
   {
      cublasGemmEx(cublasH, transa, transb, m, n, k, &hf_alpha, d_A, CUDA_R_16F, lda, d_B,
      CUDA_R_16F, ldb, &hf_beta, d_C, CUDA_R_16F, ldc,
      CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
   }
   cublas_timer.Stop();
   double cublas_time = cublas_timer.ElapsedMillis()/TEST_TIMES;

   /* step 4: copy data to host */
   CHECK_CUDA( cudaMemcpy(hC, d_C, sizeof(__half) * M * N, cudaMemcpyDeviceToHost) )

   // CUDA_CHECK(cudaStreamSynchronize(stream));

   /*
   *   C = | 19.0 | 22.0 |
   *       | 43.0 | 50.0 |
   */

   // printf("C\n");
   // print_matrix(m, n, C.data(), ldc);
   // printf("=====\n");

   /* free resources */
   // CUDA_CHECK(cudaFree(d_A));
   // CUDA_CHECK(cudaFree(d_B));
   // CUDA_CHECK(cudaFree(d_C));

   cublasDestroy(cublasH);

   // CUDA_CHECK(cudaStreamDestroy(stream));

   // CUDA_CHECK(cudaDeviceReset());
   return cublas_time;
}

double evalCuSPARSELtMatmul(__half *hA, __half *hB, __half *hC, int M, int N, int K)
{
   // Host problem definition, row-major order
   // bigger sizes may require dynamic allocations
   int m            = M;
   int n            = N;
   int k            = K;
   auto          order        = CUSPARSE_ORDER_ROW;
   auto          opA          = CUSPARSE_OPERATION_NON_TRANSPOSE;
   auto          opB          = CUSPARSE_OPERATION_NON_TRANSPOSE;
   auto          type         = CUDA_R_16F;
   auto          compute_type = CUSPARSE_COMPUTE_16F;

   bool     is_rowmajor    = (order == CUSPARSE_ORDER_ROW);
   bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
   bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
   auto     num_A_rows     = (isA_transposed) ? k : m;
   auto     num_A_cols     = (isA_transposed) ? m : k;
   auto     num_B_rows     = (isB_transposed) ? n : k;
   auto     num_B_cols     = (isB_transposed) ? k : n;
   auto     num_C_rows     = m;
   auto     num_C_cols     = n;
   unsigned alignment      = 16;
   auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows;
   auto     ldb            = (is_rowmajor) ? num_B_cols : num_B_rows;
   auto     ldc            = (is_rowmajor) ? num_C_cols : num_C_rows;
   auto     A_height       = (is_rowmajor) ? num_A_rows : num_A_cols;
   auto     B_height       = (is_rowmajor) ? num_B_rows : num_B_cols;
   auto     C_height       = (is_rowmajor) ? num_C_rows : num_C_cols;
   auto     A_size         = A_height * lda * sizeof(__half);
   auto     B_size         = B_height * ldb * sizeof(__half);
   auto     C_size         = C_height * ldc * sizeof(__half);

   //--------------------------------------------------------------------------
   // Device memory management
   __half *dA, *dB, *dC, *dD, *dA_compressed;
   int    *d_valid;
   CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
   CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
   CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )
   CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(int)) )
   dD = dC;

   CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) )
   CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) )
   CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )
   //--------------------------------------------------------------------------
   cusparseLtHandle_t             handle;
   cusparseLtMatDescriptor_t      matA, matB, matC;
   cusparseLtMatmulDescriptor_t   matmul;
   cusparseLtMatmulAlgSelection_t alg_sel;
   cusparseLtMatmulPlan_t         plan;
   cudaStream_t                   stream = nullptr;
   CHECK_CUSPARSE( cusparseLtInit(&handle) )

   // matrix descriptor initialization
   CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(
                                          &handle, &matA, num_A_rows,
                                          num_A_cols, lda, alignment,
                                          type, order,
                                          CUSPARSELT_SPARSITY_50_PERCENT) )
   CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                          &handle, &matB, num_B_rows,
                                          num_B_cols, ldb, alignment,
                                          type, order) )
   CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                          &handle, &matC, num_C_rows,
                                          num_C_cols, ldc, alignment,
                                          type, order) )
   // matmul, algorithm selection, and plan initialization
   CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(
                                          &handle, &matmul, opA, opB,
                                          &matA, &matB, &matC, &matC,
                                          compute_type) )
   CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit(
                                          &handle, &alg_sel, &matmul,
                                          CUSPARSELT_MATMUL_ALG_DEFAULT) )
   CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel))

   //--------------------------------------------------------------------------
   // Prune the A matrix (in-place) and check the correctness
   CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dA, dA,
                                       CUSPARSELT_PRUNE_SPMMA_TILE, stream) )
   CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul, dA,
                                             d_valid, stream) )
   int is_valid;
   CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(int),
                              cudaMemcpyDeviceToHost, stream) )
   CHECK_CUDA( cudaStreamSynchronize(stream) )
   if (is_valid != 0) {
      std::printf("!!!! The matrix has been pruned in a wrong way. "
                  "cusparseLtMatmul will not provide correct results\n");
      return EXIT_FAILURE;
   }

   //--------------------------------------------------------------------------
   // Compress the A matrix
   size_t compressed_size, compressed_buffer_size;
   void*  dA_compressedBuffer;
   CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan,
                                                &compressed_size,
                                                &compressed_buffer_size) )
   CHECK_CUDA( cudaMalloc((void**) &dA_compressed, compressed_size) )
   CHECK_CUDA( cudaMalloc((void**) &dA_compressedBuffer,
                        compressed_buffer_size) )

   CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed,
                                          dA_compressedBuffer,stream) )
   //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   // Search the best kernel
   int           num_streams = 0;
   cudaStream_t* streams     = nullptr;

   CHECK_CUSPARSE( cusparseLtMatmulSearch(&handle, &plan, &alpha,
                                          dA_compressed, dB, &beta,
                                          dC, dD, nullptr,
                                          streams, num_streams) )
                                          
   // otherwise, it is possible to set it directly:
   int alg = 0;
   CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel,
                                                   CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                                   &alg, sizeof(alg)))
   //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   size_t workspace_size;
   CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel))

   CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &plan,
                                                &workspace_size))
   void* d_workspace;
   CHECK_CUDA( cudaMalloc((void**) &d_workspace, workspace_size) )


   // ===========================================================
   // warm up
   cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
                  &beta, dC, dD, d_workspace, streams,
                  num_streams);


   float milliseconds = 0;
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start);
   for (int i=0; i<TEST_TIMES; i++)
   {
      // Perform the matrix multiplication
      cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
                        &beta, dC, dD, d_workspace, streams,
                        num_streams);
   }
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   milliseconds = 0;
   cudaEventElapsedTime(&milliseconds,start,stop);
   double cusparselt_time = (milliseconds)/double(TEST_TIMES);
   // ===========================================================
   cudaMemcpy(hC, dC, M*N*sizeof(__half), cudaMemcpyDeviceToHost);

   //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   // destroy plan and handle
   CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matA) )
   CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matB) )
   CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matC) )
   CHECK_CUSPARSE( cusparseLtMatmulPlanDestroy(&plan) )
   CHECK_CUSPARSE( cusparseLtDestroy(&handle) )

   //--------------------------------------------------------------------------
   // device memory deallocation
   CHECK_CUDA( cudaFree(dA_compressed) )
   // CHECK_CUDA( cudaFree(dA) )
   // CHECK_CUDA( cudaFree(dB) )
   // CHECK_CUDA( cudaFree(dC) )
   CHECK_CUDA( cudaFree(d_valid) )
   CHECK_CUDA( cudaFree(d_workspace) )
   CHECK_CUDA( cudaFree(dA_compressedBuffer) )

   return cusparselt_time;
}


double evalCuSPARSESpMMBlockedell(int *ell_columns, __half *ell_values, int ell_width,
                                 __half *hB, __half *hC,
                                 int M, int N, int K, int block_dim=16)
{
   cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT; // CUSPARSE_SPMM_BLOCKED_ELL_ALG1;

   // Host problem definition
   int   A_num_rows      = M;
   int   A_num_cols      = K;
   int   A_ell_blocksize = block_dim;
   int   A_ell_cols      = ell_width;
   int   A_num_blocks    = A_ell_cols * A_num_rows /
                        (A_ell_blocksize * A_ell_blocksize);
   int   B_num_rows      = A_num_cols;
   int   B_num_cols      = N;
   int   ldb             = B_num_rows;
   int   ldc             = A_num_rows;
   int   B_size          = ldb * B_num_cols;
   int   C_size          = ldc * B_num_cols;
   // int   *hA_columns     = h_ell_columns;
   // __half *hA_values     = h_ell_values;

   // Device memory management
   int    *dA_columns;
   __half *dA_values, *dB, *dC;
   dA_columns = ell_columns;
   dA_values = ell_values;

   // CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_num_blocks * sizeof(int)) )
   // CHECK_CUDA( cudaMalloc((void**) &dA_values,
   //                               A_ell_cols * A_num_rows * sizeof(__half)) )
   // CHECK_CUDA( cudaMemcpy(dA_columns, ell_columns,
   //                      A_num_blocks * sizeof(int),
   //                      cudaMemcpyDeviceToDevice) )
   // CHECK_CUDA( cudaMemcpy(dA_values, ell_values,
   //                      A_ell_cols * A_num_rows * sizeof(__half),
   //                      cudaMemcpyDeviceToDevice) )

   CHECK_CUDA( cudaMalloc((void**) &dB, B_size * sizeof(__half)) )
   CHECK_CUDA( cudaMalloc((void**) &dC, C_size * sizeof(__half)) )
   CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(__half),
                           cudaMemcpyHostToDevice) )
   CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(__half),
                           cudaMemcpyHostToDevice) )

   //--------------------------------------------------------------------------
   // CUSPARSE APIs
   cusparseHandle_t     bhandle = NULL;
   cusparseSpMatDescr_t bmatA;
   cusparseDnMatDescr_t bmatB, bmatC;
   void*                bdBuffer    = NULL;
   size_t               bbufferSize = 0;
   CHECK_CUSPARSE( cusparseCreate(&bhandle) )

   // Create sparse matrix A in blocked ELL format
   CHECK_CUSPARSE( cusparseCreateBlockedEll(&bmatA,
                                             A_num_rows, A_num_cols, A_ell_blocksize,
                                             A_ell_cols, dA_columns, dA_values,
                                             CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F) )
   // Create dense matrix B
   CHECK_CUSPARSE( cusparseCreateDnMat(&bmatB, A_num_cols, B_num_cols, ldb, dB,
                                       CUDA_R_16F, CUSPARSE_ORDER_COL) )
   // Create dense matrix C
   CHECK_CUSPARSE( cusparseCreateDnMat(&bmatC, A_num_rows, B_num_cols, ldc, dC,
                                       CUDA_R_16F, CUSPARSE_ORDER_COL) )
   
   // allocate an external buffer if needed
   __half hf_alpha = __float2half(alpha);
   __half hf_beta = __float2half(beta);

   CHECK_CUSPARSE( cusparseSpMM_bufferSize(bhandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &hf_alpha, bmatA, bmatB, &hf_beta, bmatC, CUDA_R_16F,
                                           alg, &bbufferSize) )
   CHECK_CUDA( cudaMalloc(&bdBuffer, bbufferSize) )

   // execute SpMM
   // warm-up
   CHECK_CUSPARSE( cusparseSpMM(bhandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &hf_alpha, bmatA, bmatB, &hf_beta, bmatC, CUDA_R_16F,
                                 alg, bdBuffer) )

   GpuTimer cusparse_timer;
   cusparse_timer.Start();
   for (int i=0; i<TEST_TIMES; i++)
   {
      CHECK_CUSPARSE( cusparseSpMM(bhandle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &hf_alpha, bmatA, bmatB, &hf_beta, bmatC, CUDA_R_16F,
                                    alg, bdBuffer) )
   }
   cusparse_timer.Stop();
   double cusparse_time = cusparse_timer.ElapsedMillis()/TEST_TIMES;

   // destroy matrix/vector descriptors
   CHECK_CUSPARSE( cusparseDestroySpMat(bmatA) )
   CHECK_CUSPARSE( cusparseDestroyDnMat(bmatB) )
   CHECK_CUSPARSE( cusparseDestroyDnMat(bmatC) )
   CHECK_CUSPARSE( cusparseDestroy(bhandle) )

   //--------------------------------------------------------------------------
   // device result check
   CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(__half), cudaMemcpyDeviceToHost) )

   // device memory deallocation
   CHECK_CUDA( cudaFree(bdBuffer) )
   CHECK_CUDA( cudaFree(dA_columns) )
   CHECK_CUDA( cudaFree(dA_values) )
   // CHECK_CUDA( cudaFree(dB) )
   // CHECK_CUDA( cudaFree(dC) )

   return cusparse_time;
}

double evalCustomBsrwmma(int *bsrRowPtr, int *bsrColInd, __half *hbsrVal,
                        __half *hB, __half *hC,
                        int M, int N, int K, int block_dim=16)
{
   __half *dB, *dC;
   cudaMalloc((void**)&dB, K * N * sizeof(__half));
   cudaMalloc((void**)&dC, M * N * sizeof(__half));
   cudaMemcpy(dB, hB, K * N * sizeof(__half), cudaMemcpyHostToDevice);
   cudaMemcpy(dC, hC, M * N * sizeof(__half), cudaMemcpyHostToDevice);

   __half hf_alpha = __float2half(alpha);
   __half hf_beta = __float2half(beta);

   // blockDim.x must be a multple of warpSize
   // 128x4 means we have 16 (4x4) warps and a block computes a 64x64 output tile
   dim3 gridDim;
   dim3 blockDim;

   blockDim.x = 128;
   blockDim.y = 4;
   gridDim.x = (M + (16 * blockDim.x / 32 - 1)) / (16 * blockDim.x / 32);
   gridDim.y = (N + 16 * blockDim.y - 1) / (16 * blockDim.y);

   // ------
   // warm up
   bsr_wmma_half_half_half<<<gridDim, blockDim>>>(hbsrVal, dB, dC, 
                                                   M, N, K, 
                                                   hf_alpha, hf_beta,
                                                   bsrRowPtr, bsrColInd);

   GpuTimer bsrwmma_timer;
   bsrwmma_timer.Start();
   for (int i = 0; i < TEST_TIMES; i++)
   {
      bsr_wmma_half_half_half<<<gridDim, blockDim>>>(hbsrVal, dB, dC, 
                                                      M, N, K, 
                                                      hf_alpha, hf_beta,
                                                      bsrRowPtr, bsrColInd);
   }
   bsrwmma_timer.Stop();
   double bsrwmma_time = bsrwmma_timer.ElapsedMillis() / double(TEST_TIMES);
   // ------
   cudaMemcpy(hC, dC, M * N * sizeof(__half), cudaMemcpyDeviceToHost);
   cudaFree(dB);
   cudaFree(dC);

   return bsrwmma_time;
}