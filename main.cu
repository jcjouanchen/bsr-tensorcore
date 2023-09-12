#include <stdio.h>
#include <assert.h>
#include <vector>

#include "backend/readMtx.hpp" // Sparse Matrix IO
#include "backend/spmm.cu"

// csr metadata for readMtx
std::vector<int> row_indices;
std::vector<int> col_indices;
std::vector<float> values;

// csr metadata
int nrows, ncols, nnz;

// csr host
int *h_csrRowPtr, *h_csrColInd;
float *h_csrVal;

// csr device
int *csrRowPtr, *csrColInd;
float *csrVal;

// B metadata
int nBcols;

// bsr metadata & storage
int nblocks, nblockrows;
int *bsrRowPtr, *bsrColInd; 
float *bsrVal;
__half *hbsrVal;

// blocked-ell storage
int ell_width;
int *ell_columns;
__half *ell_values;

void readMtxCSR(const char *filename)
{
    // mmio interface
    char *dat_name;
    readMtx(filename, &row_indices, &col_indices, &values,
            &nrows, &ncols, &nnz, 0, false, &dat_name);
    
    h_csrRowPtr = (int *)malloc(sizeof(int) * (nrows + 1));
    h_csrColInd = (int *)malloc(sizeof(int) * nnz);
    h_csrVal = (float *)malloc(sizeof(float) * nnz);
    coo2csr(h_csrRowPtr, h_csrColInd, h_csrVal,
            row_indices, col_indices, values, nrows, ncols);

    // copy csr to device
    cudaMalloc(&csrRowPtr, sizeof(int) * (nrows + 1));
    cudaMalloc(&csrColInd, sizeof(int) * nnz);
    cudaMalloc(&csrVal, sizeof(float) * nnz);
    cudaMemcpy(csrRowPtr, h_csrRowPtr, sizeof(int) * (nrows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(csrColInd, h_csrColInd, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(csrVal, h_csrVal, sizeof(float) * nnz, cudaMemcpyHostToDevice);
}

void freeCSR()
{
   free(h_csrRowPtr);
   free(h_csrColInd);
   free(h_csrVal);
   cudaFree(csrRowPtr);
   cudaFree(csrColInd);
   cudaFree(csrVal);
}

void freeBSR()
{
   cudaFree(bsrRowPtr);
   cudaFree(bsrColInd);
   cudaFree(bsrVal);
}

void freeBlockedELL()
{
   cudaFree(ell_columns);
   cudaFree(ell_values);
}

void CSR2BSRhalf(int block_dim=16)
{
    // transform from csr to bsr using cuSPARSE API
    int mb = (nrows + block_dim - 1) / block_dim;
    int nb = (ncols + block_dim - 1) / block_dim;
    nblockrows = mb;

    // cuSPARSE API metadata setup
    cusparseMatDescr_t csr_descr = 0;
    cusparseMatDescr_t bsr_descr = 0;
    cudaStream_t streamId = 0;
    cusparseHandle_t handle = 0;
    cusparseCreateMatDescr(&csr_descr);
    cusparseSetMatType(csr_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(csr_descr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseCreateMatDescr(&bsr_descr);
    cusparseSetMatType(bsr_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(bsr_descr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseCreate(&handle);
    cusparseSetStream(handle, streamId);
    cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

    // csr2bsr in row-major order
    cudaMalloc((void **)&bsrRowPtr, sizeof(int) * (nblockrows + 1));
    cusparseXcsr2bsrNnz(handle, dirA, nrows, ncols, 
                        csr_descr, csrRowPtr, csrColInd, block_dim, 
                        bsr_descr, bsrRowPtr, &nblocks);
    cudaMalloc((void **)&bsrColInd, sizeof(int) * nblocks);
    cudaMalloc((void **)&bsrVal, nblocks * block_dim * block_dim * sizeof(float));
    cusparseScsr2bsr(handle, dirA, nrows, ncols,
                     csr_descr, csrVal, csrRowPtr, csrColInd, block_dim,
                     bsr_descr, bsrVal, bsrRowPtr, bsrColInd);
    
    // convert bsrVal to half
    cudaMalloc((void**)&hbsrVal, nblocks * block_dim * block_dim * sizeof(__half));
    convertFp32ToFp16<<< (nblocks * block_dim * block_dim  + 255) / 256, 256 >>>(hbsrVal, bsrVal, nblocks * block_dim * block_dim);

    // free cusparse bsr metadata
    cusparseDestroyMatDescr(csr_descr);
    cusparseDestroyMatDescr(bsr_descr);
    cusparseDestroy(handle);
}

void BSR2BlockedELLhalf(__half *hA, int block_dim=16)
{
    // dense A info
    int   num_rows     = nrows;
    int   num_cols     = ncols;
    int   ld           = num_cols;
    int   dense_size   = ld * num_rows;
    __half *h_dense = hA;

    // bsr to host for conversion need
    int *h_bsrRowPtr = (int *) malloc(sizeof(int) * (nblockrows + 1));
    int *h_bsrColInd = (int *) malloc(sizeof(int) * nblocks);
    cudaMemcpy(h_bsrRowPtr, bsrRowPtr, sizeof(int) * (nblockrows + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bsrColInd, bsrColInd, sizeof(int) * nblocks, cudaMemcpyDeviceToHost);

    int ell_blk_size = block_dim;
    ell_width = get_ell_width(h_bsrRowPtr, nblockrows) * ell_blk_size;
    int nnz = ell_width * num_rows;

    // set h_ell_columns
    int *h_ell_columns = (int*)malloc(sizeof(int) * nnz / (ell_blk_size * ell_blk_size));
    memset(h_ell_columns, 0, (nnz / (ell_blk_size * ell_blk_size)) * sizeof(int));
    fill_h_ell_columns(h_ell_columns, h_bsrRowPtr, h_bsrColInd, nblockrows, ell_width/ell_blk_size);
    free(h_bsrRowPtr);
    free(h_bsrColInd);

    // set empty h_ell_values
    __half* h_ell_values = (__half*)malloc(nnz * sizeof(__half));
    memset(h_ell_values, static_cast<__half>(0.0f), nnz*sizeof(__half));

    //--------------------------dense2sparse using cuSPARSE APIs--------------------------------
    // Device memory management
    int   *d_ell_columns;
    __half *d_ell_values,  *d_dense;
    cudaMalloc((void**) &d_dense, dense_size * sizeof(__half));
    cudaMalloc((void**) &d_ell_columns, nnz / (ell_blk_size * ell_blk_size) * sizeof(int));
    cudaMalloc((void**) &d_ell_values, nnz * sizeof(__half));
    cudaMemcpy(d_dense, h_dense, dense_size * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ell_columns, h_ell_columns, 
               nnz / (ell_blk_size * ell_blk_size) * sizeof(int), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ell_values, h_ell_values, 
               nnz * sizeof(__half),
               cudaMemcpyHostToDevice);
    
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matB;
    cusparseDnMatDescr_t matA;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    cusparseCreate(&handle);

    // Create dense matrix A
    cusparseCreateDnMat(&matA, num_rows, num_cols, ld, d_dense,
                        CUDA_R_16F, CUSPARSE_ORDER_ROW);

    // Create sparse matrix B in Blocked ELL format
    cusparseCreateBlockedEll(&matB, num_rows, num_cols,
                            ell_blk_size, ell_width,
                            d_ell_columns, d_ell_values,
                            CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_BASE_ZERO,
                            CUDA_R_16F);

    // allocate an external buffer if needed
    cusparseDenseToSparse_bufferSize(handle, matA, matB,
                                    CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                    &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // analyze Sparse to Dense conversion
    cusparseDenseToSparse_analysis(handle, matA, matB,
                                   CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                   dBuffer);

    // execute Sparse to Dense conversion
    cusparseDenseToSparse_convert(handle, matA, matB,
                                 CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                 dBuffer);

    // destroy matrix/vector descriptors
    cusparseDestroyDnMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroy(handle);

    ell_columns = d_ell_columns;
    ell_values = d_ell_values;

    // free unused storage
    cudaFree(dBuffer);
    cudaFree(d_dense);
}

void printBSR(int block_dim=16)
{
    int *h_bsrRowPtr = (int *)malloc(sizeof(int) * (nblockrows + 1));
    int *h_bsrColInd = (int *)malloc(sizeof(int) * nblocks);
    __half *h_hbsrVal = (__half *)malloc(sizeof(__half) * nblocks * block_dim * block_dim);
    cudaMemcpy(h_bsrRowPtr, bsrRowPtr, sizeof(int) * (nblockrows + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bsrColInd, bsrColInd, sizeof(int) * nblocks, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hbsrVal, hbsrVal, sizeof(__half) * nblocks * block_dim * block_dim, cudaMemcpyDeviceToHost);
    printf("nblockrows: %d\n", nblockrows);
    printf("nblocks: %d\n", nblocks);
    printf("h_bsrRowPtr:\n");
    for(int i=0; i<nblockrows+1; i++) printf("%d ", h_bsrRowPtr[i]);
    printf("\nh_bsrColInd:\n");
    for(int i=0; i<nblocks; i++) printf("%d ", h_bsrColInd[i]);
    printf("\n_hbsrVal:\n");
    for(int i=0; i<nblocks; i++) 
    {
        printf("[%d]\n", i);
        for(int j=0; j<block_dim; j++)
        {
            for(int k=0; k<block_dim; k++)
            {
                printf("%.2f ", static_cast<float>(h_hbsrVal[i*block_dim*block_dim+j*block_dim+k]));
            }
            printf("\n");
        }
        printf("\n\n");
    }
    free(h_bsrRowPtr);
    free(h_bsrColInd);
    free(h_hbsrVal);
}

void printBlockedELL(int block_dim=16)
{
    // copy results to host
    int ell_blk_size = block_dim;
    int nnz = ell_width * nrows;
    int *h_ell_columns = (int *)malloc(sizeof(int) * (nnz/(ell_blk_size*ell_blk_size)));
    __half *h_ell_values = (__half *)malloc(sizeof(__half) * nnz);
    cudaMemcpy(h_ell_columns, ell_columns,
               nnz/(ell_blk_size*ell_blk_size) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ell_values, ell_values,
               nnz * sizeof(__half), cudaMemcpyDeviceToHost);

    printf("ell_width: %d\n", ell_width);
    printf("nnz: %d\n", nnz);
    printf("\nh_ell_columns:\n");
    for(int i=0; i<(nnz/(ell_blk_size*ell_blk_size)); i++) printf("%d ", h_ell_columns[i]);
    printf("\nh_ell_values:\n");
    for(int i=0; i<(nrows/ell_blk_size); i++) 
    {
        for(int j=0; j<(ell_width/ell_blk_size); j++)
        {
            printf("[%d, %d]\n", i, j);
            for(int k=0; k<ell_blk_size; k++)
            {
                for(int l=0; l<ell_blk_size; l++)
                {
                    printf("%.2f ", 
                    static_cast<float>(h_ell_values[i*ell_blk_size*ell_width+k*ell_width+j*ell_blk_size+l]));
                }
                printf("\n");
            }
            printf("\n\n");
        }
    }
    free(h_ell_columns);
    free(h_ell_values);
}

bool verifyResult(__half *res1, __half *res2, bool print=false)
{
    // verify
    bool pass = true;
    for(int i=0; i<nrows; i++)
    {
        for(int j=0; j<nBcols; j++)
        {
            // diff tolerence between half and float
            if (static_cast<float>(res1[i*nBcols+j]) != static_cast<float>(res2[i*nBcols+j])) 
                pass = false;
        }
    }

    if (print)
    {
        printf("TEST PASSED: %d\n", pass);

        printf("--------------res1--------------\n");
        for (int i=0; i<5; i++) 
        {
            for (int j=0; j<nBcols; j++)
            {
                // std::cout << static_cast<float>(res1[i * nBcols + j]) << " ";
                printf("%.2f ", static_cast<float>(res1[i * nBcols + j]));
            }
            printf("\n");
        }

        printf("--------------res2--------------\n");
        for (int i=0; i<5; i++) 
        {
            for (int j=0; j<nBcols; j++)
            {
                // std::cout << static_cast<float>(res2[i * nBcols + j]) << " ";
                printf("%.2f ", static_cast<float>(res2[i * nBcols + j]));
            }
            printf("\n");
        }
    }

    return pass;
}

int main(int argc, char *argv[])
{
    cudaSetDevice(0);

    // ====================== input setup ====================== //
    char *Amtxfile = argv[1]; // e.g. "G43.mtx"
    nBcols = atoi(argv[2]); // e.g. "64"

    // read A matrix as CSR
    readMtxCSR(Amtxfile);

    // 16-align for all dimension, this is strictly required for cuSPARSELt Matmul
    nrows     = ((nrows+16-1)/16)*16;
    ncols     = ((ncols+16-1)/16)*16;
    nBcols    = ((nBcols+16-1)/16)*16;

    // set shape and host storage
    int M = nrows;
    int N = nBcols;
    int K = ncols;

    __half *hA = (__half*)malloc(M * K * sizeof(__half));
    __half *hB = (__half*)malloc(K * N * sizeof(__half));
    __half *hC = (__half*)malloc(M * N * sizeof(__half));

    // convert A to dense for dense MM evaluation
    memset(hA, 0.0f, M * K * sizeof(__half));
    for (int i = 0; i < nnz; i++)
        hA[row_indices[i]*K+col_indices[i]] = __float2half(values[i]);

    // randomized B matrix
    srand(time(0));
    for (int i = 0; i < K * N; i++)
        hB[i] = static_cast<__half>(static_cast<float>((float)rand() / RAND_MAX));

    // init result C matrix
    memset(hC, 0.0f, M * N * sizeof(__half));

    // ====================== evaluation ====================== //
    /* run SpMM as dense MM (A Mtx as dense) with cuBLAS */
    // cuBLAS HGemm
    // double cublas_hgemm_time = evalCuBLASHGemm(hA, hB, hC, M, N, K);
    // printf("cublas_hgemm_time: %.3f\n", cublas_hgemm_time);
    // __half *result_cublas_hgemm = (__half*)malloc(M * N * sizeof(__half));
    // cudaMemcpy(result_cublas_hgemm, hC, M * N * sizeof(__half), cudaMemcpyHostToHost);
    // memset(hC, 0.0f, M * N * sizeof(__half));

    // cuBLAS GemmEX
    double cublas_gemmex_time = evaluCuBLASGemmex(hA, hB, hC, M, N, K);
    // printf("cublas_gemmex_time: %.3f\n", cublas_gemmex_time);
    __half *result_cublas_gemmex= (__half*)malloc(M * N * sizeof(__half));
    cudaMemcpy(result_cublas_gemmex, hC, M * N * sizeof(__half), cudaMemcpyHostToHost);
    memset(hC, 0.0f, M * N * sizeof(__half));

    /* run SpMM as N:M-sparse (A Mtx as sparse) with cuSPARSELt */
    // double cusparselt_matmul_time = evalCuSPARSELtMatmul(hA, hB, hC, M, N, K);
    // // printf("cusparselt_matmul_time: %.3f\n", cusparselt_matmul_time);
    // __half *result_cusparselt_matmul = (__half*)malloc(M * N * sizeof(__half));
    // cudaMemcpy(result_cusparselt_matmul, hC, M * N * sizeof(__half), cudaMemcpyHostToHost);
    // memset(hC, 0.0f, M * N * sizeof(__half));


    // convert A into BSR with 16x16 blocks in half, row-major order
    CSR2BSRhalf(/*block_dim=*/16);
    // printBSR(/*block_dim=*/16);

    // convert A into blockedELL with 16x16 blocks in half
    BSR2BlockedELLhalf(hA, /*block_dim=*/16);
    // printBlockedELL(/*block_dim=*/16); 

    /* run SpMM as block-sparse-16x16 (A Mtx as sparse) with cuSPARSE */
    double cusparse_blockedell_time = evalCuSPARSESpMMBlockedell(ell_columns, ell_values, ell_width,
                                                                 hB, hC,
                                                                 M, N, K, /*block_dim=*/16);
    // printf("cusparse_blockedell_time: %.3f\n", cusparse_blockedell_time);
    __half *result_cusparse_blockedell = (__half*)malloc(M * N * sizeof(__half));
    cudaMemcpy(result_cusparse_blockedell, hC, M * N * sizeof(__half), cudaMemcpyHostToHost);
    memset(hC, 0.0f, M * N * sizeof(__half));

    /* run SpMM as block-sparse (A Mtx as sparse) with custom bsrwmma kernel */
    double custom_bsrwmma_time = evalCustomBsrwmma(bsrRowPtr, bsrColInd, hbsrVal,
                                                   hB, hC,
                                                   M, N, K, /*block_dim=*/16);
    // printf("custom_bsrwmma_time: %.3f\n", custom_bsrwmma_time);
    __half *result_custom_bsrwmma = (__half*)malloc(M * N * sizeof(__half));
    cudaMemcpy(result_custom_bsrwmma, hC, M * N * sizeof(__half), cudaMemcpyHostToHost);
    memset(hC, 0.0f, M * N * sizeof(__half));

    // verify result
    // verifyResult(result_cublas_hgemm, result_custom_bsrwmma, true); // <-- hgemm fail when bcols !=32
    bool pass1 = verifyResult(result_cublas_gemmex, result_custom_bsrwmma, false); // pass
    // verifyResult(result_cusparselt_matmul, result_custom_bsrwmma, true); // <-- need to check
    bool pass2 = verifyResult(result_cusparse_blockedell, result_custom_bsrwmma, false); // pass

    printf("%.3f %.3f %.3f ", cublas_gemmex_time, cusparse_blockedell_time, custom_bsrwmma_time);
    printf("%d %d ", pass1=0?0:1, pass2=0?0:1);

    // free mem
    freeCSR();
    freeBSR();
    freeBlockedELL();
    free(hA);
    free(hB);
    free(hC);
}