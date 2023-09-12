#include <cusparse_v2.h>
#include <iostream>
#include <sys/time.h>

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return EXIT_FAILURE;                                       \
        }                                                              \
    }

#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                  \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            return EXIT_FAILURE;                                           \
        }                                                                  \
    }

/**
* The GPU Timer
*/
struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float ElapsedMillis()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

// blockell conversion helper
int get_ell_width(int *bsrRowPtr, int nblockrows)
{
   // get the maximum nblocks on a blockrow 
   int max = 0;
   for (int i=0; i<nblockrows; i++)
   {
      if ((bsrRowPtr[i+1] - bsrRowPtr[i]) > max)
         max = bsrRowPtr[i+1] - bsrRowPtr[i];
   }
   return max;
}

void fill_h_ell_columns(int *h_ell_columns, int *bsrRowPtr, int *bsrColInd, int nblockrows, int ell_width)
{
   for (int i=0; i<nblockrows; i++)
   {
      int cnt = 0;
      for (int j = bsrRowPtr[i]; j<bsrRowPtr[i+1]; j++)
      {
         h_ell_columns[i*ell_width+cnt] = bsrColInd[j];
         cnt++;
      }

      for (; cnt<ell_width; cnt++)
      {
         h_ell_columns[i*ell_width+cnt] = -1;
      }
   }
}

// row-major to column-major
void transpose(__half *dest, __half *source, int M, int N)
{
    int cnt = 0;
    for (int j=0; j<N; j++)
    {
        for (int i=0; i<M; i++)
        {

            dest[cnt++] = source[i*N+j];
        }
    }
}