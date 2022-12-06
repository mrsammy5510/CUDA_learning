#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#define CHECK(call)                                                         \
{                                                                           \
    const cudaError_t error =call;                                          \
    if(error!=cudaSuccess)                                                  \
    {                                                                       \
        printf("Error: %s:%d, ",__FILE__,__LINE__);                         \
        printf("code:%d, reason: %s\n",error, cudaGetErrorString(error));   \
        exit(1);                                                            \
    }                                                                       \
}                                                                           
void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = true;
    for (int i=0; i<N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) 
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n",hostRef[i],gpuRef[i],i);
            break;
        }
    }
    if (match) printf("Arrays match.\n\n");
}

void sumArraysonHost(float* A, float* B,float* C, const int N)
{
    for(int idx = 0;idx<N;idx++)
    {
        C[idx] = A[idx]+B[idx];
    }
    return;
}

__global__ void sumArraysonDevice(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i+1024*blockIdx.x] = A[i+1024*blockIdx.x]+B[i+1024*blockIdx.x];
    //printf("block.x:%d block.y:%d block.y:%d\n", blockIdx.x, blockIdx.y, blockIdx.z);   //Same as threadIdx, Can only be called inside the kernel function
}

void initialData(float* ip,int size)
{
    //generate different seed for random number
    time_t t;
    srand((unsigned int)time(&t));
    for(int i = 0;i<size;i++)
    {
        ip[i] = (float)(rand() & 0XFF)/10.0F;
    }
}
int main (int argc, char** argv)
{
    clock_t total = clock();
    int nElem = 478989150;   //Each thread block has it's limit of 1024 threads
    clock_t start;
    size_t nBytes = nElem*sizeof(float);
    float* h_A = (float*)malloc(nBytes);
    float* h_B = (float*)malloc(nBytes);
    float* h_C = (float*)malloc(nBytes);

    start = clock();
    initialData(h_A,nElem);
    initialData(h_B,nElem);
    printf("Intialize data cost %10.7f s\n",(float)(clock()-start)/CLOCKS_PER_SEC);

    float* d_A, *d_B, *d_C;
    float* gpuRef = (float *)malloc(nBytes);
    memset(gpuRef, 0, nBytes);
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);
    
    start = clock();
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    printf("cudaMemcpy cost %10.7f s\n",(float)(clock()-start)/CLOCKS_PER_SEC);

    dim3 block (1024);
    dim3 grid ((nElem+block.x-1)/block.x);

    start = clock();
    sumArraysonDevice<<<grid, block>>>(d_A,d_B,d_C);    //first dim is the grid, that is, the number of blocks that are going to be launch
    CHECK(cudaDeviceSynchronize());
    printf("To sum on device cost %10.7f s\n",(float)(clock()-start)/CLOCKS_PER_SEC);
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    
    
    start = clock();
    sumArraysonHost(h_A,h_B,h_C,nElem);
    printf("To sum on host cost %10.7f s\n",(float)(clock()-start)/CLOCKS_PER_SEC);
    
    
    checkResult(h_C, gpuRef, nElem);
    
    printf("Total time this program cost is %10.7f s\n",(float)(clock()-total)/CLOCKS_PER_SEC);
    /*for(int i = 0;i<nElem;i++)
    {
        printf("%5.2f\t",gpuRef[i]);
    }*/
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // free host memory
    free(h_A);
    free(h_B);
    free(gpuRef);

    return 0;
}