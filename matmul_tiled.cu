#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmulKernelTiled(float *A, float *B, float *C, int m, int k, int n);

int main(void)
{
    int m = 4096, k = 4096, n = 4096;
    printf("[Tiled Matrix multiplication, C = AB]\n");
    printf("Size of Matrix A: %d x %d\n", m, k);
    printf("Size of Matrix B: %d x %d\n", k, n);
    printf("Size of Matrix C: %d x %d\n", m, n);
    printf("Running...\n");

    srand(42);
    float *h_A = (float*)malloc(m * k * sizeof(float));
    float *h_B = (float*)malloc(k * n * sizeof(float));
    float *h_C = (float*)malloc(m * n * sizeof(float));

    // Randomly populate host matrices h_A, h_B, h_C
    for (int i = 0; i < m * k; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < k * n; i++) {
        h_B[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < m * n; i++) {
        h_C[i] = 0.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * n * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int BLOCK_SIZE = TILE_SIZE;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    printf("Launching kernel with %d blocks and %d threads per block\n", grid.x, dimBlock.x);

    cudaDeviceSynchronize();
    cudaEventRecord(start);

    matmulKernelTiled<<<grid, dimBlock>>>(d_A, d_B, d_C, m, k, n);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0.0f;
    cudaEventElapsedTime(&total, start, stop);
    printf("Time taken: %f milliseconds\n", total);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

__global__ void matmulKernelTiled(float *A, float *B, float *C, int m, int k, int n)
{
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int tiles = (k + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < tiles; t++)
    {
        int colA = t * TILE_SIZE + threadIdx.x;
        int rowB = t * TILE_SIZE + threadIdx.y;
        if (row < m && colA < k)
            s_A[threadIdx.y][threadIdx.x] = A[row * k + colA];
        else
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        if (rowB < k && col < n)
            s_B[threadIdx.y][threadIdx.x] = B[rowB * n + col];
        else
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++)
            sum += s_A[threadIdx.y][j] * s_B[j][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < n)
        C[row * n + col] = sum;
}