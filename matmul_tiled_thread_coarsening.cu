#include "common.cuh"

#define TILE_SIZE 16
#define COARSE_FACTOR 4

__global__ void matmulKernelTiledThreadCoarsening(float *A, float *B, float *C, int m, int k, int n);

int main(void)
{
    int m = 8192, k = 8192, n = 8192;
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
    dim3 grid((n + TILE_SIZE * COARSE_FACTOR - 1) / (TILE_SIZE * COARSE_FACTOR), (m + TILE_SIZE - 1) / TILE_SIZE);
    printf("Launching kernel with %d blocks and %d threads per block\n", grid.x, dimBlock.x);

    cudaDeviceSynchronize();
    cudaEventRecord(start);

    matmulKernelTiledThreadCoarsening<<<grid, dimBlock>>>(d_A, d_B, d_C, m, k, n);

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

__global__ void matmulKernelTiledThreadCoarsening(float *A, float *B, float *C, int m, int k, int n)
{
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int colStart = bx * TILE_SIZE * COARSE_FACTOR + tx;

    float Cvalue[COARSE_FACTOR];
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        Cvalue[c] = 0.0f;
    }

    for (int ph = 0; ph < k / TILE_SIZE; ++ph) {
        s_A[ty][tx] = A[row * k + ph * TILE_SIZE + tx];

        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int col = colStart + c * TILE_SIZE;
            s_B[ty][tx] = B[(ph * TILE_SIZE + ty) * n + col];
            __syncthreads();

            for (int l = 0; l < TILE_SIZE; ++l) {
                Cvalue[c] += s_A[ty][l] * s_B[l][tx];
            }
            __syncthreads();
        }
    }

    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int col = colStart + c * TILE_SIZE;
        C[row * n + col] = Cvalue[c];
    }
}