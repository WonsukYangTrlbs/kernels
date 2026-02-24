#include <stdio.h>
#include <stdlib.h>
#include <chrono>

void matmul_cpu(float* A, float* B, float* C, int m, int k, int n);

int main(void)
{
    int m = 512, k = 512, n = 512;
    printf("[Matrix multiplication, C = AB]\n");
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

    auto start = std::chrono::high_resolution_clock::now();
    matmul_cpu(h_A, h_B, h_C, m, k, n);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = stop - start;
    printf("Time taken: %f milliseconds\n", elapsed.count());

    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}

void matmul_cpu(float* A, float* B, float* C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int l = 0; l < k; l++) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}