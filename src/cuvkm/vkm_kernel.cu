#include "vkm_kernel.h"
#include <cublas_v2.h>

__global__ void vkm_memory_efficient_kernel(
    float *P, 
    float *Q, 
    c10::complex<float> *eP, 
    c10::complex<float> *eQ, 
    c10::complex<float> *G, 
    int n, int N, int d, float beta2
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n*d) return;

    c10::complex<float> tempG(0.0, 0.0);

    for (int j = 0; j < N; ++j) {
        float dist = 0.0f;
        for (int k = 0; k < 3; ++k) {
            float diff = P[i/d * 3 + k] - Q[j * 3 + k];
            dist += diff * diff;
        }

        tempG += exp(-beta2 * dist) * eQ[j*d + i%d];
    }

    G[i] = tempG / eP[i];
}

void vkm_memory_efficient_launch(
    float *P, 
    float *Q, 
    c10::complex<float> *eP, 
    c10::complex<float> *eQ, 
    c10::complex<float> *G, 
    int n, int N, int d, float beta2
) {
    int blockSize = 256;
    int numBlocks = (n * d + blockSize - 1) / blockSize;
    vkm_memory_efficient_kernel<<<numBlocks, blockSize>>>(
        P, Q, eP, eQ, G, 
        n, N, d, beta2);
}

__global__ void compute_adj_matrix_kernel(
    float *P, float *Q, float *J, int n, int N, float beta2
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n*N) return;

    int j = i % N;
    int k = i / N;
    float tempD2 = 0.0f;
    for (int p = 0; p < 3; ++p) {
        float diff = P[k*3 + p] - Q[j*3 + p];
        tempD2 += diff * diff;
    }

    J[k*N+j] = exp(-beta2 * tempD2);
}

void compute_adj_matrix_launch(
    float *P, float *Q, float *J, int n, int N, float beta2
) {
    int blockSize = 256;
    int numBlocks = (n*N + blockSize - 1) / blockSize;
    compute_adj_matrix_kernel<<<numBlocks, blockSize>>>(
        P, Q, J, n, N, beta2
    );
}
