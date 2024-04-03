#ifndef VKM_KERNEL_H
#define VKM_KERNEL_H
#include <torch/serialize/tensor.h>
#include <complex.h>

void compute_adj_matrix(
    torch::Tensor P,
    torch::Tensor Q,
    torch::Tensor J,
    float beta2
);

void compute_adj_matrix_launch(
    float *P, 
    float *Q, 
    float *J, 
    int n, int N, float beta2
);

void vkm_memory_efficient(
    torch::Tensor P, 
    torch::Tensor Q, 
    torch::Tensor eP, 
    torch::Tensor eQ, 
    torch::Tensor G, 
    float beta2
);

void vkm_memory_efficient_launch(
    float *P, 
    float *Q, 
    c10::complex<float> *eP, 
    c10::complex<float> *eQ, 
    c10::complex<float> *G, 
    int n, int N, int d, float beta2
);

#endif // VKM_KERNEL_H