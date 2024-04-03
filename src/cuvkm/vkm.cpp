#include <torch/extension.h>
#include "vkm_kernel.h"

void compute_adj_matrix(
    torch::Tensor P,
    torch::Tensor Q,
    torch::Tensor J,
    float beta2
) {
    P = P.to(torch::kCUDA, torch::kFloat);
    Q = Q.to(torch::kCUDA, torch::kFloat);
    J = J.to(torch::kCUDA, torch::kFloat);
    float *P_data = P.data_ptr<float>();
    float *Q_data = Q.data_ptr<float>();
    float *J_data = J.data_ptr<float>();

    int n = P.size(0);
    int N = Q.size(0);

    compute_adj_matrix_launch(P_data, Q_data, J_data, n, N, beta2);
}

void vkm_memory_efficient(
    torch::Tensor P, 
    torch::Tensor Q, 
    torch::Tensor eP, 
    torch::Tensor eQ, 
    torch::Tensor G, 
    float beta2
) {
    P = P.to(torch::kCUDA, torch::kFloat);
    Q = Q.to(torch::kCUDA, torch::kFloat);
    eP = eP.to(torch::kCUDA, torch::kComplexFloat);
    eQ = eQ.to(torch::kCUDA, torch::kComplexFloat);
    G = G.to(torch::kCUDA, torch::kComplexFloat);

    float *P_data = P.data_ptr<float>();
    float *Q_data = Q.data_ptr<float>();
    c10::complex<float> *eP_data = eP.data_ptr<c10::complex<float>>();
    c10::complex<float> *eQ_data = eQ.data_ptr<c10::complex<float>>();
    c10::complex<float> *G_data = G.data_ptr<c10::complex<float>>();

    int n = P.size(0);
    int N = Q.size(0);
    int d = G.size(1);

    vkm_memory_efficient_launch(P_data, Q_data, eP_data, eQ_data, G_data, n, N, d, beta2);
} 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vkm_memory_efficient", &vkm_memory_efficient, "Compute vkm with memory-efficient implementation.");
    m.def("compute_adj_matrix", &compute_adj_matrix, "Compute adjacency matrix between point sets.");
}
