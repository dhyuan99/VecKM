import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm

def strict_standard_normal(d):
    # this function generate very similar outcomes as torch.randn(d)
    # but the numbers are strictly standard normal, no randomness.
    y = np.linspace(0, 1, d+2)
    x = norm.ppf(y)[1:-1]
    np.random.shuffle(x)
    x = torch.tensor(x).float()
    return x

class ComplexReLU(nn.Module):
     def forward(self, real, imag):
         return F.relu(real), F.relu(imag)

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, real, imag):
        return self.fc_r(real) - self.fc_i(imag), self.fc_r(imag) + self.fc_i(real) 

class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ComplexConv1d, self).__init__()
        self.conv_r = nn.Conv1d(in_channels, out_channels, 1)
        self.conv_i = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, real, imag):
        return self.conv_r(real) - self.conv_i(imag), self.conv_r(imag) + self.conv_i(real)

class NaiveComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, \
                 track_running_stats=True):
        super(NaiveComplexBatchNorm1d, self).__init__()
        self.bn_r = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, real, imag):
        return self.bn_r(real), self.bn_i(imag)

class VecKM(nn.Module):
    def __init__(self, d=1024, alpha_list=[10,20], beta_list=[5,10]):
        super().__init__()
        self.alpha_list = alpha_list
        self.beta_list = beta_list
        self.n_reso = len(alpha_list)
        self.n_scales = len(self.beta_list)
        self.sqrt_d = d ** 0.5

        self.T = []
        for alpha in self.alpha_list:
            T = torch.stack(
                [strict_standard_normal(d) for _ in range(3)], 
                dim=0
            ) * alpha
            self.T.append(T)
        self.T = torch.stack(self.T, dim=0)
        self.T = nn.Parameter(self.T, False)                                    # (n_reso, 3, d)

        self.W = []
        for beta in self.beta_list:
            W = torch.stack(
                [strict_standard_normal(2048) for _ in range(3)], 
                dim=0
            ) * beta
            self.W.append(W)
        self.W = torch.stack(self.W, dim=0)
        self.W = nn.Parameter(self.W, False)                                    # (n_scales, 3, d)

    def forward(self, pts):
        """ Compute the dense local geometry encodings of the given point cloud.
        Args:
            pts: (bs, n, 3) or (n, 3) tensor, the input point cloud.

        Returns:
            G: (bs, n_scales*n_reso, n, d) or (n_scales*n_reso, n, d) tensor
               the dense local geometry encodings. 
               note: it is complex valued. 
        """
        pts = pts.unsqueeze(-3)                                                 # (bs, 1, n, 3)
        eT = torch.exp(1j * (pts @ self.T)).unsqueeze(-4)                       # (bs, 1, n_reso, n, d)
        eW = torch.exp(1j * (pts @ self.W)).unsqueeze(-3)                       # (bs, n_scales, 1, n, d)

        G = torch.matmul(
            eW, 
            eW.transpose(-1,-2).conj() @ eT
        ) / eT                                                                  # (bs, n_scales, n_reso, n, d)
        G = G / torch.norm(G, dim=-1, keepdim=True) * self.sqrt_d               # (bs, n_scales, n_reso, n, d)

        G = G.reshape(
            *G.shape[:-4], 
            G.shape[-4]*G.shape[-3], 
            G.shape[-2], 
            G.shape[-1]
        )                                                                       # (bs, n_scales*n_reso, n, d)

        return G

    def __repr__(self):
        return f'VecKM(d={self.T.shape[-1]}, alpha_list={self.alpha_list}, beta_list={self.beta_list})'

