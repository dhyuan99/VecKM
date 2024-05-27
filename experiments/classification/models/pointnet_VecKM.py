import torch
import torch.nn as nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexLinear, ComplexReLU

import numpy as np
from scipy.stats import norm

def strict_standard_normal(d):
    # this function generate very similar outcomes as torch.randn(d)
    # but the numbers are strictly standard normal, no randomness.
    y = np.linspace(0, 1, d+2)
    x = norm.ppf(y)[1:-1]
    np.random.shuffle(x)
    x = torch.tensor(x).float()
    return x

class VecKM(nn.Module):
    def __init__(self, d, alpha, beta, positional_encoding=True):
        super().__init__()
        self.sqrt_d = d ** 0.5
        self.alpha, self.beta2 = alpha, beta**2 / 2
        self.positional_encoding = positional_encoding

        self.A = torch.stack(
            [strict_standard_normal(d) for _ in range(3)], 
            dim=0
        ) * alpha
        self.A = nn.Parameter(self.A, False)                                    # (3, d): for geometry encoding

        if positional_encoding:
            self.P = torch.stack(
                [strict_standard_normal(d) for _ in range(3)], 
                dim=0
            )                                                                      
            self.P = nn.Parameter(self.P, False)                                # (3, d): for position encoding

    def forward(self, pts):
        D2 = ((pts.unsqueeze(-2)-pts.unsqueeze(-3)) ** 2).sum(-1)
        J = torch.exp(-self.beta2 * D2) + 0*1j                                  # (..., n, n)
        eA = torch.exp(1j * (pts @ self.A))                                     # (..., n, d)
        G = (J @ eA) / eA
        G = G / torch.norm(G, dim=-1, keepdim=True) * self.sqrt_d               # (..., n, d)

        if self.positional_encoding:
            eP = torch.exp(1j * (pts @ self.P))                                 # (..., n, d)
            return G + eP
        else:
            return G

    def __repr__(self):
        vkm = (
            f"VecKM("
            f"alpha={self.alpha}, "
            f"beta={self.beta2 ** 0.5}, "
            f"d={self.A.shape[-1]}, "
            f"positional_encoding={self.positional_encoding}"
        )
        return vkm

class get_model(nn.Module):
    def __init__(self, output_channels=40):
        super(get_model, self).__init__()
        self.vkm_feat = nn.Sequential(
            VecKM(256, 30, 6, positional_encoding=True),
            ComplexLinear(256, 256),
            ComplexReLU(),
            ComplexLinear(256, 1024)
        )
        self.clf1 = nn.Linear(1024, 512)
        self.clf2 = nn.Linear(512, 256)
        self.clf3 = nn.Linear(256, output_channels)
        self.dropout = nn.Dropout(p=0.4)
        self.bn_clf1 = nn.BatchNorm1d(512)
        self.bn_clf2 = nn.BatchNorm1d(256)

    def forward(self, x):
        B, _, N = x.shape                                                       # R(B, 3, N)
        x = self.vkm_feat(x.permute(0,2,1))                                     # C(B, N, 1024)
        x = x.real ** 2 + x.imag ** 2
        x = torch.max(x, 1, keepdim=True)[0]                                    # R(B, 1, 1024)
        x = x.view(-1, 1024)                                                    # R(B, 1024)
        x = F.relu(self.bn_clf1(self.clf1(x)))                                  # R(B, 512)
        x = F.relu(self.bn_clf2(self.dropout(self.clf2(x))))                    # R(B, 256)
        x = self.clf3(x)                                                        # R(B, 40)
        return x