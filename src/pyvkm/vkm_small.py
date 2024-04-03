import numpy as np
import torch
import torch.nn as nn
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
