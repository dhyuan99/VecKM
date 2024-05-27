import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm
from complexPyTorch.complexLayers import ComplexLinear, ComplexReLU

def strict_standard_normal(d):
    # this function generate very similar outcomes as torch.randn(d)
    # but the numbers are strictly standard normal, no randomness.
    y = np.linspace(0, 1, d+2)
    x = norm.ppf(y)[1:-1]
    np.random.shuffle(x)
    x = torch.tensor(x).float()
    return x

class VecKM(nn.Module):
    def __init__(self, d=256, alpha=6, beta=1.8, p=4096):
        """ I tested empirically, here are some general suggestions for selecting parameters d and p:
        (alpha=6, beta=1.8) works for the data scale that your neighbordhood radius = 1.
        Please ensure your point cloud is appropriately scaled!
        d = 256, p = 4096 is for point cloud size ~20k. Runtime is about 28ms.
        d = 128, p = 8192 is for point cloud size ~50k. Runtime is about 76ms.
        For larger point cloud size, please enlarge p, but if that costs too much, please reduce d.
        A general empirical phenomenon is (d*p) is postively correlated with the encoding quality.

        For the selection of parameter alpha and beta, please see the github section below.
        """
        super().__init__()
        self.alpha, self.beta, self.d, self.p = alpha, beta, d, p
        self.sqrt_d = d ** 0.5

        self.A = torch.stack(
            [strict_standard_normal(d) for _ in range(3)], 
            dim=0
        ) * alpha
        self.A = nn.Parameter(self.A, False)                                    # (3, d)

        self.B = torch.stack(
            [strict_standard_normal(p) for _ in range(3)], 
            dim=0
        ) * beta
        self.B = nn.Parameter(self.B, False)                                    # (3, d)

    def forward(self, pts):
        """ Compute the dense local geometry encodings of the given point cloud.
        Args:
            pts: (bs, n, 3) or (n, 3) tensor, the input point cloud.

        Returns:
            G: (bs, n, d) or (n, d) tensor
               the dense local geometry encodings. 
               note: it is complex valued. 
        """
        pA = pts @ self.A                                                       # Real(..., n, d)
        pB = pts @ self.B                                                       # Real(..., n, p)
        eA = torch.concatenate((torch.cos(pA), torch.sin(pA)), dim=-1)          # Real(..., n, 2d)
        eB = torch.concatenate((torch.cos(pB), torch.sin(pB)), dim=-1)          # Real(..., n, 2p)
        G = torch.matmul(
            eB,                                                                 # Real(..., n, 2p)
            eB.transpose(-1,-2) @ eA                                            # Real(..., 2p, 2d)
        )                                                                       # Real(..., n, 2d)
        G = torch.complex(
            G[..., :self.d], G[..., self.d:]
        ) / torch.complex(
            eA[..., :self.d], eA[..., self.d:]
        )                                                                       # Complex(..., n, d)
        G = G / torch.norm(G, dim=-1, keepdim=True) * self.sqrt_d
        return G

class NormalEstimator(nn.Module):
    def __init__(self, args, d=256, p=4096):
        super().__init__()
        self.point_batch_size = 100000
        self.sqrt_d = d ** 0.5

        self.vkm = VecKM(d, 60, 18, p)

        self.feat_trans = nn.Sequential(
            ComplexLinear(d, 128),
            ComplexReLU(),
            ComplexLinear(128, 128),
            ComplexReLU(),
            ComplexLinear(128, 3)
        )

    def forward(self, pts, normal):
        G = self.vkm(pts)                                                       # C(n, d)
        G = self.feat_trans(G)                                                  # C(n, 3)
        pred = G.real + G.imag
        return pred, normal
    


