import torch
import torch.nn as nn
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
    """ VecKM implementation for large point cloud (>10k).
    Args:
        d: int, the dimension of the dense local geometry encoding.
            * small d (e.g. 128, 256) already encodes the local geometry well.
            * typically, a larger alpha or a smaller beta requires larger d.
            * The size of the point cloud DOES NOT affect the choice of d.
        alpha: float, controlling the level of details retained.
            Assume your point cloud is normalized within a unit ball with radius 1, here is a general suggestion:
            * if accurate local geometry is needed (e.g. normal estimation), use large alpha (e.g. 60).
            * if only geometric feature is needed (e.g. classification), use small alpha (e.g. 30).
        beta: float, controlling the receptive field of the local geometry.
            Please see the table in https://github.com/dhyuan99/VecKM for the guidance of choosing beta.
        p: int, used to approximate the exponential decaying. See VecKM paper for explanation.
            * large p (e.g. 2048, 4096) is recommended.
            * the runtime and memory usage is linear to p.
            * The size of the point cloud DOES affect the choice of p.

    Several Tips:
        * The runtime, memory usage, and the quality of encoding is empirically proportional to (d * p).
        * It is recommended to use a small d and large p, to make the encoding more compact.
        * When the point cloud size increases, please increase p without adjusting d too much.
        * Check out examples/test.py for the usage of this class and how to evaluate the performance.
    """
    def __init__(self, d=1024, alpha=20, beta=6, p=1024):
        super().__init__()
        self.sqrt_d = d ** 0.5
        self.alpha, self.beta, self.d, self.p = alpha, beta, d, p

        self.A = torch.stack(
            [strict_standard_normal(d) for _ in range(3)], 
            dim=0
        ) * alpha
        self.A = nn.Parameter(self.A, False)                                    # Real(3, d)

        self.B = torch.stack(
            [strict_standard_normal(p) for _ in range(3)], 
            dim=0
        ) * beta
        self.B = nn.Parameter(self.B, False)                                    # Real(3, d)
          
    @torch.no_grad()
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
        eA = torch.concatenate((torch.cos(pA), torch.sin(pA)), dim=1)           # Real(..., n, 2d)
        eB = torch.concatenate((torch.cos(pB), torch.sin(pB)), dim=1)           # Real(..., n, 2p)
        G = torch.matmul(
            eB,                                                                 # Real(..., n, 2p)
            eB.transpose(-1,-2) @ eA                                            # Real(..., 2p, 2d)
        )                                                                       # Real(..., n, 2d)
        G = torch.complex(
            G[:,:self.d], G[:,self.d:]
        ) / torch.complex(
            eA[:,:self.d], eA[:,self.d:]
        )                                                                       # Complex(..., n, d)
        G = G / torch.norm(G, dim=-1, keepdim=True) * self.sqrt_d
        return G
    
    def __repr__(self):
        vkm = (
            f"VecKM_Large("
            f"alpha={self.alpha}, "
            f"beta={self.beta}, "
            f"d={self.d}, "
            f"p={self.p})"
        )
        return vkm
