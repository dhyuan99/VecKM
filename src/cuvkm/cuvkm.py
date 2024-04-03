import torch.nn as nn
import numpy as np
from scipy.stats import norm

import torch # Remeber to import torch before importing vkm_ops!!!
import vkm_ops

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
        self.d = d
        self.sqrt_d, self.alpha, self.beta2 = d**0.5, alpha, beta**2 / 2
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

    def forward(self, pointset1, pointset2=None, memory_efficient=False):
        """ Compute the local geometry of pointset1, in the context of pointset2.
        Args:
            pointset1 (torch.Tensor): (n, 3) tensor, the input point set.
            pointset2 (torch.Tensor): (N, 3) tensor, the input point set.
            memory_efficient: bool, False means runtime efficient, True means memory efficient.
            if pointset2 is None, it means pointset2 = pointset1.
        Return:
            torch.Tensor: (n, d) tensor, the local geometry of pointset1.
        """

        assert (
            len(pointset1.shape) == 2 and pointset1.shape[1] == 3
        ), f'only support input shape (n, 3), but get pointset1 with shape {pointset1.shape}.'
        assert (
            (pointset2 is None) or
            (len(pointset2.shape) == 2 and pointset2.shape[1] == 3)
        ), f'only support input shape (n, 3), but get pointset2 with shape {pointset2.shape}.'

        if memory_efficient:
            return self.forward_memory_efficient(pointset1, pointset2)
        else:
            return self.forward_runtime_efficient(pointset1, pointset2)

    @torch.no_grad()
    def forward_runtime_efficient(self, pointset1, pointset2):

        eP1 = torch.exp(1j * (pointset1 @ self.A))
        if pointset2 is not None:
            eP2 = torch.exp(1j * (pointset2 @ self.A))
        else:
            pointset2 = pointset1
            eP2 = eP1
            
        J = torch.zeros(
            eP1.shape[0], eP2.shape[0], 
            dtype=torch.float32, device=eP1.device)                             # adjacency matrix (n, N)
        vkm_ops.compute_adj_matrix(
            pointset1, pointset2, J, 
            self.beta2)                                                         # fill in the adjacency matrix
                                    
        G = torch.complex(
            (J @ eP2.real),
            (J @ eP2.imag)
        ) / eP1 
        G = G / torch.norm(G, dim=-1, keepdim=True) * self.sqrt_d               # C(n, d)

        if self.positional_encoding:
            eP = torch.exp(1j * (pointset1 @ self.P))                           # C(n, d)
            return G + eP
        else:
            return G
    
    @torch.no_grad()
    def forward_memory_efficient(self, pointset1, pointset2):
        eP1 = torch.exp(1j * (pointset1 @ self.A))
        if pointset2 is not None:
            eP2 = torch.exp(1j * (pointset2 @ self.A))                          # C(N, d)
        else:
            pointset2 = pointset1
            eP2 = eP1
            
        G = torch.zeros_like(eP1)                                               # local geometry encoding place holder.
        vkm_ops.vkm_memory_efficient(
            pointset1, pointset2, eP1, eP2, G, 
            self.beta2)                                                  # fill in the local geometry encoding.
        G = G / torch.norm(G, dim=-1, keepdim=True) * self.sqrt_d               # C(n, d)

        if self.positional_encoding:
            eP = torch.exp(1j * (pointset1 @ self.P))                           
            return G + eP
        else:
            return G

    @torch.no_grad()
    def baseline_forward(self, pointset1, pointset2=None):
        """ You would never use this function in practice, it is both memory and computationally expensive.
        It is just for testing the implementation of the previous two forward functions output correct results.
        """
        eP1 = torch.exp(1j * (pointset1 @ self.A))
        if pointset2 is not None:
            eP2 = torch.exp(1j * (pointset2 @ self.A))
        else:
            pointset2 = pointset1
            eP2 = eP1
          
        D2 = ((pointset1.unsqueeze(1)-pointset2.unsqueeze(0)) ** 2).sum(-1)
        J = torch.exp(-self.beta2 * D2)                                         # (n, N)
        G = torch.complex(
            (J @ eP2.real),
            (J @ eP2.imag)
        ) / eP1 

        G = G / torch.norm(G, dim=-1, keepdim=True) * self.sqrt_d               # (..., n, d)
        return G

    def __repr__(self):
        vkm = (
            f"VecKM("
            f"alpha={self.alpha}, "
            f"beta2={self.beta2}, "
            f"d={self.A.shape[-1]}, "
            f"positional_encoding={self.positional_encoding})"
        )
        return vkm

