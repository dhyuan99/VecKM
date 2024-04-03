import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm
import sys
sys.path.append('..')
from VecKM_large import VecKM
from complexPyTorch.complexLayers import ComplexConv1d, ComplexReLU, NaiveComplexBatchNorm1d

def strict_standard_normal(d):
    # this function generate very similar outcomes as torch.randn(d)
    # but the numbers are strictly standard normal, no randomness.
    y = np.linspace(0, 1, d+2)
    x = norm.ppf(y)[1:-1]
    np.random.shuffle(x)
    x = torch.tensor(x).float()
    return x

class NormalEstimator(nn.Module):
    def __init__(self, args, d=1024, p=1024):
        super().__init__()
        self.point_batch_size = 100000
        self.alpha_list = args.alpha_list
        self.beta_list = args.beta_list
        self.n_reso = len(self.alpha_list)
        self.n_scales = len(self.beta_list)
        self.sqrt_d = d ** 0.5

        self.vkm = VecKM(d, args.alpha_list, args.beta_list, p)

        self.feat_trans = nn.Sequential(
            ComplexConv1d(d, 128),
            NaiveComplexBatchNorm1d(128),
            ComplexReLU(),
            ComplexConv1d(128, 128),
            NaiveComplexBatchNorm1d(128)
        )

        self.out_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, pts, normal):
        G = self.vkm(pts)                                                       # C(n, n_scales * n_reso, d)
        G = G.transpose(-2,-1)                                                  # C(n, d, n_scales * n_reso)
        G = self.feat_trans(G)                                                  # C(n, 128, n_scales * n_reso)
        G = G.real**2 + G.imag**2                                               # R(n, 128, n_scales * n_reso)
        G = torch.max(G, dim=-1)[0]                                             # R(n, 128)
        pred = self.out_fc(G)
        return pred, normal
    


