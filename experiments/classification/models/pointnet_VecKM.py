import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')
from VecKM_small import VecKM
from complexPyTorch.complexLayers import ComplexLinear, ComplexReLU

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