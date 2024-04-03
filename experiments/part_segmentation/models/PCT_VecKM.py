import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
sys.path.append('..')
from VecKM_small import VecKM
from complexPyTorch.complexLayers import ComplexLinear, ComplexReLU

class get_model(nn.Module):
    def __init__(self, num_classes, args):
        super(get_model, self).__init__()
        self.vkm_feat = nn.Sequential(
            VecKM(256, 30, 9),
            ComplexLinear(256, 256),
            ComplexReLU(),
            ComplexLinear(256, 256)
        )
        self.lbl_emb = torch.randn(16, 256)
        self.lbl_emb = nn.Parameter(self.lbl_emb, requires_grad=True)
        self.feat_trans = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
        )

        self.attn1 = SA_Layer(128)
        self.attn2 = SA_Layer(128)
        self.attn3 = SA_Layer(128)
        self.attn4 = SA_Layer(128)
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(512, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, cls_label):
        B, _, N = xyz.size()

        G = self.vkm_feat(xyz.permute(0,2,1))
        G = G.real**2 + G.imag**2
        G = G.permute(0, 2, 1)                                                  # R(B, 256, N)

        reweight = (cls_label.view(B, 16) @ self.lbl_emb).unsqueeze(-1)
        x = G * reweight
        
        x = self.feat_trans(x)

        x1 = self.attn1(x)                                                      # B, 128, N
        x2 = self.attn2(x1)                                                     # B, 128, N
        x3 = self.attn3(x2)                                                     # B, 128, N
        x4 = self.attn4(x3)    
        x = torch.concat((x1, x2, x3, x4), dim=1)                               # B, 512, N
        x = self.conv_fuse(x)                                                   # B, 128, N

        feat =  F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, None


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        
        x_q = self.q_conv(x).permute(0, 2, 1)                                   # R(B, N, 64)
        x_k = self.k_conv(x)                                                    # R(B, 64, N)
        x_v = self.v_conv(x)                                                    # R(B, 256, N)
        energy = torch.bmm(x_q, x_k)                                            # R(B, N, N)

        attention = self.softmax(energy)                                        # R(B, N, N)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))     # R(B, N, N)

        x_r = torch.bmm(x_v, attention)                                         # R(B, 256, N)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))               # R(B, 256, N)
        x = x + x_r                                                             # R(B, 256, N)
        return x

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, _):
        total_loss = F.nll_loss(pred, target)

        return total_loss