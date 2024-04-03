import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes, args):
        super(get_model, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+16+6, mlp=[128, 128, 128])

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
        B,C,N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)

        x1 = self.attn1(l0_points)                                              # B, 128, N
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