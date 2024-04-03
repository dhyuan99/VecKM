import torch.nn as nn
import torch
import torch.nn.functional as F
from models.utils import PointNetSetAbstraction,PointNetFeaturePropagation
import sys
sys.path.append('..')
from VecKM_small import VecKM
from complexPyTorch.complexLayers import ComplexLinear, ComplexReLU

class get_model(nn.Module):
    def __init__(self, num_classes, args):
        super(get_model, self).__init__()
        self.vkm_feat = nn.Sequential(
            VecKM(256, 30, 9, False),
            ComplexLinear(256, 128),
            ComplexReLU(),
            ComplexLinear(128, 128)
        )
        
        # self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=128 + 3, mlp=[128, 128, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+16+6, mlp=[128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, cls_label):
        B,C,N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz
        
        l1_xyz = l0_xyz
        l1_points = self.vkm_feat(xyz.permute(0,2,1))
        l1_points = l1_points.real**2 + l1_points.imag**2
        l1_points = l1_points.permute(0, 2, 1)

        # l1_xyz, l1_points = self.sa1(l0_xyz, G)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)

        feat =  F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, _):
        total_loss = F.nll_loss(pred, target)

        return total_loss