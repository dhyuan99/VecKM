import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import sys
sys.path.append('..')
from VecKM_small import VecKM
from complexPyTorch.complexLayers import ComplexLinear, ComplexReLU

class get_model(nn.Module):
    def __init__(self, part_num, args):
        super(get_model, self).__init__()
        self.vkm_feat = nn.Sequential(
            VecKM(256, 30, 6),
            ComplexLinear(256, 256),
            ComplexReLU(),
            ComplexLinear(256, 1024)
        )

        self.part_num = part_num
        self.conv1 = torch.nn.Conv1d(2048+16, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, part_num, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud, label):
        B, D, N = point_cloud.size()

        x = self.vkm_feat(point_cloud.permute(0,2,1))                           # C(B, N, 1024)
        x = x.real ** 2 + x.imag ** 2                                           # R(B, N, 1024)
        pt_feat = x.permute(0, 2, 1)                                            # R(B, 1024, N)

        global_feat = torch.max(x, 1, keepdim=True)[0]                          # R(B, 1, 1024)
        global_feat = global_feat.view(-1, 1024)                                # R(B, 1024)

        global_feat = torch.cat([global_feat, label.squeeze(1)], 1)             # R(B, 1024+16)
        expand = global_feat.view(-1, 1024+16, 1).repeat(1, 1, N)               # R(B, 1024+16, N)
        concat = torch.cat([expand, pt_feat], 1)                                # R(B, 2048+16, N)

        net = F.relu(self.bn1(self.conv1(concat)))
        net = F.relu(self.bn2(self.conv2(net)))
        net = F.relu(self.bn3(self.conv3(net)))
        net = self.conv4(net)
        net = net.transpose(2, 1).contiguous()
        net = F.log_softmax(net.view(-1, self.part_num), dim=-1)
        net = net.view(B, N, self.part_num)

        return net, None


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, _):
        loss = F.nll_loss(pred, target)
        return loss