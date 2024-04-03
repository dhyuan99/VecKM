import torch
import torch.nn as nn

class NormalEstimator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.point_batch_size = 1000
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.num_neighbors = args.num_neighbors

        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 3)
        
    def forward(self, pts, normal):
        index = torch.randperm(pts.shape[0])[:self.point_batch_size]
        dist = ((pts[index].unsqueeze(1)-pts.unsqueeze(0)).abs()).max(dim=-1).values

        _, neighbor_idx = torch.topk(-dist, self.num_neighbors, dim=1)
        pts = pts[neighbor_idx] - pts[index].unsqueeze(1)
        pts = pts.transpose(1, 2)
        pts = self.bn1(torch.relu(self.conv1(pts)))
        pts = self.bn2(torch.relu(self.conv2(pts)))
        pts = self.bn3(torch.relu(self.conv3(pts)))
        
        feat = pts.max(dim=-1).values
        feat = self.bn4(torch.relu(self.fc1(feat)))
        feat = self.bn5(torch.relu(self.fc2(feat)))
        feat = self.fc3(feat)
        return feat, normal[index]
