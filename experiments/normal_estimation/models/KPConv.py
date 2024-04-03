import torch
import torch.nn as nn

class KPConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernel_points):
        super(KPConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernel_points = num_kernel_points

        # Initialize kernel points and weights
        self.kernel_points = nn.Parameter(torch.randn(num_kernel_points, 3))
        self.weights = nn.Parameter(torch.randn(num_kernel_points, in_channels, out_channels))
        self.sigma = nn.Parameter(torch.tensor(1.0))  # Learnable parameter for Gaussian kernel

    def forward(self, x):
        # x shape: (B, n, 3)
        B, n, _ = x.shape

        # Expand the kernel points and the input
        expanded_kernel_points = self.kernel_points.unsqueeze(0).unsqueeze(0).expand(B, n, -1, -1)
        expanded_x = x.unsqueeze(2).expand(-1, -1, self.num_kernel_points, -1)

        # Calculate distances between input points and kernel points
        distances = torch.norm(expanded_kernel_points - expanded_x, dim=3)

        # Apply a weighting function based on distances (e.g., Gaussian)
        weights = torch.exp(-0.5 * (distances ** 2) / self.sigma ** 2)

        # Reshape and apply the weights to the input features
        weighted_features = x.view(B, n, 1, 3).repeat(1, 1, self.num_kernel_points, 1) * weights.unsqueeze(-1)
        weighted_features = weighted_features.view(B * n * self.num_kernel_points, 3)

        # Reshape the weights for batch matrix multiplication
        reshaped_weights = self.weights.view(self.num_kernel_points, self.in_channels, self.out_channels)

        # Perform batch matrix multiplication
        # We need to ensure the first dimension of both tensors aligns correctly for bmm
        weighted_sum = torch.bmm(weighted_features.unsqueeze(1), reshaped_weights.repeat(B * n, 1, 1))
        weighted_sum = weighted_sum.view(B, n, self.num_kernel_points, self.out_channels)

        # Sum over the kernel points
        x = weighted_sum.sum(dim=2)  # Shape: (B, n, out_channels)

        return x

class NormalEstimator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.point_batch_size = 500
        self.kpconv1 = KPConv(in_channels=3, out_channels=128, num_kernel_points=args.kp)
        self.kpconv2 = KPConv(in_channels=128, out_channels=128, num_kernel_points=args.kp)
        self.conv1 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(1024)
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
        pts = torch.relu(self.kpconv1(pts))
        pts = pts.transpose(1, 2)
        pts = self.bn1(torch.relu(self.conv1(pts)))
        feat = pts.max(dim=-1).values
        
        feat = self.bn4(torch.relu(self.fc1(feat)))
        feat = self.bn5(torch.relu(self.fc2(feat)))
        feat = self.fc3(feat)
        return feat, normal[index] 


