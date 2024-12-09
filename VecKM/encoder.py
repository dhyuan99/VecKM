import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

def strict_standard_normal(d, seed=0):
    """
    this function generate very similar outcomes as torch.randn(d)
    but the numbers are strictly standard normal.
    """
    np.random.seed(seed)
    y = np.linspace(0, 1, d+2)
    x = norm.ppf(y)[1:-1]
    np.random.shuffle(x)
    x = torch.tensor(x).float()
    return x

def get_adj_matrix(pts, r):
    """ Compute the sparse adjacency matrix of the given point cloud.
    Args:
        pts: (n, ?) tensor, the input point cloud.
        r:   float, the radius of the ball.
    Returns:
        adj_matrix: sparse (n, n) matrix, the adjacency matrix. 
                    adj_matrix[i,j] equals 1 if ||pts[i] - pts[j]|| < r, else 0.
    """
    
    # This is the batch size when computing the adjacency matrix.
    # It can adjusted based on your GPU memory. 8192 ** 2 is for 12GB GPU.
    MAX_SIZE = 8192 ** 2

    N = pts.shape[0]
    if N > MAX_SIZE ** 0.5:
        step_size = MAX_SIZE // N
        slice_grid = torch.arange(0, N, step_size)
        slice_grid = torch.cat([slice_grid, torch.tensor([N])])
        non_zero_indices = []
        for j in range(1, len(slice_grid)):
            dist = torch.cdist(pts[slice_grid[j-1]:slice_grid[j]], pts)
            indices = torch.nonzero(dist < r, as_tuple=False)
            indices[:,0] += slice_grid[j-1]
            non_zero_indices.append(indices)
        non_zero_indices = torch.cat(non_zero_indices).T
        adj_matrix = torch.sparse_coo_tensor(
            non_zero_indices, 
            torch.ones_like(non_zero_indices[0], dtype=torch.float32), 
            size=(N, N)
        )
        return adj_matrix
    else:
        dist = torch.cdist(pts, pts)
        adj_matrix = torch.where(dist < r, torch.ones_like(dist), torch.zeros_like(dist))
        return adj_matrix

class ExactVecKM(nn.Module):
    def __init__(self, pt_dim, enc_dim, radius, alpha=6., seed=0):
        """ 
        Use explicitly computed adjacency matrix to compute local geometry encoding. 
        ** Eqn. (3) in the paper. **
        This will result in accurate but slow computation.
        Use this if accurate local geometry encoding is required, such as normal estimation.
        Args:
            pt_dim:  int, dimension of input point cloud, typically 3.
            enc_dim: int, dimension of local geometry encoding, typically 256~512 is sufficient.
            radius:  float, radius of the ball query. Points within this radius will be considered as neighbors.
            alpha:   float, control the sharpness of the kernel function. Default is 6.
        """
        super(ExactVecKM, self).__init__()
        self.pt_dim     = pt_dim
        self.enc_dim    = enc_dim
        self.sqrt_d     = enc_dim ** 0.5
        self.radius     = radius
        self.alpha      = alpha
        
        self.A = torch.stack(
            [strict_standard_normal(enc_dim, seed+i) for i in range(self.pt_dim)], 
            dim=0
        ) * alpha
        self.A = nn.Parameter(self.A, False)                                    # (3, d)
        
    @torch.no_grad()
    def forward(self, pts):
        """ Given a point set, compute local geometry encoding for each point.
        Args:
            pts: torch.Tensor, (N, self.pt_dim), input point cloud. 
                 ** X in Eqn. (3) in the paper **
        Returns:
            enc: torch.Tensor, (N, self.enc_dim), local geometry encoding.
                 ** G in Eqn. (3) in the paper **
        """
        assert pts.dim() == 2 and pts.size(1) == self.pt_dim, "Input tensor should be (N, self.pt_dim)"
        J   = get_adj_matrix(pts, self.radius)                                  # (N, N)
        pA  = (pts / self.radius) @ self.A                                      # (N, 3) @ (3, d) = (N, d)
        epA = torch.cat([torch.cos(pA), torch.sin(pA)], dim=1)                  # (N, 2d)
        G   = J @ epA                                                           # (N, N) @ (N, 2d) = (N, 2d)
        G   = torch.complex(
            G[:, :self.enc_dim], G[:, self.enc_dim:]
        ) / torch.complex(
            epA[:, :self.enc_dim], epA[:, self.enc_dim:]
        )                                                                       # Complex(n, d)
        G = G / torch.norm(G, dim=-1, keepdim=True) * self.sqrt_d               # Complex(n, d)
        return G
    
    def __repr__(self):
        return f"ExactVecKM(pt_dim={self.pt_dim}, enc_dim={self.enc_dim}, radius={self.radius}, alpha={self.alpha})"

class FastVecKM(nn.Module):
    def __init__(self, pt_dim, enc_dim, radius, p=4096, alpha=6., seed=0):
        """ 
        Use implicitly computed adjacency matrix to compute local geometry encoding. 
        ** Eqn. (2) in the paper. **
        This will result in fast but approximate computation, especially when the point cloud is large.
        Use this if only rough local geometry encoding is required, such as classification.
        Args:
            pt_dim:  int, dimension of input point cloud, typically 3.
            enc_dim: int, dimension of local geometry encoding, typically 256~512 is sufficient.
                     ** d in Eqn. (2) in the paper. **
            radius:  float, radius of the ball query. Points within this radius will be considered as neighbors.
            alpha:   float, control the sharpness of the kernel function. Default is 6.
            p:       int, larger p -> more accurate but slower computation. Default is 4096, good for 50000~80000 points.
                     ** p in Eqn. (2) in the paper. **
        """
        super(FastVecKM, self).__init__()
        self.pt_dim     = pt_dim
        self.enc_dim    = enc_dim
        self.sqrt_d     = enc_dim ** 0.5
        self.radius     = radius
        self.alpha      = alpha
        self.p          = p

        self.A = torch.stack(
            [strict_standard_normal(enc_dim, seed+i) for i in range(pt_dim)], 
            dim=0
        ) * alpha
        self.A = nn.Parameter(self.A, False)                                    # (3, d)

        self.B = torch.stack(
            [strict_standard_normal(p, seed+i) for i in range(pt_dim)], 
            dim=0
        ) * 1.8
        self.B = nn.Parameter(self.B, False)                                    # (3, d)

    @torch.no_grad()
    def forward(self, pts):
        """ Given a point set, compute local geometry encoding for each point.
        Args:
            pts: torch.Tensor, (N, self.pt_dim), input point cloud. 
                 ** X in Eqn. (2) in the paper **
        Returns:
            enc: torch.Tensor, (N, self.enc_dim), local geometry encoding.
                 ** G in Eqn. (2) in the paper **
        """
        assert pts.dim() == 2 and pts.size(1) == self.pt_dim, "Input tensor should be (N, self.pt_dim)"
        pA = (pts / self.radius) @ self.A                                       # (N, 3) @ (3, d) = (N, d)
        pB = (pts / self.radius) @ self.B                                       # (N, 3) @ (3, p) = (N, p)
        eA = torch.concatenate((torch.cos(pA), torch.sin(pA)), dim=-1)          # (N, 2d)
        eB = torch.concatenate((torch.cos(pB), torch.sin(pB)), dim=-1)          # (N, 2p)
        G = torch.matmul(
            eB,                                                                  
            eB.transpose(-1,-2) @ eA                                            
        )                                                                       # (N, 2p) @ (N, 2p).T @ (N, 2d) = (N, 2d)
        G = torch.complex(
            G[..., :self.enc_dim], G[..., self.enc_dim:]
        ) / torch.complex(
            eA[..., :self.enc_dim], eA[..., self.enc_dim:]
        )                                                                       # Complex(N, d)
        G = G / torch.norm(G, dim=-1, keepdim=True) * self.sqrt_d
        return G

    def __repr__(self):
        return f"FastVecKM(pt_dim={self.pt_dim}, enc_dim={self.enc_dim}, radius={self.radius}, alpha={self.alpha}, p={self.p})"