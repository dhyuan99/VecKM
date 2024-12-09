import numpy as np
import torch
from encoder import ExactVecKM, FastVecKM
from visualize import check_vkm_quality_3d

pts = np.loadtxt('Liberty100k.xyz')
print('The shape of pts is (n, 3):', pts.shape)
print('Preprocessing the data... Normalize the points into a unit ball.')
pts = pts - pts.mean(axis=0, keepdims=1)
r = np.linalg.norm(pts, axis=1).max()
pts = pts / r
pts[0] = np.array([0.00622046,0.62610329,0.07458451]) # this will be the specific point we look into later on.
print(pts)

pts = torch.tensor(pts).float()
pts = pts.cuda()

vkm = ExactVecKM(pt_dim=3, enc_dim=384, radius=0.1)
vkm = vkm.cuda()
# G = vkm(pts)

# vkm = FastVecKM(pt_dim=3, enc_dim=384, radius=0.1)
# vkm = vkm.cuda()

check_vkm_quality_3d(vkm, pts, 0)