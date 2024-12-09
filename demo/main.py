import numpy as np
import torch
from VecKM.encoder import ExactVecKM, FastVecKM
from VecKM.visualize import check_vkm_quality_3d

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

exact_vkm = ExactVecKM(pt_dim=3, enc_dim=384, radius=0.1)
exact_vkm = exact_vkm.cuda()
G = exact_vkm(pts)
print(f"ExactVecKM: The shape of per-point local geometry encoding is (n, d): {G.shape}")

fast_vkm = FastVecKM(pt_dim=3, enc_dim=384, radius=0.1)
fast_vkm = fast_vkm.cuda()
G = fast_vkm(pts)
print(f"FastVecKM: The shape of per-point local geometry encoding is (n, d): {G.shape}")

check_vkm_quality_3d(exact_vkm, pts, 0, vis_path='exact_vkm_quality_check')
check_vkm_quality_3d(fast_vkm, pts, 0, vis_path='fast_vkm_quality_check')
