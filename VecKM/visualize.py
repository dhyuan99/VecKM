import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def create_gif(image_folder, output_file, duration=500):
    """ Create a GIF from images in a folder.
    Args:
    - image_folder (str): Path to the folder containing images.
    - output_file (str): Path for the output GIF file.
    - duration (int): Duration of each frame in milliseconds.
    """
    image_files = sorted(
        [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg', 'bmp'))]
    )
    images = [Image.open(img) for img in image_files]
    images[0].save(
        output_file, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0
    )
    print(f"GIF saved at {output_file}")

def check_vkm_quality_3d(vkm, pts, which_pt, vis_path='quality_check'):
    """ Check the quality of VecKM by visualizing the local geometry encoding of a specific point.
    This will generate a 3D plot showing:
    1. local point cloud around pts[which_pt].
    2. distribution reconstruction of the local point cloud, from the VecKM encoding.
       ** Figure 5 in the paper. **
    """
    assert vkm.__class__.__name__ in ['ExactVecKM', 'FastVecKM'], "Only support ExactVecKM or FastVecKM"
    assert vkm.pt_dim == 3, "Only support 3D point cloud for visualization."
    assert pts.dim() == 2 and pts.size(1) == 3, "Input tensor should be (N, 3)"

    if os.path.exists(vis_path):
        os.system(f'rm -rf {vis_path}')
    os.makedirs(vis_path)
    
    # Get the local point cloud around pts[which_pt]
    dist        = torch.cdist(pts[which_pt].unsqueeze(0), pts)
    nbr_indices = torch.where(dist < vkm.radius)[1]
    nbr_pts     = pts[nbr_indices] - pts[which_pt]
    nbr_pts     = nbr_pts.cpu().numpy()

    # Compute the distribution reconstruction of the local point cloud
    # from the VecKM encoding.
    g = vkm(pts)[which_pt]
    p = np.linspace(-vkm.radius, vkm.radius, 50)
    xx, yy, zz = np.meshgrid(p, p, p)                                           # xx, yy, zz are (50, 50, 50)
    xx_c = (xx[1:,1:,1:] + xx[:-1,:-1,:-1]) / 2                                 # xx_c, yy_c, zz_c are (49, 49, 49)
    yy_c = (yy[1:,1:,1:] + yy[:-1,:-1,:-1]) / 2
    zz_c = (zz[1:,1:,1:] + zz[:-1,:-1,:-1]) / 2
    p = np.stack([xx_c.reshape(-1), yy_c.reshape(-1), zz_c.reshape(-1)], axis=1)# p is (117649, 3)
    p = torch.from_numpy(p).float().to(pts.device)
    p = torch.exp(1j * ((p / vkm.radius) @ vkm.A))                              # (117649, 3) @ (3, d) = (117649, d)
    score = (p @ g.conj()).real / torch.norm(g) / torch.norm(p, dim=1)          # (117649, d) @ (d,) = (117649,)
                                                                                # distribution reconstruction score
    thres = torch.sort(score)[0][-5000].item()                                  # plot the voxel with top 5000 scores   
    score = score.detach().cpu().numpy()
    score = score.reshape(xx_c.shape)
    score_bool = score > thres

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.voxels(xx, yy, zz, score_bool)
    ax.scatter(0, 0, 0, s=100, color='blue', alpha=1)
    ax.scatter(nbr_pts[:,0], nbr_pts[:,1], nbr_pts[:,2], s=10, color='red', alpha=0.5)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xlim3d(-vkm.radius, vkm.radius)
    ax.set_ylim3d(-vkm.radius, vkm.radius)
    ax.set_zlim3d(-vkm.radius, vkm.radius)
    for view_angle in tqdm(range(0, 360, 10), desc='Generating images...'):
        ax.view_init(30, view_angle)
        plt.savefig(os.path.join(
            vis_path, f'{str(view_angle).zfill(3)}.png'
            )
        )
    plt.close()

    create_gif(vis_path, f'{vis_path}.gif', duration=100)