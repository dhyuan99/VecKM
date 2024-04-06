<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> VecKM: A Linear Time and Space Local Point Cloud Geometry Encoder </h1>

<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://www.cs.umd.edu/~dhyuan" target="_blank" style="text-decoration: none;">Dehao Yuan</a>&nbsp;,&nbsp;
    <a href="http://users.umiacs.umd.edu/~fer/" target="_blank" style="text-decoration: none;">Cornelia Fermüller</a>&nbsp;,&nbsp;
    <a href="https://www.cs.umd.edu/people/trabbani" target="_blank" style="text-decoration: none;">Tahseen Rabbani</a>&nbsp;,&nbsp;
    <a href="https://furong-huang.com" target="_blank" style="text-decoration: none;">Furong Huang</a>&nbsp;,&nbsp;
    <a href="http://users.umiacs.umd.edu/~yiannis/" target="_blank" style="text-decoration: none;">Yiannis Aloimonos</a>&nbsp;&nbsp;
</p>

<p align='center';>
<b>
<em>arXiv-Preprint, 2024</em> &nbsp&nbsp&nbsp&nbsp <a href="http://arxiv.org/abs/2404.01568" target="_blank" style="text-decoration: none;">[arXiv]</a>
</b>
</p>

## Highlighted Features
<img src="assets/teasor_explained.jpg" style="width:100%">

## Installation
First, install the dependencies:
```
conda create -n VecKM python=3.11
conda activate VecKM
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge cudatoolkit-dev
pip install scipy
pip install complexPyTorch
```
If you want to use the pure PyTorch implementation (slower but more convenient), without CUDA runtime and memory optimization, simply install by:
```
pip install .
```
If you want to use the CUDA optimized implementation, please run:
```
cd src/cuvkm
python setup.py install
cd -
pip install .
```

## Usage
#### Case 1: If you have small point cloud size, e.g. < 5000, it is recommended to use the following implementation:
```
from VecKM.cuvkm.cuvkm import VecKM
vkm = VecKM(d=128, alpha=30, beta=9, positional_encoding=False).cuda()
```
Or if you want to use the slower Python implementation without installation,
```
from VecKM.pyvkm.vkm_small import VecKM
vkm = VecKM(d=128, alpha=30, beta=9, positional_encoding=False).cuda()
```
#### Case 2: If you have large point cloud size, e.g. > 10000, it is recommended to use the following implementation:
```
from VecKM.pyvkm.vkm_large import VecKM
vkm = VecKM(d=256, alpha=30, beta=9, p=2048).cuda()
# Please refer to the "Implementation by Yourself" section for suggestion to pick d and p.
```
Then you will get a local geometry encoding by:
```
pts = torch.randn(n, 3).cuda() # your input point cloud.
G = vkm(pts)
```
#### Caution: VecKM is sensitive to scaling. Please make sure your data is properly scaled before passing into VecKM.

## Implementation by Yourself

If you are struggled with installation (e.g. due to some environment issues), it is very simple to implement VecKM if you want to incorporate it into your own code. Suppose your input point cloud `pts` has shape `(n,3)` or `(b,n,3)`, then the following code will give you the VecKM local geometry encoding with output shape `(n,d)` or `(b,n,d)`. It is recommended to have PyTorch >= 1.13.0 since it has better support for complex tensors, but lower versions shall also work.

``` python
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm

def strict_standard_normal(d):
    # this function generate very similar outcomes as torch.randn(d)
    # but the numbers are strictly standard normal, no randomness.
    y = np.linspace(0, 1, d+2)
    x = norm.ppf(y)[1:-1]
    np.random.shuffle(x)
    x = torch.tensor(x).float()
    return x

class VecKM(nn.Module):
    def __init__(self, d=256, alpha=30, beta=9, p=4096):
        """ I tested empirically, here are some general suggestions for selecting parameters d and p: 
        d = 256, p = 4096 is for point cloud size ~20k. Runtime is about 28ms.
        d = 128, p = 8192 is for point cloud size ~50k. Runtime is about 76ms.
        For larger point cloud size, please enlarge p, but if that costs too much, please reduce d.
        A general empirical phenomenon is (d*p) is postively correlated with the encoding quality.

        For the selection of parameter alpha and beta, please see the github section below.
        """
        super().__init__()
        self.sqrt_d = d ** 0.5

        self.A = torch.stack(
            [strict_standard_normal(d) for _ in range(3)], 
            dim=0
        ) * alpha
        self.A = nn.Parameter(self.A, False)                                    # (3, d)

        self.B = torch.stack(
            [strict_standard_normal(p) for _ in range(3)], 
            dim=0
        ) * beta
        self.B = nn.Parameter(self.B, False)                                    # (3, d)

    def forward(self, pts):
        """ Compute the dense local geometry encodings of the given point cloud.
        Args:
            pts: (bs, n, 3) or (n, 3) tensor, the input point cloud.

        Returns:
            G: (bs, n, d) or (n, d) tensor
               the dense local geometry encodings. 
               note: it is complex valued. 
        """
        pA = pts @ self.A                                                       # Real(..., n, d)
        pB = pts @ self.B                                                       # Real(..., n, p)
        eA = torch.concatenate((torch.cos(pA), torch.sin(pA)), dim=1)           # Real(..., n, 2d)
        eB = torch.concatenate((torch.cos(pB), torch.sin(pB)), dim=1)           # Real(..., n, 2p)
        G = torch.matmul(
            eB,                                                                 # Real(..., n, 2p)
            eB.transpose(-1,-2) @ eA                                            # Real(..., 2p, 2d)
        )                                                                       # Real(..., n, 2d)
        G = torch.complex(
            G[:,:self.d], G[:,self.d:]
        ) / torch.complex(
            eA[:,:self.d], eA[:,self.d:]
        )                                                                       # Complex(..., n, d)
        G = G / torch.norm(G, dim=-1, keepdim=True) * self.sqrt_d
        return G

vkm = VecKM()
pts = torch.rand((10,1000,3))
print(vkm(pts).shape) # it will be Complex(10,1000,256)
pts = torch.rand((1000,3))
print(vkm(pts).shape) # it will be Complex(1000, 256)

from complexPyTorch.complexLayers import ComplexLinear, ComplexReLU
# You may want to use apply two-layer feature transform to the encoding.
feat_trans = nn.Sequential(
    ComplexLinear(256, 128),
    ComplexReLU(),
    ComplexLinear(128, 128)
)
G = feat_trans(vkm(pts))
G = G.real**2 + G.imag**2 # it will be Real(10, 1000, 128) or Real(1000, 1024).
```

## Effect of Parameters $\alpha$ and $\beta$
There are two parameters `alpha` and `beta` in the VecKM encoding. They are controlling the **resolution** and **receptive field** of VecKM, respectively. A higher `alpha` will produce a more detailed encoding of the local geometry, and a smaller `alpha` will produce a more abstract encoding. A higher `beta` will result in a smaller receptive field. You could look at the figure below for a rough understanding.

<img src="assets/parameters.jpg" style="width:80%">

**Assuming your input is normalized within a ball with radius 1.** The overall advice for picking `alpha` and `beta` will be, if your task is low-level, such as feature matching, normal estimation, then `alpha` in range ``(60, 120)`` is suggested. If your task is high-level, such as classification and segmentation, then `alpha` in range ``(20, 30)`` is suggested. For `beta`, it is closely related to the neighborhood radius. We provide a table of the correspondence. For example, if you want to extract the local geometry encoding with radius 0.3, then you would select beta to be 6.

<table>
<thead>
  <tr>
    <td>beta</td>
    <td>1</td>
    <td>2</td>
    <td>3</td>
    <td>4</td>
    <td>5</td>
    <td>6</td>
    <td>7</td>
    <td>8</td>
    <td>9</td>
    <td>10</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>radius</td>
    <td>1.800</td>
    <td>0.900</td>
    <td>0.600</td>
    <td>0.450</td>
    <td>0.360</td>
    <td>0.300</td>
    <td>0.257</td>
    <td>0.225</td>
    <td>0.200</td>
    <td>0.180</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>beta</td>
    <td>11</td>
    <td>12</td>
    <td>13</td>
    <td>14</td>
    <td>15</td>
    <td>16</td>
    <td>17</td>
    <td>18</td>
    <td>19</td>
    <td>20</td>
  </tr>
  <tr>
    <td>radius</td>
    <td>0.163</td>
    <td>0.150</td>
    <td>0.138</td>
    <td>0.129</td>
    <td>0.120</td>
    <td>0.113</td>
    <td>0.106</td>
    <td>0.100</td>
    <td>0.095</td>
    <td>0.090</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>beta</td>
    <td>21</td>
    <td>22</td>
    <td>23</td>
    <td>24</td>
    <td>25</td>
    <td>26</td>
    <td>27</td>
    <td>28</td>
    <td>29</td>
    <td>30</td>
  </tr>
  <tr>
    <td>radius</td>
    <td>0.086</td>
    <td>0.082</td>
    <td>0.078</td>
    <td>0.075</td>
    <td>0.072</td>
    <td>0.069</td>
    <td>0.067</td>
    <td>0.065</td>
    <td>0.062</td>
    <td>0.060</td>
  </tr>
</tbody>
</table>

## Examples
Check out the [examples](examples) for the example analysis of VecKM.

## Experiments
Check out the applications of VecKM to [normal estimation](experiments/normal_estimation), [classification](experiments/classification), [part segmentation](experiments/part_segmentation). The overall architecture change will be like:

<img src="assets/deep_VecKM.jpg" style="width:80%">

<!-- # Todo List
- [ ] VecKM library documentation.
- [ ] VecKM CUDA library efficiency improvement.
- [ ] Normal estimation baseline outputs.
- [ ] Use VecKM CUDA library to rerun classification experiments.
- [ ] Use VecKM CUDA library to rerun part segmentation experiments.
- [ ] Semantic segmentation code organization and documentation.

I am actively maintaining this repository. If you have any question, please feel free to raise an issue or contact me through my email [dhyuan@umd.edu](dhyuan@umd.edu). If the issue is important, I will post it to the Todo list. -->

# Citation
If you find it helpful, please consider citing our papers:
```
@misc{yuan2024linear,
      title={A Linear Time and Space Local Point Cloud Geometry Encoder via Vectorized Kernel Mixture (VecKM)}, 
      author={Dehao Yuan and Cornelia Fermüller and Tahseen Rabbani and Furong Huang and Yiannis Aloimonos},
      year={2024},
      eprint={2404.01568},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
