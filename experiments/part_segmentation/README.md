## ShapeNet Part Segmentation Experiment

This repo implements the ShapeNet part segmentation experiment presented in the paper. The experiment involves six architectures: 

1. PointNet: `models/pointnet.py`. This is adopted from [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) for comparison with Architecture 2.
2. PointNet_VecKM: `models/pointnet_VecKM.py`. We replace the `PointNetEncoder` in the original architecture with our VecKM module.
3. Pointnet2 `models/pointnet2.py`. This is adopted from [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) without changing the architecture. It is compared with Architecture 4.
4. PointNet2_VecKM: `models/pointnet2_VecKM.py`. We replace the first set abstraction layer before inputing the point cloud to PointNet++.
5. PCT: Point Cloud Transformer `models/PCT.py`. This is a reproduced version based on the descriptions in [Point Cloud Transformer](https://arxiv.org/abs/2012.09688) and their official [codes](https://github.com/MenghaoGuo/PCT).
6. PCT_VecKM: `models/PCT_VecKM.py` We replace the initial point embedding module in PCT with our VecKM.

The data augmentation and training strategies are borrowed from [PCT_Pytorch](https://github.com/Strawberry-Eat-Mango/PCT_Pytorch). Many thanks to their great codes!
The data augmentation and training strategies are borrowed from [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch). Many thanks to their great codes!

### Data Preparation

Please download the data [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) and unzip the file into `./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`. The file structure shall look like:

```
├── data
│   └── shapenetcore_partanno_segmentation_benchmark_v0_normal
│       ├── 02691156
│       ├── 02773838
│       ├── 02954340
│       ├── 02958343
│       ├── 03001627
│       ├── ...
├── models
│   ├── pointnet.py
│   ├── pointnet_VecKM.py
│   ├── PCT.py
│   ├── PCT_VecKM.py
│   ├── pointnet2.py
│   ├── pointnet2_VecKM.py
│   └── utils.py
├── provider.py
├── README.md
└── main.py
└── data.py
```

### Requirements

```
python >= 3.9
pytorch >= 1.13
scipy
```

### Models

We get the following accuracies by setting random seed as 0. Different GPUs will produce different results. My result is given by an RTXA5000 GPU.

|               |   Instance mIoU  |  Avg. Class mIoU | Inference Time (ms) (1 batch) | # parameters |
|---------------|:----------------:|:----------------:|:-----------------------------:|:------------:|
| PointNet      |      83.1\%      |      77.6\%      |              15.1             |     8.34M    |
| VecKM -- PN   |      84.9\%      |      81.8\%      |              40.8             |     1.29M    |
| PointNet++    |      85.0\%      |      81.9\%      |             130.8             |     1.41M    |
| VecKM -- PN++ |      85.3\%      |      82.0\%      |              65.9             |     1.50M    |
| PCT           |      85.7\%      |      82.6\%      |             145.2             |     1.63M    |
| VecKM -- PCT  |      85.6\%      |      82.3\%      |              46.6             |     1.71M    |


### Training

```shell script
python main.py --model pointnet
python main.py --model pointnet_VecKM
python main.py --model pointnet2
python main.py --model pointnet2_VecKM
python main.py --model PCT
python main.py --model PCT_VecKM
```