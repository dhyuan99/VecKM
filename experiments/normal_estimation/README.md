# Normal Estimation on PCPNet Dataset

This repo imiplements the normal estimation on PCPNet dataset presented in the paper. The experiment includes the comparison over four architectures:

1. PointNet: `models/PointNet.py`. Network architecture implementation is borrowed from [here](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).
2. KPConv: `models/KPConv.py`. We implement the KPConv by ourselves according to the original KPConv paper.
3. DGCNN: `models/DGCNN.py`. Network architecture implementation is borrowed from [here](https://github.com/WangYueFt/dgcnn).
4. VecKM: `models/VecKM.py`. Our newly proposed $O(n)$ local geometry encoder.

Since computing the dense normal vectors from 100,000 points is not feasible for every architecture, we randomly select `model.point_batch_size` points out of the 100k points and compute their normals. Different architectures have different `point_batch_size` values. A higher `point_batch_size` value means larger scalability.

To reproduce the training, please download the PCPNet dataset from [PCPNet](https://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/) and unzip them into the `./data` folder. The file structure shall look like:

```
normal_estimation
├── data
│   └── PCPNet
│       ├── *.xyz
│       ├── *.normals
│       ├── list
│       └── npy
├── main.py
├── models
│   ├── DGCNN.py
│   ├── KPConv.py
│   ├── PointNet.py
│   └── VecKM.py
├── outputs
│   └── VecKM.out
└── README.md
```

### Models

We get the following performance by training the models for three days. Further training can very likely further improve the performance. While this experiment is used for comparing the performance and inference speed, so we didn't train the models until their convergence, which may take far more time. You could check out the training logs at [./outputs/*.out](./outputs).

|                          |  Average RMSE  | Point Batch Size | Inference Time (100k points) |
| ------------------------ | :-------: | :--------------: | :------------: |
| KPConv, #nbr=700, #kp=16 |   26.67   |       0.5k       |      5.942      |
| DGCNN, #nbr=128          |   21.24   |       10k        |      OOM      |
| PointNet, #nbr=700       |   20.27   |        1k        |      6.136      |
| VecKM (Ours)             | **17.34** |     **100k**     |    **0.149**    |

The table is interpreted in this way: VecKM takes 0.149 second to compute the the normal vectors of 100k points, and similarly for others. So when the input size is 100k, VecKM is **40x** faster than PointNet, and **41x** faster than KPConv.

### Training

```
python main.py --model VecKM
python main.py --model PointNet --num_neighbors 700
python main.py --model KPConv --num_neighbors 700 --kp 16
python main.py --model DGCNN --num_neighbors 128
```