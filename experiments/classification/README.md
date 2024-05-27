## ModelNet-40 Classification Experiment

This repo implements the ModelNet-40 classification experiments presented in the paper. The experiment involves six architectures: 

1. PointNet: `models/pointnet.py`. This is directly copied from [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) for comparison with Architecture 2.
2. PointNet_VecKM: `models/pointnet_VecKM.py`. We replace the `PointNetEncoder` in the original architecture with our VecKM module.
3. PointNet2: `models/pointnet2.py`. This is directly copied from [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) for comparison with Architecture 4.
4. PointNet2_VecKM: `models/pointnet2_VecKM.py`. We replace the first set abstraction layer with VecKM.
5. PCT: Point Cloud Transformer `models/PCT.py`. This is directly copied from [PCT_Pytorch](https://github.com/Strawberry-Eat-Mango/PCT_Pytorch) for comparison with Architecture 6.
6. PCT_VecKM: `models/PCT_VecKM.py` We replace the initial point embedding module in PCT with our VecKM.

The data augmentation and training strategies are borrowed from [PCT_Pytorch](https://github.com/Strawberry-Eat-Mango/PCT_Pytorch). Many thanks to their great codes!

### Data Preparation

Please download the data [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) and unzip the file into `./data/modelnet40_ply_hdf5_2048`. The file structure shall look like:

```
./
├── data
│   └── modelnet40_ply_hdf5_2048
│       ├── ply_data_test0.h5
│       ├── ply_data_test_0_id2file.json
│       ├── ply_data_test1.h5
│       ├── ply_data_test_1_id2file.json
│       ├── ply_data_train0.h5
│       ├── ply_data_train_0_id2file.json
│       ├── ply_data_train1.h5
│       ├── ply_data_train_1_id2file.json
│       ├── ply_data_train2.h5
│       ├── ply_data_train_2_id2file.json
│       ├── ply_data_train3.h5
│       ├── ply_data_train_3_id2file.json
│       ├── ply_data_train4.h5
│       ├── ply_data_train_4_id2file.json
│       ├── shape_names.txt
│       ├── test_files.txt
│       └── train_files.txt
├── data.py
├── main.py
├── models
│   ├── pointnet.py
│   ├── pointnet_VecKM.py
│   ├── pointnet2.py
│   ├── pointnet2_VecKM.py
│   ├── PCT.py
│   ├── PCT_VecKM.py
├── README.md
└── util.py
```

### Requirements

```
python >= 3.8
pytorch >= 1.9
h5py
scikit-learn
scipy
```

### Models

We get the following accuracies by setting random seed as 0. Different GPUs will produce different results. My result is given by an RTXA5000 GPU.

|                   | Instance Accuracy | Avg. Class Accuracy | Inference Time (1 batch)  | # parameters |
| ----------------- | :---------------: | :-----------------: | :-----------------------: | :----------: |
| PointNet          |       90.8%       |        87.1%        |          3.03 ms          |    1.61M     |
| PointNet + VecKM  |   92.9% (+2.1%)   |    89.7% (+2.6%)    |  14.3 ms   |    9.06M     |
| PointNet2         |       92.8%       |        89.4%        |         117 ms          |    1.48M     |
| PointNet2 + VecKM |   93.0% (+0.2%)   |    89.7% (+0.3%)    | 40.67 ms (1.9x faster) |    3.94M     |
| PCT               |       92.5%       |        89.2%        |         149.72 ms          |    2.88M     |
| PCT + VecKM       |   93.0% (+0.5%)   |    90.5% (+1.3%)    |     21.4 (6x faster)      |    5.07M     |

### Training

```shell script
python main.py --model pointnet
python main.py --model pointnet_VecKM
python main.py --model pointnet2
python main.py --model pointnet2_VecKM
python main.py --model PCT
python main.py --model PCT_VecKM
```
