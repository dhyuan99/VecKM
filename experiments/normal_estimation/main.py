import time
import sys
import importlib
import torch
import torch.nn.functional as F
import numpy as np

def get_filenames(path):
    with open(path, 'r') as f:
        filenames = f.readlines()
        filenames = [x.strip() for x in filenames]
    return filenames
    
def get_dataset(filenames):
    pts_list, normal_list = [], []
    for filename in filenames:
        pts = np.loadtxt(f'data/PCPNet/{filename}.xyz')
        normal = np.loadtxt(f'data/PCPNet/{filename}.normals')
        pts = pts - np.mean(pts, axis=0, keepdims=True)
        pts = pts / np.max(np.linalg.norm(pts, axis=1))
        pts_list.append(torch.from_numpy(pts).float())
        normal_list.append(torch.from_numpy(normal).float())
    return pts_list, normal_list

def random_rotate(pts, normal):
    alpha, beta, gamma = np.random.rand(3) * 2 * np.pi
    Rx = torch.tensor([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    Ry = torch.tensor([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    Rz = torch.tensor([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    R = (Rx @ Ry @ Rz).float().cuda()
    pts = pts @ R.T
    normal = normal @ R.T
    return pts, normal

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='VecKM')
# PointNet, DGCNN parameters.
parser.add_argument('--num_neighbors', type=int, default=500)
# KPConv parameters.
parser.add_argument('--kp', type=int, default=16)
args = parser.parse_args()
print(args)
sys.path.append('./models')
MODEL = importlib.import_module(args.model)

import os
import shutil
from datetime import datetime
cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
if not os.path.exists('log'):
    os.mkdir('log')
if not os.path.exists(f'log/{args.model}'):
    os.mkdir(f'log/{args.model}')
cur_dir = f'log/{args.model}/{str(cur_time)}'
os.mkdir(cur_dir)
shutil.copy('main.py', f'{cur_dir}/main.py')
shutil.copy(f'models/{args.model}.py', f'{cur_dir}/{args.model}.py')
print(f'source code saved to {cur_dir}/*.py')

train_files_list = get_filenames('data/PCPNet/list/trainingset_whitenoise.txt')
train_pts_list, train_normal_list = get_dataset(train_files_list)
test_filenames = [
    'data/PCPNet/list/testset_no_noise.txt',
    'data/PCPNet/list/testset_low_noise.txt',
    'data/PCPNet/list/testset_med_noise.txt',
    'data/PCPNet/list/testset_high_noise.txt',
    'data/PCPNet/list/testset_vardensity_gradient.txt',
    'data/PCPNet/list/testset_vardensity_striped.txt', 
]
test_data_all = []
for filename in test_filenames:
    test_files_list = get_filenames(filename)
    test_pts_list, test_normal_list = get_dataset(test_files_list)
    test_data_all.append((test_pts_list, test_normal_list))
    
model = MODEL.NormalEstimator(args).cuda()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_loss = [100] * 6
for epoch in range(9999999):
    train_total_loss = 0
    model = model.train()
    for pts, normal in zip(train_pts_list, train_normal_list):
        # data augmentation
        pts, normal = pts.cuda(), normal.cuda()
        pts, normal = random_rotate(pts, normal)

        # training.
        pred_normal, gt_normal = model(pts, normal)
        cos_sim = F.cosine_similarity(pred_normal, gt_normal, dim=1).abs()
        loss = -cos_sim.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_total_loss += loss.item()

        # training loss computation.
        cos_sim[cos_sim>1] = 1
        angle = torch.acos(cos_sim) * 180 / np.pi
        rmse = torch.sqrt((angle ** 2).mean())
        train_total_loss += rmse.item()
        
    count = len(train_pts_list)
    print(f'Epoch {epoch * model.point_batch_size / 1e5}. train loss: {train_total_loss/count}. Each shape computes {model.point_batch_size} normals.')

    if (epoch+1) % 100 == 0:
        model = model.eval()
        for i, (test_pts_list, test_normal_list) in enumerate(test_data_all):
            test_total_loss = 0
            for pts, normal in zip(test_pts_list, test_normal_list):
                pts, normal = pts.cuda(), normal.cuda()

                with torch.no_grad():
                    pred_normal, gt_normal = model(pts, normal)

                cos_sim = F.cosine_similarity(pred_normal, gt_normal, dim=1).abs()
                cos_sim[cos_sim>1] = 1
                angle = torch.acos(cos_sim) * 180 / np.pi
                rmse = torch.sqrt((angle ** 2).mean())
                test_total_loss += rmse.item()

            count = len(test_pts_list)
            print(f'\ttest loss: {test_total_loss/count}. each shape computes {model.point_batch_size} normals.')

            if test_total_loss/count < best_loss[i]:
                best_loss[i] = test_total_loss/count
                torch.save(model.state_dict(), f'{cur_dir}/best_model_{i}.pth')
                print(f'best model saved to {cur_dir}/best_model_{i}.pth')

        print(f'best loss: {best_loss}')
