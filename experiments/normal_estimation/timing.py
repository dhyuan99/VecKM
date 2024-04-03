"""
Use this script to time the forward pass of the normal estimation model.
"""

import numpy as np
import torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='VecKM_')
parser.add_argument('--alpha_list', nargs='+', type=float, default=[90])
parser.add_argument('--beta_list', nargs='+', type=float, default=[18])
# PointNet, DGCNN parameters.
parser.add_argument('--num_neighbors', type=int, default=500)
# KPConv parameters.
parser.add_argument('--kp', type=int, default=16)
args = parser.parse_args()
print(args)

import sys
sys.path.append('./models')
import importlib
MODEL = importlib.import_module(args.model)
model = MODEL.NormalEstimator(args).cuda()

def timing(model, pts, repititions=30):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        model(pts, pts)
    timings = []
    with torch.no_grad():
        for rep in range(repititions):
            starter.record()
            model(pts, pts)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings.append(curr_time)
    return np.mean(timings), np.std(timings)

for input_size in [1000, 5000, 10000, 50000, 100000]:
    pts = torch.randn(input_size, 3).cuda() 
    mean_time, std_time = timing(model, pts)
    print(f"Input size: {input_size}, Mean time: {mean_time}, Std: {std_time}")