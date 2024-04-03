import torch    
import numpy as np
import matplotlib.pyplot as plt
import VecKM.cuvkm.cuvkm as cuvkm
import VecKM.pyvkm.vkm_large as Lvkm

def error(a, b):
    """relative error between two complex vectors/matrices"""
    return torch.norm(a-b) / (torch.norm(a)+torch.norm(b))

def similarity(a, b):
    """similarity between two copmlex vectors/matrices"""
    return torch.abs(torch.sum(a*b.conj())) / (torch.norm(a)*torch.norm(b))

pointset1 = np.loadtxt('Liberty100k.xyz')
pointset1 = pointset1[np.random.choice(len(pointset1), 20000, replace=False)]
pointset1 = torch.tensor(pointset1).float().cuda()
vkm = cuvkm.VecKM(d=768, alpha=30, beta=9, positional_encoding=False).cuda()
lvkm = Lvkm.VecKM(d=256, alpha=30, beta=9, p=4096).cuda()
print(vkm)
print(lvkm)

res_runtime_eff = vkm(pointset1, memory_efficient=False)
res_memory_eff = vkm(pointset1, memory_efficient=True)
res_baseline = vkm.baseline_forward(pointset1)

print('\nTest 1: both implementation gives the same result as the pytorch implementation.')
print('runtime efficient err:', error(res_runtime_eff, res_baseline).item())
print('memory efficient err:', error(res_memory_eff, res_baseline).item())

res_runtime_eff_trans = vkm(pointset1+1, None, False)
print('\nTest 2: Local geometry encoding is shift invariant.')
print('after shifting err:', error(res_runtime_eff_trans, res_baseline).item())

res_linear = lvkm(pointset1)
res_memory_eff = res_memory_eff / torch.norm(res_memory_eff, dim=-1, keepdim=True)
res_linear = res_linear / torch.norm(res_linear, dim=-1, keepdim=True)
sim_between_points_linear = torch.abs(res_linear[1:] @ res_linear[0].conj()).cpu().numpy()
sim_between_points_baseline = torch.abs(res_memory_eff[1:] @ res_memory_eff[0].conj()).cpu().numpy()
print('\nTest 3: Isometry is consistent across the baseline and the linear time and space implementation.')
print('corrcoef between geometry similarity.', np.corrcoef(sim_between_points_linear, sim_between_points_baseline)[0,1])
plt.scatter(sim_between_points_linear, sim_between_points_baseline)
plt.xlabel('linear')
plt.ylabel('baseline')
plt.savefig('hi.jpg')
plt.close()

def timing(func, repititions=30):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        func(pointset1)
    timings = []
    with torch.no_grad():
        for rep in range(repititions):
            starter.record()
            func(pointset1)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings.append(curr_time)
    return np.mean(timings), np.std(timings)

print('\nTime the implementation.')
func = lambda x: vkm.baseline_forward(x)
mean_time, std_time = timing(func)
print(f"Baseline (pytorch) Implementation. Mean time (ms): {mean_time}, Std: {std_time}")

func = lambda x: vkm(x, memory_efficient=True)
mean_time, std_time = timing(func)
print(f"Memory Efficient Implementation. Mean time (ms): {mean_time}, Std: {std_time}")

func = lambda x: vkm(x, memory_efficient=False)
mean_time, std_time = timing(func)
print(f"Runtime Efficient Implementation. Mean time (ms): {mean_time}, Std: {std_time}")

func = lambda x: lvkm(x)
mean_time, std_time = timing(func)
print(f"Linear Time and Space Implementation. Mean time (ms): {mean_time}, Std: {std_time}")