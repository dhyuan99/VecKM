# Examples
You could checkout the [example.ipynb](example.ipynb) for some visualizations, explanations. You could also checkout the [example.py](example.py) for the usage of VecKM library.

By running `python example.py`, here is the result I get:

```
VecKM(alpha=30, beta2=40.5, d=256, positional_encoding=False)
VecKM(alpha=30, beta=9, d=256, p=4096)

Test 1: both implementation gives the same result as the pytorch implementation.
runtime efficient err: 3.93712156210313e-08
memory efficient err: 2.9681839919248887e-07

Test 2: Local geometry encoding is shift invariant.
after shifting err: 3.2316979741153773e-06

Test 3: Isometry is consistent across the baseline and the linear time and space implementation.
corrcoef between geometry similarity. 0.9254295054641608

Time the implementation.
Baseline (pytorch) Implementation. Mean time: 70.33217544555664, Std: 0.2942326244448031
Memory Efficient Implementation. Mean time: 475.9087361653646, Std: 3.192807127700145
Runtime Efficient Implementation. Mean time: 34.13845621744792, Std: 0.25986100521941197
Linear Time and Space Implementation. Mean time: 29.085452842712403, Std: 0.32273889358448765
```

You can also shoot for faster execution with lower encoding quality. For example,
```
VecKM(alpha=30, beta=9, d=128, p=4096)

Test 3: Isometry is consistent across the baseline and the linear time and space implementation.
corrcoef between geometry similarity. 0.8952530351296223

Linear Time and Space Implementation. Mean time: 16.553675587972005, Std: 0.07141736938634709
```