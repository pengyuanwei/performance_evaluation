import cupy as cp

x = cp.arange(10)
print(x)
print(cp.__version__)
print(cp.cuda.runtime.getDeviceProperties(0)['name'])