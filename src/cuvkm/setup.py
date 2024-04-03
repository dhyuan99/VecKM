from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='vkm_ops',
    author='Dehao Yuan',
    ext_modules=[
        CUDAExtension('vkm_ops', [
            'vkm.cpp',
            'vkm_kernel.cu',
            ],
            extra_compile_args={'cxx': ['-std=c++17'], 'nvcc': ['-std=c++17', '-O3']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
