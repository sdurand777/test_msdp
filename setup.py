import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='msdpdpvo',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('cuda_corr',
            sources=['multi_slam/msdp_dpvo/altcorr/correlation.cpp', 'multi_slam/msdp_dpvo/altcorr/correlation_kernel.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3',
                         '-gencode=arch=compute_60,code=sm_60',
                         '-gencode=arch=compute_61,code=sm_61',
                         '-gencode=arch=compute_70,code=sm_70',
                         '-gencode=arch=compute_75,code=sm_75',
                         '-gencode=arch=compute_80,code=sm_80',
                         '-gencode=arch=compute_86,code=sm_86',
                         ],
            }),
        CUDAExtension('cuda_ba',
            sources=['multi_slam/msdp_dpvo/fastba/ba.cpp', 'multi_slam/msdp_dpvo/fastba/ba_cuda.cu', 'multi_slam/msdp_dpvo/fastba/block_e.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3',
                         '-gencode=arch=compute_60,code=sm_60',
                         '-gencode=arch=compute_61,code=sm_61',
                         '-gencode=arch=compute_70,code=sm_70',
                         '-gencode=arch=compute_75,code=sm_75',
                         '-gencode=arch=compute_80,code=sm_80',
                         '-gencode=arch=compute_86,code=sm_86',
                         ],
            },
            include_dirs=[
                osp.join(ROOT, 'thirdparty/eigen-3.4.0')]
            ),
        CUDAExtension('lietorch_backends', 
            include_dirs=[
                osp.join(ROOT, 'multi_slam/msdp_dpvo/lietorch/include'), 
                osp.join(ROOT, 'thirdparty/eigen-3.4.0')],
            sources=[
                'multi_slam/msdp_dpvo/lietorch/src/lietorch.cpp', 
                'multi_slam/msdp_dpvo/lietorch/src/lietorch_gpu.cu',
                'multi_slam/msdp_dpvo/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={'cxx': ['-O3'], 
                                'nvcc': ['-O3',
                                         '-gencode=arch=compute_60,code=sm_60',
                                         '-gencode=arch=compute_61,code=sm_61',
                                         '-gencode=arch=compute_70,code=sm_70',
                                         '-gencode=arch=compute_75,code=sm_75',
                                         '-gencode=arch=compute_80,code=sm_80',
                                         '-gencode=arch=compute_86,code=sm_86',
                                         ],
                                }),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

