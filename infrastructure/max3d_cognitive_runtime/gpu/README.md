# MAX-3D GPU Kernels

## bitdrop_gpu_kernel.cu
* RTX 4090 optimized BitDrop v3 collapse
* Vectorized 256-byte block processing
* Warp-level reductions for entropy
* Pattern-Tag-Signature generation
* 4-byte fused collapse output

## turbvec_backend.cu
* TurbVec hybrid backend GPU kernels
* Local deterministic reasoning with tanh activation
* Collapse-expand with masking
* Fused attention kernel

Build with:
```
nvcc -O3 -arch=sm_89 -o bitdrop_kernel bitdrop_gpu_kernel.cu
nvcc -O3 -arch=sm_89 -o turbvec_kernel turbvec_backend.cu
```

Integration point: `max3d_runtime.py` -> `BitDropV3.compress()` can dispatch to CUDA kernels when `torch.cuda.is_available()`.
