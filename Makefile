gpu:
	/sw/wombat/Nvidia_HPC_SDK/Linux_aarch64/22.1/compilers/bin/nvcc -I/ccsopen/home/mhawkins60/local/zfp/include -L/ccsopen/home/mhawkins60/local/zfp/build_gpu_staging/lib64 -lzfp  zfp_bench.cpp -o zfp_bench_gpu
