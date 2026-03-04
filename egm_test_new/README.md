# GB200 EGM Memory Bandwidth Test

针对 NVIDIA GB200 (Blackwell) 的 Extended GPU Memory (EGM) 特性的内存带宽测试工具。

## 功能特性

测试以下内存类型的读写性能：

1. **HBM (High Bandwidth Memory)** - GPU 本地内存，使用 `cudaMalloc`
2. **Host Pinned Memory** - 固定页的主机内存，使用 `cudaMallocHost`
3. **EGM (VMM API)** - 使用 Virtual Memory Management API 的扩展 GPU 内存
4. **EGM (Memory Pool)** - 使用 Stream-Ordered Memory Pool 的扩展 GPU 内存
5. **H2D/D2H Transfer** - 主机与设备之间的数据传输

## 硬件要求

- NVIDIA GB200 或其他支持 EGM 的 Grace-Blackwell 系统
- NVLink-C2C 连接
- CUDA 12.9+ (推荐 CUDA 13.0)

## GPU 架构对照

| 架构 | Compute Capability | 代表 GPU | Makefile 目标 |
|------|-------------------|---------|--------------|
| Blackwell | sm_100 (CC 10.0) | B100, B200, GB200 | `make sm100` / `make gb200` |
| Hopper | sm_90 (CC 9.0) | H100, H200 | `make sm90` |
| Ampere | sm_80 (CC 8.0) | A100, A30 | `make sm80` |

## 编译

```bash
# 针对 GB200/Blackwell 编译
make gb200

# 或者
make sm100

# 编译支持多种架构（推荐）
make all

# 清理
make clean
```

## 使用方法

### 基本用法

```bash
# 运行默认测试 (256MB, 100 iterations)
./egm_bandwidth_test

# 自定义参数
./egm_bandwidth_test -s 512 -i 200 -d 0
```

### 参数说明

```
-s, --size <MB>       测试缓冲区大小（MB），默认: 256
-i, --iterations <N>  迭代次数，默认: 100
-d, --device <ID>     GPU 设备 ID，默认: 0
-h, --help            显示帮助信息
```

### 使用示例

```bash
# 小规模快速测试
./egm_bandwidth_test -s 64 -i 50

# 大规模测试
./egm_bandwidth_test -s 1024 -i 200

# 指定 GPU
./egm_bandwidth_test -d 1 -s 512 -i 100
```

## 预期输出

```
╔══════════════════════════════════════════════════════════════╗
║     GB200 EGM Memory Bandwidth Test                          ║
║     Testing HBM, EGM, and Host Pinned Memory                 ║
╚══════════════════════════════════════════════════════════════╝

========================================
GPU Device Information
========================================
Device Name: NVIDIA B200
Compute Capability: 10.0
Total Global Memory: 192.00 GB
Multiprocessors: 256
Host NUMA ID: 0 (EGM Supported)
GPU Direct RDMA: Supported
========================================

╔══════════════════════════════════════════════════════════════════════════════╗
║                           Summary of Results                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Memory Type          │ Read (GB/s) │ Write (GB/s) │ Copy (GB/s) │ Latency(us)║
╠══════════════════════════════════════════════════════════════════════════════╣
║ HBM                  │     7842.15 │      7856.32 │     15698.47│       2.56 ║
║ Host Pinned          │      180.45 │       175.23 │       355.68│      14.23 ║
║ EGM (VMM)            │      218.34 │       221.56 │       439.90│      11.78 ║
║ EGM (Pool)           │      220.12 │       219.87 │       440.00│      11.65 ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

## 性能预期

根据 GB200 硬件规格：

| 内存类型 | 理论带宽 | 实测预期 |
|---------|---------|---------|
| HBM | ~8000 GB/s | ~7500-8000 GB/s |
| EGM (NVLink-C2C) | ~225 GB/s (单向) | ~200-220 GB/s |
| Host Pinned | 取决于 PCIe/C2C | ~150-200 GB/s |

## 技术说明

### EGM (Extended GPU Memory)

EGM 允许 GPU 直接访问 CPU 附加的内存（LPDDR），通过高速 NVLink-C2C 连接：

- **容量**: GB200 配备 480GB LPDDR5X
- **带宽**: 单向 225 GB/s (双向 450 GB/s)
- **延迟**: 比 HBM 高，但比 PCIe 低得多

### 两种 EGM API

1. **Virtual Memory Management (VMM) API**
   - 使用 CUDA Driver API
   - `cuMemCreate` + `CU_MEM_LOCATION_TYPE_HOST_NUMA`
   - 更底层，控制更精细

2. **Stream-Ordered Memory Pool**
   - 使用 CUDA Runtime API
   - `cudaMemPoolCreate` + `cudaMemLocationTypeHostNuma`
   - 更高层，使用更简单

## 故障排除

### EGM 不可用

如果看到 "EGM may not be supported"：

1. 确认硬件支持 (GB200 或类似平台)
2. 检查 NUMA ID: `numactl --hardware`
3. 确认 CUDA 版本 >= 12.9

### 编译错误

如果遇到编译错误：

```bash
# 检查 CUDA 版本
nvcc --version

# 确保 CUDA 13.0+
# 如果是旧版本，可能需要升级或使用兼容的目标
make sm90  # 对于 Hopper
make sm80  # 对于 Ampere
```

## 参考文档

- [NVIDIA Extended GPU Memory Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-gpu-memory)
- [GB200 NVL72 Technical Brief](https://www.nvidia.com/en-us/data-center/grace-blackwell-superchip/)
- [CUDA Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#virtual-memory-management)

## 许可

MIT License

## 作者

开发用于 GB200 EGM 特性测试和性能评估。
