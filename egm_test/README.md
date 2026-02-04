# EGM 性能测试工具

独立的 CUDA C++ 测试程序，用于验证 NVIDIA GB200 EGM 特性支持并测量性能。

## 概述

**EGM (Extended GPU Memory)** 是 NVIDIA GB200 架构的特性，允许 GPU 直接访问 Grace CPU 的 LPDDR5X 内存。本测试程序可以：

1. ✅ 检测 CUDA 环境和设备信息
2. ✅ 检测 EGM (`CU_MEM_LOCATION_TYPE_HOST_NUMA`) 支持
3. ✅ 测试三种内存类型的性能：
   - **GPU HBM** - 显存 (cudaMalloc)
   - **Pinned Host** - 锁页内存 (cudaMallocHost) + H2D/D2H
   - **EGM** - Grace CPU NUMA 内存 (cuMemCreate)
4. ✅ 输出性能对比报告

## 目录结构

```
egm_test/
├── CMakeLists.txt   # CMake 构建配置
├── egm_test.cu      # 测试程序源码
└── README.md        # 本文档
```

## 系统要求

- **硬件**: NVIDIA GPU（EGM 需要 GB200/GH200）
- **系统**: Linux
- **软件**:
  - CUDA Toolkit >= 12.0
  - CMake >= 3.18
  - GCC/G++ >= 9.0

## 编译

```bash
# 进入项目目录
cd egm_test

# 创建构建目录
mkdir build && cd build

# 配置
cmake ..

# 编译
make -j$(nproc)
```

## 运行

### 基础用法

```bash
./egm_test
```

默认参数：256 MB 数据，NUMA 节点 0，10 次迭代

### 命令行选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-s, --size <MB>` | 测试数据大小 | 256 |
| `-n, --numa <node>` | NUMA 节点 ID | 0 |
| `-i, --iterations <N>` | 迭代次数 | 10 |
| `-h, --help` | 显示帮助 | - |

### 示例

```bash
# 测试 512 MB 数据
./egm_test -s 512

# 测试 1GB 数据，NUMA 节点 1，20 次迭代
./egm_test -s 1024 -n 1 -i 20

# 查看帮助
./egm_test --help
```

## 输出示例

```
============================================================
 EGM (Extended GPU Memory) 性能测试
============================================================
测试参数:
  数据大小: 256 MB
  NUMA 节点: 0
  迭代次数: 10

============================================================
 CUDA 环境检测
============================================================
CUDA Driver 版本: 12.3
CUDA Runtime 版本: 12.3
CUDA 设备数量: 1

设备 0: NVIDIA GB200
  计算能力: SM 10.0
  显存: 192.00 GB
  架构: Hopper/Blackwell (可能支持 EGM)

============================================================
 EGM 支持检测
============================================================
测试 NUMA 节点 0...
✓ EGM 支持确认
  分配粒度: 2.00 MB

============================================================
 运行性能测试
============================================================

测试 GPU HBM 内存 (cudaMalloc)...
  写入: 2.85 TB/s (0.09 ms)
  读取: 1.42 TB/s (0.18 ms)

测试 Pinned Host 内存 (cudaMallocHost)...
  H2D: 25.60 GB/s (10.00 ms)
  D2H: 25.60 GB/s (10.00 ms)

测试 EGM 内存 (NUMA 节点 0)...
  写入: 900.00 GB/s (0.28 ms)
  读取: 450.00 GB/s (0.57 ms)

============================================================
 性能对比总结
============================================================
测试数据大小: 256.00 MB

内存类型                          写入带宽        读取带宽
------------------------------------------------------------
GPU HBM (cudaMalloc)              2.85 TB/s       1.42 TB/s
Pinned Host (cudaMallocHost)     25.60 GB/s      25.60 GB/s
EGM (HOST_NUMA)                 900.00 GB/s     450.00 GB/s

相对 GPU HBM 的效率:
------------------------------------------------------------
Pinned Host (cudaMallocHost)   写   0.9%      读   1.8%
EGM (HOST_NUMA)                写  31.6%      读  31.7%

测试完成
```

## 关键代码说明

### EGM 内存分配

```cpp
// 设置分配属性 - 指定分配到 Grace CPU NUMA 节点
CUmemAllocationProp prop = {};
prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;  // EGM 关键
prop.location.id = numa_node;  // NUMA 节点 ID

// 创建物理内存
cuMemCreate(&handle, size, &prop, 0);

// 映射到 GPU 可访问的虚拟地址
cuMemAddressReserve(&ptr, size, 0, 0, 0);
cuMemMap(ptr, size, 0, handle, 0);

// 设置 GPU 访问权限
CUmemAccessDesc access = {};
access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
access.location.id = 0;  // GPU 设备 ID
access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
cuMemSetAccess(ptr, size, &access, 1);
```

### 性能测试方法

- **写入测试**: GPU 内核填充数据，测量时间
- **读取测试**: GPU 内核 reduce 求和，测量时间
- **带宽计算**: `带宽 = 数据大小 / 时间`

## 常见问题

### EGM 不支持

如果输出显示 "✗ EGM 不支持"，可能原因：

1. **硬件不支持**: 需要 NVIDIA GB200/GH200 架构
2. **NUMA 节点无效**: 尝试其他节点 ID
3. **权限不足**: 可能需要 root 权限
4. **驱动版本低**: 需要 CUDA 12.0+ 驱动

### 性能低于预期

1. 确保使用正确的 NUMA 节点
2. 增大数据量以减少启动开销
3. 增加迭代次数以获得稳定结果

## 与 egm_system 的区别

| | egm_test | egm_system |
|--|----------|------------|
| 目的 | 硬件验证和性能测试 | 生产级检查点系统 |
| 复杂度 | 单文件，独立运行 | 多组件架构 |
| 功能 | 仅测试 | 内存管理、IPC、持久化 |
