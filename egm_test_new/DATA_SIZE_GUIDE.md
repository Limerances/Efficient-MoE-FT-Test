# EGM Bandwidth Test - 数据量选择指南

## 📊 当前配置

### 默认设置（已更新）
```cpp
DEFAULT_SIZE = 1024 MB (1 GB)  // 之前是 256 MB
DEFAULT_ITERATIONS = 50         // 之前是 100
WARMUP_ITERATIONS = 10
```

**为什么从 256 MB 改为 1 GB？**
- ✅ 更接近实际应用场景
- ✅ 充分测试 EGM 的 TLB 和页表性能
- ✅ 避免缓存效应掩盖真实带宽
- ✅ 对 480 GB 的 EGM 来说，1 GB 更有代表性

## 🎯 不同数据量的测试目的

### 小数据量：64-256 MB

**适用场景**：
- 快速验证测试程序正确性
- 开发和调试阶段
- CI/CD 自动化测试

**特点**：
- ✅ 测试快 (~5秒)
- ⚠️ 可能受缓存影响
- ⚠️ 对 EGM 不够真实

**命令**：
```bash
./egm_bandwidth_test -s 128 -i 50  # 快速测试
```

### 中等数据量：512 MB - 1 GB

**适用场景**：
- **日常性能测试（推荐）** ⭐
- 开发过程中的性能验证
- 基准测试和对比

**特点**：
- ✅ 平衡测试精度和时间
- ✅ 充分测试内存带宽
- ✅ 适合 EGM TLB 压力测试
- ✅ 测试时间适中 (~10-20秒)

**命令**：
```bash
./egm_bandwidth_test              # 使用默认 1GB
./egm_bandwidth_test -s 512 -i 50 # 512MB
```

### 大数据量：2-4 GB

**适用场景**：
- **深度 EGM 性能分析** ⭐⭐
- 模拟实际大规模应用
- 研究 NUMA 和页表性能
- 发现性能瓶颈

**特点**：
- ✅ 最接近实际应用
- ✅ 充分暴露 TLB Miss
- ✅ 真实的 NUMA 开销
- ⚠️ 测试时间较长 (~30-60秒)
- ⚠️ 需要足够内存

**命令**：
```bash
./egm_bandwidth_test -s 2048 -i 30  # 2GB
./egm_bandwidth_test -s 4096 -i 20  # 4GB
```

### 超大数据量：8+ GB

**适用场景**：
- 极限性能测试
- 研究超大数据集的行为
- 验证内存系统稳定性

**特点**：
- ✅ 测试极限场景
- ⚠️ 可能超出 HBM 容量（186 GB）
- ⚠️ 测试时间很长 (>2分钟)
- ⚠️ 仅对 EGM 有意义

**命令**：
```bash
./egm_bandwidth_test -s 8192 -i 10   # 8GB (仅 EGM)
./egm_bandwidth_test -s 16384 -i 5   # 16GB (仅 EGM)
```

## 📈 推荐测试策略

### 策略 1: 快速验证
```bash
# 编译
make gb200

# 快速测试（~10秒）
./egm_bandwidth_test -s 256 -i 30
```

**用途**: 验证程序能跑、EGM 可用

---

### 策略 2: 标准测试（推荐）⭐
```bash
# 使用默认参数（1GB）
./egm_bandwidth_test

# 或明确指定
./egm_bandwidth_test -s 1024 -i 50
```

**用途**: 日常性能测试、基准对比

---

### 策略 3: 完整性能曲线 ⭐⭐
```bash
# 使用提供的脚本
chmod +x test_bandwidth_sweep.sh
./test_bandwidth_sweep.sh
```

**测试范围**: 64MB → 4GB，7个数据点

**用途**: 
- 发现性能随数据量的变化
- 找到最优工作点
- 研究缓存、TLB 效应

---

### 策略 4: 深度 EGM 分析 ⭐⭐⭐
```bash
# 专注于 EGM 的大数据测试
for size in 1024 2048 4096 8192; do
    echo "Testing ${size}MB..."
    ./egm_bandwidth_test -s $size -i 20
    sleep 3
done
```

**用途**: EGM 性能研究、论文数据

## 🔍 预期性能趋势

### HBM (GPU 本地内存)

| 数据量 | 预期带宽 | 说明 |
|--------|---------|------|
| 64 MB  | 7800-8000 GB/s | 可能略低（启动开销） |
| 256 MB | 7900-8000 GB/s | 接近峰值 |
| 1 GB   | 7900-8000 GB/s | **峰值性能** |
| 4 GB   | 7900-8000 GB/s | 持平 |

**结论**: HBM 在 256MB 以上就能达到峰值，**不需要** GB 级别

### EGM (Extended GPU Memory)

| 数据量 | 预期带宽 | TLB Miss 率 | 说明 |
|--------|---------|------------|------|
| 64 MB  | 210-220 GB/s | 低 | TLB 覆盖率高 |
| 256 MB | 200-215 GB/s | 中 | 开始出现 TLB Miss |
| 1 GB   | 190-210 GB/s | 中高 | **更真实的性能** ⭐ |
| 4 GB   | 180-200 GB/s | 高 | 大量 TLB Miss |
| 8+ GB  | 170-190 GB/s | 很高 | 极限压力 |

**结论**: EGM **非常需要** GB 级别测试！

### 为什么 EGM 带宽会随数据量下降？

1. **TLB Miss** 增加
   - EGM 使用 2MB 页
   - 1 GB 需要 512 个 TLB 条目
   - 4 GB 需要 2048 个条目（超出 TLB 容量）

2. **页表遍历开销**
   - 更多 page walk
   - 访问多级页表

3. **NUMA 效应**
   - 跨 NUMA 节点访问
   - 远程内存控制器压力

## 🎯 最终推荐

### 对于 HBM 测试
```bash
./egm_bandwidth_test -s 256 -i 100  # 256MB 足够
```

### 对于 EGM 测试（重点）
```bash
# 单次测试（推荐 1GB）
./egm_bandwidth_test -s 1024 -i 50

# 完整分析（推荐）
./test_bandwidth_sweep.sh
```

### 对于论文/报告
```bash
# 完整的性能曲线
./test_bandwidth_sweep.sh

# 加上超大数据测试
./egm_bandwidth_test -s 8192 -i 10
./egm_bandwidth_test -s 16384 -i 5
```

## 📊 数据量 vs 测试时间估算

### HBM (8000 GB/s)

| 数据量 | 单次耗时 | 50次迭代 | 100次迭代 |
|--------|---------|---------|----------|
| 256 MB | ~32 ms  | ~2 秒   | ~3 秒    |
| 1 GB   | ~128 ms | ~6 秒   | ~13 秒   |
| 4 GB   | ~512 ms | ~26 秒  | ~51 秒   |

### EGM (200 GB/s)

| 数据量 | 单次耗时 | 50次迭代 | 100次迭代 |
|--------|---------|---------|----------|
| 256 MB | ~1.3 秒 | ~65 秒  | ~130 秒  |
| 1 GB   | ~5 秒   | ~250 秒 | ~500 秒  |
| 4 GB   | ~20 秒  | ~1000秒 | ~2000秒  |

**注意**: 以上是纯传输时间，加上 warmup 和其他开销，实际时间会更长。

## 💡 实用建议

1. **开发阶段**: 使用 256 MB，快速迭代
   ```bash
   make gb200 && ./egm_bandwidth_test -s 256 -i 30
   ```

2. **正式测试**: 使用 1 GB (默认)
   ```bash
   ./egm_bandwidth_test
   ```

3. **性能分析**: 使用数据量扫描
   ```bash
   ./test_bandwidth_sweep.sh
   ```

4. **极限测试**: 使用 4-16 GB
   ```bash
   ./egm_bandwidth_test -s 4096 -i 20
   ```

## 🔬 实验建议

建议做以下对比实验：

```bash
# 实验 1: 数据量对 HBM 的影响（应该很小）
./egm_bandwidth_test -s 128 -i 50
./egm_bandwidth_test -s 1024 -i 50
./egm_bandwidth_test -s 4096 -i 20

# 实验 2: 数据量对 EGM 的影响（应该明显）
# 重点关注 EGM 的性能下降
```

预期结果：
- **HBM**: 带宽基本不变（7500-8000 GB/s）
- **EGM**: 带宽随数据量下降（220 → 180 GB/s）

这个下降曲线就是 **TLB 和页表性能的体现**！

## 📚 参考

- GB200 EGM 使用 2MB 页面
- 典型 TLB 容量：数百到数千条目
- EGM 理论带宽：225 GB/s (单向)
- HBM 理论带宽：8000 GB/s
