/**
 * @file egm_test.cu
 * @brief EGM (Extended GPU Memory) 独立测试程序
 *
 * 功能：
 * 1. 检测 CUDA 环境和设备信息
 * 2. 检测 EGM (CU_MEM_LOCATION_TYPE_HOST_NUMA) 支持
 * 3. 测试不同内存类型的分配和读写性能：
 *    - GPU HBM 内存 (cudaMalloc)
 *    - Pinned Host 内存 (cudaMallocHost)
 *    - EGM 内存 (cuMemCreate + HOST_NUMA) [如果支持]
 * 4. 输出性能对比报告
 *
 * 编译：
 *   mkdir build && cd build
 *   cmake .. && make
 *
 * 运行：
 *   ./egm_test [选项]
 *
 * 选项：
 *   -s, --size <MB>      测试数据大小 (默认: 256 MB)
 *   -n, --numa <node>    NUMA 节点 ID (默认: 0)
 *   -i, --iterations <N> 迭代次数 (默认: 10)
 *   -h, --help           显示帮助
 */

#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// 错误检查宏
// ============================================================================

#define CHECK_CUDA_DRIVER(call)                                                \
  do {                                                                         \
    CUresult err = call;                                                       \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *errStr;                                                      \
      cuGetErrorString(err, &errStr);                                          \
      fprintf(stderr, "CUDA Driver Error at %s:%d: %s (%d)\n", __FILE__,       \
              __LINE__, errStr, err);                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_CUDA_RUNTIME(call)                                               \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Runtime Error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ============================================================================
// 工具函数
// ============================================================================

void print_separator() {
  printf("------------------------------------------------------------\n");
}

void print_header(const char *title) {
  printf("\n");
  printf("============================================================\n");
  printf(" %s\n", title);
  printf("============================================================\n");
}

void format_size(size_t bytes, char *buf, size_t buf_size) {
  const char *units[] = {"B", "KB", "MB", "GB", "TB"};
  int unit_idx = 0;
  double size = (double)bytes;

  while (size >= 1024 && unit_idx < 4) {
    size /= 1024;
    unit_idx++;
  }
  snprintf(buf, buf_size, "%.2f %s", size, units[unit_idx]);
}

void format_bandwidth(double bytes_per_sec, char *buf, size_t buf_size) {
  const char *units[] = {"B/s", "KB/s", "MB/s", "GB/s", "TB/s"};
  int unit_idx = 0;

  while (bytes_per_sec >= 1024 && unit_idx < 4) {
    bytes_per_sec /= 1024;
    unit_idx++;
  }
  snprintf(buf, buf_size, "%.2f %s", bytes_per_sec, units[unit_idx]);
}

// ============================================================================
// CUDA 内核
// ============================================================================

// 写入内核：填充数据
__global__ void write_kernel(float *data, size_t count, float value) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  for (size_t i = idx; i < count; i += stride) {
    data[i] = value;
  }
}

// 读取内核：求和
__global__ void reduce_kernel(float *data, size_t count, float *result) {
  __shared__ float shared[256];

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  float local_sum = 0.0f;

  for (size_t i = idx; i < count; i += gridDim.x * blockDim.x) {
    local_sum += data[i];
  }

  shared[threadIdx.x] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      shared[threadIdx.x] += shared[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(result, shared[0]);
  }
}

// ============================================================================
// 测试结果结构
// ============================================================================

struct BenchmarkResult {
  const char *name;
  bool available;
  double write_bandwidth; // bytes/sec
  double read_bandwidth;  // bytes/sec
  double write_time_ms;
  double read_time_ms;
};

// ============================================================================
// 环境检测
// ============================================================================

bool check_cuda_environment() {
  print_header("CUDA 环境检测");

  CHECK_CUDA_DRIVER(cuInit(0));

  int driver_version;
  CHECK_CUDA_DRIVER(cuDriverGetVersion(&driver_version));
  printf("CUDA Driver 版本: %d.%d\n", driver_version / 1000,
         (driver_version % 1000) / 10);

  int runtime_version;
  CHECK_CUDA_RUNTIME(cudaRuntimeGetVersion(&runtime_version));
  printf("CUDA Runtime 版本: %d.%d\n", runtime_version / 1000,
         (runtime_version % 1000) / 10);

  int device_count;
  CHECK_CUDA_DRIVER(cuDeviceGetCount(&device_count));
  printf("CUDA 设备数量: %d\n", device_count);

  if (device_count == 0) {
    printf("错误: 没有可用的 CUDA 设备\n");
    return false;
  }

  for (int i = 0; i < device_count; i++) {
    CUdevice device;
    CHECK_CUDA_DRIVER(cuDeviceGet(&device, i));

    char name[256];
    CHECK_CUDA_DRIVER(cuDeviceGetName(name, sizeof(name), device));

    int major, minor;
    CHECK_CUDA_DRIVER(cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CHECK_CUDA_DRIVER(cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

    size_t total_mem;
    CHECK_CUDA_DRIVER(cuDeviceTotalMem(&total_mem, device));

    char mem_str[32];
    format_size(total_mem, mem_str, sizeof(mem_str));

    printf("\n设备 %d: %s\n", i, name);
    printf("  计算能力: SM %d.%d\n", major, minor);
    printf("  显存: %s\n", mem_str);

    if (major >= 9) {
      printf("  架构: Hopper/Blackwell (可能支持 EGM)\n");
    }
  }

  return true;
}

bool check_egm_support(int numa_node, size_t *granularity_out) {
  print_header("EGM 支持检测");

  printf("测试 NUMA 节点 %d...\n", numa_node);

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.location.id = numa_node;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  size_t granularity = 0;
  CUresult result = cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

  if (result == CUDA_SUCCESS && granularity > 0) {
    char gran_str[32];
    format_size(granularity, gran_str, sizeof(gran_str));
    printf("✓ EGM 支持确认\n");
    printf("  分配粒度: %s\n", gran_str);
    *granularity_out = granularity;
    return true;
  } else {
    const char *errStr;
    cuGetErrorString(result, &errStr);
    printf("✗ EGM 不支持\n");
    printf("  错误: %s\n", errStr);
    printf("\n可能原因:\n");
    printf("  - 需要 NVIDIA GB200/GH200 架构\n");
    printf("  - NUMA 节点 %d 不存在\n", numa_node);
    printf("  - CUDA 版本过低\n");
    *granularity_out = 0;
    return false;
  }
}

// ============================================================================
// 基准测试：GPU HBM 内存
// ============================================================================

BenchmarkResult benchmark_gpu_hbm(size_t size, int iterations) {
  BenchmarkResult result = {"GPU HBM (cudaMalloc)", true, 0, 0, 0, 0};

  printf("\n测试 GPU HBM 内存 (cudaMalloc)...\n");

  size_t count = size / sizeof(float);
  float *d_data;
  float *d_result;

  CHECK_CUDA_RUNTIME(cudaMalloc(&d_data, size));
  CHECK_CUDA_RUNTIME(cudaMalloc(&d_result, sizeof(float)));

  int block_size = 256;
  int grid_size = min((int)((count + block_size - 1) / block_size), 65535);

  // 预热
  write_kernel<<<grid_size, block_size>>>(d_data, count, 1.0f);
  cudaDeviceSynchronize();

  // 写入测试
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    write_kernel<<<grid_size, block_size>>>(d_data, count, (float)i);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  result.write_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count() /
      iterations;
  result.write_bandwidth = size / (result.write_time_ms / 1000.0);

  // 读取测试
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    float zero = 0.0f;
    cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);
    reduce_kernel<<<grid_size, block_size>>>(d_data, count, d_result);
  }
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();

  result.read_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count() /
      iterations;
  result.read_bandwidth = size / (result.read_time_ms / 1000.0);

  cudaFree(d_data);
  cudaFree(d_result);

  char write_bw[32], read_bw[32];
  format_bandwidth(result.write_bandwidth, write_bw, sizeof(write_bw));
  format_bandwidth(result.read_bandwidth, read_bw, sizeof(read_bw));
  printf("  写入: %s (%.2f ms)\n", write_bw, result.write_time_ms);
  printf("  读取: %s (%.2f ms)\n", read_bw, result.read_time_ms);

  return result;
}

// ============================================================================
// 基准测试：Pinned Host 内存 (H2D/D2H)
// ============================================================================

BenchmarkResult benchmark_pinned_host(size_t size, int iterations) {
  BenchmarkResult result = {"Pinned Host (cudaMallocHost)", true, 0, 0, 0, 0};

  printf("\n测试 Pinned Host 内存 (cudaMallocHost)...\n");

  float *h_pinned;
  float *d_data;

  CHECK_CUDA_RUNTIME(cudaMallocHost(&h_pinned, size));
  CHECK_CUDA_RUNTIME(cudaMalloc(&d_data, size));

  memset(h_pinned, 0, size);

  // H2D 测试
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    cudaMemcpy(d_data, h_pinned, size, cudaMemcpyHostToDevice);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  result.write_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count() /
      iterations;
  result.write_bandwidth = size / (result.write_time_ms / 1000.0);

  // D2H 测试
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    cudaMemcpy(h_pinned, d_data, size, cudaMemcpyDeviceToHost);
  }
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();

  result.read_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count() /
      iterations;
  result.read_bandwidth = size / (result.read_time_ms / 1000.0);

  cudaFreeHost(h_pinned);
  cudaFree(d_data);

  char write_bw[32], read_bw[32];
  format_bandwidth(result.write_bandwidth, write_bw, sizeof(write_bw));
  format_bandwidth(result.read_bandwidth, read_bw, sizeof(read_bw));
  printf("  H2D: %s (%.2f ms)\n", write_bw, result.write_time_ms);
  printf("  D2H: %s (%.2f ms)\n", read_bw, result.read_time_ms);

  return result;
}

// ============================================================================
// 基准测试：EGM 内存
// ============================================================================

BenchmarkResult benchmark_egm(size_t size, int numa_node, int iterations) {
  BenchmarkResult result = {"EGM (HOST_NUMA)", false, 0, 0, 0, 0};

  printf("\n测试 EGM 内存 (NUMA 节点 %d)...\n", numa_node);

  // 检查 EGM 支持
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.location.id = numa_node;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  size_t granularity = 0;
  CUresult err = cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

  if (err != CUDA_SUCCESS || granularity == 0) {
    printf("  ✗ EGM 不可用，跳过测试\n");
    return result;
  }

  // 对齐大小
  size_t aligned_size = ((size + granularity - 1) / granularity) * granularity;

  // 分配 EGM 内存
  CUmemGenericAllocationHandle handle;
  err = cuMemCreate(&handle, aligned_size, &prop, 0);
  if (err != CUDA_SUCCESS) {
    printf("  ✗ cuMemCreate 失败\n");
    return result;
  }

  // 映射到虚拟地址
  CUdeviceptr ptr;
  CHECK_CUDA_DRIVER(cuMemAddressReserve(&ptr, aligned_size, 0, 0, 0));
  CHECK_CUDA_DRIVER(cuMemMap(ptr, aligned_size, 0, handle, 0));

  // 设置访问权限
  CUmemAccessDesc access = {};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = 0;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CHECK_CUDA_DRIVER(cuMemSetAccess(ptr, aligned_size, &access, 1));

  result.available = true;

  size_t count = aligned_size / sizeof(float);
  float *data = (float *)ptr;

  float *d_result;
  CHECK_CUDA_RUNTIME(cudaMalloc(&d_result, sizeof(float)));

  int block_size = 256;
  int grid_size = min((int)((count + block_size - 1) / block_size), 65535);

  // 预热
  write_kernel<<<grid_size, block_size>>>(data, count, 1.0f);
  cudaDeviceSynchronize();

  // 写入测试
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    write_kernel<<<grid_size, block_size>>>(data, count, (float)i);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  result.write_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count() /
      iterations;
  result.write_bandwidth = aligned_size / (result.write_time_ms / 1000.0);

  // 读取测试
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    float zero = 0.0f;
    cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);
    reduce_kernel<<<grid_size, block_size>>>(data, count, d_result);
  }
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();

  result.read_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count() /
      iterations;
  result.read_bandwidth = aligned_size / (result.read_time_ms / 1000.0);

  // 清理
  cudaFree(d_result);
  cuMemUnmap(ptr, aligned_size);
  cuMemAddressFree(ptr, aligned_size);
  cuMemRelease(handle);

  char write_bw[32], read_bw[32];
  format_bandwidth(result.write_bandwidth, write_bw, sizeof(write_bw));
  format_bandwidth(result.read_bandwidth, read_bw, sizeof(read_bw));
  printf("  写入: %s (%.2f ms)\n", write_bw, result.write_time_ms);
  printf("  读取: %s (%.2f ms)\n", read_bw, result.read_time_ms);

  return result;
}

// ============================================================================
// 打印总结报告
// ============================================================================

void print_summary(BenchmarkResult *results, int count, size_t test_size) {
  print_header("性能对比总结");

  char size_str[32];
  format_size(test_size, size_str, sizeof(size_str));
  printf("测试数据大小: %s\n\n", size_str);

  printf("%-30s %15s %15s\n", "内存类型", "写入带宽", "读取带宽");
  print_separator();

  for (int i = 0; i < count; i++) {
    if (!results[i].available)
      continue;

    char write_bw[32], read_bw[32];
    format_bandwidth(results[i].write_bandwidth, write_bw, sizeof(write_bw));
    format_bandwidth(results[i].read_bandwidth, read_bw, sizeof(read_bw));
    printf("%-30s %15s %15s\n", results[i].name, write_bw, read_bw);
  }

  // 计算相对 HBM 的效率
  if (count > 1 && results[0].available) {
    printf("\n相对 GPU HBM 的效率:\n");
    print_separator();

    for (int i = 1; i < count; i++) {
      if (!results[i].available)
        continue;

      double write_ratio =
          results[i].write_bandwidth / results[0].write_bandwidth * 100;
      double read_ratio =
          results[i].read_bandwidth / results[0].read_bandwidth * 100;
      printf("%-30s   写 %5.1f%%      读 %5.1f%%\n", results[i].name,
             write_ratio, read_ratio);
    }
  }
}

// ============================================================================
// 主函数
// ============================================================================

void print_usage(const char *prog) {
  printf("用法: %s [选项]\n\n", prog);
  printf("选项:\n");
  printf("  -s, --size <MB>       测试数据大小 (默认: 256)\n");
  printf("  -n, --numa <node>     NUMA 节点 ID (默认: 0)\n");
  printf("  -i, --iterations <N>  迭代次数 (默认: 10)\n");
  printf("  -h, --help            显示此帮助\n");
  printf("\n示例:\n");
  printf("  %s                    使用默认参数测试\n", prog);
  printf("  %s -s 512 -n 1        测试 512MB，NUMA 节点 1\n", prog);
}

int main(int argc, char **argv) {
  // 默认参数
  size_t size_mb = 256;
  int numa_node = 0;
  int iterations = 10;

  // 解析命令行参数
  static struct option long_options[] = {
      {"size", required_argument, 0, 's'},
      {"numa", required_argument, 0, 'n'},
      {"iterations", required_argument, 0, 'i'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "s:n:i:h", long_options, NULL)) != -1) {
    switch (opt) {
    case 's':
      size_mb = atoi(optarg);
      break;
    case 'n':
      numa_node = atoi(optarg);
      break;
    case 'i':
      iterations = atoi(optarg);
      break;
    case 'h':
      print_usage(argv[0]);
      return 0;
    default:
      print_usage(argv[0]);
      return 1;
    }
  }

  size_t size = size_mb * 1024 * 1024;

  // 打印标题
  printf("\n");
  printf("============================================================\n");
  printf(" EGM (Extended GPU Memory) 性能测试\n");
  printf("============================================================\n");
  printf("测试参数:\n");
  printf("  数据大小: %zu MB\n", size_mb);
  printf("  NUMA 节点: %d\n", numa_node);
  printf("  迭代次数: %d\n", iterations);

  // 检查 CUDA 环境
  if (!check_cuda_environment()) {
    return 1;
  }

  // 检查 EGM 支持
  size_t granularity;
  bool egm_supported = check_egm_support(numa_node, &granularity);

  // 运行基准测试
  print_header("运行性能测试");

  BenchmarkResult results[3];
  int result_count = 0;

  results[result_count++] = benchmark_gpu_hbm(size, iterations);
  results[result_count++] = benchmark_pinned_host(size, iterations);

  if (egm_supported) {
    results[result_count++] = benchmark_egm(size, numa_node, iterations);
  }

  // 打印总结
  print_summary(results, result_count, size);

  printf("\n测试完成\n");
  return 0;
}
