#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CU_CHECK(call) \
    do { \
        CUresult err = call; \
        if (err != CUDA_SUCCESS) { \
            const char* errStr; \
            cuGetErrorString(err, &errStr); \
            fprintf(stderr, "CUDA Driver Error at %s:%d: %s\n", \
                    __FILE__, __LINE__, errStr); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define ROUND_UP(x, y) (((x) + (y) - 1) / (y) * (y))

constexpr size_t DEFAULT_SIZE = 256 * 1024 * 1024;
constexpr int DEFAULT_ITERATIONS = 100;
constexpr int WARMUP_ITERATIONS = 10;

__global__ void memcpy_kernel_read(const float* __restrict__ src, float* __restrict__ dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    for (size_t i = idx; i < n; i += stride) {
        sum += src[i];
    }
    if (idx == 0) {
        dst[0] = sum;
    }
}

__global__ void memcpy_kernel_write(float* __restrict__ dst, size_t n, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        dst[i] = value;
    }
}

__global__ void memcpy_kernel_copy(const float* __restrict__ src, float* __restrict__ dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

struct BenchmarkResult {
    double read_bandwidth_gbps;
    double write_bandwidth_gbps;
    double copy_bandwidth_gbps;
    double avg_latency_us;
};

class MemoryBenchmark {
public:
    virtual ~MemoryBenchmark() = default;
    virtual const char* getName() const = 0;
    virtual bool allocate(size_t size) = 0;
    virtual void deallocate() = 0;
    virtual void* getDevicePtr() const = 0;
    virtual size_t getAllocatedSize() const = 0;
};

class HBMBenchmark : public MemoryBenchmark {
private:
    void* d_ptr_ = nullptr;
    size_t size_ = 0;
    
public:
    const char* getName() const override { return "HBM (cudaMalloc)"; }
    
    bool allocate(size_t size) override {
        CUDA_CHECK(cudaMalloc(&d_ptr_, size));
        size_ = size;
        return true;
    }
    
    void deallocate() override {
        if (d_ptr_) {
            CUDA_CHECK(cudaFree(d_ptr_));
            d_ptr_ = nullptr;
            size_ = 0;
        }
    }
    
    void* getDevicePtr() const override { return d_ptr_; }
    size_t getAllocatedSize() const override { return size_; }
};

class HostPinnedBenchmark : public MemoryBenchmark {
private:
    void* h_ptr_ = nullptr;
    size_t size_ = 0;
    
public:
    const char* getName() const override { return "Host Pinned (cudaMallocHost)"; }
    
    bool allocate(size_t size) override {
        CUDA_CHECK(cudaMallocHost(&h_ptr_, size));
        size_ = size;
        return true;
    }
    
    void deallocate() override {
        if (h_ptr_) {
            CUDA_CHECK(cudaFreeHost(h_ptr_));
            h_ptr_ = nullptr;
            size_ = 0;
        }
    }
    
    void* getDevicePtr() const override { return h_ptr_; }
    size_t getAllocatedSize() const override { return size_; }
};

class EGMBenchmark : public MemoryBenchmark {
private:
    CUdeviceptr dptr_ = 0;
    CUmemGenericAllocationHandle allocHandle_;
    size_t padded_size_ = 0;
    size_t size_ = 0;
    int numaId_ = -1;
    int deviceId_ = 0;
    bool allocated_ = false;
    
public:
    const char* getName() const override { return "EGM (Extended GPU Memory)"; }
    
    bool allocate(size_t size) override {
        CU_CHECK(cuInit(0));
        
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, deviceId_));
        
        CU_CHECK(cuDeviceGetAttribute(&numaId_, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device));
        
        if (numaId_ < 0) {
            printf("  [EGM] Warning: HOST_NUMA_ID not available (numaId=%d), EGM may not be supported on this platform.\n", numaId_);
            return false;
        }
        
        printf("  [EGM] Using NUMA node ID: %d\n", numaId_);
        
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        prop.location.id = numaId_;
        
        size_t granularity = 0;
        CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        
        padded_size_ = ROUND_UP(size, granularity);
        size_ = size;
        
        printf("  [EGM] Requested size: %zu MB, Padded size: %zu MB, Granularity: %zu KB\n",
               size / (1024*1024), padded_size_ / (1024*1024), granularity / 1024);
        
        CU_CHECK(cuMemCreate(&allocHandle_, padded_size_, &prop, 0));
        
        CU_CHECK(cuMemAddressReserve(&dptr_, padded_size_, 0, 0, 0));
        
        CU_CHECK(cuMemMap(dptr_, padded_size_, 0, allocHandle_, 0));
        
        CUmemAccessDesc accessDesc[2]{{}};
        accessDesc[0].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        accessDesc[0].location.id = numaId_;
        accessDesc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        accessDesc[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc[1].location.id = deviceId_;
        accessDesc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU_CHECK(cuMemSetAccess(dptr_, padded_size_, accessDesc, 2));
        
        allocated_ = true;
        printf("  [EGM] Memory allocated successfully at 0x%llx\n", (unsigned long long)dptr_);
        return true;
    }
    
    void deallocate() override {
        if (allocated_) {
            cuMemUnmap(dptr_, padded_size_);
            cuMemAddressFree(dptr_, padded_size_);
            cuMemRelease(allocHandle_);
            dptr_ = 0;
            padded_size_ = 0;
            size_ = 0;
            allocated_ = false;
        }
    }
    
    void* getDevicePtr() const override { return (void*)dptr_; }
    size_t getAllocatedSize() const override { return size_; }
};

class EGMStreamOrderedBenchmark : public MemoryBenchmark {
private:
    void* ptr_ = nullptr;
    cudaMemPool_t memPool_ = nullptr;
    cudaStream_t stream_ = nullptr;
    size_t size_ = 0;
    int numaId_ = -1;
    int deviceId_ = 0;
    bool allocated_ = false;
    
public:
    const char* getName() const override { return "EGM Stream-Ordered (cudaMemPool)"; }
    
    bool allocate(size_t size) override {
        CUDA_CHECK(cudaSetDevice(deviceId_));
        
        cudaDeviceGetAttribute(&numaId_, cudaDevAttrHostNumaId, deviceId_);
        
        if (numaId_ < 0) {
            printf("  [EGM-Pool] Warning: HOST_NUMA_ID not available (numaId=%d), EGM may not be supported.\n", numaId_);
            return false;
        }
        
        printf("  [EGM-Pool] Using NUMA node ID: %d\n", numaId_);
        
        cudaMemPoolProps props{};
        props.allocType = cudaMemAllocationTypePinned;
        props.location.type = cudaMemLocationTypeHostNuma;
        props.location.id = numaId_;
        
        CUDA_CHECK(cudaMemPoolCreate(&memPool_, &props));
        
        cudaMemAccessDesc desc{};
        desc.flags = cudaMemAccessFlagsProtReadWrite;
        desc.location.type = cudaMemLocationTypeDevice;
        desc.location.id = deviceId_;
        CUDA_CHECK(cudaMemPoolSetAccess(memPool_, &desc, 1));
        
        CUDA_CHECK(cudaStreamCreate(&stream_));
        
        CUDA_CHECK(cudaMallocFromPoolAsync(&ptr_, size, memPool_, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        
        size_ = size;
        allocated_ = true;
        printf("  [EGM-Pool] Memory allocated successfully at %p\n", ptr_);
        return true;
    }
    
    void deallocate() override {
        if (allocated_) {
            if (ptr_) {
                cudaFreeAsync(ptr_, stream_);
                cudaStreamSynchronize(stream_);
                ptr_ = nullptr;
            }
            if (stream_) {
                cudaStreamDestroy(stream_);
                stream_ = nullptr;
            }
            if (memPool_) {
                cudaMemPoolDestroy(memPool_);
                memPool_ = nullptr;
            }
            size_ = 0;
            allocated_ = false;
        }
    }
    
    void* getDevicePtr() const override { return ptr_; }
    size_t getAllocatedSize() const override { return size_; }
};

BenchmarkResult runBenchmark(MemoryBenchmark* bench, size_t size, int iterations) {
    BenchmarkResult result{0, 0, 0, 0};
    
    printf("\n--- Benchmarking: %s ---\n", bench->getName());
    
    if (!bench->allocate(size)) {
        printf("  Failed to allocate memory, skipping benchmark.\n");
        return result;
    }
    
    void* ptr = bench->getDevicePtr();
    size_t n = size / sizeof(float);
    
    void* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, sizeof(float)));
    
    void* d_src_hbm;
    void* d_dst_hbm;
    CUDA_CHECK(cudaMalloc(&d_src_hbm, size));
    CUDA_CHECK(cudaMalloc(&d_dst_hbm, size));
    
    int blockSize = 256;
    int numBlocks = std::min((int)((n + blockSize - 1) / blockSize), 65535);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    printf("  Warming up...\n");
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        memcpy_kernel_read<<<numBlocks, blockSize>>>((float*)ptr, (float*)d_temp, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("  Testing READ bandwidth...\n");
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        memcpy_kernel_read<<<numBlocks, blockSize>>>((float*)ptr, (float*)d_temp, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float read_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&read_time_ms, start, stop));
    result.read_bandwidth_gbps = (double)size * iterations / (read_time_ms / 1000.0) / 1e9;
    
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        memcpy_kernel_write<<<numBlocks, blockSize>>>((float*)ptr, n, 1.0f);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("  Testing WRITE bandwidth...\n");
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        memcpy_kernel_write<<<numBlocks, blockSize>>>((float*)ptr, n, 1.0f);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float write_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&write_time_ms, start, stop));
    result.write_bandwidth_gbps = (double)size * iterations / (write_time_ms / 1000.0) / 1e9;
    
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        memcpy_kernel_copy<<<numBlocks, blockSize>>>((float*)d_src_hbm, (float*)ptr, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("  Testing COPY bandwidth (HBM -> Target)...\n");
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        memcpy_kernel_copy<<<numBlocks, blockSize>>>((float*)d_src_hbm, (float*)ptr, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float copy_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&copy_time_ms, start, stop));
    result.copy_bandwidth_gbps = (double)size * 2 * iterations / (copy_time_ms / 1000.0) / 1e9;
    
    result.avg_latency_us = (read_time_ms * 1000.0) / iterations;
    
    printf("  Results:\n");
    printf("    Read Bandwidth:  %.2f GB/s\n", result.read_bandwidth_gbps);
    printf("    Write Bandwidth: %.2f GB/s\n", result.write_bandwidth_gbps);
    printf("    Copy Bandwidth:  %.2f GB/s (bidirectional)\n", result.copy_bandwidth_gbps);
    printf("    Avg Latency:     %.2f us\n", result.avg_latency_us);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_src_hbm));
    CUDA_CHECK(cudaFree(d_dst_hbm));
    bench->deallocate();
    
    return result;
}

void runH2DBenchmark(size_t size, int iterations) {
    printf("\n--- Benchmarking: Host-to-Device (H2D) Transfer ---\n");
    
    void* h_ptr;
    void* d_ptr;
    CUDA_CHECK(cudaMallocHost(&h_ptr, size));
    CUDA_CHECK(cudaMalloc(&d_ptr, size));
    
    memset(h_ptr, 0, size);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    printf("  Warming up...\n");
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));
    }
    
    printf("  Testing H2D transfer bandwidth...\n");
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float h2d_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_time_ms, start, stop));
    double h2d_bandwidth = (double)size * iterations / (h2d_time_ms / 1000.0) / 1e9;
    
    printf("  Warming up...\n");
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost));
    }
    
    printf("  Testing D2H transfer bandwidth...\n");
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float d2h_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_time_ms, start, stop));
    double d2h_bandwidth = (double)size * iterations / (d2h_time_ms / 1000.0) / 1e9;
    
    printf("  Results:\n");
    printf("    H2D Bandwidth: %.2f GB/s\n", h2d_bandwidth);
    printf("    D2H Bandwidth: %.2f GB/s\n", d2h_bandwidth);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFreeHost(h_ptr));
    CUDA_CHECK(cudaFree(d_ptr));
}

void printDeviceInfo(int deviceId) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    
    printf("\n========================================\n");
    printf("GPU Device Information\n");
    printf("========================================\n");
    printf("Device Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    
    int numaId = -1;
    cudaDeviceGetAttribute(&numaId, cudaDevAttrHostNumaId, deviceId);
    printf("Host NUMA ID: %d", numaId);
    if (numaId >= 0) {
        printf(" (EGM Supported)\n");
    } else {
        printf(" (EGM Not Available)\n");
    }
    
    int c2cSupported = 0;
    cudaDeviceGetAttribute(&c2cSupported, cudaDevAttrGPUDirectRDMASupported, deviceId);
    printf("GPU Direct RDMA: %s\n", c2cSupported ? "Supported" : "Not Supported");
    
    printf("========================================\n\n");
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -s, --size <MB>       Test buffer size in MB (default: %zu)\n", DEFAULT_SIZE / (1024*1024));
    printf("  -i, --iterations <N>  Number of iterations (default: %d)\n", DEFAULT_ITERATIONS);
    printf("  -d, --device <ID>     GPU device ID (default: 0)\n");
    printf("  -h, --help            Show this help message\n");
    printf("\nBenchmark Types:\n");
    printf("  - HBM: Standard GPU memory (cudaMalloc)\n");
    printf("  - Host Pinned: Page-locked host memory (cudaMallocHost)\n");
    printf("  - EGM VMM: Extended GPU Memory using Virtual Memory Management API\n");
    printf("  - EGM Pool: Extended GPU Memory using Stream-Ordered Memory Pool\n");
    printf("  - H2D/D2H: Host-Device transfer using cudaMemcpy\n");
}

int main(int argc, char* argv[]) {
    size_t size = DEFAULT_SIZE;
    int iterations = DEFAULT_ITERATIONS;
    int deviceId = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--size") == 0) {
            if (i + 1 < argc) {
                size = atoi(argv[++i]) * 1024 * 1024;
            }
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--iterations") == 0) {
            if (i + 1 < argc) {
                iterations = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--device") == 0) {
            if (i + 1 < argc) {
                deviceId = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        }
    }
    
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceId >= deviceCount) {
        fprintf(stderr, "Error: Invalid device ID %d. Available devices: 0-%d\n",
                deviceId, deviceCount - 1);
        return 1;
    }
    
    CUDA_CHECK(cudaSetDevice(deviceId));
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     GB200 EGM Memory Bandwidth Test                          ║\n");
    printf("║     Testing HBM, EGM, and Host Pinned Memory                 ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    printDeviceInfo(deviceId);
    
    printf("Test Configuration:\n");
    printf("  Buffer Size: %zu MB\n", size / (1024 * 1024));
    printf("  Iterations: %d\n", iterations);
    printf("  Warmup Iterations: %d\n", WARMUP_ITERATIONS);
    printf("  Device ID: %d\n\n", deviceId);
    
    std::vector<std::pair<const char*, BenchmarkResult>> results;
    
    HBMBenchmark hbmBench;
    results.push_back({"HBM", runBenchmark(&hbmBench, size, iterations)});
    
    HostPinnedBenchmark pinnedBench;
    results.push_back({"Host Pinned", runBenchmark(&pinnedBench, size, iterations)});
    
    EGMBenchmark egmBench;
    BenchmarkResult egmResult = runBenchmark(&egmBench, size, iterations);
    if (egmResult.read_bandwidth_gbps > 0) {
        results.push_back({"EGM (VMM)", egmResult});
    }
    
    EGMStreamOrderedBenchmark egmPoolBench;
    BenchmarkResult egmPoolResult = runBenchmark(&egmPoolBench, size, iterations);
    if (egmPoolResult.read_bandwidth_gbps > 0) {
        results.push_back({"EGM (Pool)", egmPoolResult});
    }
    
    runH2DBenchmark(size, iterations);
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                           Summary of Results                                  ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Memory Type          │ Read (GB/s) │ Write (GB/s) │ Copy (GB/s) │ Latency(us)║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════════╣\n");
    
    for (const auto& r : results) {
        printf("║ %-20s │ %11.2f │ %12.2f │ %11.2f │ %10.2f ║\n",
               r.first,
               r.second.read_bandwidth_gbps,
               r.second.write_bandwidth_gbps,
               r.second.copy_bandwidth_gbps,
               r.second.avg_latency_us);
    }
    
    printf("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    
    printf("\n");
    printf("Notes:\n");
    printf("  - HBM: On-chip High Bandwidth Memory, highest bandwidth (~8000 GB/s on GB200)\n");
    printf("  - Host Pinned: Page-locked CPU memory, accessed via PCIe/C2C\n");
    printf("  - EGM: Extended GPU Memory on CPU DRAM, accessed via NVLink-C2C (~225 GB/s)\n");
    printf("  - Copy bandwidth is bidirectional (read + write)\n");
    printf("\n");
    
    return 0;
}
