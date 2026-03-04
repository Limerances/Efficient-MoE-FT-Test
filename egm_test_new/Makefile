NVCC := nvcc
NVCC_FLAGS := -O3 -std=c++17
CUDA_ARCH := -gencode arch=compute_100,code=sm_100 \
             -gencode arch=compute_90,code=sm_90 \
             -gencode arch=compute_80,code=sm_80
LDFLAGS := -lcuda

TARGET := egm_bandwidth_test
SRC := egm_bandwidth_test.cu

.PHONY: all clean run help

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) -o $@ $< $(LDFLAGS)

sm100: $(SRC)
	$(NVCC) $(NVCC_FLAGS) -gencode arch=compute_100,code=sm_100 -o $(TARGET) $< $(LDFLAGS)

gb200: sm100

sm90: $(SRC)
	$(NVCC) $(NVCC_FLAGS) -gencode arch=compute_90,code=sm_90 -o $(TARGET) $< $(LDFLAGS)

sm80: $(SRC)
	$(NVCC) $(NVCC_FLAGS) -gencode arch=compute_80,code=sm_80 -o $(TARGET) $< $(LDFLAGS)

debug: $(SRC)
	$(NVCC) -g -G -O0 -std=c++17 $(CUDA_ARCH) -o $(TARGET)_debug $< $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

run-small: $(TARGET)
	./$(TARGET) -s 64 -i 50

run-large: $(TARGET)
	./$(TARGET) -s 1024 -i 100

clean:
	rm -f $(TARGET) $(TARGET)_debug

help:
	@echo "GB200 EGM Memory Bandwidth Test - Build System"
	@echo ""
	@echo "GPU Architecture Reference:"
	@echo "  sm_100 (CC 10.0) - Blackwell: B100, B200, GB200"
	@echo "  sm_90  (CC 9.0)  - Hopper: H100, H200"
	@echo "  sm_89  (CC 8.9)  - Ada Lovelace: RTX 4090, L40"
	@echo "  sm_80  (CC 8.0)  - Ampere: A100, A30"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build with multiple GPU architectures (sm_80, sm_90, sm_100)"
	@echo "  sm100     - Build specifically for Blackwell (sm_100)"
	@echo "  gb200     - Alias for sm100 (GB200 = Blackwell architecture)"
	@echo "  sm90      - Build for Hopper (sm_90)"
	@echo "  sm80      - Build for Ampere (sm_80)"
	@echo "  debug     - Build with debug symbols"
	@echo "  run       - Build and run with default parameters"
	@echo "  run-small - Run with 64MB buffer, 50 iterations"
	@echo "  run-large - Run with 1GB buffer, 100 iterations"
	@echo "  clean     - Remove built files"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make gb200 && ./egm_bandwidth_test -s 256 -i 100"
	@echo "  make sm100 && ./egm_bandwidth_test --help"
	@echo ""
	@echo "Note: EGM (Extended GPU Memory) requires GB200 or similar"
	@echo "      Grace-Blackwell systems with NVLink-C2C support."
