#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     GB200 EGM Bandwidth Test - Data Size Sweep              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

if [ ! -f "./egm_bandwidth_test" ]; then
    echo "Error: egm_bandwidth_test not found. Please compile first:"
    echo "  make gb200"
    exit 1
fi

OUTPUT_FILE="bandwidth_sweep_results_$(date +%Y%m%d_%H%M%S).txt"

echo "Results will be saved to: $OUTPUT_FILE"
echo ""

sizes=(64 128 256 512 1024 2048 4096)
iterations=50

{
    echo "EGM Bandwidth Test - Data Size Sweep"
    echo "Test Date: $(date)"
    echo "=========================================="
    echo ""
    
    for size in "${sizes[@]}"; do
        echo ""
        echo "=========================================="
        echo "Testing with ${size}MB buffer, ${iterations} iterations"
        echo "=========================================="
        ./egm_bandwidth_test -s $size -i $iterations
        
        sleep 2
    done
    
    echo ""
    echo "=========================================="
    echo "Test completed at: $(date)"
    echo "=========================================="
} | tee "$OUTPUT_FILE"

echo ""
echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "To analyze results, you can use:"
echo "  grep 'Read Bandwidth' $OUTPUT_FILE"
echo "  grep 'EGM' $OUTPUT_FILE"
