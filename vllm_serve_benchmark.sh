#!/bin/bash

# ============================================================================
# vLLM Serving Concurrency Benchmark
# ============================================================================
# Uses vLLM's native serving benchmark for production HTTP simulation
# More reliable than custom AsyncOpenAI implementation
# ============================================================================

set -e

# --- CONFIGURATION ---
MODEL_NAME="openai/gpt-oss-20b"
VLLM_PORT=8000
DATASET_PATH="/workspace/data/ShareGPT_V3_unfiltered_cleaned_split.json"

# Concurrency levels to test
CONCURRENCY_LEVELS=(1 2 4 8 16 32 64 128 256 512 1024)

# Test parameters
NUM_PROMPTS=1000
INPUT_TOKENS=1000
OUTPUT_TOKENS=1000

# Test duration
TEST_DURATION=180  # seconds
WARMUP_DURATION=30
COOLDOWN_DURATION=30

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/workspace/results/vllm_serve_benchmark_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# GPU info
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1)
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)

echo "============================================================================"
echo "vLLM Serving Concurrency Benchmark"
echo "============================================================================"
echo "Model: $MODEL_NAME"
echo "GPU: $GPU_MODEL (${GPU_COUNT}x)"
echo "Concurrency Levels: ${CONCURRENCY_LEVELS[@]}"
echo "Test Duration: ${TEST_DURATION}s per level"
echo "Output: $OUTPUT_DIR"
echo "============================================================================"
echo ""

# --- FUNCTION: Start vLLM Server ---
start_vllm_server() {
    echo "Starting vLLM server on port $VLLM_PORT..."

    # Kill any existing vLLM server
    pkill -f "vllm serve" || true
    sleep 2

    # Start vLLM server in background
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_NAME" \
        --port "$VLLM_PORT" \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.85 \
        --dtype auto \
        --trust-remote-code \
        > "$OUTPUT_DIR/vllm_server.log" 2>&1 &

    VLLM_PID=$!
    echo "vLLM server started (PID: $VLLM_PID)"

    # Wait for server to be ready
    echo "Waiting for server to be ready..."
    MAX_WAIT=300  # 5 minutes
    WAITED=0

    while ! curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; do
        sleep 5
        WAITED=$((WAITED + 5))

        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "ERROR: Server failed to start within ${MAX_WAIT}s"
            cat "$OUTPUT_DIR/vllm_server.log"
            exit 1
        fi

        echo "  Waiting... (${WAITED}s)"
    done

    echo "✓ vLLM server ready!"
    echo ""
}

# --- FUNCTION: Stop vLLM Server ---
stop_vllm_server() {
    echo "Stopping vLLM server..."
    pkill -f "vllm.entrypoints.openai.api_server" || true
    sleep 2
    echo "✓ Server stopped"
}

# --- FUNCTION: Run Benchmark for Concurrency Level ---
run_benchmark() {
    local concurrency=$1
    local output_file="$OUTPUT_DIR/benchmark_concurrency_${concurrency}.json"
    local log_file="$OUTPUT_DIR/benchmark_concurrency_${concurrency}.log"

    echo "=========================================================================="
    echo "Testing Concurrency: ${concurrency}"
    echo "=========================================================================="

    echo "Configuration:"
    echo "  - Concurrency: $concurrency (constant concurrent requests)"
    echo "  - Total Prompts: $NUM_PROMPTS"
    echo "  - Mode: Sustained concurrency (no rate limit)"
    echo "  - Estimated Duration: Variable (depends on GPU performance)"
    echo ""

    # Run vLLM serving benchmark
    # NOTE: We do NOT use --request-rate to sustain full concurrency
    # The benchmark will keep exactly 'concurrency' requests active at all times
    python -m vllm.entrypoints.benchmarks.benchmark_serving \
        --model "$MODEL_NAME" \
        --backend vllm \
        --base-url "http://localhost:${VLLM_PORT}/v1" \
        --dataset-name sharegpt \
        --dataset-path "$DATASET_PATH" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$concurrency" \
        --save-results \
        --results-filename "$output_file" \
        > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        echo "✓ Benchmark completed successfully"

        # Extract key metrics from log
        if [ -f "$log_file" ]; then
            echo ""
            echo "--- Results Summary ---"
            grep -E "(Throughput|Latency|Success)" "$log_file" || true
        fi
    else
        echo "✗ Benchmark failed! Check log: $log_file"
        tail -20 "$log_file"
    fi

    echo ""
}

# --- MAIN BENCHMARK LOOP ---

# Start vLLM server
start_vllm_server

# Warmup
echo "--- Warmup Phase (${WARMUP_DURATION}s) ---"
echo "Running warmup requests..."
curl -s http://localhost:$VLLM_PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL_NAME"'",
        "prompt": "Hello, how are you?",
        "max_tokens": 50
    }' > /dev/null || true

sleep "$WARMUP_DURATION"
echo "✓ Warmup complete"
echo ""

# Run benchmarks for each concurrency level
for concurrency in "${CONCURRENCY_LEVELS[@]}"; do
    run_benchmark "$concurrency"

    # Cool down between tests
    echo "Cooling down for ${COOLDOWN_DURATION}s..."
    sleep "$COOLDOWN_DURATION"
done

# Stop server
stop_vllm_server

# --- GENERATE SUMMARY REPORT ---
echo ""
echo "============================================================================"
echo "Generating Summary Report"
echo "============================================================================"

SUMMARY_FILE="$OUTPUT_DIR/BENCHMARK_SUMMARY.txt"

{
    echo "============================================================================"
    echo "vLLM Serving Concurrency Benchmark - Summary Report"
    echo "============================================================================"
    echo "Date: $(date)"
    echo "Model: $MODEL_NAME"
    echo "GPU: $GPU_MODEL (${GPU_COUNT}x)"
    echo ""
    echo "--- Test Configuration ---"
    echo "Concurrency Levels: ${CONCURRENCY_LEVELS[@]}"
    echo "Prompts per Test: $NUM_PROMPTS"
    echo "Test Duration: ${TEST_DURATION}s"
    echo "Input Tokens: $INPUT_TOKENS"
    echo "Output Tokens: $OUTPUT_TOKENS"
    echo ""
    echo "============================================================================"
    echo "Results by Concurrency Level"
    echo "============================================================================"
    printf "%-12s | %-20s | %-18s | %-15s\n" \
        "Concurrency" "Throughput (tok/s)" "Latency (ms)" "Success Rate"
    echo "-------------|----------------------|--------------------|----------------"

    for concurrency in "${CONCURRENCY_LEVELS[@]}"; do
        log_file="$OUTPUT_DIR/benchmark_concurrency_${concurrency}.log"

        if [ -f "$log_file" ]; then
            # Extract metrics from log
            throughput=$(grep -oP "Throughput:.*\K[0-9.]+" "$log_file" | head -1 || echo "N/A")
            latency=$(grep -oP "Average latency:.*\K[0-9.]+" "$log_file" | head -1 || echo "N/A")
            success=$(grep -oP "Success rate:.*\K[0-9.]+%" "$log_file" | head -1 || echo "N/A")

            printf "%-12s | %-20s | %-18s | %-15s\n" \
                "$concurrency" "$throughput" "$latency" "$success"
        else
            printf "%-12s | %-20s | %-18s | %-15s\n" \
                "$concurrency" "FAILED" "FAILED" "FAILED"
        fi
    done

    echo ""
    echo "============================================================================"
    echo "Detailed Results Location"
    echo "============================================================================"
    echo "Output Directory: $OUTPUT_DIR"
    echo ""
    echo "Files:"
    echo "  - benchmark_concurrency_N.json : Detailed metrics per concurrency level"
    echo "  - benchmark_concurrency_N.log  : Benchmark run logs"
    echo "  - vllm_server.log              : vLLM server logs"
    echo "  - BENCHMARK_SUMMARY.txt        : This summary report"
    echo "============================================================================"

} | tee "$SUMMARY_FILE"

echo ""
echo "============================================================================"
echo "BENCHMARK COMPLETE!"
echo "============================================================================"
echo "Summary: $SUMMARY_FILE"
echo ""
echo "To view detailed results:"
echo "  cat $OUTPUT_DIR/benchmark_concurrency_*.log"
echo "============================================================================"
