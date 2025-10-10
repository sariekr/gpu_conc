# RunPod vLLM Concurrency Benchmark Guide

## Overview

This benchmark tests vLLM performance under **high concurrent load** across multiple concurrency levels (1 → 1024). It simulates real-world production scenarios where hundreds/thousands of users hit your LLM endpoint simultaneously.

**What it measures:**
- System throughput (tokens/second)
- Time to First Token (TTFT)
- End-to-end latency (median, P95, P99)
- Request success rate
- GPU memory usage

**GitHub:** https://github.com/sariekr/gpu_conc

---

## Quick Start (Choose Your Platform)

- [NVIDIA H100/H200](#nvidia-h100h200) - Standard setup
- [NVIDIA B200](#nvidia-b200) - Build from source
- [AMD MI300X](#amd-mi300x) - ROCm Docker

---

## NVIDIA H100/H200

### Step 1: Create RunPod Pod

**Template:**
```
runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
```

**GPU:** H100 or H200 (any count)

**Volume:** Minimum 50 GB

**Environment Variables:**
```bash
DISABLE_RUNPOD_MONITOR=1
```

### Step 2: Installation

SSH into pod:

```bash
cd /workspace

# Install dependencies
apt-get update && apt-get install -y git wget bc

# Install vLLM
pip install vllm --no-cache-dir

# Install Python dependencies
pip install openai psutil numpy pynvml

# Clone benchmark repo
git clone https://github.com/sariekr/gpu_conc.git
cd gpu_conc

# Setup HuggingFace
export HF_HOME=/workspace/.cache/huggingface
huggingface-cli login --token YOUR_HF_TOKEN
```

### Step 3: Start vLLM Server

```bash
# Start vLLM server in background
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key test-key \
    --max-model-len 8192 \
    --max-num-seqs 512 \
    --gpu-memory-utilization 0.85 > vllm_server.log 2>&1 &

# Wait for server to start (~2-5 minutes)
sleep 120

# Verify server is running
curl http://localhost:8000/v1/models
```

### Step 4: Run Benchmark

```bash
python multi_concurrency_benchmark.py
```

**Expected duration:** ~4-5 hours (tests 10 concurrency levels: 1, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

---

## NVIDIA B200

B200 requires building vLLM from source for `sm_100` compute capability.

### Step 1: Create RunPod Pod

**Template:**
```
runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
```

**GPU:** B200 (any count)

**Volume:** Minimum 50 GB

**Environment Variables:**
```bash
DISABLE_RUNPOD_MONITOR=1
```

### Step 2: Build vLLM

```bash
cd /workspace

# Install build tools
apt-get update && apt-get install -y git ninja-build cmake wget bc

# Install UV package manager
pip install -U uv pip setuptools wheel

# Build vLLM from source (~10 min)
git clone --depth 1 https://github.com/vllm-project/vllm /opt/vllm
cd /opt/vllm
uv pip install -e . --system

# Verify B200 support
python -c "import torch; print('sm_100' in torch.cuda.get_arch_list())"
# Should print: True
```

### Step 3: Install Dependencies & Clone Repo

```bash
cd /workspace

# Install Python dependencies
pip install openai psutil numpy pynvml

# Clone benchmark
git clone https://github.com/sariekr/gpu_conc.git
cd gpu_conc

# Setup HuggingFace
export HF_HOME=/workspace/.cache/huggingface
huggingface-cli login --token YOUR_HF_TOKEN
```

### Step 4: Start vLLM Server

```bash
# Start vLLM server in background
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key test-key \
    --max-model-len 8192 \
    --max-num-seqs 512 \
    --gpu-memory-utilization 0.85 > vllm_server.log 2>&1 &

# Wait for server to start
sleep 120

# Verify server
curl http://localhost:8000/v1/models
```

### Step 5: Run Benchmark

```bash
python multi_concurrency_benchmark.py
```

---

## AMD MI300X

### Step 1: Create RunPod Pod

**Template:**
```
rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909
```

**GPU:** MI300X (any count)

**Volume:** Minimum 100 GB

**Environment Variables:**
```bash
HUGGINGFACE_HUB_CACHE=/workspace
HF_TOKEN=your_hf_token_here
```

### Step 2: Installation

```bash
cd /workspace

# Install dependencies
apt-get update && apt-get install -y git wget bc

# ⚠️ DO NOT install vLLM - it's pre-installed in Docker image!

# Install Python dependencies (if missing)
pip install openai psutil numpy

# Clone benchmark
git clone https://github.com/sariekr/gpu_conc.git
cd gpu_conc

# Verify ROCm GPUs
rocm-smi
```

### Step 3: Start vLLM Server

```bash
# ROCm environment variables
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1
export VLLM_ROCM_USE_AITER_RMSNORM=0

# Start vLLM server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key test-key \
    --max-model-len 8192 \
    --max-num-seqs 1024 \
    --gpu-memory-utilization 0.9 > vllm_server.log 2>&1 &

# Wait for server
sleep 120

# Verify
curl http://localhost:8000/v1/models
```

### Step 4: Run Benchmark

```bash
python multi_concurrency_benchmark.py
```

---

## Configuration

Edit `multi_concurrency_benchmark.py` to customize:

### Concurrency Levels (Line 22)

```python
# Default: tests 10 levels
self.concurrency_levels = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Quick test (3 levels):
self.concurrency_levels = [1, 64, 256]

# Custom range:
self.concurrency_levels = [16, 32, 64, 128]
```

### Test Duration (Line 27-31)

```python
# Default: 3 minutes per level
'phase_duration': 180,      # Main test (measured)
'ramp_up_duration': 30,     # Warmup (not measured)
'cool_down_duration': 30,   # Cooldown

# Quick test:
'phase_duration': 60,
'ramp_up_duration': 10,
'cool_down_duration': 10,
```

### Token Configuration (Line 32-33)

```python
# Default:
'input_tokens': 1000,
'output_tokens': 1000,

# Shorter (faster):
'input_tokens': 500,
'output_tokens': 500,

# Longer (more realistic):
'input_tokens': 2000,
'output_tokens': 2000,
```

### Server Settings (Line 34-36)

```python
'vllm_url': "http://localhost:8000/v1",
'api_key': "test-key",
'model': "meta-llama/Llama-3.1-8B-Instruct"
```

---

## Understanding Results

### Output Files

After benchmark completes, you'll have:

```
benchmark_concurrency_1.json      # Individual test results
benchmark_concurrency_4.json
benchmark_concurrency_8.json
...
benchmark_concurrency_1024.json

benchmark_summary_YYYYMMDD_HHMMSS.json  # Comprehensive summary
```

### Summary File Structure

```json
{
  "test_info": {
    "timestamp": "20250110_143000",
    "total_duration_hours": 4.2,
    "concurrency_levels_tested": [1, 4, 8, ...],
    "config": {...},
    "gpu_info": {
      "model": "NVIDIA H100 80GB",
      "memory": "81920MB",
      "count": 1
    }
  },
  "results": [...],
  "performance_summary": {
    "1": {
      "throughput_tokens_per_sec": 1250.5,
      "requests_per_sec": 1.25,
      "success_rate": 0.99,
      "median_latency_ms": 800,
      "p95_latency_ms": 950,
      "peak_memory_gb": 24.5
    },
    "64": {
      "throughput_tokens_per_sec": 15340.2,
      "requests_per_sec": 15.34,
      "success_rate": 0.98,
      "median_latency_ms": 4200,
      "p95_latency_ms": 5800,
      "peak_memory_gb": 76.2
    },
    ...
    "analysis": {
      "optimal_concurrency": 64,
      "max_stable_throughput": 15340.2,
      "recommendation": "Optimal concurrency level is 64 with 15340.2 tokens/s"
    }
  }
}
```

### Key Metrics Explained

**1. Throughput (tokens/second)**
- Total tokens generated per second across all requests
- **Higher is better**
- Example: 15,340 tok/s at concurrency 64

**2. Requests per Second**
- Complete requests finished per second
- Example: 15.34 req/s at concurrency 64

**3. Success Rate**
- Percentage of requests that completed successfully
- **Should be >95%**
- Example: 0.98 = 98% success rate

**4. Median Latency (milliseconds)**
- Middle value of all request completion times
- **Lower is better**
- Example: 4,200ms at concurrency 64

**5. P95 Latency (milliseconds)**
- 95% of requests completed within this time
- Used to measure worst-case performance
- Example: 5,800ms at concurrency 64

**6. Peak Memory (GB)**
- Maximum GPU memory used during test
- Monitor to avoid OOM errors

### Finding Optimal Concurrency

The benchmark automatically identifies optimal concurrency based on:
- **Maximum throughput** achieved
- **Success rate >95%**
- **Stable latency** (not exponentially increasing)

**Example Analysis:**

```
Concurrency     Throughput      Success%    Median Latency
1               1,250 tok/s     99%         800ms
4               4,800 tok/s     99%         900ms
8               9,200 tok/s     98%         1,100ms
16              14,500 tok/s    98%         1,800ms
32              16,800 tok/s    97%         3,200ms
64              15,340 tok/s    98%         4,200ms ← OPTIMAL
128             14,200 tok/s    94%         9,500ms (success rate drops)
256             12,800 tok/s    89%         18,200ms (too many failures)
```

**Optimal:** Concurrency 64 (highest throughput with >95% success rate)

---

## Monitoring During Benchmark

Open a second terminal to monitor:

### NVIDIA GPUs

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Server logs
tail -f vllm_server.log
```

### AMD GPUs

```bash
# Real-time GPU monitoring
watch -n 1 rocm-smi

# Server logs
tail -f vllm_server.log
```

**What to look for:**
- GPU utilization: Should be 90-100%
- GPU memory: Should be stable (not growing)
- Server logs: No OOM errors

---

## Troubleshooting

### Issue 1: Server Won't Start

**Symptoms:**
```bash
curl http://localhost:8000/v1/models
# Returns: Connection refused
```

**Check logs:**
```bash
tail -100 vllm_server.log
```

**Common causes:**
- Model still downloading (check HF cache)
- OOM error (reduce `--gpu-memory-utilization` to 0.75)
- Port 8000 already in use (change port)

**Fix:**
```bash
# Kill any existing vLLM processes
pkill -9 -f vllm

# Restart server with lower memory
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --gpu-memory-utilization 0.75 \
    --max-num-seqs 256 \
    --port 8000 > vllm_server.log 2>&1 &
```

### Issue 2: Low Throughput

**Symptoms:**
```json
"throughput_tokens_per_sec": 500  // Very low!
```

**Check GPU utilization:**
```bash
nvidia-smi -l 1
```

**Expected:** GPU Util should be 90-100%

**If low (<70%):**
- Server batch size too small → Increase `--max-num-seqs`
- Test duration too short → Increase `phase_duration`
- CPU bottleneck → Check with `htop`

**Fix in `multi_concurrency_benchmark.py`:**
```python
'phase_duration': 300,  # Longer test
```

### Issue 3: High Failure Rate

**Symptoms:**
```json
"success_rate": 0.65  // 35% failures!
```

**Check error types:**
```bash
grep -i error benchmark_concurrency_*.json
```

**Common errors:**
- Timeout errors → Increase `request_timeout`
- OOM errors → Reduce concurrency or batch size
- Connection errors → Server crashed (check logs)

**Fix:**
```python
# In multi_concurrency_benchmark.py
'request_timeout': 60,  # Increase from 30s

# Or test fewer levels
self.concurrency_levels = [1, 4, 8, 16, 32, 64]  # Skip 128+
```

### Issue 4: B200 "No kernel image" Error

**Error:**
```
RuntimeError: CUDA error: no kernel image is available for execution
```

**Cause:** vLLM not built for sm_100

**Fix:**
```bash
cd /opt/vllm
uv pip install -e . --system --force-reinstall

# Verify
python -c "import torch; print(torch.cuda.get_arch_list())"
# Must include 'sm_100'
```

### Issue 5: AMD ROCm Issues

**Problem:** Lower performance than expected

**Check environment variables:**
```bash
echo $VLLM_V1_USE_PREFILL_DECODE_ATTENTION  # Should be: 1
echo $VLLM_ROCM_USE_AITER_RMSNORM           # Should be: 0
```

**Fix:**
```bash
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1
export VLLM_ROCM_USE_AITER_RMSNORM=0

# Restart server
pkill -9 -f vllm
vllm serve ... > vllm_server.log 2>&1 &
```

---

## Best Practices

### Before Running Benchmark

1. ✅ vLLM server is warmed up and stable
2. ✅ Run test inference to verify server works
3. ✅ GPU memory cleared (`nvidia-smi` shows low usage)
4. ✅ No other GPU processes running
5. ✅ Sufficient disk space (>10GB for results)

### During Benchmark

- **Don't interrupt:** Let it run completely (4-5 hours)
- **Monitor resources:** Check GPU/CPU/memory in separate terminal
- **Avoid other workloads:** Don't run other GPU tasks
- **Stay connected:** Use `tmux` or `screen` to prevent SSH disconnects

### After Benchmark

```bash
# Backup results immediately (RunPod pods can be terminated!)
scp -r root@runpod_ip:/workspace/gpu_conc/*.json ./backup/

# Or compress and download
cd /workspace/gpu_conc
tar -czf results_$(date +%Y%m%d).tar.gz benchmark_*.json
```

---

## Expected Performance (Reference)

### NVIDIA H100 80GB (8B Model)

| Concurrency | Throughput    | Success Rate | Median Latency |
|-------------|---------------|--------------|----------------|
| 1           | 1,200 tok/s   | >99%         | ~800ms         |
| 8           | 9,000 tok/s   | >99%         | ~1,100ms       |
| 64          | 15,000 tok/s  | >98%         | ~4,200ms       |
| 256         | 18,000 tok/s  | >96%         | ~14,000ms      |
| 1024        | 20,000 tok/s  | >90%         | ~50,000ms      |

### NVIDIA H200 141GB (8B Model)

| Concurrency | Throughput    | Success Rate | Median Latency |
|-------------|---------------|--------------|----------------|
| 1           | 1,300 tok/s   | >99%         | ~750ms         |
| 8           | 10,000 tok/s  | >99%         | ~1,000ms       |
| 64          | 17,000 tok/s  | >98%         | ~3,800ms       |
| 256         | 21,000 tok/s  | >96%         | ~12,000ms      |
| 1024        | 24,000 tok/s  | >92%         | ~42,000ms      |

*Note: Actual performance depends on model, GPU count, vLLM version, etc.*

---

## Quick Command Reference

### NVIDIA (H100/H200)

```bash
# Setup
cd /workspace
apt-get update && apt-get install -y git wget bc
pip install vllm openai psutil numpy pynvml
git clone https://github.com/sariekr/gpu_conc.git && cd gpu_conc
huggingface-cli login --token YOUR_TOKEN

# Start server
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --api-key test-key > vllm_server.log 2>&1 &

# Wait & verify
sleep 120 && curl http://localhost:8000/v1/models

# Run benchmark
python multi_concurrency_benchmark.py

# Monitor
watch -n 1 nvidia-smi
```

### NVIDIA B200

```bash
# Build vLLM
cd /workspace
apt-get update && apt-get install -y git ninja-build cmake wget bc
pip install -U uv
git clone --depth 1 https://github.com/vllm-project/vllm /opt/vllm
cd /opt/vllm && uv pip install -e . --system

# Setup benchmark
cd /workspace
pip install openai psutil numpy pynvml
git clone https://github.com/sariekr/gpu_conc.git && cd gpu_conc
huggingface-cli login --token YOUR_TOKEN

# Start server & run
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --api-key test-key > vllm_server.log 2>&1 &
sleep 120 && python multi_concurrency_benchmark.py
```

### AMD MI300X

```bash
# Setup (vLLM pre-installed in Docker)
cd /workspace
apt-get update && apt-get install -y git wget bc
pip install openai psutil
git clone https://github.com/sariekr/gpu_conc.git && cd gpu_conc
huggingface-cli login --token YOUR_TOKEN

# ROCm environment
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1
export VLLM_ROCM_USE_AITER_RMSNORM=0

# Start server
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --api-key test-key --max-num-seqs 1024 --gpu-memory-utilization 0.9 > vllm_server.log 2>&1 &

# Wait & run
sleep 120 && python multi_concurrency_benchmark.py

# Monitor
watch -n 1 rocm-smi
```

---

## Using `tmux` to Prevent Disconnects

RunPod SSH can disconnect. Use `tmux`:

```bash
# Start tmux session
tmux new -s benchmark

# Inside tmux, run benchmark
python multi_concurrency_benchmark.py

# Detach from session: Ctrl+B, then D
# Session keeps running even if SSH disconnects

# Reconnect later
tmux attach -t benchmark

# View all sessions
tmux ls
```

---

## Support

**GitHub Issues:** https://github.com/sariekr/gpu_conc/issues

**Key Files:**
- `multi_concurrency_benchmark.py` - Main runner
- `concurrency.py` - Core benchmark module
- `README.md` - Detailed documentation

---

**Happy Benchmarking! 🚀**
