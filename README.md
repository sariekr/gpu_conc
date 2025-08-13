# vLLM Multi-Level Concurrency Benchmark

A comprehensive benchmarking suite for testing vLLM performance across multiple concurrency levels. This tool systematically evaluates throughput, latency, and resource utilization to help you find the optimal configuration for your deployment.

## 🚀 Features

- **Multi-Level Testing**: Automatically tests concurrency levels from 1 to 1024
- **Comprehensive Metrics**: Tracks throughput, latency, memory usage, and success rates
- **Synthetic Dataset**: Generates consistent, token-controlled test prompts
- **GPU Memory Monitoring**: Real-time GPU memory usage tracking
- **Detailed Analytics**: Statistical analysis with percentiles and performance summaries
- **JSON Output**: Machine-readable results for further analysis
- **Production-Ready**: Includes proper warm-up, measurement, and cool-down phases

## 📋 Requirements

### System Requirements
- Python 3.8+
- NVIDIA GPU with CUDA support
- vLLM server running and accessible

### Python Dependencies
```bash
pip install asyncio openai psutil numpy pynvml
```

### Hardware Monitoring
- NVIDIA drivers with `nvidia-smi` support
- `nvidia-ml-py` for GPU memory monitoring

## 🛠️ Installation

1. **Clone or download the benchmark files:**
   ```bash
   # Download the two main files:
   # - multi_concurrency_benchmark.py
   # - concurrency.py
   ```

2. **Install dependencies:**
   ```bash
   pip install openai psutil numpy pynvml
   ```

3. **Verify vLLM is running:**
   ```bash
   curl http://localhost:8000/v1/models
   ```

## 🎯 Quick Start

### Basic Usage

1. **Start your vLLM server:**
   ```bash
   vllm serve your-model --host 0.0.0.0 --port 8000
   ```

2. **Run the benchmark:**
   ```bash
   python multi_concurrency_benchmark.py
   ```

### Custom Configuration

Edit the configuration in `multi_concurrency_benchmark.py`:

```python
self.config = {
    'phase_duration': 180,        # Test duration per concurrency level
    'ramp_up_duration': 30,       # Warm-up time (not measured)
    'cool_down_duration': 30,     # Cool-down time
    'input_tokens': 1000,         # Target input tokens per request
    'output_tokens': 1000,        # Target output tokens per request
    'request_timeout': 30,        # Request timeout in seconds
    'vllm_url': "http://localhost:8000/v1",
    'api_key': "test-key",
    'model': "your-model-name"
}
```

## 📊 Output and Results

### Individual Test Results

Each concurrency level generates a detailed JSON file:
```
benchmark_concurrency_1.json
benchmark_concurrency_4.json
benchmark_concurrency_8.json
...
```

### Comprehensive Summary

A final summary file contains all results:
```
benchmark_summary_YYYYMMDD_HHMMSS.json
```

### Key Metrics Tracked

| Metric | Description |
|--------|-------------|
| **Throughput** | Tokens per second (system-wide) |
| **Requests/sec** | Successful requests per second |
| **Success Rate** | Percentage of successful requests |
| **Latency** | End-to-end response time (median, P95, P99) |
| **TTFT** | Time to First Token (median, P95, P99) |
| **Memory Usage** | Peak and average GPU memory consumption |
| **Output Speed** | Tokens per second per individual request |

### Sample Output

```
🚀 TESTING CONCURRENCY LEVEL: 32
════════════════════════════════════════════════════════════
⏱️  Duration: 180s
🔧 Ramp-up: 30s
❄️  Cool-down: 30s
📝 Input tokens: 1000
📤 Output tokens: 1000

📊 RESULTS FOR CONCURRENCY 32:
├── ✅ Success Rate: 98.5%
├── 🚀 Throughput: 1245.3 tokens/s
├── 📈 Requests/s: 1.25
├── ⏱️  Median Latency: 2580ms
├── 📊 P95 Latency: 4200ms
├── 🧠 Peak Memory: 24.8GB
└── 📁 Saved to: benchmark_concurrency_32.json
```

## 🔧 Configuration Options

### Concurrency Levels

Default levels tested: `[1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]`

Modify `self.concurrency_levels` to test specific ranges:
```python
self.concurrency_levels = [1, 8, 32, 128]  # Custom levels
```

### Test Duration

Adjust test phases:
- `phase_duration`: Main measurement period
- `ramp_up_duration`: System warm-up (not counted in metrics)
- `cool_down_duration`: System rest between tests

### Token Configuration

- `input_tokens`: Target prompt length
- `output_tokens`: Expected response length
- Actual token counts may vary due to tokenization

### Server Configuration

```python
'vllm_url': "http://your-server:8000/v1",
'api_key': "your-api-key",
'model': "your-model-name"
```

## 📈 Interpreting Results

### Finding Optimal Concurrency

The benchmark automatically identifies the optimal concurrency level based on:
- **Throughput**: Maximum tokens/second achieved
- **Success Rate**: Must be >95% for consideration
- **Stability**: Consistent performance without errors

### Performance Indicators

**Good Performance:**
- High throughput (tokens/sec)
- Success rate >95%
- Stable latency across requests
- Efficient memory usage

**Warning Signs:**
- Declining success rates
- Exponentially increasing latency
- Memory exhaustion
- High error rates

### Example Analysis

```json
{
  "analysis": {
    "optimal_concurrency": 64,
    "max_stable_throughput": 1847.3,
    "recommendation": "Optimal concurrency level is 64 with 1847.3 tokens/s"
  }
}
```

## 🔍 Troubleshooting

### Common Issues

**Connection Errors:**
```bash
# Check vLLM server status
curl http://localhost:8000/health

# Verify model is loaded
curl http://localhost:8000/v1/models
```

**GPU Memory Issues:**
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check vLLM GPU memory settings
vllm serve --help | grep -i memory
```

**Timeout Errors:**
- Increase `request_timeout` for large models
- Reduce `output_tokens` for faster responses
- Lower concurrency levels

**Permission Errors:**
```bash
# Ensure nvidia-ml-py can access GPU
python -c "import pynvml; pynvml.nvmlInit(); print('GPU access OK')"
```

### Performance Issues

**Low Throughput:**
- Check model size vs. GPU memory
- Verify vLLM configuration (tensor parallelism, etc.)
- Monitor CPU and memory usage
- Consider model quantization

**High Memory Usage:**
- Reduce batch size in vLLM
- Enable memory optimization flags
- Check for memory leaks

## 🧪 Advanced Usage

### Custom Dataset

Modify `SyntheticDatasetGenerator` to use your own prompts:

```python
# In concurrency.py
self.custom_prompts = [
    "Your custom prompt 1",
    "Your custom prompt 2",
    # ... more prompts
]
```

### Extended Monitoring

Add custom metrics by extending the `RequestMetrics` class:

```python
@dataclass
class CustomRequestMetrics(RequestMetrics):
    custom_metric: float = 0.0
    # Add your metrics here
```

### Integration with CI/CD

```bash
# Run benchmark and check performance regression
python multi_concurrency_benchmark.py
python check_performance_regression.py benchmark_summary_*.json
```

## 📝 Best Practices

### Pre-Benchmark Checklist

1. ✅ vLLM server is warm and stable
2. ✅ GPU memory is cleared (`nvidia-smi` shows low usage)
3. ✅ No other intensive processes running
4. ✅ Sufficient disk space for result files
5. ✅ Network connectivity is stable

### During Benchmark

- Monitor system resources with `htop` and `nvidia-smi`
- Avoid running other GPU workloads
- Ensure stable network conditions
- Let the benchmark run uninterrupted

### Post-Benchmark

- Archive result files with timestamps
- Compare results across different configurations
- Document optimal settings for your use case
- Share results with your team

## 🤝 Contributing

Feel free to enhance the benchmark by:

- Adding new metrics
- Supporting additional model formats
- Improving error handling
- Adding visualization tools
- Creating analysis scripts

For questions or issues, please check the troubleshooting section or review the detailed logs generated during benchmark execution.
