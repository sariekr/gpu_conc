#!/usr/bin/env python3
"""
Multi-Level Concurrency Benchmark for vLLM - FIXED VERSION
Doğru metodoloji ile performans testi
"""

import asyncio
import logging
import json
import time
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from concurrency_fixed import run_benchmark_fixed, TestMode


class MultiConcurrencyBenchmarkFixed:
    """Fixed version with proper methodology"""

    def __init__(self, test_mode: TestMode = TestMode.CLOSED_LOOP):
        self.concurrency_levels = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.results = []
        self.start_time = time.time()
        self.test_mode = test_mode

        # Test configuration
        self.config = {
            'phase_duration': 180,      # 3 dakika gerçek test
            'ramp_up_duration': 30,     # 30s warm-up
            'cool_down_duration': 30,   # 30s cool-down
            'input_tokens': 1000,
            'output_tokens': 1000,
            'request_timeout': 30,
            'vllm_url': "http://localhost:8000/v1",
            'api_key': "test-key",
            'model': "openai/gpt-oss-20b"
        }

        # Sonuçları kaydet
        self.summary_file = f"benchmark_summary_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    def get_gpu_info(self) -> Dict:
        """Get GPU information"""
        try:
            gpu_result = subprocess.check_output([
                "nvidia-smi", "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits"
            ]).decode().strip()

            lines = gpu_result.split('\n')
            if lines:
                gpu_name, gpu_memory = lines[0].split(', ')
                return {
                    "model": gpu_name.strip(),
                    "memory": f"{int(gpu_memory.strip())/1024:.0f}GB",
                    "count": len(lines)
                }
        except Exception as e:
            logging.warning(f"Could not get GPU info: {e}")

        return {"model": "Unknown GPU", "memory": "Unknown", "count": 1}

    async def run_single_benchmark(self, concurrency: int) -> Dict:
        """Run benchmark for a single concurrency level"""

        print(f"\n{'='*60}")
        print(f"🚀 TESTING CONCURRENCY LEVEL: {concurrency}")
        print(f"📋 Mode: {self.test_mode.value}")
        print(f"{'='*60}")
        print(f"⏱️  Test Duration: {self.config['phase_duration']}s")
        print(f"🔥 Warm-up: {self.config['ramp_up_duration']}s")
        print(f"❄️  Cool-down: {self.config['cool_down_duration']}s")
        print(f"📝 Target Input: {self.config['input_tokens']} tokens")
        print(f"📤 Target Output: {self.config['output_tokens']} tokens")

        output_file = f"benchmark_concurrency_{concurrency}_fixed.json"

        try:
            # Run the fixed benchmark
            result = await run_benchmark_fixed(
                concurrency=concurrency,
                phase_duration=self.config['phase_duration'],
                ramp_up_duration=self.config['ramp_up_duration'],
                cool_down_duration=self.config['cool_down_duration'],
                input_tokens=self.config['input_tokens'],
                output_tokens=self.config['output_tokens'],
                request_timeout=self.config['request_timeout'],
                vllm_url=self.config['vllm_url'],
                api_key=self.config['api_key'],
                gpu_info=self.get_gpu_info(),
                model=self.config['model'],
                output_file=output_file,
                test_mode=self.test_mode
            )

            # Add metadata
            result['test_timestamp'] = time.time()
            result['test_duration_total'] = time.time() - self.start_time

            # Print results
            self._print_results(concurrency, result)

            return result

        except Exception as e:
            logging.error(f"Benchmark failed for concurrency {concurrency}: {e}")
            return {
                'concurrency': concurrency,
                'error': str(e),
                'timestamp': time.time()
            }

    def _print_results(self, concurrency: int, result: Dict):
        """Print formatted results"""
        print(f"\n📊 RESULTS FOR CONCURRENCY {concurrency}:")
        print(f"├── ✅ Success Rate: {result.get('success_rate', 0):.1%}")
        print(f"├── 🚀 Throughput: {result.get('system_output_throughput_tps', 0):.1f} tokens/s")
        print(f"├── 📈 Requests/s: {result.get('requests_per_second', 0):.2f}")

        # Latency stats
        if 'latency_ms' in result:
            latency = result['latency_ms']
            print(f"├── ⏱️  Median Latency: {latency.get('median', 0):.0f}ms")
            print(f"├── 📊 P95 Latency: {latency.get('p95', 0):.0f}ms")
            print(f"├── 📊 P99 Latency: {latency.get('p99', 0):.0f}ms")

        # TTFT stats
        if 'ttft_ms' in result:
            ttft = result['ttft_ms']
            print(f"├── ⚡ Median TTFT: {ttft.get('median', 0):.0f}ms")

        # Memory stats
        if 'gpu_memory_gb' in result:
            memory = result['gpu_memory_gb']
            print(f"├── 🧠 Peak Memory: {memory.get('peak', 0):.1f}GB")

        # Duration accuracy
        print(f"├── ⏰ Duration Accuracy: {result.get('duration_accuracy', 0):.1%}")
        print(f"└── 📁 Saved to: benchmark_concurrency_{concurrency}_fixed.json")

    async def run_all_benchmarks(self):
        """Run benchmarks for all concurrency levels"""
        print(f"\n{'='*70}")
        print(f"STARTING MULTI-LEVEL BENCHMARK (FIXED VERSION)")
        print(f"Test Mode: {self.test_mode.value}")
        print(f"Concurrency Levels: {self.concurrency_levels}")
        print(f"Total estimated time: ~{len(self.concurrency_levels) * (self.config['phase_duration'] + self.config['ramp_up_duration'] + self.config['cool_down_duration']) / 60:.0f} minutes")
        print(f"{'='*70}\n")

        for concurrency in self.concurrency_levels:
            result = await self.run_single_benchmark(concurrency)
            self.results.append(result)

            # Checkpoint save
            self._save_summary()

            # Rest between tests
            if concurrency < self.concurrency_levels[-1]:
                print(f"\n⏸️  Resting 10 seconds before next test...")
                await asyncio.sleep(10)

        # Final analysis
        self._analyze_results()
        self._save_summary()

        print(f"\n{'='*70}")
        print(f"✅ ALL BENCHMARKS COMPLETED!")
        print(f"📁 Results saved to: {self.summary_file}")
        print(f"⏱️  Total time: {(time.time() - self.start_time) / 60:.1f} minutes")
        print(f"{'='*70}")

    def _analyze_results(self):
        """Analyze results and find optimal concurrency"""
        valid_results = [r for r in self.results if 'error' not in r and r.get('success_rate', 0) > 0.95]

        if not valid_results:
            logging.warning("No valid results with >95% success rate!")
            return

        # Find optimal by throughput
        optimal = max(valid_results, key=lambda x: x.get('system_output_throughput_tps', 0))

        print(f"\n🎯 OPTIMAL CONFIGURATION:")
        print(f"├── Concurrency: {optimal['concurrency']}")
        print(f"├── Throughput: {optimal.get('system_output_throughput_tps', 0):.1f} tokens/s")
        print(f"├── Success Rate: {optimal.get('success_rate', 0):.1%}")
        print(f"└── Median Latency: {optimal.get('latency_ms', {}).get('median', 0):.0f}ms")

        # Throughput curve analysis
        print(f"\n📈 THROUGHPUT CURVE:")
        for r in valid_results:
            throughput = r.get('system_output_throughput_tps', 0)
            bar = '█' * int(throughput / 50)  # Scale for display
            print(f"C={r['concurrency']:4d}: {bar} {throughput:.1f} tokens/s")

    def _save_summary(self):
        """Save summary of all results"""
        summary = {
            'test_mode': self.test_mode.value,
            'start_time': self.start_time,
            'end_time': time.time(),
            'total_duration_minutes': (time.time() - self.start_time) / 60,
            'config': self.config,
            'gpu_info': self.get_gpu_info(),
            'results': self.results,
            'analysis': self._get_analysis()
        }

        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=4)

    def _get_analysis(self) -> Dict:
        """Get analysis of results"""
        valid_results = [r for r in self.results if 'error' not in r and r.get('success_rate', 0) > 0.95]

        if not valid_results:
            return {'error': 'No valid results'}

        optimal = max(valid_results, key=lambda x: x.get('system_output_throughput_tps', 0))

        return {
            'optimal_concurrency': optimal['concurrency'],
            'max_throughput_tps': optimal.get('system_output_throughput_tps', 0),
            'optimal_latency_ms': optimal.get('latency_ms', {}).get('median', 0),
            'optimal_success_rate': optimal.get('success_rate', 0),
            'total_valid_tests': len(valid_results),
            'total_failed_tests': len([r for r in self.results if 'error' in r]),
            'recommendation': f"Optimal concurrency is {optimal['concurrency']} with {optimal.get('system_output_throughput_tps', 0):.1f} tokens/s throughput"
        }


async def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Fixed vLLM Multi-Concurrency Benchmark')
    parser.add_argument('--mode', choices=['closed_loop', 'open_loop'],
                        default='closed_loop', help='Test mode')
    parser.add_argument('--concurrency', type=int, nargs='+',
                        help='Custom concurrency levels (e.g., --concurrency 1 8 32)')
    parser.add_argument('--duration', type=int, default=180,
                        help='Test duration per level in seconds')
    parser.add_argument('--url', default='http://localhost:8000/v1',
                        help='vLLM server URL')
    parser.add_argument('--model', default='openai/gpt-oss-20b',
                        help='Model name')

    args = parser.parse_args()

    # Create benchmark
    test_mode = TestMode.CLOSED_LOOP if args.mode == 'closed_loop' else TestMode.OPEN_LOOP
    benchmark = MultiConcurrencyBenchmarkFixed(test_mode=test_mode)

    # Override settings if provided
    if args.concurrency:
        benchmark.concurrency_levels = args.concurrency
    if args.duration:
        benchmark.config['phase_duration'] = args.duration
    if args.url:
        benchmark.config['vllm_url'] = args.url
    if args.model:
        benchmark.config['model'] = args.model

    # Run benchmarks
    try:
        await benchmark.run_all_benchmarks()
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        benchmark._save_summary()
        print(f"📁 Partial results saved to: {benchmark.summary_file}")
    except Exception as e:
        logging.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())