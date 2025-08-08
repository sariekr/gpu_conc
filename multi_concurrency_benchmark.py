#!/usr/bin/env python3
"""
Multi-Level Concurrency Benchmark for vLLM
Tests performance across different concurrency levels: 1, 4, 8, 16, 32, 64, 128, 256, 512, 1024
"""

import asyncio
import logging
import json
import time
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
from concurrency import run_benchmark  # Import your benchmark module


class MultiConcurrencyBenchmark:
    """Run benchmark across multiple concurrency levels"""
    
    def __init__(self):
        self.concurrency_levels = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.results = []
        self.start_time = time.time()
        
        # Test configuration
        self.config = {
            'phase_duration': 180,      # 3 minutes per test
            'ramp_up_duration': 30,     # 30s warm-up
            'cool_down_duration': 30,   # 30s cool-down
            'input_tokens': 1000,       # Standard input size
            'output_tokens': 1000,      # Standard output size
            'request_timeout': 30,      # 30s timeout
            'vllm_url': "http://localhost:8000/v1",
            'api_key': "test-key",
            'model': "openai/gpt-oss-20b"
        }
    
    def get_gpu_info(self) -> Dict:
        """Get GPU information"""
        try:
            # Get GPU name and memory
            gpu_result = subprocess.check_output([
                "nvidia-smi", "--query-gpu=name,memory.total", 
                "--format=csv,noheader,nounits"
            ]).decode().strip()
            
            lines = gpu_result.split('\n')
            if lines:
                gpu_name, gpu_memory = lines[0].split(', ')
                return {
                    "model": gpu_name.strip(),
                    "memory": f"{gpu_memory.strip()}MB",
                    "count": len(lines)
                }
        except Exception as e:
            logging.warning(f"Could not get GPU info: {e}")
        
        return {"model": "Unknown GPU", "memory": "Unknown", "count": 1}
    
    def check_vllm_server(self) -> bool:
        """Check if vLLM server is running"""
        try:
            import requests
            response = requests.get(f"{self.config['vllm_url'].replace('/v1', '')}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def run_single_benchmark(self, concurrency: int) -> Dict:
        """Run benchmark for a single concurrency level"""
        
        print(f"\n{'='*60}")
        print(f"🚀 TESTING CONCURRENCY LEVEL: {concurrency}")
        print(f"{'='*60}")
        print(f"⏱️  Duration: {self.config['phase_duration']}s")
        print(f"🔧 Ramp-up: {self.config['ramp_up_duration']}s")
        print(f"❄️  Cool-down: {self.config['cool_down_duration']}s")
        print(f"📝 Input tokens: {self.config['input_tokens']}")
        print(f"📤 Output tokens: {self.config['output_tokens']}")
        
        output_file = f"benchmark_concurrency_{concurrency}.json"
        
        try:
            # Run the benchmark
            result = await run_benchmark(
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
                output_file=output_file
            )
            
            # Add timestamp and test info
            result['test_timestamp'] = time.time()
            result['test_duration_total'] = time.time() - self.start_time
            
            # Print quick results
            print(f"\n📊 RESULTS FOR CONCURRENCY {concurrency}:")
            print(f"├── ✅ Success Rate: {result['response_rate']:.1%}")
            print(f"├── 🚀 Throughput: {result['system_output_throughput']:.1f} tokens/s")
            print(f"├── 📈 Requests/s: {result['requests_per_second']:.2f}")
            print(f"├── ⏱️  Median Latency: {result['median_end_to_end_latency']:.0f}ms")
            print(f"├── 📊 P95 Latency: {result['p95_end_to_end_latency']:.0f}ms")
            print(f"├── 🧠 Peak Memory: {result['peak_memory_gb']:.1f}GB")
            print(f"└── 📁 Saved to: {output_file}")
            
            return result
            
        except Exception as e:
            logging.error(f"Benchmark failed for concurrency {concurrency}: {e}")
            return {
                'concurrency': concurrency,
                'error': str(e),
                'success': False,
                'test_timestamp': time.time()
            }
    
    def save_summary_results(self):
        """Save comprehensive summary of all results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        summary_file = f"benchmark_summary_{timestamp}.json"
        
        # Create summary
        summary = {
            'test_info': {
                'timestamp': timestamp,
                'total_duration_hours': (time.time() - self.start_time) / 3600,
                'concurrency_levels_tested': self.concurrency_levels,
                'config': self.config,
                'gpu_info': self.get_gpu_info()
            },
            'results': self.results,
            'performance_summary': self._create_performance_summary()
        }
        
        # Save to file
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\n📋 Comprehensive summary saved to: {summary_file}")
        return summary_file
    
    def _create_performance_summary(self) -> Dict:
        """Create performance summary table"""
        successful_results = [r for r in self.results if r.get('success', True) and 'error' not in r]
        
        if not successful_results:
            return {"error": "No successful tests"}
        
        summary = {}
        for result in successful_results:
            concurrency = result['concurrency']
            summary[concurrency] = {
                'throughput_tokens_per_sec': result.get('system_output_throughput', 0),
                'requests_per_sec': result.get('requests_per_second', 0),
                'success_rate': result.get('response_rate', 0),
                'median_latency_ms': result.get('median_end_to_end_latency', 0),
                'p95_latency_ms': result.get('p95_end_to_end_latency', 0),
                'peak_memory_gb': result.get('peak_memory_gb', 0),
                'successful_requests': result.get('successful_requests', 0),
                'failed_requests': result.get('failed_requests', 0)
            }
        
        # Find optimal concurrency (best throughput with >95% success rate)
        best_throughput = 0
        optimal_concurrency = 1
        
        for concurrency, metrics in summary.items():
            if (metrics['success_rate'] > 0.95 and 
                metrics['throughput_tokens_per_sec'] > best_throughput):
                best_throughput = metrics['throughput_tokens_per_sec']
                optimal_concurrency = concurrency
        
        summary['analysis'] = {
            'optimal_concurrency': optimal_concurrency,
            'max_stable_throughput': best_throughput,
            'total_tests_run': len(successful_results),
            'recommendation': f"Optimal concurrency level is {optimal_concurrency} with {best_throughput:.1f} tokens/s"
        }
        
        return summary
    
    def print_final_summary(self):
        """Print final test summary"""
        print(f"\n{'='*80}")
        print(f"🏁 BENCHMARK COMPLETE - FINAL SUMMARY")
        print(f"{'='*80}")
        
        successful_results = [r for r in self.results if r.get('success', True) and 'error' not in r]
        failed_results = [r for r in self.results if not r.get('success', True) or 'error' in r]
        
        print(f"📊 Tests Completed: {len(successful_results)}/{len(self.results)}")
        print(f"⏱️  Total Duration: {(time.time() - self.start_time)/3600:.1f} hours")
        
        if successful_results:
            print(f"\n🎯 PERFORMANCE OVERVIEW:")
            print(f"{'Concurrency':<12} {'Throughput':<12} {'Req/s':<8} {'Success%':<9} {'Latency':<10}")
            print(f"{'-'*60}")
            
            for result in successful_results:
                conc = result['concurrency']
                tput = result.get('system_output_throughput', 0)
                rps = result.get('requests_per_second', 0)
                success = result.get('response_rate', 0) * 100
                latency = result.get('median_end_to_end_latency', 0)
                
                print(f"{conc:<12} {tput:<12.1f} {rps:<8.2f} {success:<8.1f}% {latency:<10.0f}ms")
        
        if failed_results:
            print(f"\n❌ FAILED TESTS:")
            for result in failed_results:
                print(f"├── Concurrency {result['concurrency']}: {result.get('error', 'Unknown error')}")
    
    async def run_all_benchmarks(self):
        """Run benchmarks for all concurrency levels"""
        print(f"🏁 Starting Multi-Level Concurrency Benchmark")
        print(f"📊 Testing {len(self.concurrency_levels)} concurrency levels")
        print(f"⏱️  Estimated duration: {len(self.concurrency_levels) * (self.config['phase_duration'] + self.config['ramp_up_duration'] + self.config['cool_down_duration']) / 60:.1f} minutes")
        
        # Check vLLM server
        if not self.check_vllm_server():
            print("❌ vLLM server not responding. Please start the server first:")
            print("python -m vllm.entrypoints.openai.api_server --model NousResearch/Meta-Llama-3.1-8B-Instruct --port 8000 --api-key test-key")
            return
        
        print("✅ vLLM server is running")
        
        # Run benchmarks
        for i, concurrency in enumerate(self.concurrency_levels, 1):
            print(f"\n🔄 Progress: {i}/{len(self.concurrency_levels)}")
            
            result = await self.run_single_benchmark(concurrency)
            self.results.append(result)
            
            # Small break between tests
            if i < len(self.concurrency_levels):
                print(f"😴 Resting 30 seconds before next test...")
                await asyncio.sleep(30)
        
        # Save and display final results
        self.save_summary_results()
        self.print_final_summary()


async def main():
    """Main execution function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    benchmark = MultiConcurrencyBenchmark()
    
    try:
        await benchmark.run_all_benchmarks()
        print(f"\n✅ All benchmarks completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Benchmark interrupted by user")
        if benchmark.results:
            print(f"💾 Saving partial results...")
            benchmark.save_summary_results()
            benchmark.print_final_summary()
    
    except Exception as e:
        logging.error(f"Benchmark suite failed: {e}")
        raise


if __name__ == "__main__":
    print("🚀 Multi-Level Concurrency Benchmark for vLLM")
    print("=" * 60)
    asyncio.run(main())