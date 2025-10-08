#!/usr/bin/env python3
"""
GPU vLLM Benchmark Core Module - FIXED VERSION
Düzeltilmiş metodoloji ile doğru performans ölçümü
"""

import asyncio
import time
import random
import statistics
import logging
from typing import List, Dict, Optional, Tuple
from openai import AsyncOpenAI
import psutil
import numpy as np
import pynvml
from dataclasses import dataclass, asdict
import json
from enum import Enum


class TestMode(Enum):
    """Benchmark test modları"""
    CLOSED_LOOP = "closed_loop"  # Sabit concurrency seviyesi
    OPEN_LOOP = "open_loop"      # Sabit request rate


@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    request_id: int
    start_time: float
    first_token_time: Optional[float]
    end_time: Optional[float]
    output_tokens: int
    success: bool
    error_message: Optional[str] = None
    prompt_tokens: int = 0

    @property
    def ttft_ms(self) -> Optional[float]:
        """Time to first token in milliseconds"""
        if self.first_token_time and self.start_time:
            return (self.first_token_time - self.start_time) * 1000
        return None

    @property
    def end_to_end_latency_ms(self) -> Optional[float]:
        """End-to-end latency in milliseconds"""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return None

    @property
    def output_speed_tps(self) -> Optional[float]:
        """Output tokens per second after first token"""
        if (self.first_token_time and self.end_time and
            self.output_tokens > 0 and self.end_time > self.first_token_time):
            duration = self.end_time - self.first_token_time
            return self.output_tokens / duration if duration > 0 else 0
        return None


class SyntheticDatasetGenerator:
    """Generate synthetic datasets with consistent token counts"""

    def __init__(self, target_tokens: int = 1000):
        self.target_tokens = target_tokens

        # Orijinal benchmark'tan alınan prompts
        self.base_prompts = [
            "Explain the concept of artificial intelligence in simple terms.",
            "What are the main causes of climate change?",
            "Describe the process of photosynthesis in plants.",
            "How does the human immune system work?",
            "What were the main causes of World War II?",
            "Explain the theory of relativity in layman's terms.",
            "What are the key principles of effective leadership?",
            "How does blockchain technology work?",
            "What are the main theories about the origin of the universe?",
            "Describe the water cycle and its importance for life on Earth.",
        ]

    def generate_prompt(self) -> str:
        """Generate a prompt with approximately target token count"""
        base = random.choice(self.base_prompts)

        # Token sayısını tahmin et (yaklaşık 1.3 karakter = 1 token)
        estimated_tokens = len(base) / 4

        if estimated_tokens >= self.target_tokens:
            return base

        # Prompt'u genişlet
        expansion = []
        tokens_needed = self.target_tokens - estimated_tokens

        context = (
            "Please provide a detailed and comprehensive answer to the following question. "
            "Include relevant examples, explanations, and consider multiple perspectives. "
            "The response should be thorough and well-structured. "
        )

        # Context'i token hedefine göre tekrarla
        while len(' '.join(expansion)) / 4 < tokens_needed:
            expansion.append(context)

        return ' '.join(expansion) + "\n\n" + base


class GPUMemoryMonitor:
    """Monitor GPU memory usage during benchmarking"""

    def __init__(self):
        self.memory_samples = []
        self.peak_memory = 0.0
        self.monitoring = False
        self._monitor_task = None

    async def start_monitoring(self, interval: float = 1.0):
        """Start memory monitoring in background"""
        self.monitoring = True
        self.memory_samples = []
        self.peak_memory = 0.0

        try:
            pynvml.nvmlInit()
            while self.monitoring:
                try:
                    current_memory = self._get_gpu_memory_usage()
                    timestamp = time.time()
                    self.memory_samples.append({
                        'timestamp': timestamp,
                        'memory_gb': current_memory
                    })
                    self.peak_memory = max(self.peak_memory, current_memory)
                except Exception as e:
                    logging.warning(f"GPU memory monitoring error: {e}")

                await asyncio.sleep(interval)
            pynvml.nvmlShutdown()
        except Exception as e:
            logging.error(f"Failed to initialize GPU memory monitoring: {e}")
            self.monitoring = False

    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False

    def reset_stats(self):
        """Reset statistics after warm-up"""
        self.memory_samples = []
        self.peak_memory = 0.0
        if self.memory_samples:
            # Son değeri koru ama önceki samples'ı temizle
            last_memory = self.memory_samples[-1]['memory_gb']
            self.peak_memory = last_memory

    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in GB"""
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            total_memory = 0
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_memory += mem_info.used
            return total_memory / (1024 ** 3)  # Convert to GB
        except Exception as e:
            logging.error(f"Failed to get GPU memory usage: {e}")
            return 0.0

    def get_average_memory_usage(self) -> float:
        """Get average memory usage during monitoring"""
        if not self.memory_samples:
            return 0.0
        return statistics.mean(sample['memory_gb'] for sample in self.memory_samples)


async def process_streaming_response(stream, request_metrics: RequestMetrics):
    """Process streaming response and collect metrics"""
    collected_content = ""

    try:
        async for chunk in stream:
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                choice = chunk.choices[0]

                if hasattr(choice, 'delta') and choice.delta.content:
                    if request_metrics.first_token_time is None:
                        request_metrics.first_token_time = time.time()
                    collected_content += choice.delta.content
                    request_metrics.output_tokens += 1

                if hasattr(choice, 'finish_reason') and choice.finish_reason is not None:
                    break

        request_metrics.end_time = time.time()
        request_metrics.success = bool(collected_content.strip())

    except Exception as e:
        request_metrics.end_time = time.time()
        request_metrics.success = False
        request_metrics.error_message = str(e)


async def make_single_request(
    client: AsyncOpenAI,
    request_id: int,
    prompt: str,
    output_tokens: int,
    request_timeout: int,
    model: str
) -> RequestMetrics:
    """Make a single request and collect detailed metrics"""
    metrics = RequestMetrics(
        request_id=request_id,
        start_time=time.time(),
        first_token_time=None,
        end_time=None,
        output_tokens=0,
        success=False,
        prompt_tokens=len(prompt) // 4  # Rough estimate
    )

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=output_tokens,
            stream=True
        )

        await asyncio.wait_for(
            process_streaming_response(stream, metrics),
            timeout=request_timeout
        )

    except asyncio.TimeoutError:
        metrics.end_time = time.time()
        metrics.success = False
        metrics.error_message = "Request timeout"

    except Exception as e:
        metrics.end_time = time.time()
        metrics.success = False
        metrics.error_message = str(e)

    return metrics


async def closed_loop_worker(
    client: AsyncOpenAI,
    request_queue: asyncio.Queue,
    results_queue: asyncio.Queue,
    dataset_generator: SyntheticDatasetGenerator,
    output_tokens: int,
    request_timeout: int,
    model: str,
    worker_id: int
):
    """Worker for closed-loop testing (maintains constant concurrency)"""
    while True:
        try:
            request_data = await request_queue.get()
            if request_data is None:  # Sentinel to stop
                break

            request_id, is_warmup = request_data
            prompt = dataset_generator.generate_prompt()

            metrics = await make_single_request(
                client, request_id, prompt, output_tokens, request_timeout, model
            )

            # Result'ı queue'ya koy
            await results_queue.put((metrics, is_warmup))

        except Exception as e:
            logging.error(f"Worker {worker_id} error: {e}")
        finally:
            request_queue.task_done()


async def run_benchmark_fixed(
    concurrency: int,
    phase_duration: int,
    ramp_up_duration: int,
    cool_down_duration: int,
    input_tokens: int,
    output_tokens: int,
    request_timeout: int,
    vllm_url: str,
    api_key: str,
    gpu_info: Dict,
    model: str = "openai/gpt-oss-20b",
    output_file: str = "benchmark_results.json",
    test_mode: TestMode = TestMode.CLOSED_LOOP,
    target_request_rate: Optional[float] = None
) -> Dict:
    """
    FIXED benchmark with proper methodology

    Key fixes:
    1. Proper measurement boundaries - only measure what happens during test phase
    2. Clean separation between warm-up and test phases
    3. Controlled request generation
    4. Accurate throughput calculation
    """

    logging.info(f"Starting FIXED benchmark: concurrency={concurrency}, mode={test_mode.value}")

    async with AsyncOpenAI(base_url=vllm_url, api_key=api_key) as client:
        # Initialize components
        dataset_generator = SyntheticDatasetGenerator(target_tokens=input_tokens)
        memory_monitor = GPUMemoryMonitor()

        # Queues
        request_queue = asyncio.Queue()
        results_queue = asyncio.Queue()

        # Start memory monitoring
        memory_task = asyncio.create_task(memory_monitor.start_monitoring())

        # Create workers
        workers = [
            asyncio.create_task(
                closed_loop_worker(
                    client, request_queue, results_queue, dataset_generator,
                    output_tokens, request_timeout, model, worker_id=i
                )
            )
            for i in range(concurrency)
        ]

        # ============= WARM-UP PHASE =============
        logging.info(f"Starting warm-up phase ({ramp_up_duration}s)...")
        warmup_start = time.time()
        warmup_request_id = 0

        # Send warm-up requests
        while time.time() - warmup_start < ramp_up_duration:
            # Queue'da çok fazla istek birikmesini engelle
            if request_queue.qsize() < concurrency * 2:
                await request_queue.put((f"warmup_{warmup_request_id}", True))
                warmup_request_id += 1
            await asyncio.sleep(0.1)

        # Warm-up isteklerinin bitmesini bekle
        logging.info("Waiting for warm-up requests to complete...")
        await request_queue.join()

        # Warm-up sonuçlarını temizle
        warmup_results = []
        while not results_queue.empty():
            try:
                result, is_warmup = results_queue.get_nowait()
                if is_warmup:
                    warmup_results.append(result)
            except asyncio.QueueEmpty:
                break

        logging.info(f"Warm-up completed: {len(warmup_results)} requests processed")

        # Memory stats'ları resetle (warm-up'ı sayma)
        memory_monitor.reset_stats()

        # ============= MAIN TEST PHASE =============
        logging.info("Starting MAIN TEST PHASE (measured)...")
        test_start_time = time.time()
        test_results = []
        request_id = 0

        if test_mode == TestMode.CLOSED_LOOP:
            # Closed-loop: Sabit concurrency seviyesi
            # Hedef: Test süresi boyunca sürekli 'concurrency' kadar aktif istek

            # Başlangıçta concurrency kadar istek gönder
            for i in range(concurrency):
                await request_queue.put((request_id, False))
                request_id += 1

            # Test süresi boyunca tamamlanan isteklerin yerine yenilerini gönder
            test_end_time = test_start_time + phase_duration

            while time.time() < test_end_time:
                try:
                    # Tamamlanan bir istek bekle (timeout ile)
                    result, is_warmup = await asyncio.wait_for(
                        results_queue.get(),
                        timeout=1.0
                    )

                    if not is_warmup:
                        test_results.append(result)

                        # Yeni istek gönder (concurrency'yi koru)
                        if time.time() < test_end_time:
                            await request_queue.put((request_id, False))
                            request_id += 1

                except asyncio.TimeoutError:
                    # Timeout'ta devam et
                    continue

        else:  # OPEN_LOOP mode
            # Open-loop: Sabit request rate
            if not target_request_rate:
                target_request_rate = concurrency / 2  # Default rate

            request_interval = 1.0 / target_request_rate
            next_request_time = test_start_time

            while time.time() < test_start_time + phase_duration:
                current_time = time.time()

                if current_time >= next_request_time:
                    await request_queue.put((request_id, False))
                    request_id += 1
                    next_request_time += request_interval

                # Kısa süre bekle
                await asyncio.sleep(min(0.01, request_interval / 2))

        # Test süresi doldu, kalan istekleri topla
        actual_test_end = time.time()
        actual_test_duration = actual_test_end - test_start_time

        logging.info(f"Test phase ended. Collecting remaining results...")

        # Kalan sonuçları topla (max 30 saniye bekle)
        collection_timeout = 30
        collection_start = time.time()

        while time.time() - collection_start < collection_timeout:
            try:
                result, is_warmup = await asyncio.wait_for(
                    results_queue.get(),
                    timeout=1.0
                )
                if not is_warmup:
                    test_results.append(result)
            except asyncio.TimeoutError:
                if request_queue.empty() and results_queue.empty():
                    break

        # ============= CLEANUP =============
        # Stop workers
        for _ in range(concurrency):
            await request_queue.put(None)

        await asyncio.gather(*workers, return_exceptions=True)

        # Stop memory monitoring
        memory_monitor.stop_monitoring()
        await memory_task

        # ============= CALCULATE METRICS =============
        # Sadece test fazındaki istekleri kullan
        successful_requests = [r for r in test_results if r.success]
        failed_requests = [r for r in test_results if not r.success]

        # Gerçek test süresini hesapla (ilk istek başlangıcı - son istek bitişi)
        if successful_requests:
            actual_start = min(r.start_time for r in successful_requests)
            actual_end = max(r.end_time for r in successful_requests if r.end_time)
            actual_duration = actual_end - actual_start
        else:
            actual_duration = actual_test_duration

        logging.info(f"Actual test duration: {actual_duration:.2f}s (planned: {phase_duration}s)")

        if not successful_requests:
            logging.warning("No successful requests in test phase!")
            results_dict = {
                'concurrency': concurrency,
                'test_mode': test_mode.value,
                'planned_duration': phase_duration,
                'actual_duration': actual_duration,
                'total_requests': len(test_results),
                'successful_requests': 0,
                'failed_requests': len(test_results),
                'error': 'No successful requests'
            }
        else:
            # Metrikleri hesapla
            total_output_tokens = sum(r.output_tokens for r in successful_requests)
            total_input_tokens = sum(r.prompt_tokens for r in successful_requests)

            end_to_end_latencies = [r.end_to_end_latency_ms for r in successful_requests if r.end_to_end_latency_ms]
            ttft_values = [r.ttft_ms for r in successful_requests if r.ttft_ms]
            output_speeds = [r.output_speed_tps for r in successful_requests if r.output_speed_tps]

            results_dict = {
                'concurrency': concurrency,
                'test_mode': test_mode.value,
                'planned_duration': phase_duration,
                'actual_duration': actual_duration,
                'duration_accuracy': actual_duration / phase_duration,  # Ne kadar doğru ölçtük?

                # Request metrics
                'total_requests': len(test_results),
                'successful_requests': len(successful_requests),
                'failed_requests': len(failed_requests),
                'success_rate': len(successful_requests) / len(test_results) if test_results else 0,

                # Throughput - DOĞRU HESAPLAMA!
                'system_output_throughput_tps': total_output_tokens / actual_duration,
                'system_input_throughput_tps': total_input_tokens / actual_duration,
                'requests_per_second': len(successful_requests) / actual_duration,

                # Latency metrics
                'latency_ms': {
                    'median': statistics.median(end_to_end_latencies) if end_to_end_latencies else 0,
                    'mean': statistics.mean(end_to_end_latencies) if end_to_end_latencies else 0,
                    'p95': float(np.percentile(end_to_end_latencies, 95)) if end_to_end_latencies else 0,
                    'p99': float(np.percentile(end_to_end_latencies, 99)) if end_to_end_latencies else 0,
                    'min': min(end_to_end_latencies) if end_to_end_latencies else 0,
                    'max': max(end_to_end_latencies) if end_to_end_latencies else 0,
                },

                # TTFT metrics
                'ttft_ms': {
                    'median': statistics.median(ttft_values) if ttft_values else 0,
                    'mean': statistics.mean(ttft_values) if ttft_values else 0,
                    'p95': float(np.percentile(ttft_values, 95)) if ttft_values else 0,
                    'p99': float(np.percentile(ttft_values, 99)) if ttft_values else 0,
                },

                # Per-request output speed
                'output_speed_tps': {
                    'median': statistics.median(output_speeds) if output_speeds else 0,
                    'mean': statistics.mean(output_speeds) if output_speeds else 0,
                },

                # Memory metrics
                'gpu_memory_gb': {
                    'peak': memory_monitor.peak_memory,
                    'average': memory_monitor.get_average_memory_usage(),
                },

                # System info
                'gpu_info': gpu_info,
                'model': model,
                'test_config': {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'request_timeout': request_timeout,
                    'ramp_up_duration': ramp_up_duration,
                    'cool_down_duration': cool_down_duration,
                },

                # Timestamp
                'timestamp': time.time(),
            }

        # Save results
        try:
            with open(output_file, 'w') as f:
                json.dump(results_dict, f, indent=4)
            logging.info(f"Results saved to {output_file}")
        except Exception as e:
            logging.error(f"Failed to save results: {e}")

        return results_dict


# Export
__all__ = ['run_benchmark_fixed', 'TestMode']


if __name__ == "__main__":
    async def main():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        results = await run_benchmark_fixed(
            concurrency=32,
            phase_duration=180,
            ramp_up_duration=30,
            cool_down_duration=30,
            input_tokens=1000,
            output_tokens=1000,
            request_timeout=30,
            vllm_url="http://localhost:8000/v1",
            api_key="test-key",
            gpu_info={"model": "NVIDIA H100", "memory": "80GB"},
            model="your-model",
            output_file="benchmark_fixed.json",
            test_mode=TestMode.CLOSED_LOOP
        )

        print(f"\nBenchmark completed!")
        print(f"Throughput: {results.get('system_output_throughput_tps', 0):.2f} tokens/s")
        print(f"Success rate: {results.get('success_rate', 0):.1%}")

    asyncio.run(main())