#!/usr/bin/env python3
"""
GPU vLLM Benchmark Core Module - AMD GPU Version
Comprehensive benchmarking with synthetic dataset generation and detailed metrics
"""

import asyncio
import time
import random
import statistics
import logging
from typing import List, Dict, Optional
from openai import AsyncOpenAI
import psutil
import numpy as np
import subprocess
import os
from dataclasses import dataclass, asdict
import json


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
    """Generate synthetic datasets with consistent token counts using original dataset"""
    
    def __init__(self, target_tokens: int = 1000):
        """
        Initialize the dataset generator with a target token count.
        
        Args:
            target_tokens (int): Target number of tokens for generated prompts
        """
        self.target_tokens = target_tokens
        
        # Use exact same prompts from original vllm_benchmark.py
        self.short_prompts = [
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
            "What are the major differences between capitalism and socialism?",
            "How does the human brain process and store memories?",
            "What are the main challenges in space exploration?",
            "Explain the concept of supply and demand in economics.",
        ]
        
        # Use exact same long context pairs from original vllm_benchmark.py
        self.long_prompt_pairs = [
            {
                "prompt": "Explain the concept of artificial intelligence in simple terms.",
                "context": "Artificial intelligence (AI) is a rapidly evolving field of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence. These tasks include visual perception, speech recognition, decision-making, and language translation. AI systems are designed to learn from experience, adjust to new inputs, and perform human-like tasks. The field of AI encompasses various subfields, including machine learning, neural networks, and deep learning, which have led to significant advancements in areas such as autonomous vehicles, virtual assistants, and recommendation systems."
            },
            {
                "prompt": "What are the main causes of climate change?",
                "context": "Climate change is a complex global phenomenon primarily driven by human activities that release greenhouse gases into the atmosphere. The burning of fossil fuels for energy, deforestation, industrial processes, and agriculture are major contributors to the increased concentration of carbon dioxide and other heat-trapping gases. These gases form a 'blanket' around the Earth, causing the planet to warm at an unprecedented rate. The resulting changes in temperature patterns lead to more frequent and severe weather events, rising sea levels, and disruptions to ecosystems worldwide."
            },
            {
                "prompt": "Describe the process of photosynthesis in plants.",
                "context": "Photosynthesis is a fundamental biological process that allows plants to convert light energy into chemical energy. This process occurs in the chloroplasts of plant cells, specifically in structures called thylakoids. Chlorophyll, the pigment that gives plants their green color, is crucial in capturing light energy. During photosynthesis, plants take in carbon dioxide from the air through tiny pores called stomata and water from the soil through their roots. Using light energy, they combine these ingredients to produce glucose and oxygen. This process not only provides energy for the plant but also releases oxygen as a byproduct, which is essential for most life on Earth."
            },
            {
                "prompt": "How does the human immune system work?",
                "context": "The human immune system is a complex network of cells, tissues, and organs that work together to defend the body against harmful pathogens. It consists of two main parts: the innate immune system, which provides a quick, non-specific response to invaders, and the adaptive immune system, which develops targeted defenses against specific pathogens. Key components include white blood cells (such as neutrophils, macrophages, and lymphocytes), antibodies, and the complement system. The immune system has the remarkable ability to distinguish between the body's own cells and foreign invaders, allowing it to target threats while minimizing damage to healthy tissue."
            },
            {
                "prompt": "What were the main causes of World War II?",
                "context": "World War II, which lasted from 1939 to 1945, was one of the deadliest conflicts in human history. Its origins can be traced to several complex factors. The harsh terms of the Treaty of Versailles, which ended World War I, left Germany economically devastated and resentful. This paved the way for the rise of fascism and the Nazi Party under Adolf Hitler. Aggressive expansionist policies by Nazi Germany, Fascist Italy, and Imperial Japan, combined with the policy of appeasement by Western powers, allowed these regimes to gain territory unchecked. The immediate trigger for the war in Europe was Germany's invasion of Poland in September 1939, while the attack on Pearl Harbor in 1941 brought the United States into the conflict."
            },
            {
                "prompt": "Explain the theory of relativity in layman's terms.",
                "context": "Albert Einstein's theory of relativity, developed in the early 20th century, revolutionized our understanding of space, time, and gravity. It consists of two parts: special relativity and general relativity. Special relativity, introduced in 1905, deals with objects moving at very high speeds. It proposes that the speed of light is constant for all observers and that time and space are not absolute but relative to the observer's motion. This leads to phenomena like time dilation and length contraction. General relativity, published in 1915, extends these ideas to include gravity. Einstein proposed that massive objects curve the fabric of spacetime, and this curvature is what we experience as gravity. These theories have been consistently supported by experimental evidence and have practical applications in technologies like GPS satellites."
            },
            {
                "prompt": "What are the key principles of effective leadership?",
                "context": "Effective leadership is crucial in guiding organizations, teams, and individuals towards achieving their goals. While leadership styles may vary, several key principles are widely recognized as essential for success. These include clear communication, which ensures that vision and expectations are understood by all; integrity, which builds trust and respect; adaptability, allowing leaders to navigate changing environments; empathy, fostering strong relationships and understanding team dynamics; decision-making skills, enabling timely and informed choices; vision, providing direction and inspiration; and the ability to empower others, encouraging growth and innovation within the team. Effective leaders also demonstrate accountability, both for their own actions and those of their team, and continuously seek personal growth and learning opportunities."
            },
            {
                "prompt": "How does blockchain technology work?",
                "context": "Blockchain is a decentralized, distributed ledger technology that underlies cryptocurrencies like Bitcoin, but has potential applications far beyond digital currencies. At its core, a blockchain is a chain of blocks, each containing a list of transactions. Every block is linked to the previous one through cryptographic hashes, creating an immutable record. The key innovation of blockchain is its ability to achieve consensus in a decentralized network without requiring trust in any single entity. This is typically achieved through consensus mechanisms like Proof of Work or Proof of Stake. When a new transaction occurs, it is broadcast to a network of computers (nodes) for validation. Once validated, the transaction is combined with others to create a new block, which is then added to the chain. This process ensures transparency, security, and resistance to tampering, making blockchain suitable for various applications beyond finance, including supply chain management, voting systems, and digital identity verification."
            },
            {
                "prompt": "What are the main theories about the origin of the universe?",
                "context": "The origin of the universe has been a subject of intense scientific inquiry and philosophical debate for centuries. Currently, the most widely accepted scientific theory is the Big Bang model, which proposes that the universe began as an infinitely dense and hot singularity about 13.8 billion years ago, and has been expanding and cooling ever since. This theory is supported by observational evidence such as the cosmic microwave background radiation and the abundance of light elements in the universe. However, questions remain about what happened before the Big Bang and what caused it. Other theories include the Steady State theory, which suggests that the universe has always existed and is constantly creating new matter as it expands, though this theory has fallen out of favor due to lack of supporting evidence. More speculative ideas include the concept of a cyclic universe, where big bangs and big crunches occur in an endless cycle, and the idea of a multiverse, where our universe is just one of many existing universes."
            },
            {
                "prompt": "Describe the water cycle and its importance for life on Earth.",
                "context": "The water cycle, also known as the hydrologic cycle, is the continuous movement of water within the Earth and atmosphere. It is a complex system involving the processes of evaporation, transpiration, condensation, precipitation, and runoff. Water evaporates from the Earth's surface, primarily from oceans, lakes, and rivers, due to solar energy. Plants also release water vapor through transpiration. As this water vapor rises in the atmosphere, it cools and condenses to form clouds. Eventually, it falls back to Earth as precipitation in the form of rain, snow, or hail. Some of this water flows over the land as surface runoff, returning to bodies of water, while some seeps into the ground, replenishing groundwater reserves. This cycle is crucial for life on Earth as it redistributes water around the globe, shapes landscapes through erosion and deposition, regulates global temperatures, and provides fresh water essential for all living organisms. Understanding and protecting the water cycle is vital for managing water resources and addressing environmental challenges like climate change and water scarcity."
            },
            {
                "prompt": "What are the major differences between capitalism and socialism?",
                "context": "Capitalism and socialism are two contrasting economic and political systems that have shaped much of modern history. Capitalism is characterized by private ownership of the means of production, where individuals or corporations own businesses and property. It operates on the principles of free market competition, with prices determined by supply and demand. Profit is a key motivator in capitalist systems, and government intervention is generally limited. In contrast, socialism advocates for collective or governmental ownership and administration of the means of production and distribution of goods. It aims to create a more equitable society by reducing class distinctions and distributing resources according to need rather than ability to pay. In socialist systems, the government plays a much larger role in economic planning and the provision of social services. While pure forms of either system are rare, many countries adopt mixed economies incorporating elements of both capitalism and socialism to varying degrees."
            },
            {
                "prompt": "How does the human brain process and store memories?",
                "context": "The human brain's ability to process and store memories is a complex and fascinating process involving various regions and neural networks. When we experience something, sensory information is first processed in the relevant cortical areas (e.g., visual cortex for sight, auditory cortex for sound). This information is then integrated in the hippocampus, a seahorse-shaped structure crucial for forming new memories. The hippocampus helps bind different aspects of an experience into a cohesive memory and plays a key role in converting short-term memories into long-term ones. Long-term memories are thought to be stored through changes in synaptic connections between neurons across widespread areas of the cortex. This process, known as consolidation, can take days or even years. Different types of memories (e.g., episodic, semantic, procedural) involve different brain regions and processes. The retrieval of memories involves reactivating these neural patterns, which explains why memories can be influenced by our current state and environment. Understanding these processes is crucial for addressing memory-related disorders and developing potential therapies."
            },
            {
                "prompt": "What are the main challenges in space exploration?",
                "context": "Space exploration, while offering immense potential for scientific discovery and technological advancement, faces numerous challenges. One of the primary obstacles is the hostile environment of space itself. The vacuum of space, extreme temperatures, and harmful radiation pose significant risks to both human astronauts and sensitive equipment. Prolonged exposure to microgravity can lead to health issues for astronauts, including muscle atrophy and bone density loss. Logistical challenges are also substantial: the enormous distances involved in space travel require advanced propulsion systems and careful resource management. Launching payloads into orbit remains extremely expensive, limiting the scope and frequency of missions. Communication delays become increasingly problematic for deep space missions, necessitating a high degree of autonomy in spacecraft and rovers. Additionally, space debris orbiting Earth poses a growing threat to satellites and spacecraft. As we look towards long-term goals like establishing bases on the Moon or Mars, we face new challenges in creating sustainable habitats and managing psychological effects on crew members during extended missions. Despite these obstacles, ongoing research and technological innovations continue to push the boundaries of what's possible in space exploration."
            },
            {
                "prompt": "Explain the concept of supply and demand in economics.",
                "context": "Supply and demand is a fundamental concept in economics that describes how the price and quantity of a good or service in a market are determined through the interaction between buyers and sellers. The law of demand states that, all else being equal, as the price of a product increases, the quantity demanded by consumers decreases. This is typically represented by a downward-sloping demand curve. Conversely, the law of supply states that as the price of a product increases, the quantity that producers are willing to supply increases, represented by an upward-sloping supply curve. The point where these two curves intersect is called the equilibrium point, determining the market price and quantity. This model helps explain how prices fluctuate in response to changes in supply or demand. For instance, if demand increases while supply remains constant, prices will rise. If supply increases while demand remains constant, prices will fall. Understanding supply and demand is crucial for analyzing market behavior, predicting price changes, and formulating economic policies."
            }
        ]
        
        self.technical_expansions = [
            "comprehensive analysis", "systematic approach", "implementation methodology",
            "optimization strategies", "performance metrics", "scalability considerations",
            "reliability factors", "efficiency improvements", "integration processes",
            "evaluation criteria", "best practices", "industry standards",
            "research findings", "experimental results", "statistical significance",
            "comparative studies", "theoretical frameworks", "practical applications"
        ]
        
        # Filler content patterns for reaching token targets
        self.filler_patterns = [
            "Furthermore, it is important to consider the implications of",
            "In addition to these factors, researchers have identified",
            "Moreover, the significance of this approach extends to",
            "Additionally, studies have demonstrated the effectiveness of",
            "Consequently, the implementation of these principles requires",
            "Nevertheless, the complexity of this system involves",
            "Subsequently, the development of new methodologies has enabled",
            "Therefore, understanding these mechanisms is crucial for"
        ]
    
    def expand_to_target_tokens(self, base_content: str) -> str:
        """
        Expand content to reach target token count using technical expansions
        
        Args:
            base_content (str): Base content to expand
            
        Returns:
            str: Expanded content reaching approximately target tokens
        """
        current_words = base_content.split()
        target_words = int(self.target_tokens * 0.75)  # Rough token to word ratio
        
        if len(current_words) >= target_words:
            return base_content
        
        words_needed = target_words - len(current_words)
        expansion_parts = []
        
        # Add technical expansions to reach target
        while len(' '.join(expansion_parts).split()) < words_needed:
            # Add filler pattern
            if random.random() < 0.6:  # 60% chance
                expansion_parts.append(random.choice(self.filler_patterns))
            
            # Add technical terms
            technical_phrase = ' '.join(random.choices(self.technical_expansions, k=random.randint(2, 4)))
            expansion_parts.append(technical_phrase)
            
            # Add connecting words
            if random.random() < 0.4:  # 40% chance
                connectors = ["and", "or", "however", "therefore", "moreover", "furthermore", "additionally", "consequently"]
                expansion_parts.append(random.choice(connectors))
        
        # Combine and trim to target length
        expansion_text = ' '.join(expansion_parts)
        expansion_words = expansion_text.split()[:words_needed]
        
        return base_content + " " + ' '.join(expansion_words)
    
    def generate_dataset_entry(self) -> str:
        """
        Generate a single dataset entry with target token count using original dataset
        
        Returns:
            str: Generated prompt with approximately target token count
        """
        # Randomly choose between short prompt expansion or long context pair
        use_long_context = random.choice([True, False])
        
        if use_long_context:
            # Use long context pairs exactly as in original code
            prompt_pair = random.choice(self.long_prompt_pairs)
            base_content = prompt_pair["context"] + "\n\n" + prompt_pair["prompt"]
        else:
            # Use short prompts exactly as in original code
            base_content = random.choice(self.short_prompts)
        
        # Expand to reach target token count
        return self.expand_to_target_tokens(base_content)


class GPUMemoryMonitor:
    """Monitor AMD GPU memory usage during benchmarking using ROCm tools"""
    
    def __init__(self):
        """Initialize memory monitor"""
        self.memory_samples = []
        self.peak_memory = 0.0
        self.monitoring = False
        self.rocm_available = self._check_rocm_availability()
    
    def _check_rocm_availability(self) -> bool:
        """Check if ROCm tools are available"""
        try:
            # Check for rocm-smi
            result = subprocess.run(['rocm-smi', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logging.info("ROCm SMI detected - using rocm-smi for GPU monitoring")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        # Check for radeontop
        try:
            result = subprocess.run(['radeontop', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logging.info("radeontop detected - using radeontop for GPU monitoring")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        # Check for AMD GPU via lspci
        try:
            result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=5)
            if 'AMD' in result.stdout and ('VGA' in result.stdout or 'Display' in result.stdout):
                logging.warning("AMD GPU detected but no ROCm tools available - memory monitoring disabled")
                return False
        except Exception:
            pass
        
        logging.warning("No AMD GPU monitoring tools available - memory monitoring disabled")
        return False
    
    async def start_monitoring(self, interval: float = 1.0):
        """
        Start memory monitoring in background
        
        Args:
            interval (float): Time interval between memory samples in seconds
        """
        self.monitoring = True
        self.memory_samples = []
        self.peak_memory = 0.0
        
        if not self.rocm_available:
            logging.warning("ROCm tools not available - skipping GPU memory monitoring")
            return
        
        while self.monitoring:
            try:
                current_memory = self._get_gpu_memory_usage()
                if current_memory > 0:  # Only record valid readings
                    self.memory_samples.append({
                        'timestamp': time.time(),
                        'memory_gb': current_memory
                    })
                    self.peak_memory = max(self.peak_memory, current_memory)
            except Exception as e:
                logging.warning(f"GPU memory monitoring error: {e}")
            
            await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
    
    def _get_gpu_memory_usage(self) -> float:
        """
        Get current AMD GPU memory usage in GB using available tools
        
        Returns:
            float: Total used GPU memory in GB across all AMD devices
        """
        if not self.rocm_available:
            return 0.0
        
        try:
            # Try rocm-smi first (most reliable for ROCm GPUs)
            return self._get_memory_rocm_smi()
        except Exception:
            try:
                # Fallback to parsing /sys filesystem for AMD GPUs
                return self._get_memory_sysfs()
            except Exception as e:
                logging.warning(f"Failed to get AMD GPU memory usage: {e}")
                return 0.0
    
    def _get_memory_rocm_smi(self) -> float:
        """Get memory usage using rocm-smi"""
        try:
            # Get memory info from rocm-smi
            result = subprocess.run([
                'rocm-smi', '--showmemuse', '--csv'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                raise Exception(f"rocm-smi failed: {result.stderr}")
            
            total_memory_gb = 0.0
            lines = result.stdout.strip().split('\n')
            
            for line in lines[1:]:  # Skip header
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            # Memory usage is typically in MB, convert to GB
                            memory_mb = float(parts[1].strip().replace('MB', '').replace('GB', ''))
                            if 'GB' in parts[1]:
                                total_memory_gb += memory_mb
                            else:
                                total_memory_gb += memory_mb / 1024.0
                        except (ValueError, IndexError):
                            continue
            
            return total_memory_gb
            
        except Exception as e:
            raise Exception(f"rocm-smi memory reading failed: {e}")
    
    def _get_memory_sysfs(self) -> float:
        """Get memory usage from /sys filesystem (fallback method)"""
        try:
            total_memory_gb = 0.0
            
            # Look for AMD GPU devices in /sys/class/drm/
            drm_path = "/sys/class/drm"
            if os.path.exists(drm_path):
                for device in os.listdir(drm_path):
                    if device.startswith('card') and not device.endswith('-'):
                        device_path = os.path.join(drm_path, device, "device")
                        
                        # Check if it's an AMD GPU
                        vendor_path = os.path.join(device_path, "vendor")
                        if os.path.exists(vendor_path):
                            with open(vendor_path, 'r') as f:
                                vendor_id = f.read().strip()
                            
                            # AMD vendor ID is 0x1002
                            if vendor_id == "0x1002":
                                # Try to read memory usage
                                mem_info_path = os.path.join(device_path, "mem_info_vram_used")
                                if os.path.exists(mem_info_path):
                                    with open(mem_info_path, 'r') as f:
                                        memory_bytes = int(f.read().strip())
                                        total_memory_gb += memory_bytes / (1024 ** 3)
            
            return total_memory_gb
            
        except Exception as e:
            raise Exception(f"sysfs memory reading failed: {e}")
    
    def get_average_memory_usage(self) -> float:
        """
        Get average memory usage during monitoring
        
        Returns:
            float: Average memory usage in GB
        """
        if not self.memory_samples:
            return 0.0
        return statistics.mean(sample['memory_gb'] for sample in self.memory_samples)


async def process_streaming_response(stream, request_metrics: RequestMetrics):
    """Process streaming response and collect metrics"""
    collected_content = ""
    reasoning_content = ""
    
    try:
        async for chunk in stream:
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                
                # Normal content
                if hasattr(choice, 'delta') and choice.delta.content:
                    if request_metrics.first_token_time is None:
                        request_metrics.first_token_time = time.time()
                    collected_content += choice.delta.content
                    request_metrics.output_tokens += 1
                
                # Reasoning content (if available)
                elif hasattr(choice, 'delta') and hasattr(choice.delta, 'reasoning_content') and choice.delta.reasoning_content:
                    if request_metrics.first_token_time is None:
                        request_metrics.first_token_time = time.time()
                    reasoning_content += choice.delta.reasoning_content
                    request_metrics.output_tokens += 1
                
                if hasattr(choice, 'finish_reason') and choice.finish_reason is not None:
                    break
        
        # Content check
        total_content = collected_content + reasoning_content
        if not total_content.strip():
            logging.warning(f"Request {request_metrics.request_id}: No content generated!")
        
        request_metrics.end_time = time.time()
        request_metrics.success = bool(total_content.strip())  # Success if content exists
        
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
    """
    Make a single request and collect detailed metrics
    
    Args:
        client: AsyncOpenAI client
        request_id: Unique request identifier
        prompt: Input prompt
        output_tokens: Expected output tokens
        request_timeout: Request timeout in seconds
        model: Model name to use for the request
        
    Returns:
        RequestMetrics: Detailed metrics for the request
    """
    metrics = RequestMetrics(
        request_id=request_id,
        start_time=time.time(),
        first_token_time=None,
        end_time=None,
        output_tokens=0,
        success=False
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
        logging.warning(f"Request {request_id} timed out")
        
    except Exception as e:
        metrics.end_time = time.time()
        metrics.success = False
        metrics.error_message = str(e)
        logging.error(f"Request {request_id} failed: {e}")
    
    return metrics


async def request_worker(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    queue: asyncio.Queue,
    results: List[RequestMetrics],
    dataset_generator: SyntheticDatasetGenerator,
    output_tokens: int,
    request_timeout: int,
    model: str
):
    """
    Worker coroutine that processes requests from the queue
    
    Args:
        client: AsyncOpenAI client
        semaphore: Concurrency control semaphore
        queue: Request queue
        results: Shared results list
        dataset_generator: Synthetic dataset generator
        output_tokens: Target output tokens
        request_timeout: Request timeout in seconds
        model: Model name to use for requests
    """
    while True:
        try:
            request_id = await queue.get()
            if request_id is None:  # Sentinel value to stop worker
                queue.task_done()
                break
            
            async with semaphore:
                # Generate synthetic prompt
                prompt = dataset_generator.generate_dataset_entry()
                
                # Make request
                metrics = await make_single_request(
                    client, request_id, prompt, output_tokens, request_timeout, model
                )
                
                results.append(metrics)
                
            queue.task_done()
            
        except Exception as e:
            logging.error(f"Worker error: {e}")
            queue.task_done()


def calculate_percentile(values: List[float], percentile: float) -> Optional[float]:
    """
    Calculate percentile from list of values
    
    Args:
        values: List of numeric values
        percentile: Desired percentile (0-100)
        
    Returns:
        Optional[float]: Calculated percentile or None if values is empty
    """
    if not values:
        return None
    return float(np.percentile(values, percentile))


async def run_benchmark(
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
    output_file: str = "benchmark_results.json"
) -> Dict:
    """
    Run a comprehensive benchmark phase and save results to a JSON file
    
    Args:
        concurrency: Number of concurrent requests
        phase_duration: Main test phase duration in seconds
        ramp_up_duration: Ramp-up period in seconds (NOT included in measurements)
        cool_down_duration: Cool-down period in seconds (NOT included in measurements)
        input_tokens: Target input tokens per request
        output_tokens: Target output tokens per request
        request_timeout: Request timeout in seconds
        vllm_url: vLLM server URL
        api_key: API key for authentication
        gpu_info: GPU information dict (e.g., {'model': 'AMD MI250X', 'memory': '128GB'})
        model: Model name to use for requests
        output_file: Path to save JSON results
        
    Returns:
        Dict: Comprehensive benchmark results
    """
    # Validate input parameters
    for param, value in [
        ('concurrency', concurrency),
        ('phase_duration', phase_duration),
        ('ramp_up_duration', ramp_up_duration),
        ('cool_down_duration', cool_down_duration),
        ('input_tokens', input_tokens),
        ('output_tokens', output_tokens),
        ('request_timeout', request_timeout)
    ]:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{param} must be a positive integer")
    
    if not vllm_url or not api_key:
        raise ValueError("vllm_url and api_key must be provided")
    
    logging.info(f"Starting benchmark phase: concurrency={concurrency}, duration={phase_duration}s")
    
    async with AsyncOpenAI(base_url=vllm_url, api_key=api_key) as client:
        # Initialize components
        dataset_generator = SyntheticDatasetGenerator(target_tokens=input_tokens)
        memory_monitor = GPUMemoryMonitor()
        
        # Control structures
        semaphore = asyncio.Semaphore(concurrency)
        queue = asyncio.Queue()
        results = []
        
        # Ramp-up phase - system initialization (NOT measured)
        logging.info(f"Starting ramp-up phase ({ramp_up_duration}s) - system initialization...")
        
        # Start memory monitoring during ramp-up but don't count this time
        memory_task = asyncio.create_task(memory_monitor.start_monitoring())
        
        # Create workers during ramp-up
        workers = [
            asyncio.create_task(
                request_worker(client, semaphore, queue, results, dataset_generator, 
                             output_tokens, request_timeout, model)
            )
            for _ in range(concurrency)
        ]
        
        # Send some warm-up requests during ramp-up (these won't be counted)
        warmup_requests = 0
        ramp_start = time.time()
        while time.time() - ramp_start < ramp_up_duration:
            await queue.put(f"warmup_{warmup_requests}")
            warmup_requests += 1
            await asyncio.sleep(0.1)  # Slower rate during warm-up
        
        # Clear results from warm-up period
        results.clear()
        
        # CRITICAL FIX: Clear the queue from pending warm-up requests
        queue_size_before = queue.qsize()
        cleared_count = 0
        while not queue.empty():
            try:
                queue.get_nowait()
                queue.task_done()
                cleared_count += 1
            except asyncio.QueueEmpty:
                break
        
        logging.info(f"Ramp-up completed. {warmup_requests} warm-up requests sent.")
        logging.info(f"Cleared {cleared_count} pending requests from queue.")
        logging.info("Starting measured test phase with clean queue...")
        
        # Main test phase - THIS IS WHAT GETS MEASURED
        test_start_time = time.time()
        logging.info("MEASUREMENT STARTED - Main test phase beginning...")
        
        # Generate requests for the measured test duration
        request_id = 0
        while time.time() - test_start_time < phase_duration:
            # Only add requests if queue isn't overwhelmed
            if queue.qsize() < concurrency * 3:  # Reasonable limit
                await queue.put(request_id)
                request_id += 1
            
            await asyncio.sleep(0.5)  # 500ms - much slower request rate
        
        # Wait for all queued requests to complete WITH TIMEOUT
        try:
            await asyncio.wait_for(queue.join(), timeout=120.0)  # Max 2 minutes wait
            logging.info("All queued requests completed successfully")
        except asyncio.TimeoutError:
            logging.warning(f"Queue join timeout after 120 seconds. Remaining queue size: {queue.qsize()}")
            # Force clear remaining queue
            while not queue.empty():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except:
                    break
        
        test_end_time = time.time()
        total_test_time = test_end_time - test_start_time
        logging.info(f"MEASUREMENT ENDED - Test phase completed in {total_test_time:.2f}s")
        
        # Stop workers
        for _ in range(concurrency):
            await queue.put(None)
        await asyncio.gather(*workers, return_exceptions=True)
        
        # Cool-down phase (NOT measured)
        logging.info(f"Starting cool-down phase ({cool_down_duration}s)...")
        memory_monitor.stop_monitoring()
        await memory_task
        await asyncio.sleep(cool_down_duration)
        
        # Calculate metrics - only from measured test phase
        successful_requests = [r for r in results if r.success]
        failed_requests = [r for r in results if not r.success]
        
        if not successful_requests:
            logging.warning("No successful requests in this phase")
            results_dict = {
                'concurrency': concurrency,
                'total_requests_sent': len(results),
                'successful_requests': 0,
                'failed_requests': len(failed_requests),
                'response_rate': 0.0,
                'system_output_throughput': 0.0,
                'median_end_to_end_latency': 0.0,
                'average_latency_per_request': 0.0,
                'median_output_speed_per_query': 0.0,
                'gpu_info': gpu_info,
                'test_duration': total_test_time,
                'peak_memory_gb': memory_monitor.peak_memory,
                'average_memory_gb': memory_monitor.get_average_memory_usage(),
                'model': model
            }
        else:
            # Extract metrics
            total_output_tokens = sum(r.output_tokens for r in successful_requests)
            end_to_end_latencies = [r.end_to_end_latency_ms for r in successful_requests if r.end_to_end_latency_ms]
            ttft_values = [r.ttft_ms for r in successful_requests if r.ttft_ms]
            output_speeds = [r.output_speed_tps for r in successful_requests if r.output_speed_tps]
            
            # Calculate final results
            results_dict = {
                'concurrency': concurrency,
                'total_requests_sent': len(results),
                'successful_requests': len(successful_requests),
                'failed_requests': len(failed_requests),
                'response_rate': len(successful_requests) / len(results) if results else 0.0,
                'test_duration_seconds': total_test_time,
                
                # Throughput metrics
                'system_output_throughput': total_output_tokens / total_test_time if total_test_time > 0 else 0,
                'requests_per_second': len(successful_requests) / total_test_time if total_test_time > 0 else 0,
                
                # Latency metrics
                'median_end_to_end_latency': statistics.median(end_to_end_latencies) if end_to_end_latencies else 0,
                'average_latency_per_request': statistics.mean(end_to_end_latencies) if end_to_end_latencies else 0,
                'p95_end_to_end_latency': calculate_percentile(end_to_end_latencies, 95) or 0,
                'p99_end_to_end_latency': calculate_percentile(end_to_end_latencies, 99) or 0,
                
                # TTFT metrics
                'median_ttft': statistics.median(ttft_values) if ttft_values else 0,
                'average_ttft': statistics.mean(ttft_values) if ttft_values else 0,
                'p95_ttft': calculate_percentile(ttft_values, 95) or 0,
                'p99_ttft': calculate_percentile(ttft_values, 99) or 0,
                
                # Output speed metrics
                'median_output_speed_per_query': statistics.median(output_speeds) if output_speeds else 0,
                'average_output_speed_per_query': statistics.mean(output_speeds) if output_speeds else 0,
                
                # Memory metrics
                'peak_memory_gb': memory_monitor.peak_memory,
                'average_memory_gb': memory_monitor.get_average_memory_usage(),
                
                # System info
                'gpu_info': gpu_info,
                'model': model,
                'test_config': {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'phase_duration': phase_duration,
                    'request_timeout': request_timeout
                },
                
                # Error summary
                'error_summary': {
                    'timeout_errors': len([r for r in failed_requests if r.error_message and 'timeout' in r.error_message.lower()]) if failed_requests else 0,
                    'other_errors': len([r for r in failed_requests if r.error_message and 'timeout' not in r.error_message.lower()]) if failed_requests else 0
                }
            }
        
        # Save results to JSON file
        try:
            with open(output_file, "w") as f:
                json.dump(results_dict, f, indent=4)
            logging.info(f"Benchmark results saved to {output_file}")
        except Exception as e:
            logging.error(f"Failed to save results to {output_file}: {e}")
        
        logging.info(f"Phase completed: {len(successful_requests)}/{len(results)} successful requests")
        
        return results_dict


# Export main function
__all__ = ['run_benchmark']


if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        
        try:
            results = await run_benchmark(
                concurrency=1024,
                phase_duration=180,
                ramp_up_duration=30,
                cool_down_duration=30,
                input_tokens=1000,
                output_tokens=1000,
                request_timeout=30,
                vllm_url="http://localhost:8000/v1",
                api_key="your-api-key",
                gpu_info={"model": "AMD MI250X", "memory": "128GB"},
                model="openai/gpt-oss-20b",
                output_file="benchmark_results.json"
            )
            logging.info("Benchmark completed successfully")
        except Exception as e:
            logging.error(f"Benchmark failed: {e}")
    
    asyncio.run(main())