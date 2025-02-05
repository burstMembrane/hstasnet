import torch
import time
import logging
from tqdm import tqdm
from hstasnet import HSTasNet  
from config.parse import parse_config
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Measure HSTasNet inference latency.")
parser.add_argument(
    "--device", type=str, default="cpu", help="Device to run the model on."
)
args = parser.parse_args()

config = parse_config("./config/train.yaml")
model_args = config.get("model_args", {})

if not model_args:
    raise ValueError("Model parameters not found in the config file.")

model = HSTasNet(**model_args).to(args.device)

DURATIONS = [1, 10, 30]  
SAMPLE_RATE = config.get("sample_rate", 44100) 
NUM_RUNS = 20  
input = torch.randn(1, 2, SAMPLE_RATE, device=args.device)
logger.info("Running warm-up passes...")
for _ in range(5):
    _ = model(input)

results = {}

for duration in DURATIONS:
    L = duration * SAMPLE_RATE 
    x = torch.randn(1, 2, L, device=args.device)  

    latencies = []
    for _ in tqdm(range(NUM_RUNS), desc=f"Testing {duration}s Audio"):
        start_time = time.perf_counter()
        _ = model(x, length=L)  # Forward pass
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms

    avg_latency = sum(latencies) / NUM_RUNS
    min_latency = min(latencies)
    max_latency = max(latencies)

    results[duration] = {
        "avg_latency": avg_latency,
        "min_latency": min_latency,
        "max_latency": max_latency,
    }

logger.info("\nFinal Latency Results:")
logger.info(f"Device: {args.device}")
for duration, stats in results.items():
    logger.info(f"\nLatency for {duration} sec of audio:")
    logger.info(f"  Avg Latency: {stats['avg_latency']:.2f} ms")
    logger.info(f"  Min Latency: {stats['min_latency']:.2f} ms")
    logger.info(f"  Max Latency: {stats['max_latency']:.2f} ms")
