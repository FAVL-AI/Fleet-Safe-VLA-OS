import torch
import time

def benchmark_inference(model, inputs, runs=20):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    latencies = []

    # warmup
    for _ in range(5):
        _ = model(**inputs)

    torch.cuda.synchronize()

    for _ in range(runs):
        start = time.time()

        _ = model(**inputs)

        torch.cuda.synchronize()
        latencies.append(time.time() - start)

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    avg_latency = sum(latencies) / len(latencies)

    return {
        "peak_vram_mb": peak_mem,
        "latency_s": avg_latency,
        "throughput": 1.0 / avg_latency
    }

if __name__ == "__main__":
    # Example usage hook
    print("VRAM Benchmarking utility ready.")
