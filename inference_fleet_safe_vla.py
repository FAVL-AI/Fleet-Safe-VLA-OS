import torch
import time
import psutil
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq
import gc

def print_vram_usage(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[{prefix}] VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
    else:
        # Fallback for Mac/MPS or CPU
        process = psutil.Process()
        mem_info = process.memory_info()
        print(f"[{prefix}] RAM Usage: {mem_info.rss / (1024 ** 3):.2f} GB")

def main():
    print("======================================================")
    print(" BASELINE INFERENCE - FLEET-SAFE VLA (NO TURBOQUANT) ")
    print("======================================================")
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print_vram_usage("Initial")

    model_id = "openvla/openvla-7b"
    print(f"Loading processor and model ({model_id}) on {device}...")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # Load model in bfloat16 or float16 for standard baseline inference
    dtype = torch.bfloat16 if device == "cuda" else torch.float16
    
    # We load standard model
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, 
        torch_dtype=dtype, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to(device)

    print_vram_usage("Model Loaded")

    # Create dummy image and text for testing rollout
    image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    prompt = "In: What action should the robot take to safely avoid the dynamic crowd? Out:"

    inputs = processor(prompt, image).to(device, dtype=dtype if device != "mps" else torch.float32)

    if device == "mps":
        # MPS requires special typing map sometimes or we can just fall back to standard
        pass

    print("\nStarting inference benchmarking...")
    
    # Warmup
    print("Warming up model...")
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=5, use_cache=True)
    
    # Benchmark
    target_tokens = 50
    start_time = time.time()
    
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=target_tokens, use_cache=True)
    
    end_time = time.time()
    latency = end_time - start_time
    
    generated_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    throughput = generated_tokens / latency if latency > 0 else 0
    
    print("\n--- BENCHMARK RESULTS ---")
    print(f"Latency: {latency:.4f} seconds")
    print(f"Throughput: {throughput:.2f} tokens/second")
    print_vram_usage("Peak Inference")
    print("-------------------------")
    
if __name__ == "__main__":
    main()
