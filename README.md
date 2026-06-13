# LLM Inference Engine — 15x Throughput on a T4

> Naive inference on a T4: **~30 tokens/sec**
> This engine: **458 tokens/sec**
> **15x faster** — built from scratch in PyTorch, inspired by vLLM.

No black boxes. No wrappers around existing engines.
Every component — scheduler, paged KV cache, continuous batcher — written and understood from first principles.

---

## Why this exists

Most tutorials show you how to *call* an LLM. Nobody shows you what happens inside the inference server.

This project answers: *how do production systems like vLLM serve hundreds of requests efficiently on a single GPU?*

The answer is three ideas working together:

| Problem | Solution | Impact |
|---|---|---|
| GPU sits idle between requests | Continuous batching | No wasted decode cycles |
| KV cache fragments memory | Paged KV cache | No memory waste, more concurrent requests |
| New requests wait for current batch to finish | Dynamic injection | Latency drops, throughput climbs |

---

## Benchmark

Tested on **Google Colab T4**, LLaMA-style small model config.

| Mode | Throughput |
|---|---|
| Naive (one request at a time) | ~30 tokens/sec |
| This engine (batch=8, continuous) | **458 tokens/sec** |
| Improvement | **~15x** |

```
===== BENCHMARK RESULTS =====
Total decode steps    : 132
Total tokens generated: 4352
Total time            : 9.49s
Throughput            : 458.26 tokens/sec
```

---

## Architecture

A request's journey through the engine:

```
Incoming Request
      │
      ▼
┌─────────────┐
│  Scheduler  │  ← maintains wait queue, controls batch size
└──────┬──────┘
       │ allocates memory
       ▼
┌─────────────┐
│  BlockPool  │  ← paged KV cache, preallocated blocks
│ (Paged KV)  │    dynamically assigned per request
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ ContinuousEngine │  ← prefill → decode loop
│                  │    injects new requests mid-decode
│                  │    frees blocks on completion
└──────────────────┘
       │
       ▼
  Generated Tokens
```

---

## Key Design Decisions

**Why paged KV cache?**
Naive inference pre-allocates KV memory for the full max sequence length — even if a request only generates 20 tokens. Paged KV cache allocates fixed-size blocks on demand, exactly like virtual memory in an OS. Result: less waste, more concurrent requests.

**Why continuous batching?**
Static batching waits for every request in a batch to finish before starting new ones. If one request generates 5 tokens and another generates 500, the GPU waits. Continuous batching injects new requests the moment a slot frees — GPU utilization stays high throughout.

**Why dynamic injection during decode?**
Prefill is compute-bound. Decode is memory-bound. Mixing them mid-flight keeps both compute and memory pipelines busy, which is exactly what a T4's architecture rewards.

---

## Project Structure

```
llm-inference-engine/
│
├── engine/
│   ├── request.py           # Request lifecycle management
│   ├── scheduler.py         # Batch scheduling & queue management
│   ├── memory.py            # BlockPool — paged KV cache
│   └── continuous_engine.py # Prefill + continuous decode loop
│
├── benchmark.py             # End-to-end throughput benchmark
├── requirements.txt
└── README.md
```

---

## Run it yourself

```bash
git clone https://github.com/aman-singh315/LLM-Inference-Engine
cd LLM-Inference-Engine
pip install -r requirements.txt
python benchmark.py
```

---

## What I learned building this

- **Occupancy ≠ performance.** Packing more blocks doesn't help if memory bandwidth is the bottleneck.
- **The scheduler is the engine.** Getting batching logic right matters more than micro-optimizations.
- **KV cache is the memory pressure point.** Paging it is not optional at scale — it's the difference between serving 8 requests and serving 80.

---

## Related Projects

- [Flash Attention CUDA Kernel](../flash-attention) — custom CUDA kernel implementing Flash Attention v2 with warp-level shuffle reductions. Benchmarked against cuBLAS baseline.
- [GEMM CUDA Kernel](../GeMM-CUDA-KERNEL) — full tiling hierarchy from block → warp → register level.

---

## Future Work

- Integrate custom Flash Attention kernel (built separately — see above)
- Speculative decoding
- INT8/FP16 quantization
- Streaming output API

---

*Built by a BCA student who wanted to understand how vLLM actually works.*
