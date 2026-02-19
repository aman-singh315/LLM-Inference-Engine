# LLM Inference Engine with Continuous Batching & Paged KV Cache

## **A minimal yet production-inspired from vLLM & implementation of a high-throughput LLM inference engine featuring:**
 - Continuous batching
 - Dynamic request scheduling
 - Paged KV-cache memory management
 - Block allocation & recycling
 - Throughput benchmarking

**This project demonstrates how modern inference systems optimize GPU utilization during autoregressive generation.**


## Motivation

**Naive LLM inference processes one request at a time.
Modern inference engines maximize hardware efficiency by:**

 - Dynamically batching requests
 - Injecting new requests during decode
 - Reusing KV-cache memory blocks
 - Avoiding full-sequence recomputation
 - This project implements those ideas from scratch in PyTorch.

## **Architechture**
Scheduler  →  BlockPool → ContinuousEngine 
  **1. Scheduler**
  - Maintains waiting queue
  - Controls active batch size
  - Supports dynamic injection during decoding
  - Tracks completed requests

  **2. BlockPool (Paged KV Cache)**
  - Preallocates KV memory blocks
  - Dynamically allocates per request
  - Frees blocks after completion
  - Enables memory reuse across generations

    **3. Continuous Engine**
  - Handles prefill stage
  - Performs step-by-step decoding
  - Manages KV-cache updates
  - Cleans up finished requests

**This simulates the design used in modern inference systems such as vLLM-style paged attention.**


## Features Implemented
  - Continuous batching
  - Dynamic request injection during decoding
  - Paged KV-cache memory management
  - Block allocation & recycling
  - Throughput measurement (tokens/sec)
  - Device-agnostic execution (CPU / CUDA)


## Project-Structure
llm-inference-engine/
│
├── engine/
│   ├── request.py
│   ├── scheduler.py
│   ├── memory.py
│   └── continuous_engine.py
│
├── benchmark.py
├── requirements.txt
└── README.md


## --- Submitted initial requests ---
Prefill batch size 8 | 0.0325s
[Alloc] Block 127
...
===== CONTINUOUS BATCH TEST COMPLETE =====
Total time: 9.8920s
Total tokens generated: 1700
Throughput: 171.86 tokens/sec
Total decode steps: 245


## Performance

 ```The benchmark measures:```
  - Total decode steps
  - Total tokens generated
  - End-to-end runtime
  - Tokens per second (throughput)
  - Throughput depends on:
  - GPU hardware
  - Batch size
  - Model size
  - Block configuration

## Technical Highlights
 **KV-cache stored as:**
   [num_layers, total_blocks, num_heads, block_size, head_dim]

## Future Improvements
  - Custom CUDA kernels
  - FlashAttention implementation
  - Speculative decoding
  - Quantization support
  - Streaming output API

## License
  MIT License
