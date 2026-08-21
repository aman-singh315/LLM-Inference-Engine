# LLM Inference Engine

A from-scratch, continuous-batching LLM inference engine built to understand — and rebuild by hand — the mechanics behind modern serving systems like vLLM and TensorRT-LLM. This isn't a wrapper around `model.generate()`. The decode loop, KV cache memory management, and paged attention integration are all hand-implemented and hand-optimized.

Final measured throughput: **~550 tok/s** decode (batch ≈ 25, NVIDIA T4), up from a **~30 tok/s** naive baseline — roughly an **18x improvement**, achieved through a paged block-based KV cache, continuous batching, FlashInfer integration, and low-level kernel-launch optimization.

---

## What this engine does

- **Continuous batching** — requests join and leave the active batch every decode step, instead of waiting for a fixed batch to fully finish before the next one starts.
- **Block-based (paged) KV cache** — memory is allocated in fixed-size blocks (16 tokens each) from a shared pool, not as one contiguous per-request tensor. This is the same idea behind vLLM's PagedAttention: no more reserving worst-case sequence length per request up front.
- **Hand-written decode step** — the decode path does not call `model.forward()`. Every layer is walked manually: RMSNorm → QKV projection → RoPE → paged attention → output projection → MLP → residual. This was necessary to control exactly where and how KV cache reads/writes happen.
- **FlashInfer paged attention** — `BatchDecodeWithPagedKVCacheWrapper` replaces a naive gather-pad-attend path, reading directly from the block pool instead of reconstructing a padded KV tensor every step.
- **Custom scheduler** — tracks waiting/active/completed requests, injects new requests into free batch slots mid-run, and frees KV blocks back to the pool the moment a request finishes.

---

## The story: how this throughput number was actually earned

### 1. A correctness fix that tanked performance

Early on, decode was silently wrong: the prompt's KV cache blocks weren't being passed into the decode step, so every decode step was only attending over the single most recent token — not the full sequence. Output was still fluent-looking (that's the dangerous part), but it wasn't actually correct autoregressive generation.

Fixing it was simple. The consequence wasn't: throughput dropped from ~450–470 tok/s to **~80 tok/s**. Correct behavior had been hiding behind a bug that was, incidentally, fast. This became the starting point for a real optimization pass, backed by profiling instead of guesswork.

### 2. The profiler lied first — so that got fixed before anything else

Before trusting any optimization, the decode step was instrumented phase-by-phase: projections, KV writes, attention, MLP sub-ops, final head. The first profiling run reported:

```
attention: 0.000 ms/step (0.0%)
```

That's not real. FlashInfer's `wrapper.run()` was invoked with no timer bracketing it — the next `torch.cuda.synchronize()` silently absorbed the entire attention kernel's execution time into a dead gap between two unrelated timers. The profiled total came out to 61.1ms/step against a real measured latency of ~52ms — the numbers didn't reconcile, which was the tell.

A second bug compounded it: the MLP block was timed twice — once as a single wrapping "mlp_core" span, and again as five separate sub-op timers (gate/up/act/mul/down proj) inside it. The two errors happened to roughly cancel out, which is exactly what made the broken profiler look plausible at a glance.

Both were fixed — attention properly bracketed with sync-timestamps, `mlp_core` retired in favor of the five sub-op timers alone — before any real optimization work started. Profiling data is only as good as the profiler; this was the first thing worth being skeptical of, not the last.

### 3. Real bottlenecks, once visible

With a trustworthy profiler, two clear inefficiencies stood out:

**KV cache writes were a Python loop of tiny scalar scatter-writes.** Every decode step, for every layer, the new token's key/value vectors were written into the block pool one request at a time in a `for i in range(batch_size)` loop — at batch≈7 and 22 layers, that's on the order of 300 individual tiny GPU write kernels per step, each paying full launch overhead for writing a handful of floats. Replacing this with a single vectorized index-tensor write per layer (`keys[layer_idx, block_ids_tensor, :, offsets_tensor, :] = k`) cut this cost from **~8.2ms → ~1.8ms per step**.

**Q/K/V were three separate matmuls where they only needed to be one.** `q_proj`, `k_proj`, and `v_proj` were called as three independent linear layers per decoder layer. At small batch and seq_len=1, these matmuls are tiny — GPU time is dominated by per-kernel launch overhead, not FLOPs. Concatenating the three weight matrices once at init time and doing a single fused matmul (then splitting the output) cut three kernel launches to one, per layer.

### 4. The real lever wasn't the kernels — it was the batch size

Even after both fixes, overall step latency barely moved (~34-37ms/step, largely flat across several optimization rounds). That flatness was itself the signal: this workload was **launch-overhead-bound, not compute-bound** — the GPU was mostly idle between tiny kernel launches, not actually crunching numbers. Shrinking individual ops further wasn't going to change that.

The benchmark script was only feeding ~8-10 concurrent requests into a scheduler configured for far more capacity, so the batch was draining down to an average of **6.91 requests/step** instead of anywhere near saturation. Submitting a much larger, continuous pool of requests (and raising the KV block pool capacity to match) pushed the average batch to **~25-28 requests/step** — and because per-step cost was overhead-bound rather than work-bound, this came at almost no extra latency cost per step:

| | Avg batch/step | Step latency (profiled) | Throughput |
|---|---|---|---|
| Before | 6.91 | ~34.7 ms | ~169 tok/s |
| After | 28.01 | ~36.4 ms | ~600 tok/s (instrumented) |

Step cost barely changed. Throughput scaled almost linearly with batch. This is the core lesson this project produced: **at small batch, optimizing kernels gives diminishing returns; the batch size itself is often the dominant lever, because it's what determines whether launch overhead gets hidden or not.**

### 5. Getting a number that could actually be trusted

Two more things stood between "one good run" and a defensible benchmark:

- **The instrumented profiler path was still measuring the profiled — not the real — `decode_step`.** Re-running against the actual, uninstrumented method (no per-layer synchronize calls) gave the true engine throughput, separate from any profiling overhead.
- **Run-to-run throughput fluctuated by 10-15%** on a shared, free-tier Colab T4 — including one run that dropped to ~244 tok/s. `nvidia-smi` showed the GPU sitting at 76°C with near-zero power draw and 0% utilization *between* runs — a strong signal of multi-tenant GPU contention outside this project's control, compounded by cold-start costs (CUDA kernel JIT compilation, cuBLAS/cuDNN autotuning, allocator ramp-up) that hit hardest on short benchmark runs. Adding an explicit warmup phase (a handful of throwaway decode steps run and discarded before the timed benchmark starts) removed the cold-start variance; increasing `MAX_NEW_TOKENS` from 50 → 100 (more decode steps per run, better statistical averaging) tightened the rest.

Final result, repeated across multiple back-to-back runs with less than 1% variance between them:

```
Decode steps:       297
Average latency:    44.5 ms
P50 latency:        45.2 ms
P95 latency:        55–67 ms
Average requests/step: 25.27
Decode throughput:  ~550 tok/s
```

---

## Architecture notes

- **Block size:** 16 tokens/block, paged KV cache pool (configurable `total_blocks`, sized to the largest batch × sequence length you expect to run concurrently — undersizing this creates a silent ceiling where requests stall waiting for free blocks).
- **Attention backend:** FlashInfer `BatchDecodeWithPagedKVCacheWrapper`, `kv_layout="HND"`, RoPE applied manually before the call (`pos_encoding_mode="NONE"` to avoid FlashInfer double-applying it).
- **Scheduler:** tracks `waiting` / `active` / `completed` request queues; supports dynamic mid-run injection of new requests into free slots.
- **Decode step, per layer:** input RMSNorm → fused QKV projection → RoPE → vectorized KV cache write → FlashInfer paged attention → output projection → residual → post-attention RMSNorm → MLP (gate/up/SiLU/mul/down) → residual.

---

## What's next

- Fuse `mlp_gate_proj` / `mlp_up_proj` the same way QKV was fused — profiling at the higher batch size showed these growing into real compute cost rather than pure launch overhead, unlike at low batch where the same fusion barely mattered.
- Investigate `mlp_layernorm` cost, which stayed flat and unexplained through every other optimization round.
- Benchmark on a dedicated (non-shared) GPU instance to get a contention-free baseline number.
- Speculative decoding, quantization (INT8/FP8), and prefix caching are natural next steps toward a more complete serving stack.

---

## Running it

The engine, profiler, and benchmark script are structured as three separate notebook cells (`ContinuousEngine`, profiling instrumentation, benchmark driver). To reproduce the benchmark:

1. Run the model/tokenizer/engine setup cell.
2. Run the benchmark cell — it submits a batch of prompts, runs a short warmup phase, then times the serving loop over `MAX_STEPS` iterations.
3. Adjust `MAX_BATCH_SIZE`, `total_blocks` (KV pool capacity), and the prompt pool size together — they're coupled: a larger batch needs proportionally more KV blocks, or requests will stall on allocation instead of running concurrently.





## Related Projects

- [Flash Attention CUDA Kernel](../flash-attention) — custom CUDA kernel implementing Flash Attention v2 with warp-level shuffle reductions. Benchmarked against cuBLAS baseline.
- [GEMM CUDA Kernel](https://github.com/aman-singh315/GeMM-CUDA-KERNEL) — full tiling hierarchy from block → warp → register level.
