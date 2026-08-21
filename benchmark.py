import time
import statistics

# Configuration
MAX_BATCH_SIZE = 32
MAX_NEW_TOKENS = 100
MAX_STEPS = 500

# Create Scheduler + Engine
scheduler = Scheduler(max_active=MAX_BATCH_SIZE)

engine = ContinuousEngine(
    model=model,
    tokenizer=tokenizer,
    scheduler=scheduler,
    device=model.device,
    max_batch_size=MAX_BATCH_SIZE
)

# engine.decode_step = types.MethodType(instrumented_decode_step_v2, engine)
print_profile_summary_v2()


# WARMUP not counted in benchmark stats

print("\n--- Running warmup ---")

warmup_prompts = [
    "What is the capital of france?.",
    "Why we breath oxygen only?",
    "what is the work of ratina in eyes?",
]

for i, prompt in enumerate(warmup_prompts):
    scheduler.submit(Request(9000 + i, prompt, 20))

warmup_steps = 0
while (scheduler.waiting or scheduler.active) and warmup_steps < 30:
    scheduler.inject_if_possible()
    engine.run_prefill()
    if engine.scheduler.active:
        engine.decode_step()
    engine.cleanup()
    warmup_steps += 1

if torch.cuda.is_available():
    torch.cuda.synchronize()

print(f"--- Warmup complete ({warmup_steps} steps) ---\n")



# Submit Initial Requests

initial_prompts = [
    "Explain quantum computing.",
    "What is artificial intelligence?",
    "Translate this to French: Hello world.",
    "Summarize World War II.",
    "Explain black holes.",
    "What is reinforcement learning?",
    "Describe climate change.",
    "What is blockchain?",
    "Explain quantum computing in simple terms.",
    "What is artificial intelligence and how does it work?",
    "Explain the difference between machine learning and deep learning.",
    "Translate this sentence to French: Hello, how are you today?",
    "Summarize the major causes of World War II.",
    "Explain how black holes are formed.",
    "What is reinforcement learning? Give a simple example.",
    "Describe the main causes and effects of climate change.",
    "Explain blockchain technology to someone who has never heard of it.",
    "What is the difference between RAM and storage?",
    "Explain how a neural network learns from data.",
    "What are the advantages and disadvantages of electric vehicles?",
    "Explain the concept of recursion using a simple programming example.",
    "What is the purpose of an operating system?",
    "Describe how the internet works from a high level.",
    "Explain the difference between HTTP and HTTPS.",
    "What is a database and why do applications need one?",
    "Explain what an API is and provide a practical example.",
    "What is cloud computing and why is it useful?",
    "Explain the difference between supervised and unsupervised learning.",
    "What are transformers in natural language processing?",
    "Explain how large language models generate text.",
    "What is tokenization in natural language processing?",
    "Why are GPUs useful for training neural networks?",
    "Explain the difference between CPU and GPU architecture.",
    "What is parallel computing and why does it improve performance?",
    "Explain what a cache is in computer architecture.",
     "What is memory fragmentation and why can it be a problem?",
    "Explain virtual memory and how operating systems use it.",
    "What is the difference between a process and a thread?",
    "Explain what a scheduler does in an operating system.",
    "What is dynamic batching in LLM inference?",
    "Explain the difference between prefill and decode in LLM inference.",
    "What is a KV cache and why is it important for LLM inference?",
    "Explain paged attention and why it can reduce KV-cache memory fragmentation.",
    "What is continuous batching and how does it improve inference throughput?",
    "Explain why generating one token at a time can become a bottleneck.",
    "What is quantization and how can it reduce model memory usage?",
    "Explain the difference between FP32, FP16, BF16, and INT8.",
    "Why does matrix multiplication dominate many neural network workloads?",
    "Explain how attention works inside a transformer.",
    "What is multi-head attention and why do transformers use it?",
    "Explain grouped-query attention and how it differs from standard multi-head attention.",
    "What is FlashAttention and what problem does it solve?",
    "Explain why memory bandwidth can become a bottleneck during inference.",
    "What is CUDA and why is it important for GPU computing?",
    "Explain how a CUDA kernel executes work on a GPU.",
    "What is kernel fusion and why can it improve neural network performance?",
    "Explain how inference throughput and latency are different metrics.",
    "What factors should be considered when optimizing an LLM inference engine?",
]

for i, prompt in enumerate(initial_prompts):
    scheduler.submit(Request(i, prompt, MAX_NEW_TOKENS))

print("\n Submitted initial requests")

# BENCHMARK METRICS

start_time = time.perf_counter()

step_latencies = []
step_token_counts = []

step_counter = 0
new_requests_added = False

# Serving Loop

while True:

    if step_counter >= MAX_STEPS:
        print("WARNING: Hit MAX_STEPS safety limit.")
        break

    scheduler.inject_if_possible()

    # Prefill
    engine.run_prefill()

    # Measure ONE complete decode step

    if engine.scheduler.active:

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        step_start = time.perf_counter()

        engine.decode_step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        step_end = time.perf_counter()

        step_latency_ms = (step_end - step_start) * 1000

        # Number of requests that participated in this step
        active_count = sum(
            1 for r in engine.scheduler.active
            if not r.finished and r.state is not None
        )

        step_latencies.append(step_latency_ms)
        step_token_counts.append(active_count)

    # Cleanup
    engine.cleanup()

    step_counter += 1

    # Dynamic Injection

    if step_counter == 5 and not new_requests_added:

        print("\n--- Injecting MORE requests during decode ---")

        more_prompts = [
            "What is quantum computing?",
            "Summarize the theory of relativity.",
            "Why do leaves change color during autumn?",
            "Explain how photosynthesis converts sunlight into energy.",
            "How does the human immune system protect the body?",
            "Explain the difference between weather and climate.",
            "How do earthquakes happen beneath the Earth's surface?",
            "Explain how airplanes generate lift and stay in the air.",
            "Why does the ocean have tides?",
            "Explain how GPS determines the location of a device.",
            "How do solar panels convert sunlight into electricity?",
            "Explain the basic principles behind nuclear energy.",
            "What causes a rainbow to appear after rainfall?",
            "Explain how antibiotics work against bacterial infections.",
            "How does a refrigerator keep food cold?",
            "Explain how satellites communicate with ground stations.",
            "Why do metals conduct electricity better than most materials?",
            "Explain how a compiler converts source code into machine code.",
            "What is garbage collection in programming languages?",
            "Explain the difference between encryption and hashing.",
            "How does a web browser load and display a webpage?",
            "Explain how distributed systems handle failures between servers.",
        ]

        for j, prompt in enumerate(more_prompts):
            scheduler.submit(
                Request(100 + j, prompt, MAX_NEW_TOKENS)
            )

        new_requests_added = True

    # Exit
    if not scheduler.waiting and not scheduler.active:
        break


# FINAL METRICS

total_time = time.perf_counter() - start_time

all_requests = (
    list(scheduler.waiting)
    + list(scheduler.active)
    + list(scheduler.completed)
)

# PRINT GENERATED OUTPUTS

print("\n===== GENERATED OUTPUTS")

for req in scheduler.completed:
    generated_text = tokenizer.decode(
        req.output_tokens,
        skip_special_tokens=True
    )
    print(f"\n--- Request {req.req_id} ---")
    print(f"Prompt:    {req.prompt}")
    print(f"Generated: {generated_text}")


total_tokens_generated = sum(
    len(req.output_tokens)
    for req in all_requests
)

prefill_tokens = sum(
    1 for req in all_requests
    if req.state is not None
)

decode_tokens = total_tokens_generated - prefill_tokens

overall_tok_per_sec = (
    total_tokens_generated / total_time
    if total_time > 0 else 0
)

decode_tok_per_sec = (
    decode_tokens / total_time
    if total_time > 0 else 0
)


# LATENCY STATISTICS

if step_latencies:

    avg_latency = statistics.mean(step_latencies)
    p50_latency = statistics.median(step_latencies)

    p90_latency = statistics.quantiles(
        step_latencies,
        n=10
    )[8]

    p95_latency = statistics.quantiles(
        step_latencies,
        n=20
    )[18]

    min_latency = min(step_latencies)
    max_latency = max(step_latencies)

else:
    avg_latency = p50_latency = p90_latency = p95_latency = 0
    min_latency = max_latency = 0


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n===== DECODE LATENCY BENCHMARK =====")

print(f"Decode steps:       {len(step_latencies)}")

print(f"Average latency:    {avg_latency:.3f} ms")
print(f"P50 latency:        {p50_latency:.3f} ms")
print(f"P90 latency:        {p90_latency:.3f} ms")
print(f"P95 latency:        {p95_latency:.3f} ms")
print(f"Min latency:        {min_latency:.3f} ms")
print(f"Max latency:        {max_latency:.3f} ms")

print("\n===== THROUGHPUT =====")

print(f"Total time:         {total_time:.4f} s")
print(f"Total tokens:       {total_tokens_generated}")
print(f"Decode tokens:      {decode_tokens}")
print(f"Overall throughput: {overall_tok_per_sec:.2f} tok/s")
print(f"Decode throughput:  {decode_tok_per_sec:.2f} tok/s")

print("\n===== BATCHING =====")

if step_token_counts:
    print(
        f"Average requests/step: "
        f"{statistics.mean(step_token_counts):.2f}"
    )

print(f"Total loop iterations: {step_counter}")
print_profile_summary_v2()
