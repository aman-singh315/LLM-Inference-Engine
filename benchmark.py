import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

from engine.request import Request
from engine.scheduler import Scheduler
from engine.continuous_engine import ContinuousEngine

# Device Agnostic code

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Model

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.config.use_cache = True
model.eval()


# Configuration
MAX_BATCH_SIZE = 42
MAX_NEW_TOKENS = 128

# Create Scheduler + Engine
scheduler = Scheduler(max_active=MAX_BATCH_SIZE)

engine = ContinuousEngine(
    model=model,
    tokenizer=tokenizer,
    scheduler=scheduler,
    device=model.device,
    max_batch_size=MAX_BATCH_SIZE
)

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
    "Explain the theory of relativity.",
    "What is quantum computing?",
    "Describe how neural networks work.",
    "What causes inflation in economics?",
    "Summarize the French Revolution.",
    "How does the internet work?",
    "What is the difference between AI and machine learning?",
    "Explain photosynthesis.",
    "What is the Big Bang theory?",
    "Describe the process of natural selection.",
    "What are cryptocurrencies?",
    "Explain how vaccines work.",
    "What is the stock market?",
    "Describe the water cycle.",
    "What is cybersecurity?",
    "Explain the concept of cloud computing.",
    "What is genetic engineering?",
    "Describe the human digestive system.",
    "What is game theory?",
    "Explain how GPS works."

]

for i, prompt in enumerate(initial_prompts):
    scheduler.Submit(
        Request(i, prompt, MAX_NEW_TOKENS)
    )

print("\n--- Submitted initial requests ---")

start_time = time.time()
step_counter = 0
new_requests_added = False
total_tokens_generated = 0

# Serving Loop
while True:

    # Inject waiting requests into active slots
    scheduler.Inject_if_possible()

    # Prefill new ones
    engine.run_prefill()

    # Decode one step for all active
    engine.decode_step()

    # Count generated tokens
    for r in scheduler.active:
        total_tokens_generated = sum(
            len(req.output_tokens)
            for req in scheduler.active + scheduler.completed
        )

    # Cleanup finished
    engine.cleanup()

    step_counter += 1

    # Dynamic Injection During Decode
    if step_counter == 5 and not new_requests_added:

        print("\n--- Injecting MORE requests during decode ---")

        more_prompts = [
            "What is quantum computing?",
            "Summarize the theory of relativity.",
            "What is Self-attention in Transformer",
            "Why sky looks blue",
            "What is backpropagation in Deep Learning",
            "what is gradient descent"
        ]

        for j, prompt in enumerate(more_prompts):
            scheduler.Submit(
                Request(100 + j, prompt, MAX_NEW_TOKENS)
            )

        new_requests_added = True

    # Exit condition
    if not scheduler.waiting and not scheduler.active:
        break

# Final Stats
total_time = time.time() - start_time

tokens_per_sec = total_tokens_generated / total_time

print("\n===== CONTINUOUS BATCH TEST COMPLETE =====")
print(f"Total time: {total_time:.4f}s")
print(f"Total tokens generated: {total_tokens_generated}")
print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
print(f"Total decode steps: {step_counter}")
