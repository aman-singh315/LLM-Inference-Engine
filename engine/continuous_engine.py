import torch
import time

class ContinuousEngine:

  def __init__(self, model, tokenizer, scheduler, device, max_batch_size):
    self.model = model
    self.tokenizer = tokenizer
    self.scheduler = scheduler
    self.device = device
    self.max_batch_size = max_batch_size

    # Block-based memory
    self.block_size = 16

    config = model.config

    self.block_pool = BlockPool(
      total_blocks=2500,
      block_size=self.block_size,
      num_layers=config.num_hidden_layers,
      num_heads=config.num_key_value_heads,
      head_dim=config.hidden_size // config.num_attention_heads,
      device=device
    )

    # ---------------- PREFILL ----------------
  @torch.no_grad()
  def run_prefill(self):

    new_reqs = [
      r for r in self.scheduler.active
      if r.state is None
    ]

    if not new_reqs:
      return

    prompts = [r.prompt for r in new_reqs]

    batch = self.tokenizer(
      prompts,
      return_tensors="pt",
      padding=True
    ).to(self.device)

    torch.cuda.synchronize()
    t0 = time.time()

    outputs = self.model(
      input_ids=batch["input_ids"],
      attention_mask=batch["attention_mask"],
      use_cache=True
    )

    torch.cuda.synchronize()
    print(f"Prefill batch size {len(new_reqs)} | {time.time() - t0:.4f}s")

    prompt_lengths = batch["attention_mask"].sum(dim=1)

    for i, req in enumerate(new_reqs):

      req.state = RequestState()

      prompt_len = prompt_lengths[i].item()

      # Allocate blocks for prompt
      num_blocks = (prompt_len + self.block_size - 1) // self.block_size

      for _ in range(num_blocks):
        block_id = self.block_pool.allocate()
        req.state.block_ids.append(block_id)

      req.state.total_tokens = prompt_len


      # Write KV into blocks
      self._write_full_prompt_kv(
        req,
        outputs.past_key_values,
        batch_index=i,
        prompt_len=prompt_len
      )
      # First generated token
      next_token = outputs.logits[i, -1].argmax()

      req.last_token = next_token.view(1, 1)
      req.output_tokens.append(next_token.item())
      req.num_generated += 1

    # ---------------- DECODE ----------------
  @torch.no_grad()
  def decode_step(self):

    active_reqs = [
      r for r in self.scheduler.active
      if not r.finished and r.state is not None
    ]

    if not active_reqs:
      return

    input_ids = torch.cat(
      [r.last_token for r in active_reqs],
      dim=0
    ).to(self.device)

    outputs = self.model(
      input_ids=input_ids,
      use_cache=True  
    )

    logits = outputs.logits
    next_tokens = logits[:, -1].argmax(dim=-1)

    for i, req in enumerate(active_reqs):

      # Allocate new block if needed
      if req.state.total_tokens % self.block_size == 0:
        block_id = self.block_pool.allocate()
        req.state.block_ids.append(block_id)

      self._write_single_token_kv(
          req,
          outputs.past_key_values,
          batch_index=i
        )

      token = next_tokens[i]



      req.last_token = token.view(1, 1)
      req.output_tokens.append(token.item())
      req.num_generated += 1
      req.state.total_tokens += 1

      if (
        token.item() == self.tokenizer.eos_token_id
        or req.num_generated >= req.max_new_tokens
      ):
        req.finished = True

    # ---------------- CLEANUP ----------------
  def cleanup(self):

    finished = [
      r for r in self.scheduler.active
      if r.finished
    ]

    if not finished:
      return

    for req in finished:
      for block_id in req.state.block_ids:
        self.block_pool.free(block_id)

    self.scheduler.completed.extend(finished)

    self.scheduler.active = [
      r for r in self.scheduler.active
      if not r.finished
    ]





   # ================= INTERNAL KV WRITERS =================
  def _write_full_prompt_kv(self, req, kv, batch_index, prompt_len):

    for layer_idx, layer_kv in enumerate(kv):

      k = layer_kv[0]
      v = layer_kv[1]
      # k: [batch, heads, seq, dim]
      k_slice = k[batch_index, :, :prompt_len, :]
      v_slice = v[batch_index, :, :prompt_len, :]

      for token_idx in range(prompt_len):

        block_id = req.state.block_ids[token_idx // self.block_size]
        offset = token_idx % self.block_size

        self.block_pool.keys[layer_idx, block_id, :, offset, :] = k_slice[:, token_idx, :]
        self.block_pool.values[layer_idx, block_id, :, offset, :] = v_slice[:, token_idx, :]

  def _write_single_token_kv(self, req, kv, batch_index):

    token_idx = req.state.total_tokens
    block_id = req.state.block_ids[token_idx // self.block_size]
    offset = token_idx % self.block_size
    for layer_idx, layer_kv in enumerate(kv):
      k = layer_kv[0]
      v = layer_kv[1]

      # k: [batch, heads, 1, dim]
      self.block_pool.keys[layer_idx, block_id, :, offset, :] = k[batch_index, :, 0, :]
      self.block_pool.values[layer_idx, block_id, :, offset, :] = v[batch_index, :, 0, :]



  def _gather_kv(self, req):

    past = []

    for layer_idx in range(self.num_layers):

      k_blocks = []
      v_blocks = []

      for block_id in req.state.block_ids:
        k_blocks.append(
        self.block_pool.keys[layer_idx, block_id]
        )
        v_blocks.append(
        self.block_pool.values[layer_idx, block_id]
        )

      k_cat = torch.cat(k_blocks, dim=1)
      v_cat = torch.cat(v_blocks, dim=1)

      k_cat = k_cat[:, :req.state.total_tokens, :]
      v_cat = v_cat[:, :req.state.total_tokens, :]

      past.append((
            k_cat.unsqueeze(0),
        v_cat.unsqueeze(0)
      ))

    return tuple(past)
