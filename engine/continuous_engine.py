import torch
import time
import flashinfer

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

    self.num_layers = config.num_hidden_layers
    self.num_kv_heads = config.num_key_value_heads
    self.num_q_heads = config.num_attention_heads
    self.head_dim = config.hidden_size // config.num_attention_heads
    self.hidden_size = config.hidden_size

    self.block_pool = BlockPool(
      total_blocks=256,
      block_size=self.block_size,
      num_layers=self.num_layers,
      num_heads=self.num_kv_heads,
      head_dim=self.head_dim,
      device=device
    )
    #right padding so prompt tokens occupy [0 : prompt_len]
    self.tokenizer.padding_side = "right"


    # NEW: grab direct references to the model's internal layers.
    # We are NOT copying weights — these are references to the
    # same nn.Module objects HF already loaded. We're just going
    # to call their sub-pieces (norm, proj, mlp) ourselves instead
    # of letting HF's LlamaModel.forward() orchestrate them.

    llama_model = model.model
    self.embed_tokens = llama_model.embed_tokens
    self.layers = llama_model.layers
    self.final_norm = llama_model.norm
    self.lm_head = model.lm_head

    if hasattr(llama_model, "rotary_emb"):
      self.rotary_emb = llama_model.rotary_emb
      self._rope_is_model_level = True
    else:
      self.rotary_emb = self.layers[0].self_attn.rotary_emb
      self._rope_is_model_level = False

    self.attn_scale = self.head_dim ** -0.5

    self.fused_qkv_weights = []
    for layer in self.layers:
      w = torch.cat([
        layer.self_attn.q_proj.weight,
        layer.self_attn.k_proj.weight,
        layer.self_attn.v_proj.weight,
      ], dim=0)
      self.fused_qkv_weights.append(w)

    self.q_dim = self.num_q_heads * self.head_dim
    self.kv_dim = self.num_kv_heads * self.head_dim

    self.flashinfer_workspace = torch.empty(
        32 * 1024 * 1024, dtype=torch.uint8, device=device
    )

    self.flashinfer_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        self.flashinfer_workspace,
        kv_layout="HND"
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

    # Actual prefill
    outputs = self.model(
      input_ids=batch["input_ids"],
      attention_mask=batch["attention_mask"],
      use_cache=True
    )

    torch.cuda.synchronize()
    #Measuring the prefill latency
    print(f"Prefill batch size {len(new_reqs)} | {time.time() - t0:.4f}s")

    prompt_lengths = batch["attention_mask"].sum(dim=1)  # here we are excluding the masks and getting real prompt length

    for i, req in enumerate(new_reqs):
      req.state = RequestState()  # this means This request now has an inference state.
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
      last_idx = prompt_len - 1;
      next_token = outputs.logits[i, last_idx].argmax()

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

    batch_size = len(active_reqs)
    device = self.device

    # --------------------------------------------------------
    # STEP 2a: figure out where the NEW token for each request
    # is going to live (which block, which offset) BEFORE we
    # write anything — we need this decided first because the
    # FlashInfer paged-attention call needs to read the cache
    # AFTER the new token's k/v are already written into it.
    # --------------------------------------------------------
    new_token_block_ids = []   # block id that will hold this step's new token, per request
    new_token_offsets = []     # offset inside that block, per request
    position_ids_list = []     # absolute position of the new token, per request

    for req in active_reqs:
        token_idx = req.state.total_tokens  # position of the token we're about to generate
        if token_idx % self.block_size == 0:
            # current last block is full (or this is the very first decode token) -> allocate a new one
            block_id = self.block_pool.allocate()
            req.state.block_ids.append(block_id)

        block_id = req.state.block_ids[token_idx // self.block_size]
        offset = token_idx % self.block_size

        new_token_block_ids.append(block_id)
        new_token_offsets.append(offset)
        position_ids_list.append(token_idx)

    # --------------------------------------------------------
    # STEP 2b: batch tensors for the new token itself
    # --------------------------------------------------------
    last_tokens = torch.tensor(
        [[req.last_token.item()] for req in active_reqs],
        dtype=torch.long, device=device
    )  # [batch, 1]

    position_ids = torch.tensor(
        [[p] for p in position_ids_list],
        dtype=torch.long, device=device
    )  # [batch, 1]

    # --------------------------------------------------------
    # STEP 2c: embed the new token -> [batch, 1, hidden_size]
    # (this replaces the embedding step that used to happen
    # silently inside self.model(...))
    # --------------------------------------------------------
    hidden_states = self.embed_tokens(last_tokens)

    # RoPE cos/sin computed ONCE per step and reused by every layer
    # (this is what llama_model.rotary_emb used to do internally per forward call)
    cos, sin = self.rotary_emb(hidden_states, position_ids)

    # --------------------------------------------------------
    # STEP 2d: build the FlashInfer BATCH metadata ONCE for this
    # step — NOT per request, NOT per layer. This is what actually
    # fixes issue #3 from the original list (the draft was calling
    # plan()/run() per-request in a loop, which defeats batching).
    # --------------------------------------------------------
    block_counts = [len(req.state.block_ids) for req in active_reqs]

    # indptr: cumulative offsets into `indices` marking where each
    # request's block list starts. e.g. block_counts=[3,2] -> indptr=[0,3,5]
    indptr = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(block_counts), dim=0)),
        dtype=torch.int32, device=device
    )

    # indices: every request's block_ids, concatenated in order
    indices = torch.tensor(
        [bid for req in active_reqs for bid in req.state.block_ids],
        dtype=torch.int32, device=device
    )

    # last_page_len: how many valid tokens are in each request's LAST
    # (most recently allocated) block, AFTER we've accounted for the
    # new token we're about to write this step.
    last_page_len = torch.tensor(
        [
            ((req.state.total_tokens + 1 - 1) % self.block_size) + 1
            for req in active_reqs
        ],
        dtype=torch.int32, device=device
    )

    seq_lens = torch.tensor(
        [req.state.total_tokens + 1 for req in active_reqs],  # +1: includes the token we're generating this step
        dtype=torch.int32, device=device
    )

    self.flashinfer_wrapper.plan(
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_len,
        num_qo_heads=self.num_q_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        page_size=self.block_size,
        pos_encoding_mode="NONE",   # we already apply RoPE ourselves below — don't let flashinfer double-apply it
        data_type=torch.float16,
    )

    # --------------------------------------------------------
    # STEP 2e: run every decoder layer BY HAND.
    # This block replaces what self.model(...) used to do internally.
    # --------------------------------------------------------
    for layer_idx, layer in enumerate(self.layers):

        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        # ---- QKV projections ----
        # q = layer.self_attn.q_proj(hidden_states)  # [batch, 1, num_q_heads * head_dim]
        # k = layer.self_attn.k_proj(hidden_states)  # [batch, 1, num_kv_heads * head_dim]
        # v = layer.self_attn.v_proj(hidden_states)  # [batch, 1, num_kv_heads * head_dim]

        # q = q.view(batch_size, self.num_q_heads, self.head_dim)   # drop the seq=1 dim, split heads
        # k = k.view(batch_size, self.num_kv_heads, self.head_dim)
        # v = v.view(batch_size, self.num_kv_heads, self.head_dim)

        # ---- fused QKV projection ----
        qkv = torch.nn.functional.linear(hidden_states, self.fused_qkv_weights[layer_idx])
        q, k, v = torch.split(qkv, [self.q_dim, self.kv_dim, self.kv_dim], dim=-1)

        q = q.view(batch_size, self.num_q_heads, self.head_dim)
        k = k.view(batch_size, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, self.num_kv_heads, self.head_dim)

        # ---- RoPE: rotate Q and K using the cos/sin computed once above ----
        q, k = apply_rotary_pos_emb(q.unsqueeze(2), k.unsqueeze(2), cos, sin)
        q = q.squeeze(2)  # back to [batch, num_q_heads, head_dim]
        k = k.squeeze(2)  # back to [batch, num_kv_heads, head_dim]

        # ---- write this new token's K/V into the paged cache BEFORE attention ----
        # (flashinfer reads directly from block_pool.keys/values, so the new
        # token has to be in there first, or it'll attend over stale/missing data)
        for i in range(batch_size):
            b_id = new_token_block_ids[i]
            off = new_token_offsets[i]
            self.block_pool.keys[layer_idx, b_id, :, off, :] = k[i]
            self.block_pool.values[layer_idx, b_id, :, off, :] = v[i]

        # ---- paged attention via FlashInfer (this is the whole point) ----
        attn_out = self.flashinfer_wrapper.run(
            q,
            (self.block_pool.keys[layer_idx],
            self.block_pool.values[layer_idx]),
        )  # [batch, num_q_heads, head_dim]

        attn_out = attn_out.reshape(batch_size, 1, self.hidden_size)
        attn_out = layer.self_attn.o_proj(attn_out)

        hidden_states = residual + attn_out

        # ---- MLP block ----
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

    # --------------------------------------------------------
    # STEP 2f: final norm + lm_head -> logits, same as HF does at the end
    # --------------------------------------------------------
    hidden_states = self.final_norm(hidden_states)
    logits = self.lm_head(hidden_states)  # [batch, 1, vocab_size]

    next_tokens = logits[:, -1, :].argmax(dim=-1)

    # --------------------------------------------------------
    # STEP 2g: request bookkeeping (same as before — no gather_kv,
    # no _write_single_token_kv, since we already wrote K/V above)
    # --------------------------------------------------------
    for i, req in enumerate(active_reqs):
        token = next_tokens[i].item()
        req.last_token = next_tokens[i].view(1, 1)
        req.output_tokens.append(token)
        req.num_generated += 1
        req.state.total_tokens += 1

        if (token == self.tokenizer.eos_token_id or
            req.num_generated >= req.max_new_tokens):
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

      for block_idx , block_id in enumerate(req.state.block_ids):
        start = block_idx * self.block_size
        end = min(start + self.block_size, prompt_len)
        length = end - start

        if length <= 0:
          break

        self.block_pool.keys[
            layer_idx, block_id, : , :length, :
        ] = k_slice[:, start:end, :]

        self.block_pool.values[
            layer_idx, block_id, : , :length, :
        ] = v_slice[:, start:end, :]

  def _write_single_token_kv(self, req, kv, batch_index):
    token_idx = req.state.total_tokens
    block_id = req.state.block_ids[token_idx // self.block_size]
    offset = token_idx % self.block_size
    for layer_idx, layer_kv in enumerate(kv):
      k = layer_kv[0]
      v = layer_kv[1]

      self.block_pool.keys[
          layer_idx, block_id, :, offset, :
      ] = k[batch_index, :, -1, :]
      self.block_pool.values[
          layer_idx, block_id, :, offset, :
      ] = v[batch_index, :, -1, :]
      # k: [batch, heads, 1, dim]


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
