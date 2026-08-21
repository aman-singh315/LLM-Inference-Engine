import types
import time
import torch

# NEW profile_stats dict — different phases than the old gather_kv version,
# since the new decode_step doesn't have gather_kv/cache_build/writeback at all
profile_stats_v2 = {
    "meta_embed_rope": 0.0,   # building indptr/indices/last_page_len + embedding + RoPE cos/sin
    "qkv_proj": 0.0,          # q_proj/k_proj/v_proj + RoPE apply, summed across all 22 layers
    "kv_write": 0.0,          # writing new token's k/v into block_pool, summed across all layers
    "attention": 0.0,         # flashinfer_wrapper.run(), summed across all layers
    "o_proj": 0.0,
    "mlp": 0.0,
    "final_head": 0.0,        # final_norm + lm_head
    "bookkeeping": 0.0,       # request state updates at the end
    "steps": 0,
    "mlp_layernorm": 0.0,
    # "mlp_core" : 0.0,
    "mlp_residual" : 0.0,
    "mlp_gate_proj": 0.0,
    "mlp_up_proj": 0.0,
    "mlp_activation": 0.0,
    "mlp_mul": 0.0,
    "mlp_down_proj": 0.0,
}

@torch.no_grad()
def instrumented_decode_step_v2(self):
    active_reqs = [
        r for r in self.scheduler.active
        if not r.finished and r.state is not None
    ]
    if not active_reqs:
        return

    batch_size = len(active_reqs)
    device = self.device

    torch.cuda.synchronize()
    t0 = time.time()

    new_token_block_ids = []
    new_token_offsets = []
    position_ids_list = []

    for req in active_reqs:
        token_idx = req.state.total_tokens
        if token_idx % self.block_size == 0:
            block_id = self.block_pool.allocate()
            req.state.block_ids.append(block_id)
        block_id = req.state.block_ids[token_idx // self.block_size]
        offset = token_idx % self.block_size
        new_token_block_ids.append(block_id)
        new_token_offsets.append(offset)
        position_ids_list.append(token_idx)

    last_tokens = torch.tensor(
        [[req.last_token.item()] for req in active_reqs],
        dtype=torch.long, device=device
    )
    position_ids = torch.tensor(
        [[p] for p in position_ids_list],
        dtype=torch.long, device=device
    )

    hidden_states = self.embed_tokens(last_tokens)
    cos, sin = self.rotary_emb(hidden_states, position_ids)

    block_counts = [len(req.state.block_ids) for req in active_reqs]
    indptr = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(block_counts), dim=0)),
        dtype=torch.int32, device=device
    )
    indices = torch.tensor(
        [bid for req in active_reqs for bid in req.state.block_ids],
        dtype=torch.int32, device=device
    )
    last_page_len = torch.tensor(
        [((req.state.total_tokens + 1 - 1) % self.block_size) + 1 for req in active_reqs],
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
        pos_encoding_mode="NONE",
        data_type=torch.float16,
    )

    torch.cuda.synchronize()
    t1 = time.time()
    profile_stats_v2["meta_embed_rope"] += (t1 - t0)

    # new insert
    new_token_block_ids_t = torch.tensor(new_token_block_ids, dtype=torch.long, device=device)
    new_token_offsets_t = torch.tensor(new_token_offsets, dtype=torch.long, device=device)

    for layer_idx, layer in enumerate(self.layers):

        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        torch.cuda.synchronize()
        ta = time.time()

        # q = layer.self_attn.q_proj(hidden_states)
        # k = layer.self_attn.k_proj(hidden_states)
        # v = layer.self_attn.v_proj(hidden_states)
        # q = q.view(batch_size, self.num_q_heads, self.head_dim)
        # k = k.view(batch_size, self.num_kv_heads, self.head_dim)
        # v = v.view(batch_size, self.num_kv_heads, self.head_dim)

        # ---- fused QKV projection ----
        qkv = torch.nn.functional.linear(hidden_states, self.fused_qkv_weights[layer_idx])
        q, k, v = torch.split(qkv, [self.q_dim, self.kv_dim, self.kv_dim], dim=-1)

        q = q.view(batch_size, self.num_q_heads, self.head_dim)
        k = k.view(batch_size, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, self.num_kv_heads, self.head_dim)



        q, k = apply_rotary_pos_emb(q.unsqueeze(2), k.unsqueeze(2), cos, sin)
        q = q.squeeze(2)
        k = k.squeeze(2)

        torch.cuda.synchronize()
        tb = time.time()
        profile_stats_v2["qkv_proj"] += (tb - ta)

        # for i in range(batch_size):
        #     b_id = new_token_block_ids[i]
        #     off = new_token_offsets[i]
        #     self.block_pool.keys[layer_idx, b_id, :, off, :] = k[i]
        #     self.block_pool.values[layer_idx, b_id, :, off, :] = v[i]
        self.block_pool.keys[layer_idx, new_token_block_ids_t, :, new_token_offsets_t, :] = k
        self.block_pool.values[layer_idx, new_token_block_ids_t, :, new_token_offsets_t, :] = v


        torch.cuda.synchronize()
        tc = time.time()
        profile_stats_v2["kv_write"] += (tc - tb)


        torch.cuda.synchronize()
        tattn_start = time.perf_counter()

        attn_out = self.flashinfer_wrapper.run(
            q,
            (self.block_pool.keys[layer_idx], self.block_pool.values[layer_idx]),
        )

        torch.cuda.synchronize()
        tattn_end = time.perf_counter()
        profile_stats_v2["attention"] += (tattn_end - tattn_start)

        #  O projection --------------------------
        torch.cuda.synchronize()
        to_start = time.perf_counter()

        attn_out = attn_out.reshape(batch_size, 1, self.hidden_size)
        attn_out = layer.self_attn.o_proj(attn_out)

        torch.cuda.synchronize()
        to_end = time.perf_counter()

        profile_stats_v2["o_proj"] += (to_end -  to_start)

        hidden_states = residual + attn_out

        #------- MLP------------------------------
        # torch.cuda.synchronize()
        # tmlp_start = time.perf_counter()

        # residual = hidden_states
        # hidden_states = layer.post_attention_layernorm(hidden_states)
        # hidden_states = layer.mlp(hidden_states)
        # hidden_states = residual + hidden_states

        # torch.cuda.synchronize()
        # tmlp_end = time.perf_counter()

        # profile_stats_v2["mlp"] += (tmlp_end - tmlp_start)

        #------------- MLP Profiling
        residual = hidden_states

        #1. post-attention layernorm
        torch.cuda.synchronize()
        tmln_start = time.perf_counter()

        hidden_states = layer.post_attention_layernorm(hidden_states)

        torch.cuda.synchronize()
        tmln_end = time.perf_counter()

        profile_stats_v2["mlp_layernorm"] += (tmln_end - tmln_start)


        # ---------------- MLP CORE PROFILING ----------------

        # gate projection
        torch.cuda.synchronize()
        tgate_start = time.perf_counter()

        gate = layer.mlp.gate_proj(hidden_states)

        torch.cuda.synchronize()
        tgate_end = time.perf_counter()

        profile_stats_v2["mlp_gate_proj"] += (tgate_end - tgate_start)


        # up projection
        torch.cuda.synchronize()
        tup_start = time.perf_counter()

        up = layer.mlp.up_proj(hidden_states)

        torch.cuda.synchronize()
        tup_end = time.perf_counter()

        profile_stats_v2["mlp_up_proj"] += (tup_end - tup_start)


        # SiLU activation
        torch.cuda.synchronize()
        tact_start = time.perf_counter()

        gate = layer.mlp.act_fn(gate)

        torch.cuda.synchronize()
        tact_end = time.perf_counter()

        profile_stats_v2["mlp_activation"] += (tact_end - tact_start)


        # Elementwise multiplication
        torch.cuda.synchronize()
        tmul_start = time.perf_counter()

        hidden_states = gate * up

        torch.cuda.synchronize()
        tmul_end = time.perf_counter()

        profile_stats_v2["mlp_mul"] += (tmul_end - tmul_start)

        # Down projection
        torch.cuda.synchronize()
        tdown_start = time.perf_counter()

        hidden_states = layer.mlp.down_proj(hidden_states)

        torch.cuda.synchronize()
        tdown_end = time.perf_counter()

        profile_stats_v2["mlp_down_proj"] += (tdown_end - tdown_start)



        # 3. Residual addition
        torch.cuda.synchronize()
        tres_start = time.perf_counter()

        hidden_states = residual + hidden_states

        torch.cuda.synchronize()
        tres_end = time.perf_counter()

        profile_stats_v2["mlp_residual"] += (tres_end - tres_start)


    # ------ final Head-------------------
    torch.cuda.synchronize()
    tf_start = time.perf_counter()
    hidden_states = self.final_norm(hidden_states)
    logits = self.lm_head(hidden_states)
    next_tokens = logits[:, -1, :].argmax(dim=-1)

    torch.cuda.synchronize()
    tf_end = time.perf_counter()
    profile_stats_v2["final_head"] += (tf_end - tf_start)

    #---- Bookkeeping-----
    torch.cuda.synchronize()
    tbk_start = time.perf_counter()

    for i, req in enumerate(active_reqs):
        token = next_tokens[i].item()
        req.last_token = next_tokens[i].view(1, 1)
        req.output_tokens.append(token)
        req.num_generated += 1
        req.state.total_tokens += 1
        if (token == self.tokenizer.eos_token_id or
            req.num_generated >= req.max_new_tokens):
              req.finished = True

    torch.cuda.synchronize()
    tbk_end = time.perf_counter()
    profile_stats_v2["bookkeeping"] += (tbk_end - tbk_start)
    profile_stats_v2["steps"] += 1


def print_profile_summary_v2():
    steps = profile_stats_v2["steps"]
    if steps == 0:
        print("No steps recorded.")
        return
    total = sum(v for k, v in profile_stats_v2.items() if k != "steps")
    print(f"\n===== FLASHINFER DECODE STEP PROFILE ({steps} steps) =====")
    for key in ["meta_embed_rope", "qkv_proj", "kv_write", "attention", "o_proj",
                "mlp_layernorm", "mlp_residual", "final_head", "bookkeeping",
                "mlp_gate_proj","mlp_up_proj","mlp_activation","mlp_mul","mlp_down_proj",]:
        ms_per_step = (profile_stats_v2[key] / steps) * 1000
        pct = (profile_stats_v2[key] / total) * 100 if total > 0 else 0
        print(f"{key:16s}: {ms_per_step:7.3f} ms/step  ({pct:5.1f}%)")
    print(f"{'TOTAL':16s}: {(total/steps)*1000:7.3f} ms/step")
