"""Track A: DeepSeek-style MTP draft module on a frozen Llama trunk.

Module (per DeepSeek-V3 §2.2, depth 1): at trunk position t,
    input_t = W_in([RMSNorm(h_t) ; RMSNorm(Emb(x_{t+1}))])
    m_t     = DecoderLayer(input_0..t)          <- ONE transformer layer
    logits  = frozen shared lm_head(m_t)        -> predicts x_{t+2}

Phase-A design decisions (mtp_scoping.md; single-variable experiment):
- Trunk is DENSE everywhere (hidden extraction, verify). Only the MODULE's
  attention varies:
    * dense  — full causal attention (prefill sdpa, decode over full cache)
    * delta  — pipeline prefill via the differentiable delta op
      (delta_forward_train: sink 1024 + window 2048 + exact anchor row every
      gamma), decode = sink+window sparse reads PLUS the WP-3 anchor
      correction: every `gamma` CONFIRMED positions an exact full-cache row
      refreshes a cached per-head correction added to sparse rows between
      anchors. Speculative chaining steps READ the correction but never
      refresh it or advance the cadence (07-21 review round 2: counting
      calls instead of confirmed positions halved the effective stride and
      let anchors fire on speculative queries over later-discarded cache).
- Module attention is computed EXPLICITLY here (q/k/v/o driven by hand, own
  cache) instead of routing through llama.py's dispatch — one layer does not
  justify inheriting a 2K-line code path, and Phase A needs zero hidden
  behavior.
- Warm start: decoder-layer weights copied from the trunk's LAST layer
  (EAGLE-style head training converges much faster than random init).
- Chaining beyond depth 1 (K>1) feeds the module its OWN output hidden as a
  stand-in for the trunk hidden (standard MTP/EAGLE inference chaining;
  trained depth-1 only, so per-position acceptance decay at K>1 is expected
  and is itself a measurement).
"""

from __future__ import annotations

import torch
from torch import nn

from .llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)

SINK = 1024


class MTPModule(nn.Module):
    def __init__(self, trunk, module_attn="dense", gamma=64, window=2048):
        super().__init__()
        cfg = trunk.config
        self.cfg = cfg
        self.module_attn = module_attn
        self.gamma = gamma
        self.window = window
        d = cfg.hidden_size
        self.h_norm = LlamaRMSNorm(d, eps=cfg.rms_norm_eps)
        self.e_norm = LlamaRMSNorm(d, eps=cfg.rms_norm_eps)
        self.w_in = nn.Linear(2 * d, d, bias=False)
        self.layer = LlamaDecoderLayer(cfg, layer_idx=0)
        # warm start from the trunk's last layer
        self.layer.load_state_dict(
            trunk.model.layers[-1].state_dict(), strict=True)
        self.final_norm = LlamaRMSNorm(d, eps=cfg.rms_norm_eps)
        self.final_norm.load_state_dict(trunk.model.norm.state_dict())
        # decode-time KV cache (post-rope), [1, kv_heads, len, head_dim]
        self._ck = None
        self._cv = None
        # delta-variant decode correction state (WP-3 semantics in the head):
        # exact full-cache row every ANCHOR_EVERY steps -> cached correction
        self._delta = None
        self._since_anchor = 0

    # ---- explicit single-layer forward pieces -------------------------
    def _mix(self, hiddens, tok_embs):
        return self.w_in(torch.cat(
            [self.h_norm(hiddens), self.e_norm(tok_embs)], dim=-1))

    def _qkv_rope(self, x, rotary, pos_ids):
        attn = self.layer.self_attn
        hs = (*x.shape[:-1], -1, attn.head_dim)
        q, k, v = attn._qkv(self.layer.input_layernorm(x), hs)
        cos, sin = rotary(v.transpose(1, 2), pos_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v

    def _finish(self, x, attn_out):
        attn = self.layer.self_attn
        b, h, s, dh = attn_out.shape
        o = attn.o_proj(attn_out.transpose(1, 2).reshape(b, s, h * dh))
        x = x + o
        x = x + self.layer.mlp(self.layer.post_attention_layernorm(x))
        return x

    # ---- parallel (training / prefill) forward ------------------------
    def forward_parallel(self, hiddens, tok_embs, rotary):
        """hiddens[1,L,d] (trunk h_0..h_{L-1}, no-grad), tok_embs[1,L,d]
        (Emb(x_1)..Emb(x_L) aligned). Returns module hidden [1,L,d] where
        position t predicts x_{t+2}. Causal within the module sequence."""
        x = self._mix(hiddens, tok_embs)
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        q, k, v = self._qkv_rope(x, rotary, pos)
        if self.module_attn == "delta":
            from .train.flex_delta import delta_forward_train
            attn_out = delta_forward_train(
                q, k, v, gamma=self.gamma, window=self.window)
            attn_out = attn_out.transpose(1, 2)  # -> [b, h, s, d]
        else:
            attn = self.layer.self_attn
            kk = repeat_kv(k, attn.num_key_value_groups)
            vv = repeat_kv(v, attn.num_key_value_groups)
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q, kk, vv, is_causal=True, scale=attn.head_dim ** -0.5)
        x = self._finish(x, attn_out)
        return self.final_norm(x)

    # ---- decode-time cache API ----------------------------------------
    @torch.no_grad()
    def prefill_cache(self, hiddens, tok_embs, rotary):
        """Build the module KV cache over the prompt (delta variant caches
        the SAME post-rope k/v — sparsity is applied at READ time)."""
        x = self._mix(hiddens, tok_embs)
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        _, k, v = self._qkv_rope(x, rotary, pos)
        self._ck, self._cv = k, v
        # correction state is per-prompt (07-21 review round 2: the module
        # object is reused across mtp_eval's prompt loop — a stale delta
        # from the previous prompt leaked into early decode steps)
        self._delta = None
        self._since_anchor = 0

    def cache_len(self):
        return 0 if self._ck is None else self._ck.size(2)

    def crop(self, n):
        self._ck = self._ck[:, :, :n]
        self._cv = self._cv[:, :, :n]

    @torch.no_grad()
    def decode_step(self, hidden, tok_emb, rotary, pos, speculative=False):
        """One module position: append to cache, return module_hidden[1,d].
        dense: attend over the full cache. delta: sink+window sparse read +
        the WP-3 anchor correction. Cadence counts CONFIRMED positions only:
        speculative steps read the current correction but never refresh it
        (their cache tail is discarded by the following crop) and never
        advance the counter (each emitted position passes through
        decode_step twice — draft, then confirmed re-append)."""
        x = self._mix(hidden.view(1, 1, -1), tok_emb.view(1, 1, -1))
        pos_ids = torch.tensor([[pos]], device=x.device)
        q, k, v = self._qkv_rope(x, rotary, pos_ids)
        self._ck = torch.cat([self._ck, k], dim=2)
        self._cv = torch.cat([self._cv, v], dim=2)
        attn = self.layer.self_attn
        scale = attn.head_dim ** -0.5
        sdpa = torch.nn.functional.scaled_dot_product_attention
        full = self.module_attn == "dense" \
            or self._ck.size(2) <= SINK + self.window
        if full:
            # q_len=1 over past-only cache: no mask; enable_gqa avoids the
            # full-cache repeat_kv copy (07-21 review)
            attn_out = sdpa(q, self._ck, self._cv, scale=scale,
                            enable_gqa=True)
        else:
            ck = torch.cat([self._ck[:, :, :SINK],
                            self._ck[:, :, -self.window:]], dim=2)
            cv = torch.cat([self._cv[:, :, :SINK],
                            self._cv[:, :, -self.window:]], dim=2)
            sparse_out = sdpa(q, ck, cv, scale=scale, enable_gqa=True)
            if speculative:
                attn_out = sparse_out if self._delta is None \
                    else sparse_out + self._delta
            else:
                self._since_anchor += 1
                # cadence = the module's trained gamma (anchor every gamma
                # CONFIRMED positions — exactly gamma, not gamma+1: the
                # counter includes the anchor position itself). Anchors here
                # always see a confirmed-only cache, so crop() (which only
                # ever removes the speculative tail) can never invalidate
                # the keys a live correction was computed over.
                if self._delta is None or self._since_anchor >= self.gamma:
                    dense_out = sdpa(q, self._ck, self._cv, scale=scale,
                                     enable_gqa=True)
                    self._delta = dense_out - sparse_out
                    self._since_anchor = 0
                    attn_out = dense_out
                else:
                    attn_out = sparse_out + self._delta
        x = self._finish(x, attn_out)
        return self.final_norm(x)[0, 0]
