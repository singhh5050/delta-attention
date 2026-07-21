"""MiMo-7B's production MTP layer loaded into our Track A machinery.

Verified against the public sources (07-21; docs/mimo_mtp_plan.md):
- HF modeling_mimo.py: MiMoMTPLayers = token_layernorm(emb) ⊕
  hidden_layernorm(trunk_hidden) → input_proj(2d→d) → ONE Qwen2 decoder
  layer → final_layernorm → shared lm_head. Same concat order as our
  MTPModule ([hidden; emb]).
- vLLM mimo_mtp.py: the layer consumes the trunk's POST-final-norm hidden
  states, keeps its own KV cache, absolute rope positions, and only K=1 is
  natively supported. (vLLM's position-0 embedding masking is a plumbing
  artifact of its shifted-input flow — the HF reference module has no such
  masking, and our explicit alignment never produces an undefined slot, so
  we deliberately do NOT replicate it.)
- Weights: model.mtp_layers.0.* — 16 tensors; q/k/v have Qwen2-style bias,
  o_proj does not. Identical names to our LlamaDecoderLayer submodules, so
  the decoder block loads strict=True after swapping o_proj to bias-free.

The trunk loads as PLAIN Qwen2ForCausalLM (same weight names; mtp_layers.*
ignored as unexpected keys) — no trust_remote_code, no dependence on
MiMo's transformers-4.40-era custom code under our 4.51.3 pin. The trunk
is dense everywhere; only the module's attention reads vary (dense vs the
anchor-corrected delta state machine inherited from MTPModule v2).
"""

from __future__ import annotations

import json

import torch
from torch import nn

from .llama import LlamaConfig, LlamaDecoderLayer, LlamaRMSNorm
from .mtp import MTPModule

MIMO_ID = "XiaomiMiMo/MiMo-7B-RL"


def _mimo_hf_config(model_id=MIMO_ID):
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(model_id, "config.json")) as f:
        return json.load(f)


def mimo_llama_config(model_id=MIMO_ID):
    """A LlamaConfig shaped like MiMo's decoder layer (the MTP block is a
    standard Qwen2 layer = Llama layer + qkv bias)."""
    c = _mimo_hf_config(model_id)
    return LlamaConfig(
        hidden_size=c["hidden_size"],
        intermediate_size=c["intermediate_size"],
        num_attention_heads=c["num_attention_heads"],
        num_key_value_heads=c["num_key_value_heads"],
        head_dim=c["head_dim"],
        rms_norm_eps=c["rms_norm_eps"],
        rope_theta=c["rope_theta"],
        max_position_embeddings=c["max_position_embeddings"],
        vocab_size=c["vocab_size"],
        hidden_act=c["hidden_act"],
        attention_bias=True,   # q/k/v biased (Qwen2); o_proj swapped below
        attention_dropout=0.0,
        num_hidden_layers=1,
    )


def build_mimo_trunk(model_id=MIMO_ID, attn_impl="flash_attention_2"):
    """Frozen DENSE MiMo trunk as vanilla Qwen2 (hiddens via
    output_hidden_states; no monkey-patching)."""
    import transformers

    c = _mimo_hf_config(model_id)
    qcfg = transformers.Qwen2Config(
        **{k: v for k, v in c.items()
           if k in transformers.Qwen2Config().to_dict()})
    qcfg.architectures = ["Qwen2ForCausalLM"]
    trunk = transformers.Qwen2ForCausalLM.from_pretrained(
        model_id, config=qcfg, torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    for p in trunk.parameters():
        p.requires_grad_(False)
    trunk.config.use_cache = True
    return trunk.eval().cuda(), tokenizer


class MiMoMTPModule(MTPModule):
    """MiMo's trained MTP layer driven by MTPModule's (twice-reviewed)
    cache + anchor-corrected read machinery. Only construction and weight
    loading differ; every forward/decode path is inherited."""

    def __init__(self, module_attn="dense", gamma=64, window=2048,
                 model_id=MIMO_ID):
        nn.Module.__init__(self)  # skip MTPModule.__init__ (llama-trunk
        # coupled warm start); build the same attribute surface explicitly
        cfg = mimo_llama_config(model_id)
        self.cfg = cfg
        self.module_attn = module_attn
        self.gamma = gamma
        self.window = window
        d = cfg.hidden_size
        self.h_norm = LlamaRMSNorm(d, eps=cfg.rms_norm_eps)   # hidden_layernorm
        self.e_norm = LlamaRMSNorm(d, eps=cfg.rms_norm_eps)   # token_layernorm
        self.w_in = nn.Linear(2 * d, d, bias=False)           # input_proj
        self.layer = LlamaDecoderLayer(cfg, layer_idx=0)
        # MiMo has NO o_proj bias (Qwen2 convention); attention_bias=True
        # would otherwise create one and break strict loading
        old = self.layer.self_attn.o_proj
        self.layer.self_attn.o_proj = nn.Linear(
            old.in_features, old.out_features, bias=False)
        self.final_norm = LlamaRMSNorm(d, eps=cfg.rms_norm_eps)
        self._ck = None
        self._cv = None
        self._delta = None
        self._since_anchor = 0
        self._load_mimo_weights(model_id)

    def _load_mimo_weights(self, model_id):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        with open(hf_hub_download(model_id, "model.safetensors.index.json")) as f:
            wmap = json.load(f)["weight_map"]
        prefix = "model.mtp_layers.0."
        shards = sorted({v for k, v in wmap.items() if k.startswith(prefix)})
        tensors = {}
        for shard in shards:
            path = hf_hub_download(model_id, shard)
            data = load_file(path)
            tensors.update({k[len(prefix):]: v for k, v in data.items()
                            if k.startswith(prefix)})
        rename = {"token_layernorm.weight": "e_norm.weight",
                  "hidden_layernorm.weight": "h_norm.weight",
                  "input_proj.weight": "w_in.weight",
                  "final_layernorm.weight": "final_norm.weight"}
        state = {}
        for k, v in tensors.items():
            state[rename.get(k, f"layer.{k}")] = v
        missing, unexpected = self.load_state_dict(state, strict=False)
        # strict accounting: every module parameter must be covered and
        # every checkpoint tensor consumed — a silent partial load would
        # produce a plausible-but-wrong head
        assert not unexpected, f"unconsumed MiMo tensors: {unexpected}"
        assert not missing, f"uninitialized module params: {missing}"
        n = sum(t.numel() for t in tensors.values())
        print(f"[mimo] loaded {len(tensors)} MTP tensors ({n/1e6:.0f}M params)",
              flush=True)
