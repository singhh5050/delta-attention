"""Qwen3 with the full delta-attention machinery — a thin subclass layer
over the vendored Llama implementation (llama.py).

Why this is safe to be thin: every custom attention path (delta_forward,
sdpa_rectangle_forward, delta_decode_forward, flex_delta_train_forward)
receives query/key/value AFTER projection and rope. Qwen3's only attention
difference vs Llama is per-head RMSNorm on q and k applied BETWEEN the
projection and rope — upstream of everything the delta machinery touches —
so overriding the _qkv hook is the whole port. MLP (SwiGLU), RMSNorm,
rotary, GQA repeat_kv, decoder-layer wiring and weight names are otherwise
identical; Qwen3 checkpoints add only self_attn.{q,k}_norm.weight keys.

Verified against transformers==4.51.3 (the env pin; Qwen3 support landed
in 4.51.0). Dense Qwen3 (8B/14B/32B) only — the 2026 Qwen3.5/3.6 hybrids
use Gated DeltaNet linear attention, which this mechanism does not apply
to (no softmax rows to correct).
"""

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from .llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
)


class Qwen3DeltaAttention(LlamaAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        # per-head q/k RMSNorm over head_dim (the Qwen3 addition)
        self.q_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _qkv(self, hidden_states, hidden_shape):
        q = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        return q, k, v


class Qwen3DeltaDecoderLayer(LlamaDecoderLayer):
    attn_cls = Qwen3DeltaAttention


class Qwen3DeltaModel(LlamaModel):
    config_class = Qwen3Config
    layer_cls = Qwen3DeltaDecoderLayer


class Qwen3ForCausalLM(LlamaForCausalLM):
    config_class = Qwen3Config
    model_cls = Qwen3DeltaModel
