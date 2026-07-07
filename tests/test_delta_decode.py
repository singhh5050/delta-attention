"""WP-3 Gate-1 extensions (GPU): T6-T9 per docs/WP3_delta_decode.md."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

cuda_only = pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA GPU")

DECODE_FIELDS = ("decode_mode", "gamma_dec", "refresh_policy",
                 "drift_threshold", "gamma_dec_max", "log_drift")


@pytest.fixture(scope="session")
def model_and_tokenizer():
    from delta_attention.config import Config
    from delta_attention.sample import init_model

    config = Config()
    config.attn_implementation = "window"
    config.mode = "delta"
    config.attn_implementation_original = config.attn_implementation
    model, tokenizer = init_model(config)
    for f in ("mode", "delta_lambda", "sliding_window", "log_drift") + DECODE_FIELDS:
        setattr(model.config, f, getattr(config, f))
    model.config.attn_implementation_original = config.attn_implementation
    model.eval()
    return model.cuda(), tokenizer


class _decode_override:
    def __init__(self, config, **values):
        self.config, self.values, self.saved = config, values, {}

    def __enter__(self):
        for k, v in self.values.items():
            self.saved[k] = getattr(self.config, k)
            setattr(self.config, k, v)

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(self.config, k, v)
        return False


def _reset(model):
    from delta_attention.llama import LlamaAttention

    for m in model.modules():
        if isinstance(m, LlamaAttention):
            m._dec_state = None
            m._dec_drift_points = None


def _decode_step(model, q, k, v):
    module = model.model.layers[0].self_attn
    with torch.no_grad():
        out, _ = module.sdpa_rectangle_forward(
            module, q, k, v, scaling=module.head_dim ** -0.5)
    return out


def _rand_step(seed, n_keys=4096, device="cuda"):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(1, 32, 1, 128, generator=gen).to(device=device, dtype=torch.bfloat16)
    k = torch.randn(1, 8, n_keys, 128, generator=gen).to(device=device, dtype=torch.bfloat16)
    v = torch.randn(1, 8, n_keys, 128, generator=gen).to(device=device, dtype=torch.bfloat16)
    return q, k, v


def _generate_ids(model, tokenizer, prompt, max_new_tokens=32):
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return ids[0, inputs["input_ids"].size(1):].tolist()


# ---------------------------------------------------------------------------
# T6 — anchor-step output equals the dense rectangle output
# ---------------------------------------------------------------------------

@cuda_only
def test_t6_anchor_step_exactness(model_and_tokenizer):
    model, _ = model_and_tokenizer
    q, k, v = _rand_step(0, n_keys=8192)  # > sink+window so sparse != dense

    _reset(model)
    with _decode_override(model.config, decode_mode="delta", gamma_dec=8,
                          refresh_policy="fixed"):
        out_anchor = _decode_step(model, q, k, v)  # fresh state => anchor
    _reset(model)
    with _decode_override(model.config, decode_mode="dense"):
        out_dense = _decode_step(model, q, k, v)

    cos = torch.nn.functional.cosine_similarity(
        out_anchor.flatten().float(), out_dense.flatten().float(), dim=0).item()
    print(f"T6: anchor-vs-dense cos {cos:.6f}")
    assert cos > 0.999


# ---------------------------------------------------------------------------
# T7 — gamma_dec=1 => generation token-ids identical to dense decode
# ---------------------------------------------------------------------------

@cuda_only
def test_t7_gamma1_matches_dense_decode(model_and_tokenizer):
    from delta_attention.validation import NIAH_UUID, build_niah_prompt

    model, tokenizer = model_and_tokenizer
    prompt = build_niah_prompt(tokenizer, filler_tokens=3800)

    _reset(model)
    with _decode_override(model.config, decode_mode="dense"):
        ids_dense = _generate_ids(model, tokenizer, prompt)
    _reset(model)
    with _decode_override(model.config, decode_mode="delta", gamma_dec=1,
                          refresh_policy="fixed"):
        ids_delta = _generate_ids(model, tokenizer, prompt)

    assert ids_delta == ids_dense, (
        f"T7: gamma_dec=1 diverged from dense decode\n{ids_dense}\n{ids_delta}")
    assert NIAH_UUID in tokenizer.decode(ids_delta)


# ---------------------------------------------------------------------------
# T8 — forced zero delta => non-anchor step equals pure sparse decode
# ---------------------------------------------------------------------------

@cuda_only
def test_t8_zero_delta_is_sparse(model_and_tokenizer):
    model, _ = model_and_tokenizer
    module = model.model.layers[0].self_attn
    q, k, v = _rand_step(1, n_keys=8192)

    _reset(model)
    with _decode_override(model.config, decode_mode="delta", gamma_dec=10_000,
                          refresh_policy="fixed", gamma_dec_max=100_000):
        _decode_step(model, q, k, v)  # anchor, populates state
        module._dec_state["last_delta"] = torch.zeros_like(
            module._dec_state["last_delta"])
        q2, _, _ = _rand_step(2)
        out_zero = _decode_step(model, q2, k, v)
    _reset(model)
    with _decode_override(model.config, decode_mode="sparse"):
        out_sparse = _decode_step(model, q2, k, v)

    assert torch.equal(out_zero, out_sparse), "T8: zero delta != pure sparse decode"


# ---------------------------------------------------------------------------
# T9 — state hygiene: repeated generate calls are identical
# ---------------------------------------------------------------------------

@cuda_only
def test_t9_state_hygiene(model_and_tokenizer):
    from delta_attention.validation import build_niah_prompt

    model, tokenizer = model_and_tokenizer
    prompt = build_niah_prompt(tokenizer, filler_tokens=2000)

    with _decode_override(model.config, decode_mode="delta", gamma_dec=8,
                          refresh_policy="fixed"):
        _reset(model)
        first = _generate_ids(model, tokenizer, prompt)
        second = _generate_ids(model, tokenizer, prompt)   # no manual reset
        _reset(model)
        fresh = _generate_ids(model, tokenizer, prompt)

    assert first == second == fresh, "T9: decode state leaked across generate calls"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
