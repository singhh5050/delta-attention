"""Gate 1 — math identity tests (WP-0 §3).

All tests run on 1 GPU in bf16 against meta-llama/Llama-3.1-8B-Instruct
attention modules with random inputs where possible. Any failure here blocks
Gate 2 and everything downstream.

T1  qsa_kernel correctness against a row-restricted causal SDPA reference.
T2  anchor exactness: delta-corrected output == dense at anchor + tail rows.
T3  gamma=1 => dense everywhere (strongest end-to-end indexing test).
T4  window >= context => delta ~ 0 and corrected ~ dense.
T5  full-model logit sanity: delta(gamma=1) vs dense FA2 top-1 agreement.

Run: pytest tests/test_math_identities.py -v
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="Gate 1 requires a CUDA GPU")

MODEL_STR = "meta-llama/Llama-3.1-8B-Instruct"


def _summary(name, cos, diff=None):
    msg = f"{name}: cos min={cos.min().item():.6f} mean={cos.mean().item():.6f} " \
          f"p1={torch.quantile(cos.float(), 0.01).item():.6f}"
    if diff is not None:
        msg += f" max_abs_diff={diff:.5f}"
    print(msg, flush=True)


@pytest.fixture(scope="session")
def model_and_tokenizer():
    from delta_attention.config import Config
    from delta_attention.sample import init_model

    config = Config()
    config.model_str = MODEL_STR
    config.attn_implementation = "window"
    config.mode = "delta"
    config.attn_implementation_original = config.attn_implementation
    model, tokenizer = init_model(config)
    model.config.attn_implementation_original = config.attn_implementation
    model.config.mode = config.mode
    model.config.delta_lambda = config.delta_lambda
    model.config.sliding_window = config.sliding_window
    model.eval()
    return model.cuda(), tokenizer


# ---------------------------------------------------------------------------
# T1 — qsa_kernel correctness
# ---------------------------------------------------------------------------

@cuda_only
def test_t1_qsa_kernel_correctness():
    from delta_attention.delta_kernel import attention as qsa_kernel

    torch.manual_seed(0)
    b, h, s, d = 1, 32, 4096, 128
    device = "cuda"
    q = torch.randn(b, h, s, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(b, h, s, d, device=device, dtype=torch.bfloat16)
    v = torch.randn(b, h, s, d, device=device, dtype=torch.bfloat16)
    scaling = d ** -0.5

    # 37 sorted query rows including the endpoints.
    mid = torch.randperm(s - 2, generator=torch.Generator().manual_seed(1))[:35] + 1
    idx = torch.cat([torch.tensor([0, s - 1]), mid]).unique().sort().values.to(device)

    out = qsa_kernel(q[:, :, idx], k, v, idx.unsqueeze(0), scaling)  # [b, h, n, d]

    # Reference: causal SDPA restricted to the selected query rows.
    key_pos = torch.arange(s, device=device)
    mask = (key_pos.unsqueeze(0) <= idx.unsqueeze(1)).view(1, 1, len(idx), s)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q[:, :, idx], k, v, attn_mask=mask, scale=scaling
    )

    cos = torch.nn.functional.cosine_similarity(out.float(), ref.float(), dim=-1)  # [b, h, n]
    max_diff = (out.float() - ref.float()).abs().max().item()
    _summary("T1", cos, max_diff)
    assert cos.min().item() > 0.999, f"T1: per-row cos {cos.min().item():.6f} <= 0.999"
    assert max_diff < 2e-2, f"T1: max abs diff {max_diff:.4f} >= 2e-2 (beyond bf16 noise)"


# ---------------------------------------------------------------------------
# T2 — anchor exactness at s=8192
# ---------------------------------------------------------------------------

@cuda_only
def test_t2_anchor_exactness(model_and_tokenizer):
    from delta_attention.validation import anchor_indices, delta_and_dense, row_cos

    model, _ = model_and_tokenizer
    s, gamma, window = 8192, 64, 2048
    out, dense = delta_and_dense(model, s=s, gamma=gamma, window=window, mode="delta")
    cos = row_cos(out, dense)[0]
    anchors, tail = anchor_indices(s, gamma)

    _summary("T2 anchors", cos[anchors])
    _summary("T2 tail", cos[tail])
    assert cos[anchors].min().item() > 0.999, "T2: anchor rows do not match dense"
    assert cos[tail].min().item() > 0.999, "T2: cut_n tail rows (dense by construction) do not match dense"


# ---------------------------------------------------------------------------
# T3 — gamma=1 => dense everywhere
# ---------------------------------------------------------------------------

@cuda_only
def test_t3_gamma1_is_dense(model_and_tokenizer):
    from delta_attention.validation import delta_and_dense, row_cos

    model, _ = model_and_tokenizer
    out, dense = delta_and_dense(model, s=8192, gamma=1, window=2048, mode="delta")
    cos = row_cos(out, dense)[0]
    _summary("T3", cos)
    assert cos.mean().item() > 0.999, f"T3: mean cos {cos.mean().item():.6f} <= 0.999"
    assert torch.quantile(cos.float(), 0.01).item() > 0.995, "T3: p1 cos <= 0.995 — indexing is broken"


# ---------------------------------------------------------------------------
# T4 — window >= context => delta ~ 0
# ---------------------------------------------------------------------------

@cuda_only
def test_t4_full_window_zero_delta(model_and_tokenizer):
    from delta_attention.delta_kernel import attention as qsa_kernel
    from delta_attention.llama import repeat_kv
    from delta_attention.validation import (anchor_indices, make_random_qkv,
                                            row_cos, run_delta_forward)

    model, _ = model_and_tokenizer
    s, gamma = 8192, 64
    window = s  # window covers everything (plus 1024 sinks)
    device = next(model.parameters()).device
    q, k, v = make_random_qkv(s, 0, device)
    scaling = 128 ** -0.5

    sparse = run_delta_forward(model, q, k, v, mode="sparse-only", gamma=gamma, window=window)
    anchors, _ = anchor_indices(s, gamma)
    anchors = anchors.to(device)
    kk = repeat_kv(k, 4)
    vv = repeat_kv(v, 4)
    with torch.no_grad():
        dense_sel = qsa_kernel(q[:, :, anchors], kk, vv, anchors.unsqueeze(0), scaling).transpose(1, 2)

    delta = dense_sel.float() - sparse[:, anchors].float()
    rel = delta.norm().item() / dense_sel.float().norm().item()
    print(f"T4: ||delta||/||output|| = {rel:.6f}", flush=True)
    assert rel < 1e-2, f"T4: delta not ~0 when window covers context (rel {rel:.4f})"

    from delta_attention.validation import delta_and_dense

    out, dense = delta_and_dense(model, s=s, gamma=gamma, window=window, mode="delta")
    cos = row_cos(out, dense)[0]
    _summary("T4 corrected-vs-dense", cos)
    assert cos.mean().item() > 0.999, "T4: corrected output does not match dense with full window"


# ---------------------------------------------------------------------------
# T5 — full-model logit sanity (catches layer-wiring bugs T2–T4 can miss)
# ---------------------------------------------------------------------------

@cuda_only
def test_t5_full_model_logit_sanity(model_and_tokenizer):
    from delta_attention.validation import _config_override, build_niah_prompt

    model, tokenizer = model_and_tokenizer
    if hasattr(model, "no_lm_head"):
        delattr(model, "no_lm_head")

    prompt = build_niah_prompt(tokenizer, filler_tokens=3800)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
    assert 3500 <= input_ids.size(1) <= 4200, f"prompt length {input_ids.size(1)} not ~4K"

    def top1(attn_impl, mode, gamma):
        with _config_override(
            model.config, _attn_implementation=attn_impl, mode=mode, delta_lambda=gamma, sliding_window=2048
        ):
            with torch.no_grad():
                logits = model(input_ids).logits
        return logits.argmax(dim=-1)[0]

    pred_delta = top1("window", "delta", 1)
    pred_dense = top1("flash_attention_2", "delta", 1)

    agree = (pred_delta == pred_dense).float().mean().item()
    print(f"T5: top-1 agreement {agree:.4f} over {input_ids.size(1)} positions", flush=True)
    assert agree >= 0.95, f"T5: top-1 next-token agreement {agree:.4f} < 0.95 — layer wiring suspect"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
