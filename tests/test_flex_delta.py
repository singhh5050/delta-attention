"""WP-2 gates (GPU): T13 equivalence (blocks all training) and T14 gradient
sanity, per docs/WP2_trainable_delta.md."""

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


@pytest.fixture(scope="session")
def model_and_tokenizer():
    from delta_attention.config import Config
    from delta_attention.sample import init_model

    config = Config()
    config.attn_implementation = "window"
    config.mode = "delta"
    config.attn_implementation_original = config.attn_implementation
    model, tokenizer = init_model(config)
    for f in ("mode", "delta_lambda", "sliding_window", "log_drift"):
        setattr(model.config, f, getattr(config, f))
    model.config.attn_implementation_original = config.attn_implementation
    model.eval()
    return model.cuda(), tokenizer


# ---------------------------------------------------------------------------
# T13 — equivalence with the inference path (blocks all training)
# ---------------------------------------------------------------------------

@cuda_only
def test_t13_train_matches_inference(model_and_tokenizer):
    from delta_attention.train.flex_delta import delta_forward_train
    from delta_attention.validation import make_random_qkv, row_cos, run_delta_forward

    model, _ = model_and_tokenizer
    device = next(model.parameters()).device
    s, gamma, window = 8192, 64, 2048
    q, k, v = make_random_qkv(s, 0, device)

    inference = run_delta_forward(model, q, k, v, mode="delta", gamma=gamma, window=window)
    with torch.no_grad():
        train = delta_forward_train(q, k, v, gamma=gamma, window=window)

    cos = row_cos(train.float(), inference.float())[0]
    p1 = torch.quantile(cos.float(), 0.01).item()
    print(f"T13: mean cos {cos.mean().item():.6f} p1 {p1:.6f} min {cos.min().item():.6f}")
    assert cos.mean().item() > 0.999, (
        "T13: train forward diverges from inference — reconcile the flex MASK "
        "to hip-attn's block-granular window semantics, not the tolerance")
    assert p1 > 0.995, "T13: per-row p1 below gate"


@cuda_only
def test_t13_gamma1_matches_dense(model_and_tokenizer):
    from delta_attention.train.flex_delta import delta_forward_train
    from delta_attention.validation import _dense_reference, make_random_qkv, row_cos

    model, _ = model_and_tokenizer
    device = next(model.parameters()).device
    s = 4096
    q, k, v = make_random_qkv(s, 1, device)
    with torch.no_grad():
        train = delta_forward_train(q, k, v, gamma=1, window=2048)
        dense = _dense_reference(q, k, v, 128 ** -0.5)
    cos = row_cos(train.float(), dense.float())[0]
    print(f"T13/gamma1: mean cos {cos.mean().item():.6f} p1 "
          f"{torch.quantile(cos.float(), 0.01).item():.6f}")
    assert cos.mean().item() > 0.999
    assert torch.quantile(cos.float(), 0.01).item() > 0.995


# ---------------------------------------------------------------------------
# T14 — gradients flow; anchor rows concentrate gradient
# ---------------------------------------------------------------------------

@cuda_only
def test_t14_gradient_sanity():
    from delta_attention.train.flex_delta import anchor_grad_ratio, delta_forward_train

    torch.manual_seed(0)
    device, s, gamma = "cuda", 4096, 64
    q = torch.randn(1, 32, s, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 8, s, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 8, s, 128, device=device, dtype=torch.bfloat16, requires_grad=True)

    out = delta_forward_train(q, k, v, gamma=gamma, window=2048)
    loss = out.float().pow(2).mean()
    loss.backward()

    for name, t in (("q", q), ("k", k), ("v", v)):
        assert t.grad is not None and torch.isfinite(t.grad.float()).all() \
            and t.grad.abs().sum() > 0, f"T14: no/invalid gradient for {name}"

    ratio = anchor_grad_ratio(q.grad, gamma)
    print(f"T14: anchor_grad_ratio {ratio:.3f}")
    assert ratio > 1.0, (
        f"T14: expected gradient concentration at anchor rows, got ratio {ratio:.3f}")


@cuda_only
def test_t14_detach_delta_ablation():
    from delta_attention.train.flex_delta import delta_forward_train

    torch.manual_seed(0)
    device, s, gamma = "cuda", 4096, 64
    q = torch.randn(1, 32, s, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 8, s, 128, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, 8, s, 128, device=device, dtype=torch.bfloat16)

    out = delta_forward_train(q, k, v, gamma=gamma, window=2048, detach_delta=True)
    out.float().pow(2).mean().backward()
    assert q.grad is not None and q.grad.abs().sum() > 0, \
        "T14/detach: sparse-path gradient must still flow"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
