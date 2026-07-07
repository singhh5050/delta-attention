"""WP-1 Gate-1 extensions (GPU): T10-T12 per docs/WP1_dynamic_stride.md.

T10 gates everything else in WP-1: with uniform idx the variable-stride
broadcast must reproduce the existing reshape path bit-for-bit.
"""

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
    for f in ("mode", "delta_lambda", "sliding_window", "log_drift", "stride_policy",
              "gamma_min", "gamma_max", "adapt_chunk", "adapt_threshold"):
        setattr(model.config, f, getattr(config, f))
    model.config.attn_implementation_original = config.attn_implementation
    model.eval()
    return model.cuda(), tokenizer


class _stride_override:
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


# ---------------------------------------------------------------------------
# T10 — uniform idx: variable broadcast == reshape path, bit for bit
# ---------------------------------------------------------------------------

@cuda_only
def test_t10_uniform_equals_reshape_path():
    from delta_attention.stride import apply_delta_variable

    torch.manual_seed(0)
    b, s_p, h, d, gamma = 1, 8064, 32, 128, 64
    sparse = torch.randn(b, s_p, h, d, device="cuda", dtype=torch.bfloat16)
    idx = torch.arange(0, s_p, gamma, device="cuda")
    dense_sel = torch.randn(b, idx.numel(), h, d, device="cuda", dtype=torch.bfloat16)

    # the existing delta_forward reshape path, replicated verbatim
    diff = dense_sel - sparse[:, idx]
    ref = (diff.reshape(b, s_p // gamma, 1, h, d)
           + sparse.reshape(b, s_p // gamma, gamma, h, d)).reshape(b, s_p, h, d)

    out = apply_delta_variable(sparse, dense_sel, idx, s_p)
    assert torch.equal(out, ref), "T10: variable-stride broadcast diverges from reshape path"


# ---------------------------------------------------------------------------
# T11 — adaptive with gamma_min == gamma_max == 64 degenerates to fixed
# ---------------------------------------------------------------------------

@cuda_only
def test_t11_adaptive_degenerate_equals_fixed(model_and_tokenizer):
    from delta_attention.validation import make_random_qkv, run_delta_forward

    model, _ = model_and_tokenizer
    device = next(model.parameters()).device
    q, k, v = make_random_qkv(8192, 0, device)

    fixed = run_delta_forward(model, q, k, v, mode="delta", gamma=64, window=2048)
    with _stride_override(model.config, stride_policy="adaptive",
                          gamma_min=64, gamma_max=64, adapt_chunk=4096,
                          adapt_threshold=0.95):
        adaptive = run_delta_forward(model, q, k, v, mode="delta", gamma=64, window=2048)

    assert torch.equal(adaptive, fixed), (
        f"T11: adaptive-degenerate != fixed (max abs diff "
        f"{(adaptive.float() - fixed.float()).abs().max().item():.3e})")


# ---------------------------------------------------------------------------
# T12 — anchor exactness holds under a random non-uniform idx
# ---------------------------------------------------------------------------

@cuda_only
def test_t12_anchor_exactness_variable_idx(model_and_tokenizer):
    from delta_attention.delta_kernel import attention as qsa_kernel
    from delta_attention.llama import repeat_kv
    from delta_attention.stride import apply_delta_variable
    from delta_attention.validation import (_dense_reference, make_random_qkv,
                                            row_cos, run_delta_forward)

    model, _ = model_and_tokenizer
    device = next(model.parameters()).device
    s, s_p = 8192, 8064
    q, k, v = make_random_qkv(s, 0, device)
    scaling = 128 ** -0.5

    sparse = run_delta_forward(model, q, k, v, mode="sparse-only", gamma=64, window=2048)

    gen = torch.Generator().manual_seed(2)
    mid = (torch.randperm(s_p - 1, generator=gen)[:127] + 1).to(device)
    idx = torch.cat([torch.zeros(1, dtype=torch.long, device=device), mid]).unique().sort().values

    kk, vv = repeat_kv(k, 4), repeat_kv(v, 4)
    with torch.no_grad():
        dense_sel = qsa_kernel(q[:, :, idx], kk, vv, idx.unsqueeze(0), scaling).transpose(1, 2)
        dense_ref = _dense_reference(q, k, v, scaling)

    corrected = apply_delta_variable(sparse[:, :s_p], dense_sel, idx, s_p)
    cos = row_cos(corrected, dense_ref[:, :s_p])[0]
    worst = cos[idx].min().item()
    print(f"T12: min anchor-row cos {worst:.6f} over {idx.numel()} non-uniform anchors")
    assert worst > 0.999, "T12: anchor rows do not match dense under variable idx"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
