"""Startup validation gate (WP-0 §4).

Every experiment entrypoint (server, runner, training) must call
``startup_validation`` within its first ~60 seconds. Any failure prints the
reason and exits with status 1. No entrypoint may skip the gate.

The module also hosts the shared numeric helpers used by Gate 1
(tests/test_math_identities.py) so the self-check and the tests exercise the
exact same code path.

Which modes skip which checks
-----------------------------
- anchor exactness: only meaningful when the delta correction is active, i.e.
  mode == "delta" and attn_implementation in {"window", "hip_attention"} and
  stride_policy == "fixed". Skipped (with a printed notice) for
  recompute / sparse-only / flash_attention_2 / mode "none".
- NIAH known-answer generation: asserted to contain the needle for
  mode in {"delta", "none"} (dense-equivalent); for "sparse-only" and
  "recompute" only non-empty output is required.
"""

from __future__ import annotations

import sys
import uuid
from typing import Any, Dict, Iterable, Optional, Tuple

# The mandatory wandb metric set from docs/00_MASTER_PLAN.md. Every run must
# log each of these at least once (zeros/nulls at init are fine).
MANDATORY_WANDB_KEYS = (
    "config_name",
    "mode",
    "gamma",
    "gamma_dec",
    "refresh_policy",
    "context_len",
    "task",
    "n_samples",
    "accuracy",
    "samples_per_sec",
    "prefill_ms_p50",
    "decode_ms_per_token_p50",
    "oom_count",
)

# Config-row schema for experiments.yaml rows (after YAML merge-key resolution).
REQUIRED_ROW_KEYS = {
    "name",
    "mode",
    "attn_implementation",
    "delta_lambda",
    "sliding_window",
    "decode_mode",
    "stride_policy",
    "context_lengths",
    "tasks",
    "n_samples",
    "log_drift",
}
OPTIONAL_ROW_KEYS = {
    "seed",
    "max_new_tokens",
    "expected_anchor",
    "gamma_dec",
    "refresh_policy",
    "drift_threshold",
    "gamma_dec_max",
    "adapt_threshold",
    "gamma_min",
    "gamma_max",
    "adapt_chunk",
    "checkpoint",
}

# Fixed needle for the built-in known-answer NIAH prompt. Never randomize:
# determinism is a rule of engagement.
NIAH_UUID = "3f2c9a17-4b8e-4d26-9c51-7a0e8f6b2d43"
_NIAH_FILLER = "The grass is green. The sky is blue. The sun is yellow. "
_NIAH_QUESTION = (
    "\n\nWhat is the special magic uuid mentioned in the text above? "
    "Answer with the uuid only.\nThe special magic uuid is:"
)


class ValidationError(Exception):
    pass


def _fail(step: str, reason: str) -> None:
    print(f"[startup_validation] FAIL at {step}: {reason}", flush=True)
    sys.exit(1)


# --------------------------------------------------------------------------
# 1. config-row schema
# --------------------------------------------------------------------------

def validate_config_row(row: Dict[str, Any]) -> None:
    """Assert a resolved experiments.yaml row has no missing/extra keys."""
    keys = set(row.keys())
    missing = REQUIRED_ROW_KEYS - keys
    extra = keys - REQUIRED_ROW_KEYS - OPTIONAL_ROW_KEYS
    if missing:
        raise ValidationError(f"config row {row.get('name')!r} missing keys: {sorted(missing)}")
    if extra:
        raise ValidationError(f"config row {row.get('name')!r} has unknown keys: {sorted(extra)}")
    if row["mode"] not in ("delta", "recompute", "sparse-only", "none"):
        raise ValidationError(f"config row {row['name']!r}: invalid mode {row['mode']!r}")
    if row["attn_implementation"] not in ("window", "hip_attention", "flash_attention_2"):
        raise ValidationError(
            f"config row {row['name']!r}: invalid attn_implementation {row['attn_implementation']!r}"
        )


# --------------------------------------------------------------------------
# 2. anchor exactness (shared with Gate 1 T2)
# --------------------------------------------------------------------------

def _dense_reference(query_states, key_states, value_states, scaling):
    """Dense causal attention reference. Returns [b, s, h, d] like delta_forward."""
    import torch
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    from .llama import repeat_kv

    n_groups = query_states.size(1) // key_states.size(1)
    k = repeat_kv(key_states, n_groups)
    v = repeat_kv(value_states, n_groups)
    out = sdpa(query_states, k, v, is_causal=True, scale=scaling)
    return out.transpose(1, 2).contiguous()


def make_random_qkv(s: int, seed: int, device, n_heads: int = 32, n_kv_heads: int = 8, head_dim: int = 128):
    """Random post-RoPE q/k/v in the shapes delta_forward expects."""
    import torch

    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(1, n_heads, s, head_dim, generator=gen, dtype=torch.float32)
    k = torch.randn(1, n_kv_heads, s, head_dim, generator=gen, dtype=torch.float32)
    v = torch.randn(1, n_kv_heads, s, head_dim, generator=gen, dtype=torch.float32)
    return (
        q.to(device=device, dtype=torch.bfloat16),
        k.to(device=device, dtype=torch.bfloat16),
        v.to(device=device, dtype=torch.bfloat16),
    )


class _config_override:
    """Temporarily set delta-path fields on model.config; always restore."""

    _FIELDS = ("_attn_implementation", "mode", "delta_lambda", "sliding_window")

    def __init__(self, config, **values):
        self.config = config
        self.values = values
        self.saved = {}

    def __enter__(self):
        for f in self._FIELDS:
            self.saved[f] = getattr(self.config, f)
            if f in self.values:
                setattr(self.config, f, self.values[f])
        return self.config

    def __exit__(self, *exc):
        for f, v in self.saved.items():
            setattr(self.config, f, v)
        return False


def run_delta_forward(model, q, k, v, *, mode: str, gamma: int, window: int, layer: int = 0):
    """Run LlamaAttention.delta_forward on crafted q/k/v. Returns [b, s, h, d]."""
    import torch

    module = model.model.layers[layer].self_attn
    s = q.size(2)
    device = q.device
    position_ids = torch.arange(s, device=device).unsqueeze(0)
    # rope sin/cos are threaded through to hip-attn args but not re-applied
    # (need_apply_rope is False on the window path); real values keep shapes honest.
    cos, sin = model.model.rotary_emb(v.transpose(1, 2), position_ids)
    scaling = module.head_dim ** -0.5

    with _config_override(
        model.config, _attn_implementation="window", mode=mode, delta_lambda=gamma, sliding_window=window
    ):
        with torch.no_grad():
            out, _ = module.delta_forward(
                module,
                q,
                k,
                v,
                None,
                scaling=scaling,
                position_ids=position_ids,
                rope_sin=sin.squeeze(0),
                rope_cos=cos.squeeze(0),
            )
    return out


def delta_and_dense(model, *, s: int, gamma: int, window: int, mode: str = "delta", seed: int = 0):
    """Delta-corrected output and dense reference on identical random inputs."""
    device = next(model.parameters()).device
    q, k, v = make_random_qkv(s, seed, device)
    out = run_delta_forward(model, q, k, v, mode=mode, gamma=gamma, window=window)
    scaling = model.model.layers[0].self_attn.head_dim ** -0.5
    import torch

    with torch.no_grad():
        dense = _dense_reference(q, k, v, scaling)
    return out, dense


def row_cos(a, b):
    """Per-row cosine similarity over flattened (h, d). Inputs [b, s, h, d]."""
    import torch

    a32 = a.reshape(a.size(0), a.size(1), -1).float()
    b32 = b.reshape(b.size(0), b.size(1), -1).float()
    return torch.nn.functional.cosine_similarity(a32, b32, dim=-1)


def anchor_indices(s: int, gamma: int):
    """Anchor rows and dense-tail rows exactly as delta_forward computes them."""
    import torch

    cut_n = s % gamma + max(128, gamma)
    s_p = s - cut_n
    anchors = torch.arange(0, s_p, step=gamma)
    tail = torch.arange(s_p, s)
    return anchors, tail


def anchor_exactness_check(model, *, gamma: int, window: int, s: int = 2048, min_cos: float = 0.999) -> float:
    """T2 fast self-check: corrected output must equal dense at anchor rows."""
    out, dense = delta_and_dense(model, s=s, gamma=gamma, window=window, mode="delta")
    cos = row_cos(out, dense)[0]
    anchors, tail = anchor_indices(s, gamma)
    worst = min(cos[anchors].min().item(), cos[tail].min().item())
    if worst < min_cos:
        raise ValidationError(
            f"anchor exactness: min anchor/tail row cos {worst:.6f} < {min_cos} (s={s}, gamma={gamma}, window={window})"
        )
    return worst


# --------------------------------------------------------------------------
# 3. built-in known-answer NIAH generation
# --------------------------------------------------------------------------

def build_niah_prompt(tokenizer, filler_tokens: int = 2048) -> str:
    """Fixed-UUID needle at ~50% depth of ~filler_tokens of filler text."""
    unit = _NIAH_FILLER
    unit_tokens = max(len(tokenizer.encode(unit, add_special_tokens=False)), 1)
    n_units = max(filler_tokens // unit_tokens, 2)
    half = n_units // 2
    needle = f"The special magic uuid is {NIAH_UUID}. "
    return unit * half + needle + unit * (n_units - half) + _NIAH_QUESTION


def niah_known_answer_check(model, tokenizer, mode: str, max_new_tokens: int = 48) -> str:
    import torch

    prompt = build_niah_prompt(tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None
        )
    text = tokenizer.decode(out_ids[0, inputs["input_ids"].size(1):], skip_special_tokens=True)
    if mode in ("delta", "none"):
        if NIAH_UUID not in text:
            raise ValidationError(f"NIAH known-answer: needle not retrieved (mode={mode}); got: {text!r}")
    else:
        if not text.strip():
            raise ValidationError(f"NIAH known-answer: empty generation (mode={mode})")
    return text


# --------------------------------------------------------------------------
# 4. wandb key coverage
# --------------------------------------------------------------------------

def assert_wandb_keys_logged(logged_keys: Iterable[str]) -> None:
    missing = set(MANDATORY_WANDB_KEYS) - set(logged_keys)
    if missing:
        raise ValidationError(f"mandatory wandb keys never logged: {sorted(missing)}")


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------

def startup_validation(
    config_row: Dict[str, Any],
    *,
    model=None,
    tokenizer=None,
    logged_wandb_keys: Optional[Iterable[str]] = None,
    smoke: bool = False,
) -> None:
    """Run the WP-0 §4 gate. Prints reason and exits(1) on any failure.

    ``model``/``tokenizer`` are optional so client-side entrypoints (the
    runner) can validate config rows without loading the model; entrypoints
    that hold the model (server, training) must pass it to get the numeric
    and generation self-checks.
    """
    try:
        validate_config_row(config_row)
    except ValidationError as e:
        _fail("config-row schema", str(e))

    mode = config_row["mode"]
    delta_active = (
        mode == "delta"
        and config_row["attn_implementation"] in ("window", "hip_attention")
        and config_row.get("stride_policy", "fixed") == "fixed"
    )

    if model is not None:
        if delta_active:
            try:
                worst = anchor_exactness_check(
                    model,
                    gamma=int(config_row["delta_lambda"]),
                    window=int(config_row["sliding_window"]),
                    s=2048,
                )
                print(f"[startup_validation] anchor exactness OK (min row cos {worst:.6f})", flush=True)
            except ValidationError as e:
                _fail("anchor exactness", str(e))
            except Exception as e:  # noqa: BLE001 - the gate must report, not raise
                _fail("anchor exactness", f"unexpected error: {e!r}")
        else:
            print(
                f"[startup_validation] skipping anchor exactness (mode={mode}, "
                f"attn={config_row['attn_implementation']}, stride={config_row.get('stride_policy')})",
                flush=True,
            )

        if tokenizer is None:
            _fail("NIAH known-answer", "model provided without tokenizer")
        try:
            text = niah_known_answer_check(model, tokenizer, mode)
            print(f"[startup_validation] NIAH known-answer OK: {text.strip()[:60]!r}", flush=True)
        except ValidationError as e:
            _fail("NIAH known-answer", str(e))
        except Exception as e:  # noqa: BLE001
            _fail("NIAH known-answer", f"unexpected error: {e!r}")

    if logged_wandb_keys is not None:
        try:
            assert_wandb_keys_logged(logged_wandb_keys)
        except ValidationError as e:
            _fail("wandb key coverage", str(e))

    print(f"[startup_validation] PASS for config {config_row['name']!r}", flush=True)


def server_startup_validation(config, model, tokenizer) -> None:
    """Model-side gate for server_hf.py (which holds a Config, not a YAML row).

    Runs the anchor-exactness self-check with the live window/gamma and the
    built-in NIAH known-answer generation. Exits(1) on failure.
    """
    attn = config.attn_implementation
    # FA2 ignores mode entirely — treat as dense ("none") for check selection.
    mode = "none" if attn == "flash_attention_2" else config.mode
    delta_active = mode == "delta" and attn in ("window", "hip_attention")

    if delta_active:
        try:
            worst = anchor_exactness_check(
                model, gamma=int(config.delta_lambda), window=int(config.sliding_window), s=2048
            )
            print(f"[startup_validation] anchor exactness OK (min row cos {worst:.6f})", flush=True)
        except ValidationError as e:
            _fail("anchor exactness", str(e))
        except Exception as e:  # noqa: BLE001
            _fail("anchor exactness", f"unexpected error: {e!r}")
    else:
        print(f"[startup_validation] skipping anchor exactness (mode={mode}, attn={attn})", flush=True)

    try:
        text = niah_known_answer_check(model, tokenizer, mode)
        print(f"[startup_validation] NIAH known-answer OK: {text.strip()[:60]!r}", flush=True)
    except ValidationError as e:
        _fail("NIAH known-answer", str(e))
    except Exception as e:  # noqa: BLE001
        _fail("NIAH known-answer", f"unexpected error: {e!r}")

    print("[startup_validation] server gate PASS", flush=True)
