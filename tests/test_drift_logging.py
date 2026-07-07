"""Drift telemetry tests (delta_interanchor_cos). GPU parts need 1 GPU;
aggregation math is covered offline in tests/test_runner_offline.py.

Run: pytest tests/test_drift_logging.py -v
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
    model.config.attn_implementation_original = config.attn_implementation
    model.config.mode = config.mode
    model.config.delta_lambda = config.delta_lambda
    model.config.sliding_window = config.sliding_window
    model.config.log_drift = False
    model.eval()
    return model.cuda(), tokenizer


@cuda_only
def test_drift_stats_populated_when_enabled(model_and_tokenizer):
    from delta_attention.drift import HIST_BINS
    from delta_attention.validation import delta_and_dense

    model, _ = model_and_tokenizer
    module = model.model.layers[0].self_attn
    module._drift_stats = None

    s, gamma = 8192, 64
    model.config.log_drift = True
    try:
        delta_and_dense(model, s=s, gamma=gamma, window=2048, mode="delta")
    finally:
        model.config.log_drift = False

    stats = module._drift_stats
    assert stats is not None, "log_drift=True did not populate _drift_stats"
    n_anchors = (s - (s % gamma + max(128, gamma))) // gamma
    n_heads = 32
    assert stats["layer"] == 0 and stats["gamma"] == gamma and stats["seq_len"] == s
    assert stats["n"] == (n_anchors - 1) * n_heads
    assert sum(stats["hist"]) == stats["n"], "histogram does not cover all cos values"
    assert len(stats["hist"]) == HIST_BINS
    assert -1.0 <= stats["p10"] <= stats["p50"] <= stats["p90"] <= 1.0
    assert -1.0 <= stats["mean"] <= 1.0
    print(f"drift layer0: mean={stats['mean']:.4f} p10={stats['p10']:.4f} "
          f"p50={stats['p50']:.4f} p90={stats['p90']:.4f} n={stats['n']}")


@cuda_only
def test_no_drift_stats_when_disabled(model_and_tokenizer):
    from delta_attention.validation import delta_and_dense

    model, _ = model_and_tokenizer
    module = model.model.layers[0].self_attn
    module._drift_stats = None
    assert model.config.log_drift is False
    delta_and_dense(model, s=4096, gamma=64, window=2048, mode="delta")
    assert module._drift_stats is None, "log_drift=False must not collect stats"


@cuda_only
def test_flush_writes_sidecar(model_and_tokenizer, tmp_path, monkeypatch):
    """The model_wrapper flush path: stats on modules -> JSONL lines -> cleared."""
    import json

    from delta_attention.model_wrapper import HuggingFaceModel
    from delta_attention.validation import delta_and_dense

    model, _ = model_and_tokenizer
    sidecar = tmp_path / "drift.jsonl"
    monkeypatch.setenv("DELTA_DRIFT_LOG", str(sidecar))

    model.config.log_drift = True
    try:
        delta_and_dense(model, s=4096, gamma=64, window=2048, mode="delta", seed=1)
    finally:
        model.config.log_drift = False

    # borrow the flush implementation without building a full wrapper
    fake = object.__new__(HuggingFaceModel)
    fake.model = model
    fake._flush_drift_stats()

    lines = [json.loads(ln) for ln in sidecar.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1 and lines[0]["layer"] == 0  # only layer 0 ran
    assert model.model.layers[0].self_attn._drift_stats is None, "stats not cleared"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
