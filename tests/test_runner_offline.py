"""Offline unit tests for the WP-0 runner plumbing (no GPU, no network).

Covers: YAML merge-key resolution, group/--configs expansion, smoke
overrides, the mode-"none" -> FA2 server-flag mapping, unsupported-config
detection via Config-field introspection, config-row schema validation, and
results.csv row shape.

Run: pytest tests/test_runner_offline.py -v
"""

import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.run_matrix import (CSV_COLUMNS, IMPLICIT_ROW_DEFAULTS,  # noqa: E402
                             append_result_row, apply_smoke_overrides,
                             base_result_row, build_server_cmd,
                             ensure_results_csv, load_configs,
                             load_experiments, resolve_config_names,
                             unsupported_reason)

# The real Config fields as of upstream HEAD (delta_attention/config.py).
# Kept static here so these tests never import torch/hip_attn.
SERVER_FIELDS = frozenset({
    "model_str", "attn_implementation", "mode", "hip_attn_args", "port",
    "host", "trust_remote_code", "temperature", "top_k", "top_p",
    "stop_words", "max_new_tokens", "repetition_penalty", "delta_lambda",
    "sliding_window", "log_drift", "checkpoint",
})


def test_yaml_merge_keys_resolve():
    """experiments.yaml is dumped RESOLVED (no merge anchors since the
    live-arms prune) — assert the equivalent semantics: every config
    carries the default fields run_matrix expects."""
    doc = load_experiments()
    by_name = {c["name"]: c for c in doc["configs"]}
    assert by_name["p32_delta_g64"]["sliding_window"] == 2048
    assert by_name["p32_delta_g64"]["attn_implementation"] == "window"
    assert by_name["p32_dense_fa2"]["mode"] == "none"
    assert by_name["p32_ft32k_delta"]["checkpoint"] == \
        "checkpoints/pilot_delta_32k"
    for c in doc["configs"]:
        for field in ("mode", "delta_lambda", "context_lengths", "tasks"):
            assert field in c, (c["name"], field)


def test_group_and_name_resolution():
    doc = load_experiments()
    names = resolve_config_names(doc, "ruler32k")
    assert names == doc["groups"]["ruler32k"]
    # mixing group + name dedupes while preserving order:
    mixed = resolve_config_names(doc, "p32_dense_fa2,ruler32k")
    assert mixed[0] == "p32_dense_fa2"
    assert len(mixed) == len(set(mixed)) == len(doc["groups"]["ruler32k"])
    assert set(resolve_config_names(doc, "all")) == {c["name"] for c in doc["configs"]}
    with pytest.raises(SystemExit):
        resolve_config_names(doc, "no_such_config")


def test_smoke_overrides():
    rows_full = load_configs("p32_delta_g64", smoke=False)
    rows_smoke = load_configs("p32_delta_g64", smoke=True)
    full, smoke = rows_full[0], rows_smoke[0]
    assert full["context_lengths"] == [32768]  # arm-level override
    assert smoke["context_lengths"] == [4096]
    assert smoke["n_samples"] == 50
    assert set(smoke["tasks"]) == {"niah_single_1", "niah_multikey_2", "qa_1"}
    # nothing else may change:
    for k in full:
        if k not in ("context_lengths", "tasks", "n_samples"):
            assert smoke[k] == full[k], k
    with pytest.raises(RuntimeError, match="smoke_overrides may only touch"):
        apply_smoke_overrides(full, {"delta_lambda": 1})


def test_implicit_defaults_applied():
    row = load_configs("p32_delta_g64", smoke=False)[0]
    for k, v in IMPLICIT_ROW_DEFAULTS.items():
        assert row[k] == v


def test_mode_none_maps_to_fa2_delta():
    row = load_configs("p32_dense_fa2", smoke=False)[0]
    cmd, passed, unknown, notes = build_server_cmd(row, 1234, SERVER_FIELDS)
    assert passed["mode"] == "delta"
    assert passed["attn_implementation"] == "flash_attention_2"
    assert any("mode 'none'" in n for n in notes)
    assert "--mode" in cmd and cmd[cmd.index("--mode") + 1] == "delta"
    # dash-style flags, matching run-server-hf.sh usage:
    assert "--attn-implementation" in cmd
    # mode none with a non-FA2 implementation is a hard error:
    bad = dict(row, attn_implementation="window")
    with pytest.raises(RuntimeError, match="mode 'none'"):
        build_server_cmd(bad, 1234, SERVER_FIELDS)


def test_unknown_keys_recorded_not_passed():
    row = load_configs("p32_delta_g64", smoke=False)[0]
    cmd, passed, unknown, _ = build_server_cmd(row, 1234, SERVER_FIELDS)
    # decode_mode / stride_policy are not (yet) Config fields:
    assert "decode_mode" in unknown and "stride_policy" in unknown
    assert "--decode-mode" not in cmd
    assert passed["delta_lambda"] == 64 and "--delta-lambda" in cmd


def test_bool_flags_are_presence_only():
    # argparse_dataclass bools are store_true flags: "--log-drift True" is a
    # parse error, so True -> bare flag, False -> omitted entirely.
    on = load_configs("p32_delta_g64", smoke=False)[0]   # log_drift: true
    off = load_configs("p32_ft32k_delta", smoke=False)[0]  # log_drift: false
    cmd_on, passed_on, _, _ = build_server_cmd(on, 1234, SERVER_FIELDS)
    cmd_off, passed_off, _, _ = build_server_cmd(off, 1234, SERVER_FIELDS)
    assert passed_on["log_drift"] is True and passed_off["log_drift"] is False
    assert "--log-drift" in cmd_on
    nxt = cmd_on.index("--log-drift") + 1
    assert nxt == len(cmd_on) or cmd_on[nxt].startswith("--"), \
        "bool flag must not take a value"
    assert "--log-drift" not in cmd_off and "False" not in cmd_off


def test_aggregate_drift():
    from delta_attention.drift import HIST_BINS, aggregate_drift, hist_bin_edges

    line = {"layer": 0, "gamma": 64, "seq_len": 8192, "n": 100, "mean": 0.9,
            "p10": 0.8, "p50": 0.9, "p90": 0.95, "hist_bins": HIST_BINS,
            "hist_range": [-1.0, 1.0], "hist": [0] * (HIST_BINS - 1) + [100]}
    other = dict(line, n=300, mean=0.5, hist=[0] * (HIST_BINS - 1) + [300])
    layer1 = dict(line, layer=1)

    agg = aggregate_drift([line, other, layer1])
    assert set(agg) == {0, 1}
    assert agg[0]["n"] == 400 and agg[0]["requests"] == 2
    assert abs(agg[0]["mean"] - (0.9 * 100 + 0.5 * 300) / 400) < 1e-12
    assert agg[0]["hist"][-1] == 400 and sum(agg[0]["hist"]) == 400
    assert agg[1]["n"] == 100
    assert len(hist_bin_edges()) == HIST_BINS + 1

    import pytest as _pytest
    with _pytest.raises(ValueError, match="mixed-version"):
        aggregate_drift([dict(line, hist_bins=10, hist=[0] * 10)])


def test_unsupported_detection():
    """Every LIVE arm must be runnable against the server's Config surface;
    a synthetic row with a field the server lacks must be flagged (the
    WP-era arms that exercised real unsupported fields are retired)."""
    rows = {r["name"]: r for r in load_configs("all", smoke=False)}
    for name, row in rows.items():
        assert unsupported_reason(row, SERVER_FIELDS) is None, name
    # negative case: a MEANINGFUL non-default field the server surface
    # lacks (arbitrary unknown keys go through build_server_cmd's unknown
    # bucket instead — tested separately)
    fake = dict(rows["p32_delta_g64"], name="fake", decode_mode="delta",
                gamma_dec=16)
    assert unsupported_reason(fake, SERVER_FIELDS) is not None

def test_validation_row_schema():
    from delta_attention.validation import ValidationError, validate_config_row

    for row in load_configs("all", smoke=False):
        validate_config_row(row)  # every real row must pass
    good = load_configs("p32_delta_g64", smoke=False)[0]
    with pytest.raises(ValidationError, match="missing"):
        validate_config_row({k: v for k, v in good.items() if k != "mode"})
    with pytest.raises(ValidationError, match="unknown keys"):
        validate_config_row({**good, "typo_key": 1})
    with pytest.raises(ValidationError, match="invalid mode"):
        validate_config_row({**good, "mode": "bogus"})


def test_results_csv_row_shape(tmp_path):
    path = tmp_path / "results.csv"
    ensure_results_csv(path)
    row = base_result_row(load_configs("p32_delta_g64", smoke=False)[0])
    row.update(status="ok", accuracy=96.5)
    append_result_row(row, path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert list(rows[0]) == CSV_COLUMNS
    assert rows[0]["config_name"] == "p32_delta_g64"
    assert rows[0]["status"] == "ok"
    assert rows[0]["git_sha"] != ""


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
