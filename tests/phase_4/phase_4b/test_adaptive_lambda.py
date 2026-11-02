"""Step 4B.2 — Unit tests for AdaptiveLambda (10 tests)."""
import sys
import math
import pytest
import torch

sys.path.insert(0, "src")
from training.adaptive_lambda import AdaptiveLambda


# ============================================================
# T1 — Initialization
# ============================================================
def test_t1_initialization():
    al = AdaptiveLambda()
    assert al.current() == 1e-4
    assert al.smoothed_cosine is None
    assert len(al.history) == 0
    assert al.alpha == 0.05
    assert al.lambda_min == 1e-5
    assert al.lambda_max == 10.0
    assert al.ema_beta == 0.1


# ============================================================
# T2 — First update (EMA init)
# ============================================================
def test_t2_first_update_ema_init():
    al = AdaptiveLambda()
    new_lam = al.update(epoch=0, raw_cosine=0.5)
    # smoothed_cosine should be exactly 0.5 (initialized, not smoothed)
    assert al.smoothed_cosine == 0.5
    # log update: log(1e-4) + 0.05 * 0.5 = log(1e-4) + 0.025
    expected_log = math.log(1e-4) + 0.025
    assert abs(new_lam - math.exp(expected_log)) < 1e-12
    # Positive cosine → lambda increases
    assert new_lam > 1e-4


# ============================================================
# T3 — Subsequent update (EMA smoothing)
# ============================================================
def test_t3_subsequent_update_ema():
    al = AdaptiveLambda()
    al.update(epoch=0, raw_cosine=0.5)
    lam_after_0 = al.current()
    al.update(epoch=1, raw_cosine=-0.3)
    # smoothed = 0.1*(-0.3) + 0.9*0.5 = 0.42 (still positive)
    assert abs(al.smoothed_cosine - 0.42) < 1e-12
    # Still increases
    assert al.current() > lam_after_0


# ============================================================
# T4 — Lambda_min clip
# ============================================================
def test_t4_lambda_min_clip():
    al = AdaptiveLambda()
    # Feed strongly negative cosine repeatedly
    for i in range(50):
        al.update(epoch=i, raw_cosine=-100.0)
    # Should clamp at lambda_min
    assert abs(al.current() - 1e-5) < 1e-10
    # Last history entry should show clip min
    assert al.history[-1]["clip_applied"] == "min"
    assert al.get_summary()["n_clip_min"] > 0


# ============================================================
# T5 — Lambda_max clip
# ============================================================
def test_t5_lambda_max_clip():
    al = AdaptiveLambda()
    # raw_cosine=100.0 is clipped to 1.0. Starting from 1e-4, need ~230 epochs
    # of sustained max cosine to reach lambda_max=10.0 (log-space: 11.51 / 0.05).
    # Use 500 epochs to ensure clip fires.
    for i in range(500):
        al.update(epoch=i, raw_cosine=100.0)
    assert abs(al.current() - 10.0) < 1e-6, \
        f"Expected lambda at max bound, got {al.current()}"
    # Somewhere in history, clip_applied='max' must appear
    clip_max_epochs = [h["epoch"] for h in al.history if h["clip_applied"] == "max"]
    assert len(clip_max_epochs) > 0, "No max clip recorded in 500 epochs"
    assert al.get_summary()["n_clip_max"] > 0


# ============================================================
# T6 — Numerical stability with extreme cosines
# ============================================================
def test_t6_extreme_cosines_no_nan():
    al = AdaptiveLambda()
    # Extreme positive
    new_lam = al.update(epoch=0, raw_cosine=1e10)
    assert math.isfinite(new_lam)
    # Allow tiny FP noise from exp(log(x)) round-tripping
    assert (1e-5 - 1e-10) <= new_lam <= (10.0 + 1e-9)
    # Extreme negative
    new_lam = al.update(epoch=1, raw_cosine=-1e10)
    assert math.isfinite(new_lam)
    assert (1e-5 - 1e-10) <= new_lam <= (10.0 + 1e-9)


# ============================================================
# T7 — Reproducibility
# ============================================================
def test_t7_reproducibility():
    cosines = [0.5, -0.3, 0.2, -0.1, 0.4]
    al1 = AdaptiveLambda()
    al2 = AdaptiveLambda()
    for i, c in enumerate(cosines):
        al1.update(epoch=i, raw_cosine=c)
        al2.update(epoch=i, raw_cosine=c)
    assert al1.current() == al2.current()
    assert al1.get_history() == al2.get_history()


# ============================================================
# T8 — Log history correctness
# ============================================================
def test_t8_log_history():
    al = AdaptiveLambda()
    N = 5
    for i in range(N):
        al.update(epoch=i, raw_cosine=0.1 * (i - 2))
    hist = al.get_history()
    assert len(hist) == N
    required_fields = [
        "epoch", "raw_cosine_input", "raw_cosine_used", "cosine_clipped",
        "smoothed_cosine", "smoothed_cosine_20_epochs_ago", "delta_smoothed_cosine",
        "log_lambda_prev", "log_lambda_raw_update", "log_lambda_after_clip",
        "lambda", "clip_applied", "alpha",
    ]
    for i, h in enumerate(hist):
        assert h["epoch"] == i
        for field in required_fields:
            assert field in h, f"epoch {i}: missing field '{field}'"
        # lambda should equal exp(log_lambda_after_clip)
        assert abs(h["lambda"] - math.exp(h["log_lambda_after_clip"])) < 1e-10
        # raw_cosine_used should equal raw_cosine_input for values in [-1, 1]
        assert h["raw_cosine_used"] == h["raw_cosine_input"]


# ============================================================
# T9 — Summary statistics
# ============================================================
def test_t9_summary_statistics():
    al = AdaptiveLambda()
    # Alternating extreme to trigger clips
    for i in range(100):
        al.update(epoch=i, raw_cosine=10.0 if i % 2 == 0 else -10.0)
    s = al.get_summary()
    assert s["n_updates"] == 100
    assert s["lambda_init"] == 1e-4
    assert 1e-5 <= s["lambda_final"] <= 10.0
    assert 1e-5 <= s["lambda_min_seen"] <= 10.0
    assert 1e-5 <= s["lambda_max_seen"] <= 10.0
    assert s["n_clip_min"] + s["n_clip_max"] <= 100


# ============================================================
# T10 — Frozen config defaults
# ============================================================
def test_t10_frozen_config_defaults():
    al = AdaptiveLambda()
    assert al.alpha == 0.05
    assert al.lambda_min == 1e-5
    assert al.lambda_max == 10.0
    assert al.lambda_init == 1e-4
    assert al.ema_beta == 0.1
    # Invalid bounds should raise
    with pytest.raises(ValueError):
        AdaptiveLambda(lambda_min=10.0, lambda_max=1.0)
    with pytest.raises(ValueError):
        AdaptiveLambda(lambda_init=100.0)  # outside [lambda_min, lambda_max]
    with pytest.raises(ValueError):
        AdaptiveLambda(lambda_init=1e-6)  # below lambda_min
    with pytest.raises(ValueError):
        AdaptiveLambda(alpha=-0.1)
    with pytest.raises(ValueError):
        AdaptiveLambda(ema_beta=1.5)


# ============================================================
# T11 — 20-epoch look-back of smoothed cosine
# ============================================================
def test_t11_ema_reference_window_lookback():
    """After 20 epochs, smoothed_cosine_20_epochs_ago should be recorded."""
    al = AdaptiveLambda()
    # Run 25 epochs with varying cosine
    for i in range(25):
        al.update(epoch=i, raw_cosine=0.5 * math.sin(i * 0.3))
    hist = al.get_history()
    assert len(hist) == 25
    # Epochs 0-19: smoothed_cosine_20_epochs_ago should be None
    for i in range(20):
        assert hist[i]["smoothed_cosine_20_epochs_ago"] is None, \
            f"epoch {i}: expected None, got {hist[i]['smoothed_cosine_20_epochs_ago']}"
        assert hist[i]["delta_smoothed_cosine"] is None
    # Epochs 20-24: smoothed_cosine_20_epochs_ago should be the smoothed value 20 epochs back
    for i in range(20, 25):
        assert hist[i]["smoothed_cosine_20_epochs_ago"] is not None
        assert hist[i]["delta_smoothed_cosine"] is not None
        expected_20_ago = hist[i - 20]["smoothed_cosine"]
        assert abs(hist[i]["smoothed_cosine_20_epochs_ago"] - expected_20_ago) < 1e-12


# ============================================================
# T12 — Cosine clipping to [-1, 1]
# ============================================================
def test_t12_cosine_clipping():
    """raw_cosine outside [-1, 1] is clipped; original preserved in history."""
    al = AdaptiveLambda()
    # Extreme positive (beyond mathematically valid range)
    al.update(epoch=0, raw_cosine=1.5)
    h0 = al.get_history()[0]
    assert h0["raw_cosine_input"] == 1.5
    assert h0["raw_cosine_used"] == 1.0
    assert h0["cosine_clipped"] is True

    # Extreme negative
    al.update(epoch=1, raw_cosine=-2.3)
    h1 = al.get_history()[1]
    assert h1["raw_cosine_input"] == -2.3
    assert h1["raw_cosine_used"] == -1.0
    assert h1["cosine_clipped"] is True

    # Valid range — no clipping
    al.update(epoch=2, raw_cosine=0.5)
    h2 = al.get_history()[2]
    assert h2["raw_cosine_input"] == 0.5
    assert h2["raw_cosine_used"] == 0.5
    assert h2["cosine_clipped"] is False

    # Summary records clip count
    summary = al.get_summary()
    assert summary["n_cosine_clipped"] == 2
    assert abs(summary["cosine_clip_fraction"] - 2/3) < 1e-9
