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
    for i in range(200):
        al.update(epoch=i, raw_cosine=100.0)
    assert abs(al.current() - 10.0) < 1e-8
    assert al.history[-1]["clip_applied"] == "max"
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
    for i, h in enumerate(hist):
        assert h["epoch"] == i
        assert "raw_cosine" in h
        assert "smoothed_cosine" in h
        assert "log_lambda_prev" in h
        assert "log_lambda_raw_update" in h
        assert "log_lambda_after_clip" in h
        assert "lambda" in h
        assert "clip_applied" in h
        # lambda should equal exp(log_lambda_after_clip)
        assert abs(h["lambda"] - math.exp(h["log_lambda_after_clip"])) < 1e-10


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
