"""Test compute_gradient_cosine handles parameter-subset mismatch."""
import sys
import torch

sys.path.insert(0, "src")
from training.train_phase_4b import compute_gradient_cosine


def test_mismatched_none_patterns():
    """Data covers only params 0-1; physics covers all 3.
    Cosine should use params 0-1 only (where both are non-None)."""
    a = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), None]
    b = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([3.0, 3.0])]
    cos = compute_gradient_cosine(a, b)
    # Both a and b on common subset {0,1}:
    #   fa = [1,0,0,1], fb = [1,0,0,1]
    #   cos = 1.0
    assert abs(cos - 1.0) < 1e-6, f"Expected cos ≈ 1, got {cos}"


def test_mismatched_orthogonal():
    a = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    b = [torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])]
    # fa = [1,0,0,1], fb = [0,1,1,0], dot = 0
    cos = compute_gradient_cosine(a, b)
    assert abs(cos) < 1e-6, f"Expected cos ≈ 0, got {cos}"


def test_no_common_params():
    a = [None, torch.tensor([1.0])]
    b = [torch.tensor([1.0]), None]
    cos = compute_gradient_cosine(a, b)
    assert cos == 0.0, f"Expected 0.0 when no common params, got {cos}"


def test_zero_norm_common():
    a = [torch.zeros(5), torch.tensor([1.0])]
    b = [torch.zeros(5), torch.tensor([1.0])]
    cos = compute_gradient_cosine(a, b)
    # common subset = {0,1}; norms: |a| = 1, |b| = 1, not zero → cos = 1
    assert abs(cos - 1.0) < 1e-6, f"Expected cos ≈ 1, got {cos}"


def test_both_zero_norm_common():
    a = [torch.zeros(5)]
    b = [torch.zeros(5)]
    cos = compute_gradient_cosine(a, b)
    assert cos == 0.0, f"Expected 0.0 when both norms zero, got {cos}"


def test_different_list_lengths_raises():
    a = [torch.tensor([1.0])]
    b = [torch.tensor([1.0]), torch.tensor([2.0])]
    try:
        compute_gradient_cosine(a, b)
        assert False, "Expected ValueError"
    except ValueError:
        pass
