"""Unit tests for test utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from tests import testing_utils


def test__assert_cosine_similarity_succeeds_for_identical_inputs() -> None:
    """Verify that identical inputs pass the similarity assertion."""
    rng = np.random.default_rng(42)
    y = rng.uniform(-1.0, 1.0, (4, 10)).astype(np.float32)

    # Identical inputs should have exactly 1.0 similarity, which is >= 0.999
    testing_utils.assert_cosine_similarity(y, y)


def test__assert_cosine_similarity_fails_for_uncorrelated_inputs() -> None:
    """Verify that uncorrelated inputs fail the similarity assertion."""
    rng = np.random.default_rng(42)
    y1 = rng.uniform(-1.0, 1.0, (10, 10)).astype(np.float32)
    # Generate an independent random array (unlikely to have 0.999 correlation)
    y2 = rng.uniform(-1.0, 1.0, (10, 10)).astype(np.float32)

    with pytest.raises(AssertionError, match="is less than required"):
        testing_utils.assert_cosine_similarity(y1, y2)


def test__assert_cosine_similarity_custom_threshold() -> None:
    """Verify that a custom threshold works as expected."""
    rng = np.random.default_rng(42)
    y1 = rng.uniform(-1.0, 1.0, (4, 5)).astype(np.float32)

    # Slightly perturb y1 to get a high but not perfect correlation
    y2 = y1 + rng.normal(0, 0.01, y1.shape).astype(np.float32)

    # Should pass with a lower threshold, e.g. 0.99
    testing_utils.assert_cosine_similarity(y2, y1, min_similarity=0.99)

    # Might fail if we set threshold very close to 1.0 depending on noise
    with pytest.raises(AssertionError):
        testing_utils.assert_cosine_similarity(y2, y1, min_similarity=0.99999)


def test__assert_cosine_similarity_handles_multidimensional_arrays() -> None:
    """Verify that multidimensional arrays are correctly flattened and compared."""
    rng = np.random.default_rng(42)
    # Shape like (batch_size, height, width, channels)
    y = rng.uniform(-1.0, 1.0, (2, 8, 8, 3)).astype(np.float32)

    # Identical multidimensional arrays should pass
    testing_utils.assert_cosine_similarity(y, y)


def test__assert_allclose_with_mismatch_tolerance_succeeds_for_identical_inputs() -> None:
    """Verify that identical inputs pass the tolerance check."""
    a = np.array([1.0, 2.0, 3.0])
    testing_utils.assert_allclose_with_mismatch_tolerance(a, a)


def test__assert_allclose_with_mismatch_tolerance_succeeds_within_mismatch_fraction() -> None:
    """Verify that inputs pass if mismatch fraction is within allowed limit."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    # One element differs by more than atol (0.1 difference on a 4-element array = 25% mismatch)
    b = np.array([1.0, 2.0, 3.1, 4.0])

    # max_mismatch_fraction=0.3 allows up to 30% mismatch, so 25% should pass
    testing_utils.assert_allclose_with_mismatch_tolerance(
        a, b, atol=0.01, max_mismatch_fraction=0.30
    )


def test__assert_allclose_with_mismatch_tolerance_fails_exceeding_mismatch_fraction() -> None:
    """Verify that inputs fail if mismatch fraction exceeds allowed limit."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0, 2.0, 3.1, 4.0])

    # max_mismatch_fraction=0.2 allows up to 20% mismatch, so 25% should fail
    with pytest.raises(AssertionError, match="Mismatched elements: 1 / 4"):
        testing_utils.assert_allclose_with_mismatch_tolerance(
            a, b, atol=0.01, max_mismatch_fraction=0.20
        )
