"""Utility functions for testing models."""

from __future__ import annotations

import numpy as np


def assert_cosine_similarity(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    min_similarity: float = 0.999,
) -> None:
    """Asserts that the cosine similarity between two outputs is above a threshold.

    Args:
        y_pred: Predicted output numpy array.
        y_true: Ground truth/reference output numpy array.
        min_similarity: Minimum allowed cosine similarity. Defaults to 0.999.
    """
    # Flatten both arrays to (batch_size, -1) to support multidimensional outputs
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)
    y_true_flat = y_true.reshape(y_true.shape[0], -1)

    # Normalize vectors
    pred_norm = y_pred_flat / np.linalg.norm(y_pred_flat, axis=-1, keepdims=True)
    true_norm = y_true_flat / np.linalg.norm(y_true_flat, axis=-1, keepdims=True)

    # Calculate average cosine similarity across the batch
    cosine_sim = np.mean(np.sum(pred_norm * true_norm, axis=-1))

    assert cosine_sim >= min_similarity, (
        f"Cosine similarity {cosine_sim:.6f} is less than required {min_similarity:.6f}"
    )


def assert_allclose_with_mismatch_tolerance(
    actual: np.ndarray,
    desired: np.ndarray,
    rtol: float = 1e-7,
    atol: float = 0.0,
    max_mismatch_fraction: float = 0.0,
    err_msg: str = "",
) -> None:
    """Asserts that two arrays are equal within tolerance, allowing a small fraction of mismatches.

    Args:
        actual: The actual numpy array.
        desired: The desired/reference numpy array.
        rtol: Relative tolerance. Defaults to 1e-7.
        atol: Absolute tolerance. Defaults to 0.0.
        max_mismatch_fraction: Maximum allowed fraction of mismatched elements.
            Defaults to 0.0 (no mismatches allowed).
        err_msg: Optional error message to prepend to the assertion message.
    """
    is_close = np.isclose(actual, desired, rtol=rtol, atol=atol)
    mismatched_mask = ~is_close
    mismatched_count = int(np.sum(mismatched_mask))
    total_elements = actual.size
    mismatch_fraction = mismatched_count / total_elements

    if mismatch_fraction > max_mismatch_fraction:
        percent_mismatched = mismatch_fraction * 100
        percent_max_allowed = max_mismatch_fraction * 100

        mismatched_indices = np.argwhere(mismatched_mask)
        num_to_show = min(5, mismatched_count)
        mismatch_details = []
        for i in range(num_to_show):
            idx = tuple(mismatched_indices[i])
            mismatch_details.append(f" {idx}: {actual[idx]} (ACTUAL), {desired[idx]} (DESIRED)")
        mismatch_str = "\n".join(mismatch_details)

        msg = (
            f"Not equal to tolerance rtol={rtol}, atol={atol}\n"
            f"Mismatched elements: {mismatched_count} / {total_elements} "
            f"({percent_mismatched:.2f}%)\n"
            f"Max allowed mismatch fraction: {percent_max_allowed:.2f}%\n"
            f"First {num_to_show} mismatches are at indices:\n{mismatch_str}"
        )
        if err_msg:
            msg = f"{err_msg}\n{msg}"
        raise AssertionError(msg)
