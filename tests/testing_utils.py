"""Utility functions for tests."""

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
