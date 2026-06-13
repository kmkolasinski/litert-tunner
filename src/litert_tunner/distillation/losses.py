from typing import TypeAlias, cast

import keras
import numpy as np
from keras import ops

TensorLike: TypeAlias = keras.KerasTensor | np.ndarray | float | int


def mse_loss(y_pred: TensorLike, y_true: TensorLike) -> keras.KerasTensor:
    """Computes the mean squared error between the predicted and true outputs.

    Assumes the inputs are logits or continuous values (not necessarily probabilities).

    Args:
        y_pred: The predicted outputs of the student model, expected shape (batch_size, ...).
        y_true: The target outputs of the teacher model, expected shape (batch_size, ...).

    Returns:
        The mean squared error across the batch.
    """
    return cast("keras.KerasTensor", ops.mean(ops.square(ops.subtract(y_pred, y_true))))


def kl_loss(y_pred: TensorLike, y_true: TensorLike) -> keras.KerasTensor:
    """Computes the KL divergence distillation loss between student and teacher.

    This loss encourages the student to match the output probability distribution
    of the teacher. It applies clipping to ensure numerical stability.

    Assumes `y_pred` and `y_true` are probabilities (e.g. outputs of a softmax).

    Args:
        y_pred: The predicted probabilities of the student model, expected shape (batch_size, ...).
        y_true: The target probabilities of the teacher model, expected shape (batch_size, ...).

    Returns:
        The mean KL divergence loss across the batch.
    """
    eps = 1e-7  # numerical stability for log of near-zero probs
    y_true_clipped = ops.clip(y_true, eps, 1.0)
    y_pred_clipped = ops.clip(y_pred, eps, 1.0)
    # Per-sample KL, then mean over batch
    kl = ops.sum(y_true_clipped * ops.log(y_true_clipped / y_pred_clipped), axis=-1)
    return cast("keras.KerasTensor", ops.mean(kl))


def cosine_loss(y_pred: TensorLike, y_true: TensorLike) -> keras.KerasTensor:
    """Computes the cosine distance loss between student and teacher.

    This loss encourages the student to match the direction of the teacher's output
    vectors, ignoring their magnitude.

    Assumes `y_pred` and `y_true` are continuous feature vectors.
    Loss is bounded between 0 (identical direction) and 2 (opposite direction).

    Args:
        y_pred: The predicted outputs of the student model, expected shape (batch_size, ...).
        y_true: The target outputs of the teacher model, expected shape (batch_size, ...).

    Returns:
        The mean cosine loss across the batch.
    """
    # L2 normalize along the last axis
    y_pred_norm = ops.normalize(y_pred, axis=-1)
    y_true_norm = ops.normalize(y_true, axis=-1)
    # Cosine similarity
    cos_sim = ops.sum(y_pred_norm * y_true_norm, axis=-1)
    # Cosine loss is 1 - cosine_similarity
    return cast("keras.KerasTensor", ops.mean(1.0 - cos_sim))
