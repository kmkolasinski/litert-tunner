from typing import TypeAlias, cast

import keras
import numpy as np
from keras import ops

TensorLike: TypeAlias = keras.KerasTensor | np.ndarray | float | int


def cosine_similarity_metric(y_pred: TensorLike, y_true: TensorLike) -> keras.KerasTensor:
    """Computes the cosine similarity between the predicted and true outputs.

    Assumes the inputs are continuous vectors. The outputs will be normalized
    before computing the dot product.

    Args:
        y_pred: The predicted outputs, expected shape (batch_size, ...).
        y_true: The true outputs, expected shape (batch_size, ...).

    Returns:
        The mean cosine similarity across the batch.
    """
    y_pred_flat = ops.reshape(y_pred, (ops.shape(y_pred)[0], -1))
    y_true_flat = ops.reshape(y_true, (ops.shape(y_true)[0], -1))

    y_true_flat = ops.cast(y_true_flat, "float32")
    y_pred_flat = ops.cast(y_pred_flat, "float32")

    pred_norm = ops.normalize(y_pred_flat, axis=-1)
    true_norm = ops.normalize(y_true_flat, axis=-1)

    cosine_sim = ops.mean(ops.sum(pred_norm * true_norm, axis=-1))
    return cast("keras.KerasTensor", cosine_sim)
