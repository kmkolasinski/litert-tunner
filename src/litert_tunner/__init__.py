"""litert_tunner package."""

import keras

from litert_tunner import flatbuffer, graph
from litert_tunner.testing_utils import (
    assert_allclose_with_mismatch_tolerance,
    assert_cosine_similarity,
)
from litert_tunner.trainer import Trainer, prepare_for_finetuning

__version__ = "0.1.0"


def load_model(path: str) -> keras.Model:
    """Load a .tflite INT8 model and return a trainable Keras replica.

    Args:
        path: Path to the .tflite file.

    Returns:
        A trainable Keras Model replica.
    """
    graph_def = flatbuffer.parse_tflite(path)
    return graph.build_keras_model(graph_def)


def save_model(model: keras.Model, path: str) -> None:
    """Save updated parameters back to a .tflite file.

    Args:
        model: The trained Keras Model replica.
        path: Path to write the updated .tflite file.
    """
    flatbuffer.save_tflite(model, path)


__all__ = [
    "Trainer",
    "assert_allclose_with_mismatch_tolerance",
    "assert_cosine_similarity",
    "load_model",
    "prepare_for_finetuning",
    "save_model",
]
