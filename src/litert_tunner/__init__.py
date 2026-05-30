"""litert_tunner package."""

import keras

from litert_tunner import flatbuffer, graph, ops, quantization

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
    "flatbuffer",
    "graph",
    "load_model",
    "ops",
    "quantization",
    "save_model",
]
