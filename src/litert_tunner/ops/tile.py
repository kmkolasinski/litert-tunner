"""TILE op implementation for litert_tunner.

Simulates TFLite's TILE op as a Keras layer.
TILE is a passthrough for quantization — it does not change
the scale or zero-point. It repeats the tensor along specified dimensions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import keras
from keras import ops

from litert_tunner.ops import registry

if TYPE_CHECKING:
    from litert_tunner.graph import types
    from litert_tunner.ops.utils import TensorLike


class Tile(keras.Layer):
    """Simulates TFLite's TILE op.

    Constructs a tensor by tiling a given tensor. This is a passthrough
    for quantization — scale and zero-point are preserved unchanged.

    The multiples tensor is expected to be statically known at build time.
    No trainable parameters. Does not implement ``Writable``.

    Args:
        multiples: A 1D array of integers specifying the number of times to
            repeat the input tensor along each dimension.
        name: Layer name.
    """

    def __init__(
        self,
        multiples: tuple[int, ...],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._multiples = multiples

    def call(self, inputs: TensorLike) -> TensorLike:
        """Forward pass applying tile.

        Args:
            inputs: The input tensor.

        Returns:
            Tiled output tensor.
        """
        return ops.tile(inputs, self._multiples)

    def get_config(self):
        """Return the configuration dictionary for serialization of the layer."""
        config = super().get_config()
        config.update({"multiples": self._multiples})
        return config


@registry.register_op("TILE")
def build_tile(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a Tile layer from parsed TFLite operator info.

    TFLite TILE inputs:
        [0] input tensor
        [1] multiples tensor (1D INT32)

    TFLite TILE outputs:
        [0] output tensor (same type as input)

    Args:
        op: Parsed operator info.
        tensors: All tensors in the graph.

    Returns:
        A configured Tile Keras layer.
    """
    multiples_tensor = tensors[op.input_indices[1]]
    if multiples_tensor.data is None:
        msg = f"TILE (output index {op.output_indices[0]}) requires a constant multiples tensor."
        raise ValueError(msg)

    multiples = tuple(int(m) for m in multiples_tensor.data.flatten())

    return Tile(
        multiples=multiples,
        name=f"tile_{op.output_indices[0]}",
    )
