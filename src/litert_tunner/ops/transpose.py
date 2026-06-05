"""TRANSPOSE op implementation for litert_tunner.

Simulates TFLite's TRANSPOSE op as a Keras layer.
TRANSPOSE permutes the dimensions of a tensor according to a
constant permutation vector. This is a passthrough for quantization —
scale and zero-point are preserved unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import keras
from keras import ops

from litert_tunner.ops import registry

if TYPE_CHECKING:
    from litert_tunner.graph import types
    from litert_tunner.ops.utils import TensorLike

    ShapeLike = tuple[int, ...] | list[int] | list[tuple[int, ...]]


class Transpose(keras.Layer):
    """Simulates TFLite's TRANSPOSE op.

    Permutes the dimensions of the input tensor according to the
    given permutation. This is a passthrough for quantization —
    scale and zero-point are preserved unchanged.

    No trainable parameters. Does not implement ``Writable``.

    Args:
        perm: Permutation of dimensions (e.g. (0, 2, 1) to swap last two axes).
        name: Layer name.
    """

    def __init__(
        self,
        perm: tuple[int, ...],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._perm = perm

    def call(self, x: TensorLike) -> TensorLike:
        """Forward pass applying transpose.

        Args:
            x: Input tensor.

        Returns:
            Transposed output tensor.
        """
        return ops.transpose(x, axes=self._perm)

    def get_config(self):
        """Return the configuration dictionary for serialization."""
        config = super().get_config()
        config.update({"perm": self._perm})
        return config


@registry.register_op("TRANSPOSE")
def build_transpose(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a Transpose layer from parsed TFLite operator info.

    TFLite TRANSPOSE inputs:
        [0] input tensor (INT8 or float32)
        [1] perm tensor (INT32, constant) — permutation vector

    TFLite TRANSPOSE outputs:
        [0] output tensor (same type as input)

    Args:
        op: Parsed operator info with input/output indices and options.
        tensors: All tensors in the graph.

    Returns:
        A configured QuantizedTranspose Keras layer.
    """
    perm_tensor = tensors[op.input_indices[1]]
    if perm_tensor.data is None:
        msg = "TRANSPOSE requires a constant permutation tensor"
        raise ValueError(msg)

    perm = tuple(int(d) for d in perm_tensor.data.flatten())

    return Transpose(
        perm=perm,
        name=f"transpose_{op.output_indices[0]}",
    )
