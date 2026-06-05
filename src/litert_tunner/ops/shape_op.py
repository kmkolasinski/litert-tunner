"""SHAPE op implementation for litert_tunner.

Simulates TFLite's SHAPE op as a Keras layer.
Returns the shape of the input tensor as a 1-D int32 tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import keras
from keras import ops

from litert_tunner.ops import registry

if TYPE_CHECKING:
    from litert_tunner.graph import types
    from litert_tunner.ops.utils import TensorLike


class Shape(keras.Layer):
    """Simulates TFLite's SHAPE op.

    Returns the shape of the input tensor as a 1-D tensor of int32 values.
    This is a metadata-only op with no trainable parameters.

    Does not implement ``Writable``.

    Args:
        name: Layer name.
    """

    def call(self, x: TensorLike) -> TensorLike:
        """Forward pass returning the shape of the input.

        Args:
            x: Input tensor.

        Returns:
            1-D tensor containing the shape of the input, cast to float32
            so it flows through the graph as a standard activation.
        """
        shape_tensor = ops.shape(x)
        # Cast to float32 to be consistent with the graph — other ops
        # (strided_slice, pack) will consume this as a float tensor.
        return ops.cast(ops.stack(shape_tensor), "float32")


@registry.register_op("SHAPE")
def build_shape(
    op: types.OperatorInfo,
    _tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a Shape layer from parsed TFLite operator info.

    TFLite SHAPE inputs:
        [0] input tensor (any type)

    TFLite SHAPE outputs:
        [0] output tensor (INT32 or INT64) — 1-D shape

    Args:
        op: Parsed operator info with input/output indices and options.
        tensors: All tensors in the graph.
        graph_def: The parsed GraphDef.

    Returns:
        A configured QuantizedShape Keras layer.
    """
    return Shape(
        name=f"shape_{op.output_indices[0]}",
    )
