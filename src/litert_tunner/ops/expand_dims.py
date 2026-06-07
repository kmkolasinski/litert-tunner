"""EXPAND_DIMS op implementation for litert_tunner.

Simulates TFLite's EXPAND_DIMS op as a Keras layer.
EXPAND_DIMS is a passthrough for quantization — it does not change
the scale or zero-point. It simply inserts a dimension of 1 at the
specified axis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import keras
from keras import ops

from litert_tunner.ops import registry

if TYPE_CHECKING:
    from litert_tunner.graph import types
    from litert_tunner.ops.utils import TensorLike


class ExpandDims(keras.Layer):
    """Simulates TFLite's EXPAND_DIMS op.

    Inserts a dimension of 1 into a tensor's shape. This is a passthrough
    for quantization — scale and zero-point are preserved unchanged.

    The axis to expand is expected to be statically known at build time.
    No trainable parameters. Does not implement ``Writable``.

    Args:
        axis: The dimension index at which to insert the singleton dimension.
            Note: This axis is relative to the non-batch dimensions when
            it's positive, but negative indices might need special handling.
            Since the Keras layer operates on batched inputs, we adjust
            the axis.
        name: Layer name.
    """

    def __init__(
        self,
        axis: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._axis = axis

    def call(self, inputs: TensorLike) -> TensorLike:
        """Forward pass applying expand_dims.

        Args:
            inputs: The input tensor.

        Returns:
            Output tensor with expanded dimension.
        """
        return ops.expand_dims(inputs, axis=self._axis)

    def get_config(self):
        """Return the configuration dictionary for serialization of the layer."""
        config = super().get_config()
        config.update({"axis": self._axis})
        return config


@registry.register_op("EXPAND_DIMS")
def build_expand_dims(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build an ExpandDims layer from parsed TFLite operator info.

    TFLite EXPAND_DIMS inputs:
        [0] input tensor
        [1] axis tensor (INT32 scalar)

    TFLite EXPAND_DIMS outputs:
        [0] output tensor (same type as input)

    Args:
        op: Parsed operator info.
        tensors: All tensors in the graph.

    Returns:
        A configured ExpandDims Keras layer.
    """
    axis_tensor = tensors[op.input_indices[1]]
    if axis_tensor.data is None:
        # Fallback if axis is dynamic. Not typical for EXPAND_DIMS in TFLite.
        msg = f"EXPAND_DIMS (output index {op.output_indices[0]}) requires a constant axis tensor."
        raise ValueError(msg)

    axis = int(axis_tensor.data.item())

    return ExpandDims(
        axis=axis,
        name=f"expand_dims_{op.output_indices[0]}",
    )
