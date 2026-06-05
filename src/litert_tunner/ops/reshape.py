"""RESHAPE op implementation for litert_tunner.

Simulates TFLite's RESHAPE op as a Keras layer.
RESHAPE is a passthrough for quantization — it does not change
the scale or zero-point. It simply reshapes the tensor to the
target shape.
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


class Reshape(keras.Layer):
    """Simulates TFLite's RESHAPE op.

    Reshapes the input tensor to the target shape. This is a passthrough
    for quantization — scale and zero-point are preserved unchanged.

    The target shape (excluding batch dimension) is always known statically
    at build time — either from the shape constant, op options, or the output
    tensor shape in the TFLite graph.

    No trainable parameters. Does not implement ``Writable``.

    Args:
        target_shape: The target shape (excluding batch dimension).
        name: Layer name.
    """

    def __init__(
        self,
        target_shape: tuple[int, ...] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._target_shape = target_shape

    def call(self, inputs: TensorLike) -> TensorLike:
        """Forward pass applying reshape.

        Args:
            inputs: Either a single tensor or a list of tensors. When a list
                is provided (e.g., [data, shape_vector] from a dynamic PACK),
                only the first element (data) is reshaped; the shape is taken
                from the static ``_target_shape`` set at build time.

        Returns:
            Reshaped output tensor.
        """
        x = inputs[0] if isinstance(inputs, (list, tuple)) else inputs

        batch_size = ops.shape(x)[0]
        if self._target_shape is not None:
            full_shape = (batch_size, *self._target_shape)
        else:
            full_shape = (batch_size, -1)
        return ops.reshape(x, full_shape)

    def get_config(self):
        """Return the configuration dictionary for serialization of the layer."""
        config = super().get_config()
        config.update({"target_shape": self._target_shape})
        return config


@registry.register_op("RESHAPE")
def build_reshape(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a Reshape layer from parsed TFLite operator info.

    TFLite RESHAPE inputs:
        [0] input tensor (INT8 or FLOAT32)
        [1] shape tensor (INT32, constant or dynamic) — target shape

    TFLite RESHAPE outputs:
        [0] output tensor (same type as input)

    The target shape is resolved statically in this priority order:
        1. Constant shape tensor (data is not None)
        2. Op options (NewShape)
        3. Output tensor shape from the graph

    Even when the shape tensor is dynamic (e.g., output of a PACK op that
    computes the shape at runtime), the output tensor shape in the TFLite
    graph always records the correct non-batch dimensions. We use that
    static shape instead of passing a symbolic shape tensor to
    ``ops.reshape``, which avoids Keras tracing failures.

    Args:
        op: Parsed operator info with input/output indices and options.
        tensors: All tensors in the graph.

    Returns:
        A configured QuantizedReshape Keras layer.
    """
    target_shape: tuple[int, ...] | None = None

    # 1. Try constant shape tensor
    if len(op.input_indices) > 1 and op.input_indices[1] >= 0:
        shape_tensor = tensors[op.input_indices[1]]
        if shape_tensor.data is not None:
            target_shape = tuple(int(d) for d in shape_tensor.data.flatten())

    # 2. Try op options
    if target_shape is None:
        new_shape = op.options.get("NewShape")
        if new_shape is not None:
            target_shape = tuple(int(d) for d in new_shape)

    # 3. Fall back to output tensor shape (always available)
    if target_shape is None:
        output_tensor = tensors[op.output_indices[0]]
        target_shape = output_tensor.shape

    # Strip batch dimension — the layer prepends it dynamically
    if len(target_shape) > 0 and target_shape[0] in (1, -1):
        target_shape = target_shape[1:]

    return Reshape(
        target_shape=target_shape,
        name=f"reshape_{op.output_indices[0]}",
    )
