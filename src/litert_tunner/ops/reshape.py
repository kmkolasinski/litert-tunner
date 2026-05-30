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


class QuantizedReshape(keras.Layer):
    """Simulates TFLite's RESHAPE op.

    Reshapes the input tensor to the target shape. This is a passthrough
    for quantization — scale and zero-point are preserved unchanged.

    Supports two modes:
        1. Static target shape known at build time (from constant tensor or options)
        2. Dynamic target shape provided as a second input at call time

    No trainable parameters. Does not implement ``Writable``.

    Args:
        target_shape: The target shape (excluding batch dimension), or None
            if the shape comes from a dynamic input at call time.
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
            inputs: Either a single tensor (static shape mode) or a list of
                [data_tensor, shape_tensor] (dynamic shape mode).

        Returns:
            Reshaped output tensor.
        """
        if isinstance(inputs, (list, tuple)):
            # Dynamic shape mode: inputs = [data, shape_vector]
            data = inputs[0]
            shape_vector = inputs[1]
            # Cast shape to int32 and use it for reshape
            target = ops.cast(shape_vector, "int32")
            return ops.reshape(data, target)

        # Static shape mode
        x = inputs
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
    """Build a QuantizedReshape layer from parsed TFLite operator info.

    TFLite RESHAPE inputs:
        [0] input tensor (INT8 or FLOAT32)
        [1] shape tensor (INT32, constant or dynamic) — target shape

    TFLite RESHAPE outputs:
        [0] output tensor (same type as input)

    If the shape tensor is constant (``data is not None``), the target shape
    is baked into the layer. If it's dynamic (e.g., output of a PACK op),
    the shape is provided at call time as a second input.

    Args:
        op: Parsed operator info with input/output indices and options.
        tensors: All tensors in the graph.
        graph_def: The parsed GraphDef.

    Returns:
        A configured QuantizedReshape Keras layer.
    """
    # Check if the shape input is constant or dynamic
    has_dynamic_shape = False
    target_shape: tuple[int, ...] | None = None

    if len(op.input_indices) > 1 and op.input_indices[1] >= 0:
        shape_tensor = tensors[op.input_indices[1]]
        if shape_tensor.data is not None:
            # Static shape — bake it into the layer
            target_shape = tuple(int(d) for d in shape_tensor.data.flatten())
        else:
            # Dynamic shape — will be provided at call time
            has_dynamic_shape = True

    # If no shape tensor at all, try op options
    if target_shape is None and not has_dynamic_shape:
        new_shape = op.options.get("NewShape")
        if new_shape is not None:
            target_shape = tuple(int(d) for d in new_shape)

    # Last resort for static mode: use output tensor shape
    if target_shape is None and not has_dynamic_shape:
        output_tensor = tensors[op.output_indices[0]]
        target_shape = output_tensor.shape

    # Strip batch dimension for static mode — the layer prepends it dynamically
    if target_shape is not None and len(target_shape) > 0 and target_shape[0] in (1, -1):
        target_shape = target_shape[1:]

    return QuantizedReshape(
        target_shape=target_shape if not has_dynamic_shape else None,
        name=f"quantized_reshape_{op.output_indices[0]}",
    )
