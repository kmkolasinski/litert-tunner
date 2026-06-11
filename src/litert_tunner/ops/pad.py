"""PAD op implementation for litert_tunner.

Simulates TFLite's PAD op as a Keras layer. PAD is a passthrough for
quantization — it does not change the scale or zero-point. It pads the
input tensor with the quantization zero-point (for INT8 models) or
zero (for float32 models) along each dimension according to a static
paddings specification.

Supports both INT8 (quantized) and float32 (unquantized) TFLite models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import keras
from keras import ops

from litert_tunner.graph import types
from litert_tunner.ops import registry

if TYPE_CHECKING:
    from litert_tunner.ops.utils import TensorLike


class Pad(keras.Layer):
    """Simulates TFLite's PAD op.

    Pads the input tensor with a constant value along each spatial dimension.
    This is a passthrough for quantization — scale and zero-point are preserved
    unchanged.

    For quantized models, the pad value is the input tensor's zero-point
    (which represents 0.0 in real space). For float32 models, it is 0.0.

    The paddings tensor is always statically known at build time from the
    TFLite graph. No trainable parameters. Does not implement ``Writable``.

    Args:
        pad_width: A tuple of ``(before, after)`` pairs for each dimension,
            e.g. ``((0, 0), (1, 1), (1, 1), (0, 0))``. This includes the
            batch dimension.
        constant_value: The value to pad with. For quantized models this
            should be the zero-point; for float32 models this is 0.0.
        name: Layer name.
    """

    def __init__(
        self,
        pad_width: tuple[tuple[int, int], ...],
        constant_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._pad_width = pad_width
        self._constant_value = constant_value

    def call(self, inputs: TensorLike) -> TensorLike:
        """Forward pass applying padding.

        Args:
            inputs: The input tensor.

        Returns:
            Padded output tensor.
        """
        return ops.pad(inputs, self._pad_width, constant_values=self._constant_value)

    def get_config(self):
        """Return the configuration dictionary for serialization of the layer."""
        config = super().get_config()
        config.update({
            "pad_width": self._pad_width,
            "constant_value": self._constant_value,
        })
        return config


@registry.register_op("PAD")
def build_pad(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a Pad layer from parsed TFLite operator info.

    TFLite PAD inputs:
        [0] input tensor (INT8 or FLOAT32)
        [1] paddings tensor (INT32/INT64, constant, shape ``[n, 2]``)

    TFLite PAD outputs:
        [0] output tensor (same type as input)

    The paddings tensor must be a constant — dynamic padding is not
    supported because TFLite always embeds padding as a static constant.

    For quantized inputs, the pad value is the zero-point of the input
    tensor. This matches TFLite's behavior where PAD fills with
    ``zero_point`` in INT8 space, representing 0.0 in real space.

    Args:
        op: Parsed operator info with input/output indices and options.
        tensors: All tensors in the graph.

    Returns:
        A configured Pad Keras layer.
    """
    paddings_tensor = tensors[op.input_indices[1]]
    if paddings_tensor.data is None:
        msg = f"PAD (output index {op.output_indices[0]}) requires a constant paddings tensor."
        raise ValueError(msg)

    # paddings_tensor.data has shape [n, 2] — convert to tuple of (before, after) pairs
    pad_width = tuple((int(row[0]), int(row[1])) for row in paddings_tensor.data)

    # For quantized models, pad with zero_point (represents 0.0 in real space).
    # For float32 models, pad with 0.0.
    input_tensor = tensors[op.input_indices[0]]
    constant_value = 0.0
    if types.is_quantized(input_tensor):
        assert input_tensor.quantization is not None  # noqa: S101
        constant_value = float(input_tensor.quantization.zero_points[0])

    return Pad(
        pad_width=pad_width,
        constant_value=constant_value,
        name=f"pad_{op.output_indices[0]}",
    )
