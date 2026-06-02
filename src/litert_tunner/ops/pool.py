"""Pooling op implementations for litert_tunner.

Simulates TFLite's quantized pooling operations (MAX_POOL_2D, etc.)
as Keras layers. Pooling ops are passthrough for quantization —
they do not change scale/zero-point and have no trainable parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import keras
from keras import ops

from litert_tunner.ops import registry, utils

if TYPE_CHECKING:
    from litert_tunner.graph import types
    from litert_tunner.ops.utils import TensorLike

# TFLite padding integer codes (shared with conv2d.py)
_PADDING_SAME = 0
_PADDING_VALID = 1

_PADDING_MAP: dict[int, str] = {
    _PADDING_SAME: "same",
    _PADDING_VALID: "valid",
}


class QuantizedMaxPool2D(keras.Layer):
    """Simulates TFLite's quantized MAX_POOL_2D op.

    MAX_POOL_2D is a passthrough for quantization: it does not change
    the scale or zero-point of the tensor. The layer simply applies
    max pooling followed by an optional fused activation.

    No trainable parameters. Does not implement ``Writable``.

    Args:
        pool_size: Pooling window as (pH, pW).
        strides: Pooling strides as (sH, sW).
        padding: Padding string, "same" or "valid".
        fused_activation: TFLite fused activation code (0=none, 1=relu, 3=relu6).
        name: Layer name.
    """

    def __init__(
        self,
        pool_size: tuple[int, int] = (2, 2),
        strides: tuple[int, int] = (2, 2),
        padding: str = "same",
        fused_activation: int = utils.FUSED_ACTIVATION_NONE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._pool_size = pool_size
        self._strides = strides
        self._padding = padding
        self._fused_activation = fused_activation

    def call(self, x: TensorLike) -> TensorLike:
        """Forward pass applying max pooling.

        Args:
            x: Input tensor of shape (batch, H, W, C).

        Returns:
            Pooled output tensor.
        """
        output = ops.max_pool(x, self._pool_size, self._strides, self._padding)
        return utils.apply_fused_activation(output, self._fused_activation)

    def get_config(self):
        """Return the configuration dictionary for serialization of the layer."""
        config = super().get_config()
        config.update({
            "pool_size": self._pool_size,
            "strides": self._strides,
            "padding": self._padding,
            "fused_activation": self._fused_activation,
        })
        return config


def _map_padding(padding_code: int) -> str:
    """Map TFLite padding integer code to Keras padding string.

    Args:
        padding_code: Integer padding code (0=SAME, 1=VALID).

    Returns:
        Keras padding string.

    Raises:
        ValueError: If the padding code is not recognized.
    """
    if padding_code not in _PADDING_MAP:
        msg = f"Unsupported padding code: {padding_code}"
        raise ValueError(msg)
    return _PADDING_MAP[padding_code]


@registry.register_op("MAX_POOL_2D")
def build_max_pool_2d(
    op: types.OperatorInfo,
    _tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a QuantizedMaxPool2D layer from parsed TFLite operator info.

    TFLite MAX_POOL_2D inputs:
        [0] input tensor (INT8), shape (batch, H, W, C)

    TFLite MAX_POOL_2D outputs:
        [0] output tensor (INT8)

    Args:
        op: Parsed operator info with input/output indices and options.
        tensors: All tensors in the graph.
        graph_def: The parsed GraphDef.

    Returns:
        A configured QuantizedMaxPool2D Keras layer.
    """
    filter_h = op.options.get("FilterHeight", 2)
    filter_w = op.options.get("FilterWidth", 2)
    stride_h = op.options.get("StrideH", 2)
    stride_w = op.options.get("StrideW", 2)
    padding_code = op.options.get("Padding", _PADDING_SAME)
    padding = _map_padding(padding_code)
    fused_activation = op.options.get("fused_activation_function", utils.FUSED_ACTIVATION_NONE)

    return QuantizedMaxPool2D(
        pool_size=(filter_h, filter_w),
        strides=(stride_h, stride_w),
        padding=padding,
        fused_activation=fused_activation,
        name=f"quantized_max_pool_2d_{op.output_indices[0]}",
    )
