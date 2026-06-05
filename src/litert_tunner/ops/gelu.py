"""GELU op implementation for litert_tunner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import keras
from keras import ops

from litert_tunner.graph import types
from litert_tunner.ops import registry, utils

if TYPE_CHECKING:
    from litert_tunner.ops.utils import TensorLike

    ShapeLike = tuple[int, ...] | list[int] | list[tuple[int, ...]]


class QuantizedGelu(keras.Layer, types.Writable):
    """Simulates TFLite's quantized GELU op.

    The forward pass performs:
        1. Dequantize INT8 input to float32
        2. Apply GELU activation
        3. Fake-quantize output to INT8
    """

    def __init__(
        self,
        input_scale: float,
        input_zero_point: float,
        output_scale: float,
        output_zero_point: float,
        *,
        approximate: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._input_scale = input_scale
        self._input_zero_point = input_zero_point
        self._output_scale = output_scale
        self._output_zero_point = output_zero_point
        self._approximate = approximate

    def build(self, input_shape: ShapeLike) -> None:
        """Create quantization params."""
        self.input_quant = utils.QuantizationVars(
            self,
            "input",
            self._input_scale,
            self._input_zero_point,
            trainable=False,
        )

        self.output_quant = utils.QuantizationVars(
            self,
            "output",
            self._output_scale,
            self._output_zero_point,
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x: TensorLike) -> TensorLike:
        """Forward pass simulating quantized GELU."""
        # 1. Dequantize
        input_float = self.input_quant.dequantize(x)
        # 2. GELU
        output_float = ops.gelu(input_float, approximate=self._approximate)
        # 3. Quantize to simulated INT8
        return self.output_quant.quantize(output_float)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_scale": self._input_scale,
            "input_zero_point": self._input_zero_point,
            "output_scale": self._output_scale,
            "output_zero_point": self._output_zero_point,
            "approximate": self._approximate,
        })
        return config

    def collect_write_ops(
        self,
        op: types.OperatorInfo,
    ) -> tuple[list[types.BufferWriteOp], list[types.QuantizationWriteOp]]:
        """Return flatbuffer write instructions for the GELU layer."""
        quant_writes: list[types.QuantizationWriteOp] = [
            self.input_quant.make_write_op(op.input_indices[0]),
            self.output_quant.make_write_op(op.output_indices[0]),
        ]
        return [], quant_writes


class FloatGelu(keras.Layer):
    """Simulates TFLite's float32 GELU op.

    The forward pass performs:
        1. Apply GELU activation

    This layer has no persistent weights to write back and emits no write ops.
    """

    def __init__(self, *, approximate: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._approximate = approximate

    def call(self, x: TensorLike) -> TensorLike:
        """Forward pass for float32 GELU."""
        return ops.gelu(x, approximate=self._approximate)

    def get_config(self):
        config = super().get_config()
        config.update({
            "approximate": self._approximate,
        })
        return config


def _build_quantized_gelu(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a QuantizedGelu layer from parsed TFLite operator info."""
    input_tensor = tensors[op.input_indices[0]]
    output_tensor = tensors[op.output_indices[0]]

    input_quant = input_tensor.quantization
    output_quant = output_tensor.quantization

    if input_quant is None or output_quant is None:
        msg = "GELU requires quantized input and output tensors"
        raise ValueError(msg)
    # TFLite GELU options might contain an approximate flag (GeluOptions).
    # Default is False.
    approximate = op.options.get("Approximate", False)

    return QuantizedGelu(
        input_scale=float(input_quant.scales[0]),
        input_zero_point=float(input_quant.zero_points[0]),
        output_scale=float(output_quant.scales[0]),
        output_zero_point=float(output_quant.zero_points[0]),
        approximate=approximate,
        name=f"quantized_gelu_{op.output_indices[0]}",
    )


def _build_float_gelu(
    op: types.OperatorInfo,
) -> keras.Layer:
    """Build a FloatGelu layer from parsed TFLite operator info."""
    approximate = op.options.get("Approximate", False)
    return FloatGelu(
        approximate=approximate,
        name=f"float_gelu_{op.output_indices[0]}",
    )


@registry.register_op("GELU")
def build_gelu(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a GELU layer from parsed TFLite operator info."""
    input_tensor = tensors[op.input_indices[0]]
    if types.is_quantized(input_tensor):
        return _build_quantized_gelu(op, tensors)
    return _build_float_gelu(op)
