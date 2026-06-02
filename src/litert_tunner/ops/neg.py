"""NEG op implementation for litert_tunner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import keras

from litert_tunner.graph import types
from litert_tunner.ops import registry, utils

if TYPE_CHECKING:
    from litert_tunner.ops.utils import TensorLike

    ShapeLike = tuple[int, ...] | list[int] | list[tuple[int, ...]]


class QuantizedNeg(keras.Layer, types.Writable):
    """Simulates TFLite's quantized NEG (negation) op.

    The forward pass performs:
        1. Dequantize INT8 input to float32 (if quantized)
        2. Apply negation (unary minus)
        3. Fake-quantize output to INT8 (if quantized)
    """

    def __init__(
        self,
        input_scale: float | None,
        input_zero_point: float | None,
        output_scale: float | None,
        output_zero_point: float | None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._input_scale = input_scale
        self._input_zero_point = input_zero_point
        self._output_scale = output_scale
        self._output_zero_point = output_zero_point

    def build(self, input_shape: ShapeLike) -> None:
        """Create quantization params."""
        if self._input_scale is not None and self._input_zero_point is not None:
            self.input_quant = utils.QuantizationVars(
                self,
                "input",
                self._input_scale,
                self._input_zero_point,
                trainable=False,
            )
        else:
            self.input_quant = None

        if self._output_scale is not None and self._output_zero_point is not None:
            self.output_quant = utils.QuantizationVars(
                self,
                "output",
                self._output_scale,
                self._output_zero_point,
                trainable=False,
            )
        else:
            self.output_quant = None
        super().build(input_shape)

    def call(self, x: TensorLike) -> TensorLike:
        """Forward pass simulating quantized NEG."""
        # 1. Dequantize
        input_float = self.input_quant.dequantize(x) if self.input_quant is not None else x
        # 2. Negate
        output_float = -input_float
        # 3. Quantize to simulated INT8
        if self.output_quant is not None:
            return self.output_quant.quantize(output_float)
        return output_float

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_scale": self._input_scale,
            "input_zero_point": self._input_zero_point,
            "output_scale": self._output_scale,
            "output_zero_point": self._output_zero_point,
        })
        return config

    def collect_write_ops(
        self,
        op: types.OperatorInfo,
    ) -> tuple[list[types.BufferWriteOp], list[types.QuantizationWriteOp]]:
        """Return flatbuffer write instructions for the NEG layer."""
        quant_writes: list[types.QuantizationWriteOp] = []
        if self.input_quant is not None:
            quant_writes.append(self.input_quant.make_write_op(op.input_indices[0]))
        if self.output_quant is not None:
            quant_writes.append(self.output_quant.make_write_op(op.output_indices[0]))
        return [], quant_writes


@registry.register_op("NEG")
def build_neg(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a QuantizedNeg layer from parsed TFLite operator info."""
    input_tensor = tensors[op.input_indices[0]]
    output_tensor = tensors[op.output_indices[0]]

    input_quant = input_tensor.quantization
    output_quant = output_tensor.quantization

    input_scale = float(input_quant.scales[0]) if input_quant is not None else None
    input_zero_point = float(input_quant.zero_points[0]) if input_quant is not None else None
    output_scale = float(output_quant.scales[0]) if output_quant is not None else None
    output_zero_point = float(output_quant.zero_points[0]) if output_quant is not None else None

    return QuantizedNeg(
        input_scale=input_scale,
        input_zero_point=input_zero_point,
        output_scale=output_scale,
        output_zero_point=output_zero_point,
        name=f"quantized_neg_{op.output_indices[0]}",
    )
