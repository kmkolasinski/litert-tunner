"""RELU op implementation for litert_tunner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import keras
from keras import ops

from litert_tunner.graph import types
from litert_tunner.ops import registry, utils

if TYPE_CHECKING:
    from litert_tunner.ops.utils import TensorLike

    ShapeLike = tuple[int, ...] | list[int] | list[tuple[int, ...]]


class QuantizedRelu(keras.Layer, types.Writable):
    """Simulates TFLite's quantized RELU op.

    The forward pass performs:
        1. Dequantize INT8 input to float32
        2. Apply ReLU activation
        3. Fake-quantize output to INT8
    """

    def __init__(
        self,
        input_scale: float,
        input_zero_point: float,
        output_scale: float,
        output_zero_point: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._input_scale = input_scale
        self._input_zero_point = input_zero_point
        self._output_scale = output_scale
        self._output_zero_point = output_zero_point

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
        """Forward pass simulating quantized RELU."""
        # 1. Dequantize
        input_float = self.input_quant.dequantize(x)
        # 2. ReLU
        output_float = ops.relu(input_float)
        # 3. Quantize to simulated INT8
        return self.output_quant.quantize(output_float)

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
        """Return flatbuffer write instructions for the RELU layer."""
        quant_writes: list[types.QuantizationWriteOp] = [
            self.input_quant.make_write_op(op.input_indices[0]),
            self.output_quant.make_write_op(op.output_indices[0]),
        ]
        return [], quant_writes


@registry.register_op("RELU")
def build_relu(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a QuantizedRelu layer from parsed TFLite operator info."""
    input_tensor = tensors[op.input_indices[0]]
    output_tensor = tensors[op.output_indices[0]]

    input_quant = input_tensor.quantization
    output_quant = output_tensor.quantization

    if input_quant is None or output_quant is None:
        msg = "RELU requires quantized input and output tensors"
        raise ValueError(msg)

    return QuantizedRelu(
        input_scale=float(input_quant.scales[0]),
        input_zero_point=float(input_quant.zero_points[0]),
        output_scale=float(output_quant.scales[0]),
        output_zero_point=float(output_quant.zero_points[0]),
        name=f"quantized_relu_{op.output_indices[0]}",
    )
