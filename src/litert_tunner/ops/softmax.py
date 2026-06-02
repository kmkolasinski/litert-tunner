"""SOFTMAX op implementation for litert_tunner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import keras
from keras import ops

from litert_tunner.graph import types
from litert_tunner.ops import registry, utils

if TYPE_CHECKING:
    from litert_tunner.ops.utils import TensorLike

    ShapeLike = tuple[int, ...] | list[int] | list[tuple[int, ...]]


class QuantizedSoftmax(keras.Layer, types.Writable):
    """Simulates TFLite's quantized SOFTMAX op.

    The forward pass performs:
        1. Dequantize INT8 input to float32
        2. Apply softmax along the specified axis
        3. Fake-quantize output to INT8

    SOFTMAX output quantization is hardcoded in TFLite (scale=1/256, zp=-128),
    so output quant params are kept frozen (non-trainable).
    """

    def __init__(
        self,
        input_scale: float,
        input_zero_point: float,
        output_scale: float,
        output_zero_point: float,
        axis: int = -1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._input_scale = input_scale
        self._input_zero_point = input_zero_point
        self._output_scale = output_scale
        self._output_zero_point = output_zero_point
        self._axis = axis

    def build(self, input_shape: ShapeLike) -> None:
        """Create quantization params."""
        self.input_quant = utils.QuantizationVars(
            self,
            "input",
            self._input_scale,
            self._input_zero_point,
            trainable=False,
        )

        # SOFTMAX output quantization is hardcoded in TFLite (scale=1/256, zp=-128)
        # So we keep it frozen.
        self.output_quant = utils.QuantizationVars(
            self,
            "output",
            self._output_scale,
            self._output_zero_point,
            trainable=False,
        )

        super().build(input_shape)

    def call(self, x: TensorLike) -> TensorLike:
        """Forward pass simulating quantized SOFTMAX."""
        # 1. Dequantize
        input_float = self.input_quant.dequantize(x)
        # 2. Softmax
        output_float = ops.softmax(input_float, axis=self._axis)
        # 3. Quantize to simulated INT8
        return self.output_quant.quantize(output_float)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_scale": self._input_scale,
            "input_zero_point": self._input_zero_point,
            "output_scale": self._output_scale,
            "output_zero_point": self._output_zero_point,
            "axis": self._axis,
        })
        return config

    def collect_write_ops(
        self,
        op: types.OperatorInfo,
    ) -> tuple[list[types.BufferWriteOp], list[types.QuantizationWriteOp]]:
        """Return flatbuffer write instructions for the SOFTMAX layer."""
        quant_writes: list[types.QuantizationWriteOp] = []
        quant_writes.append(self.input_quant.make_write_op(op.input_indices[0]))
        quant_writes.append(self.output_quant.make_write_op(op.output_indices[0]))
        return [], quant_writes


@registry.register_op("SOFTMAX")
def build_softmax(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a QuantizedSoftmax layer from parsed TFLite operator info."""
    input_tensor = tensors[op.input_indices[0]]
    output_tensor = tensors[op.output_indices[0]]

    input_quant = input_tensor.quantization
    output_quant = output_tensor.quantization

    if input_quant is None or output_quant is None:
        msg = "SOFTMAX requires quantized input and output tensors"
        raise ValueError(msg)

    # TFLite SOFTMAX always operates on the last axis by default.
    # The 'beta' option is not used for axis — TFLite SOFTMAX is always last-axis.
    return QuantizedSoftmax(
        input_scale=float(input_quant.scales[0]),
        input_zero_point=float(input_quant.zero_points[0]),
        output_scale=float(output_quant.scales[0]),
        output_zero_point=float(output_quant.zero_points[0]),
        axis=-1,
        name=f"quantized_softmax_{op.output_indices[0]}",
    )
