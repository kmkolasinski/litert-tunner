"""SQUARED_DIFFERENCE op implementation for litert_tunner."""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

import keras
import numpy as np
from keras import ops

from litert_tunner.graph import types
from litert_tunner.ops import registry, utils

if TYPE_CHECKING:
    from litert_tunner.ops.utils import TensorLike

    ShapeLike = tuple[int, ...] | list[int] | list[tuple[int, ...]]


class QuantizedSquaredDifference(keras.Layer, types.Writable):
    """Simulates TFLite's quantized SQUARED_DIFFERENCE op.

    The forward pass performs:
        1. Dequantize both INT8 inputs to float32
        2. Compute squared difference: (input1 - input2) ** 2
        3. Fake-quantize output to INT8
    """

    def __init__(
        self,
        input1_scale: float,
        input1_zero_point: float,
        input2_scale: float,
        input2_zero_point: float,
        output_scale: float,
        output_zero_point: float,
        constant_input: np.ndarray | None = None,
        constant_input_index: int = -1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._input1_scale = input1_scale
        self._input1_zero_point = input1_zero_point
        self._input2_scale = input2_scale
        self._input2_zero_point = input2_zero_point
        self._output_scale = output_scale
        self._output_zero_point = output_zero_point
        self._constant_input_data = constant_input
        self._constant_input_index = constant_input_index

    def build(self, input_shape: ShapeLike) -> None:
        """Create quantization params."""
        self.input1_quant = utils.QuantizationVars(
            self,
            "input1",
            self._input1_scale,
            self._input1_zero_point,
            trainable=False,
        )

        self.input2_quant = utils.QuantizationVars(
            self,
            "input2",
            self._input2_scale,
            self._input2_zero_point,
            trainable=False,
        )

        self.output_quant = utils.QuantizationVars(
            self,
            "output",
            self._output_scale,
            self._output_zero_point,
            trainable=True,
        )

        # Constant input (frozen), stored as simulated INT8 float32
        if self._constant_input_data is not None:
            self.constant_input = self.add_weight(
                name="constant_input",
                shape=self._constant_input_data.shape,
                initializer=keras.initializers.Constant(
                    typing.cast("float", self._constant_input_data.astype(np.float32))
                ),
                trainable=False,
            )

        super().build(input_shape)

    def call(
        self, inputs: TensorLike | tuple[TensorLike, TensorLike] | list[TensorLike]
    ) -> TensorLike:
        """Forward pass simulating quantized SQUARED_DIFFERENCE."""
        if self._constant_input_data is not None:
            dynamic_input = inputs
            if self._constant_input_index == 0:
                x1, x2 = self.constant_input, dynamic_input
            else:
                x1, x2 = dynamic_input, self.constant_input
        else:
            x1, x2 = inputs

        x1_float = self.input1_quant.dequantize(x1)
        x2_float = self.input2_quant.dequantize(x2)

        output_float = ops.square(ops.subtract(x1_float, x2_float))

        return self.output_quant.quantize(output_float)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input1_scale": self._input1_scale,
                "input1_zero_point": self._input1_zero_point,
                "input2_scale": self._input2_scale,
                "input2_zero_point": self._input2_zero_point,
                "output_scale": self._output_scale,
                "output_zero_point": self._output_zero_point,
            }
        )
        return config

    def collect_write_ops(
        self,
        op: types.OperatorInfo,
    ) -> tuple[list[types.BufferWriteOp], list[types.QuantizationWriteOp]]:
        """Return flatbuffer write instructions for the SQUARED_DIFFERENCE layer."""
        quant_writes: list[types.QuantizationWriteOp] = [
            self.input1_quant.make_write_op(op.input_indices[0]),
            self.input2_quant.make_write_op(op.input_indices[1]),
            self.output_quant.make_write_op(op.output_indices[0]),
        ]
        return [], quant_writes


@registry.register_op("SQUARED_DIFFERENCE")
def build_squared_difference(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a QuantizedSquaredDifference layer from parsed TFLite operator info."""
    input1_tensor = tensors[op.input_indices[0]]
    input2_tensor = tensors[op.input_indices[1]]
    output_tensor = tensors[op.output_indices[0]]

    input1_quant = input1_tensor.quantization
    input2_quant = input2_tensor.quantization
    output_quant = output_tensor.quantization

    if input1_quant is None or input2_quant is None or output_quant is None:
        msg = "SQUARED_DIFFERENCE requires quantized input and output tensors"
        raise ValueError(msg)

    # Detect constant inputs and pre-quantize them to simulated INT8
    constant_input, constant_input_index = utils.extract_constant_input(
        input1_tensor, input1_quant, input2_tensor, input2_quant
    )

    return QuantizedSquaredDifference(
        input1_scale=float(input1_quant.scales[0]),
        input1_zero_point=float(input1_quant.zero_points[0]),
        input2_scale=float(input2_quant.scales[0]),
        input2_zero_point=float(input2_quant.zero_points[0]),
        output_scale=float(output_quant.scales[0]),
        output_zero_point=float(output_quant.zero_points[0]),
        constant_input=constant_input,
        constant_input_index=constant_input_index,
        name=f"quantized_squared_difference_{op.output_indices[0]}",
    )
