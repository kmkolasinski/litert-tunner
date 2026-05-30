"""MUL op implementation for litert_tunner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import keras
from keras import ops

from litert_tunner.graph import types
from litert_tunner.ops import registry, utils

if TYPE_CHECKING:
    from litert_tunner.ops.utils import TensorLike

    ShapeLike = tuple[int, ...] | list[int] | list[tuple[int, ...]]


class QuantizedMul(keras.Layer, types.Writable):
    """Simulates TFLite's quantized MUL op.

    The forward pass performs:
        1. Dequantize both INT8 inputs to float32
        2. Multiply in float32
        3. Apply fused activation (if any)
        4. Fake-quantize output to INT8

    Trainable parameters: output_scale, output_zero_point.
    Frozen parameters: input scales and zero-points.
    """

    def __init__(
        self,
        input1_scale: float,
        input1_zero_point: float,
        input2_scale: float,
        input2_zero_point: float,
        output_scale: float,
        output_zero_point: float,
        fused_activation: int = utils.FUSED_ACTIVATION_NONE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._input1_scale = input1_scale
        self._input1_zero_point = input1_zero_point
        self._input2_scale = input2_scale
        self._input2_zero_point = input2_zero_point
        self._output_scale = output_scale
        self._output_zero_point = output_zero_point
        self._fused_activation = fused_activation

    def build(self, input_shape: ShapeLike) -> None:
        """Create quantization params."""
        self.input1_scale = self.add_weight(
            name="input1_scale",
            shape=(),
            initializer=keras.initializers.Constant(self._input1_scale),
            trainable=False,
        )
        self.input1_zero_point = self.add_weight(
            name="input1_zero_point",
            shape=(),
            initializer=keras.initializers.Constant(self._input1_zero_point),
            trainable=False,
        )
        self.input2_scale = self.add_weight(
            name="input2_scale",
            shape=(),
            initializer=keras.initializers.Constant(self._input2_scale),
            trainable=False,
        )
        self.input2_zero_point = self.add_weight(
            name="input2_zero_point",
            shape=(),
            initializer=keras.initializers.Constant(self._input2_zero_point),
            trainable=False,
        )

        # Output quantization params (trainable)
        self.output_scale = self.add_weight(
            name="output_scale",
            shape=(),
            initializer=keras.initializers.Constant(self._output_scale),
            trainable=True,
        )
        self.output_zero_point = self.add_weight(
            name="output_zero_point",
            shape=(),
            initializer=keras.initializers.Constant(self._output_zero_point),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tuple[TensorLike, TensorLike] | list[TensorLike]) -> TensorLike:
        """Forward pass simulating quantized MUL."""
        x1, x2 = inputs
        # 1. Dequantize
        x1_float = utils.dequantize_ste(x1, self.input1_scale, self.input1_zero_point)
        x2_float = utils.dequantize_ste(x2, self.input2_scale, self.input2_zero_point)
        # 2. Multiply
        output_float = ops.multiply(x1_float, x2_float)
        # 3. Fused activation
        output_float = utils.apply_fused_activation(output_float, self._fused_activation)
        # 4. Quantize to simulated INT8
        return utils.quantize_ste(output_float, self.output_scale, self.output_zero_point)

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
                "fused_activation": self._fused_activation,
            }
        )
        return config

    def collect_write_ops(
        self,
        op: types.OperatorInfo,
    ) -> tuple[list[types.BufferWriteOp], list[types.QuantizationWriteOp]]:
        """Return flatbuffer write instructions for the MUL layer."""
        quant_writes: list[types.QuantizationWriteOp] = []
        quant_writes.append(
            utils.make_quant_write_op(
                op.input_indices[0], self.input1_scale, self.input1_zero_point
            )
        )
        quant_writes.append(
            utils.make_quant_write_op(
                op.input_indices[1], self.input2_scale, self.input2_zero_point
            )
        )
        quant_writes.append(
            utils.make_quant_write_op(
                op.output_indices[0], self.output_scale, self.output_zero_point
            )
        )
        return [], quant_writes


@registry.register_op("MUL")
def build_mul(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a QuantizedMul layer from parsed TFLite operator info."""
    input1_tensor = tensors[op.input_indices[0]]
    input2_tensor = tensors[op.input_indices[1]]
    output_tensor = tensors[op.output_indices[0]]

    input1_quant = input1_tensor.quantization
    input2_quant = input2_tensor.quantization
    output_quant = output_tensor.quantization

    if input1_quant is None or input2_quant is None or output_quant is None:
        msg = "MUL requires quantized input and output tensors"
        raise ValueError(msg)

    fused_activation = op.options.get("fused_activation_function", utils.FUSED_ACTIVATION_NONE)

    return QuantizedMul(
        input1_scale=float(input1_quant.scales[0]),
        input1_zero_point=float(input1_quant.zero_points[0]),
        input2_scale=float(input2_quant.scales[0]),
        input2_zero_point=float(input2_quant.zero_points[0]),
        output_scale=float(output_quant.scales[0]),
        output_zero_point=float(output_quant.zero_points[0]),
        fused_activation=fused_activation,
        name=f"quantized_mul_{op.output_indices[0]}",
    )
