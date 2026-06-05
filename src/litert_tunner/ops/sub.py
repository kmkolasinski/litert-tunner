"""SUB op implementation for litert_tunner.

Supports both quantized (INT8) and float32 operations.
"""

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


class QuantizedSub(keras.Layer, types.Writable):
    """Simulates TFLite's quantized SUB op.

    The forward pass performs:
        1. Dequantize both INT8 inputs to float32
        2. Subtract in float32 (input1 - input2)
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
        self._fused_activation = fused_activation
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

        # Output quantization params (frozen)
        self.output_quant = utils.QuantizationVars(
            self,
            "output",
            self._output_scale,
            self._output_zero_point,
            trainable=False,
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
        """Forward pass simulating quantized SUB.

        When one input is a constant stored in the layer, ``inputs`` is a single
        tensor and the constant is retrieved from ``self.constant_input``.
        """
        if self._constant_input_data is not None:
            # One input is a stored constant
            dynamic_input = inputs
            if self._constant_input_index == 0:
                x1, x2 = self.constant_input, dynamic_input
            else:
                x1, x2 = dynamic_input, self.constant_input
        else:
            x1, x2 = inputs

        # 1. Dequantize
        x1_float = self.input1_quant.dequantize(x1)
        x2_float = self.input2_quant.dequantize(x2)
        # 2. Subtract
        output_float = ops.subtract(x1_float, x2_float)
        # 3. Fused activation
        output_float = utils.apply_fused_activation(output_float, self._fused_activation)
        # 4. Quantize to simulated INT8
        return self.output_quant.quantize(output_float)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input1_scale": self._input1_scale,
            "input1_zero_point": self._input1_zero_point,
            "input2_scale": self._input2_scale,
            "input2_zero_point": self._input2_zero_point,
            "output_scale": self._output_scale,
            "output_zero_point": self._output_zero_point,
            "fused_activation": self._fused_activation,
        })
        return config

    def collect_write_ops(
        self,
        op: types.OperatorInfo,
    ) -> tuple[list[types.BufferWriteOp], list[types.QuantizationWriteOp]]:
        """Return flatbuffer write instructions for the SUB layer."""
        quant_writes: list[types.QuantizationWriteOp] = []
        quant_writes.append(self.input1_quant.make_write_op(op.input_indices[0]))
        quant_writes.append(self.input2_quant.make_write_op(op.input_indices[1]))
        quant_writes.append(self.output_quant.make_write_op(op.output_indices[0]))
        return [], quant_writes


class FloatSub(keras.Layer):
    """Simulates TFLite's float32 SUB op.

    The forward pass performs:
        1. Subtract in float32
        2. Apply fused activation (if any)

    This layer has no persistent weights to write back and emits no write ops.
    """

    def __init__(
        self,
        fused_activation: int = utils.FUSED_ACTIVATION_NONE,
        constant_input: np.ndarray | None = None,
        constant_input_index: int = -1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._fused_activation = fused_activation
        self._constant_input_data = constant_input
        self._constant_input_index = constant_input_index

    def build(self, input_shape: ShapeLike) -> None:
        """Build the layer and create weights."""
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
        """Forward pass for float32 SUB."""
        if self._constant_input_data is not None:
            dynamic_input = inputs
            if self._constant_input_index == 0:
                x1, x2 = self.constant_input, dynamic_input
            else:
                x1, x2 = dynamic_input, self.constant_input
        else:
            x1, x2 = inputs

        output_float = ops.subtract(x1, x2)
        return utils.apply_fused_activation(output_float, self._fused_activation)

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "fused_activation": self._fused_activation,
        })
        return config


def _extract_float_constant(
    input1_tensor: types.TensorInfo,
    input2_tensor: types.TensorInfo,
) -> tuple[np.ndarray | None, int]:
    """Extract a constant float32 input tensor."""
    for idx, tensor in enumerate([input1_tensor, input2_tensor]):
        if tensor.data is not None:
            return tensor.data.astype(np.float32), idx
    return None, -1


def _build_quantized_sub(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a QuantizedSub layer from parsed TFLite operator info."""
    input1_tensor = tensors[op.input_indices[0]]
    input2_tensor = tensors[op.input_indices[1]]
    output_tensor = tensors[op.output_indices[0]]

    input1_quant = input1_tensor.quantization
    input2_quant = input2_tensor.quantization
    output_quant = output_tensor.quantization

    if input1_quant is None or input2_quant is None or output_quant is None:
        msg = "SUB requires quantized input and output tensors"
        raise ValueError(msg)

    fused_activation = op.options.get("fused_activation_function", utils.FUSED_ACTIVATION_NONE)

    # Detect constant inputs and pre-quantize them to simulated INT8
    constant_input, constant_input_index = utils.extract_constant_input(
        input1_tensor, input1_quant, input2_tensor, input2_quant
    )

    return QuantizedSub(
        input1_scale=float(input1_quant.scales[0]),
        input1_zero_point=float(input1_quant.zero_points[0]),
        input2_scale=float(input2_quant.scales[0]),
        input2_zero_point=float(input2_quant.zero_points[0]),
        output_scale=float(output_quant.scales[0]),
        output_zero_point=float(output_quant.zero_points[0]),
        fused_activation=fused_activation,
        constant_input=constant_input,
        constant_input_index=constant_input_index,
        name=f"quantized_sub_{op.output_indices[0]}",
    )


def _build_float_sub(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a FloatSub layer from parsed TFLite operator info."""
    input1_tensor = tensors[op.input_indices[0]]
    input2_tensor = tensors[op.input_indices[1]]

    fused_activation = op.options.get("fused_activation_function", utils.FUSED_ACTIVATION_NONE)

    constant_input, constant_input_index = _extract_float_constant(input1_tensor, input2_tensor)

    return FloatSub(
        fused_activation=fused_activation,
        constant_input=constant_input,
        constant_input_index=constant_input_index,
        name=f"float_sub_{op.output_indices[0]}",
    )


@registry.register_op("SUB")
def build_sub(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a SUB layer from parsed TFLite operator info."""
    input_tensor = tensors[op.input_indices[0]]
    if types.is_quantized(input_tensor):
        return _build_quantized_sub(op, tensors)
    return _build_float_sub(op, tensors)
