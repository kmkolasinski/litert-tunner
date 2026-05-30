"""Quantize and Dequantize op builders and Keras layers for litert_tunner."""

from __future__ import annotations

import typing

import keras

from litert_tunner.graph import types
from litert_tunner.ops import registry, utils

TensorLike = typing.Any
ShapeLike = tuple[int | None, ...] | list[int | None] | list[tuple[int | None, ...]]


class Dequantize(keras.Layer, types.Writable):
    """Dequantizes INT8 values to float32.

    Formula: real_value = scale * (int8_value - zero_point)

    This layer casts the input to float32 before dequantization.

    Args:
        scale: Dequantization scale value(s). Can be a scalar or array.
        zero_point: Zero point value(s). Can be a scalar or array.
        passthrough: If True, passes the input through unchanged (e.g. if the
            input is already dequantized).
        name: Layer name.
    """

    def __init__(
        self,
        scale: float,
        zero_point: float,
        *,
        passthrough: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._initial_scale = scale
        self._initial_zero_point = zero_point
        self._passthrough = passthrough

    def build(self, input_shape: ShapeLike) -> None:
        """Create scale and zero_point weights for the layer."""
        self.scale = self.add_weight(
            name="scale",
            shape=(),
            initializer=keras.initializers.Constant(self._initial_scale),
            trainable=False,
        )
        self.zero_point = self.add_weight(
            name="zero_point",
            shape=(),
            initializer=keras.initializers.Constant(self._initial_zero_point),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x: TensorLike) -> TensorLike:
        """Dequantize input tensor."""
        if self._passthrough:
            return x
        return utils.dequantize_ste(x, self.scale, self.zero_point)

    def get_config(self) -> dict[str, typing.Any]:
        """Return the configuration dictionary for serialization of the layer."""
        config = super().get_config()
        config.update(
            {
                "scale": self._initial_scale,
                "zero_point": self._initial_zero_point,
                "passthrough": self._passthrough,
            }
        )
        return config

    def collect_write_ops(
        self,
        op: types.OperatorInfo,
    ) -> tuple[list[types.BufferWriteOp], list[types.QuantizationWriteOp]]:
        """Return flatbuffer write instructions for the Dequantize layer.

        Updates the quantization params on the input tensor.

        Args:
            op: The OperatorInfo that this layer was built from.

        Returns:
            A tuple of (buffer_writes, quantization_writes).
        """
        if self._passthrough:
            return [], []
        input_tensor_idx = op.input_indices[0]
        quant_write = utils.make_quant_write_op(input_tensor_idx, self.scale, self.zero_point)
        return [], [quant_write]


class Quantize(keras.Layer, types.Writable):
    """Quantizes float32 values to simulated INT8 with STE gradients.

    Formula: int8_value = clamp(round(x / scale) + zero_point, -128, 127)

    The output is still float32 (to allow gradient flow), but values
    are constrained to the INT8 range.

    Args:
        scale: Quantization scale value.
        zero_point: Zero point value.
        trainable: Whether scale and zero_point are trainable.
        name: Layer name.
    """

    def __init__(
        self,
        scale: float,
        zero_point: float,
        *,
        trainable: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._initial_scale = scale
        self._initial_zero_point = zero_point
        self._trainable_params = trainable

    def build(self, input_shape: ShapeLike) -> None:
        """Create scale and zero_point weights for the layer."""
        self.scale = self.add_weight(
            name="scale",
            shape=(),
            initializer=keras.initializers.Constant(self._initial_scale),
            trainable=self._trainable_params,
        )
        self.zero_point = self.add_weight(
            name="zero_point",
            shape=(),
            initializer=keras.initializers.Constant(self._initial_zero_point),
            trainable=self._trainable_params,
        )
        super().build(input_shape)

    def call(self, x: TensorLike) -> TensorLike:
        """Quantize input to simulated INT8."""
        return utils.quantize_ste(x, self.scale, self.zero_point)

    def get_config(self) -> dict[str, typing.Any]:
        """Return the configuration dictionary for serialization of the layer."""
        config = super().get_config()
        config.update(
            {
                "scale": self._initial_scale,
                "zero_point": self._initial_zero_point,
                "trainable": self._trainable_params,
            }
        )
        return config

    def collect_write_ops(
        self,
        op: types.OperatorInfo,
    ) -> tuple[list[types.BufferWriteOp], list[types.QuantizationWriteOp]]:
        """Return flatbuffer write instructions for the Quantize layer.

        Updates the quantization params on the output tensor.

        Args:
            op: The OperatorInfo that this layer was built from.

        Returns:
            A tuple of (buffer_writes, quantization_writes).
        """
        output_tensor_idx = op.output_indices[0]
        quant_write = utils.make_quant_write_op(output_tensor_idx, self.scale, self.zero_point)
        return [], [quant_write]


@registry.register_op("QUANTIZE")
def build_quantize(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a Quantize layer from parsed TFLite operator info.

    TFLite QUANTIZE inputs:
        [0] input tensor (FLOAT32 or INT8)
    TFLite QUANTIZE outputs:
        [0] output tensor (INT8)
    """
    output_tensor = tensors[op.output_indices[0]]
    output_quant = output_tensor.quantization

    if output_quant is None:
        msg = "QUANTIZE op requires quantized output tensor"
        raise ValueError(msg)

    scale = float(output_quant.scales[0])
    zero_point = float(output_quant.zero_points[0])

    return Quantize(
        scale=scale,
        zero_point=zero_point,
        trainable=True,
        name=f"quantize_{op.output_indices[0]}",
    )


@registry.register_op("DEQUANTIZE")
def build_dequantize(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a Dequantize layer from parsed TFLite operator info.

    TFLite DEQUANTIZE inputs:
        [0] input tensor (INT8)
    TFLite DEQUANTIZE outputs:
        [0] output tensor (FLOAT32)
    """
    input_tensor = tensors[op.input_indices[0]]
    input_quant = input_tensor.quantization

    if input_quant is None:
        msg = "DEQUANTIZE op requires quantized input tensor"
        raise ValueError(msg)

    scale = float(input_quant.scales[0])
    zero_point = float(input_quant.zero_points[0])

    return Dequantize(
        scale=scale,
        zero_point=zero_point,
        name=f"dequantize_{op.output_indices[0]}",
    )
