"""Fake quantization Keras layers for litert_tunner.

These layers simulate TFLite's integer arithmetic in float32,
enabling gradient flow through quantization boundaries via
Straight-Through Estimator (STE).

All layers use keras.ops for backend-agnostic computation.
"""

import typing

import keras
import numpy as np
from keras import ops

from litert_tunner.graph import types

# INT8 quantization range constants
_INT8_MIN = -128.0
_INT8_MAX = 127.0


class FakeQuantize(keras.Layer):
    """Simulates quantize → dequantize round-trip with STE gradients.

    This layer performs:
        quantized = clamp(round(x / scale) + zero_point, -128, 127)
        output = (quantized - zero_point) * scale

    Gradients flow through via Straight-Through Estimator (the round
    and clamp operations are treated as identity during backprop).

    Args:
        scale: Initial quantization scale value(s).
        zero_point: Initial zero point value(s).
        trainable: Whether scale and zero_point are trainable.
        name: Layer name.
    """

    def __init__(
        self,
        scale: float,
        zero_point: float,
        trainable: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._initial_scale = scale
        self._initial_zero_point = zero_point
        self._trainable_params = trainable

    def build(self, input_shape):
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

    def call(self, x):
        """Forward pass with STE gradient."""
        return _fake_quantize(x, self.scale, self.zero_point)

    def get_config(self):
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


class Dequantize(keras.Layer):
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
        passthrough: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._initial_scale = scale
        self._initial_zero_point = zero_point
        self._passthrough = passthrough

    def build(self, input_shape):
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

    def call(self, x):
        """Dequantize input tensor."""
        if self._passthrough:
            return x
        return dequantize_ste(x, self.scale, self.zero_point)

    def get_config(self):
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
        tensors: tuple[types.TensorInfo, ...],
    ) -> tuple[list[types.BufferWriteOp], list[types.QuantizationWriteOp]]:
        """Return flatbuffer write instructions for the Dequantize layer.

        Updates the quantization params on the input tensor.

        Args:
            op: The OperatorInfo that this layer was built from.
            tensors: All tensors in the graph.

        Returns:
            A tuple of (buffer_writes, quantization_writes).
        """
        if self._passthrough:
            return [], []
        input_tensor_idx = op.input_indices[0]
        quant_write = make_quant_write_op(input_tensor_idx, self.scale, self.zero_point)
        return [], [quant_write]


class Quantize(keras.Layer):
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
        trainable: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._initial_scale = scale
        self._initial_zero_point = zero_point
        self._trainable_params = trainable

    def build(self, input_shape):
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

    def call(self, x):
        """Quantize input to simulated INT8."""
        return quantize_ste(x, self.scale, self.zero_point)

    def get_config(self):
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
        tensors: tuple[types.TensorInfo, ...],
    ) -> tuple[list[types.BufferWriteOp], list[types.QuantizationWriteOp]]:
        """Return flatbuffer write instructions for the Quantize layer.

        Updates the quantization params on the output tensor.

        Args:
            op: The OperatorInfo that this layer was built from.
            tensors: All tensors in the graph.

        Returns:
            A tuple of (buffer_writes, quantization_writes).
        """
        output_tensor_idx = op.output_indices[0]
        quant_write = make_quant_write_op(output_tensor_idx, self.scale, self.zero_point)
        return [], [quant_write]


def _round_ste(x):
    """Round with Straight-Through Estimator.

    Forward: round(x)
    Backward: identity (gradient passes through unchanged)
    """
    return x + ops.stop_gradient(ops.round(x) - x)


def _clip_ste(x, min_val, max_val):
    """Clip with Straight-Through Estimator.

    Forward: clip(x, min_val, max_val)
    Backward: identity (gradient passes through unchanged)
    """
    return x + ops.stop_gradient(ops.clip(x, min_val, max_val) - x)


def quantize_ste(x, scale, zero_point):
    """Quantize float32 → simulated INT8 with STE gradients."""
    scaled = x / scale
    rounded = _round_ste(scaled)
    shifted = rounded + zero_point
    clamped = _clip_ste(shifted, _INT8_MIN, _INT8_MAX)
    return clamped


def dequantize_ste(x, scale, zero_point):
    """Dequantize simulated INT8 values to float32.

    Formula: real_value = scale * (x - zero_point)
    """
    x_float = ops.cast(x, "float32")
    return scale * (x_float - zero_point)


def _fake_quantize(x, scale, zero_point):
    """Fake quantize: quantize then dequantize with STE gradients."""
    quantized = quantize_ste(x, scale, zero_point)
    dequantized = dequantize_ste(quantized, scale, zero_point)
    return dequantized


def to_float_list(tensor: typing.Any) -> list[float]:
    """Converts a tensor to a list of float values.

    Args:
        tensor: The input tensor.

    Returns:
        A list of python float values.
    """
    arr = np.asarray(typing.cast(np.ndarray, ops.convert_to_numpy(tensor)))
    if arr.ndim == 0:
        return [float(arr)]
    return [float(x) for x in arr]


def to_int_list(tensor: typing.Any) -> list[int]:
    """Converts a tensor to a list of rounded integer values.

    Args:
        tensor: The input tensor.

    Returns:
        A list of python int values.
    """
    arr = np.asarray(typing.cast(np.ndarray, ops.convert_to_numpy(tensor)))
    if arr.ndim == 0:
        return [int(np.round(arr))]
    return [int(np.round(x)) for x in arr]


def make_quant_write_op(
    tensor_index: int,
    scale_tensor: typing.Any,
    zp_tensor: typing.Any,
) -> types.QuantizationWriteOp:
    """Helper to construct a QuantizationWriteOp from Keras scale/zp weights.

    Args:
        tensor_index: The index of the tensor in the flatbuffer.
        scale_tensor: The scale weight/tensor.
        zp_tensor: The zero point weight/tensor.

    Returns:
        A QuantizationWriteOp populated with Python list values.
    """
    scales = to_float_list(scale_tensor)
    zps = to_int_list(zp_tensor)
    return types.QuantizationWriteOp(
        tensor_index=tensor_index,
        scales=scales,
        zero_points=zps,
    )
