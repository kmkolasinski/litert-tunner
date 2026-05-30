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
        x_float = ops.cast(x, "float32")
        return self.scale * (x_float - self.zero_point)

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
        scale_val = float(typing.cast(typing.Any, ops.convert_to_numpy(self.scale)))
        zp_val = int(np.round(typing.cast(typing.Any, ops.convert_to_numpy(self.zero_point))))
        quant_write = types.QuantizationWriteOp(
            tensor_index=input_tensor_idx, scales=[scale_val], zero_points=[zp_val]
        )
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
        return _quantize_ste(x, self.scale, self.zero_point)

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
        scale_val = float(typing.cast(typing.Any, ops.convert_to_numpy(self.scale)))
        zp_val = int(np.round(typing.cast(typing.Any, ops.convert_to_numpy(self.zero_point))))
        quant_write = types.QuantizationWriteOp(
            tensor_index=output_tensor_idx, scales=[scale_val], zero_points=[zp_val]
        )
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


def _quantize_ste(x, scale, zero_point):
    """Quantize float32 → simulated INT8 with STE gradients."""
    scaled = x / scale
    rounded = _round_ste(scaled)
    shifted = rounded + zero_point
    clamped = _clip_ste(shifted, _INT8_MIN, _INT8_MAX)
    return clamped


def _fake_quantize(x, scale, zero_point):
    """Fake quantize: quantize then dequantize with STE gradients."""
    quantized = _quantize_ste(x, scale, zero_point)
    dequantized = (quantized - zero_point) * scale
    return dequantized
