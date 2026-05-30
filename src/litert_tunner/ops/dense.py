"""FullyConnected (Dense) op implementation for litert_tunner.

Simulates TFLite's quantized FullyConnected op as a Keras layer.
The forward pass replicates the integer arithmetic in float32,
enabling gradient flow through trainable parameters (bias, scales,
zero-points).
"""

from __future__ import annotations

import keras
import numpy as np
from keras import ops

from litert_tunner.graph import types
from litert_tunner.ops import registry
from litert_tunner.quantization import fake_quant

# Fused activation function codes from TFLite schema
_FUSED_ACTIVATION_NONE = 0
_FUSED_ACTIVATION_RELU = 1
_FUSED_ACTIVATION_RELU_N1_TO_1 = 2
_FUSED_ACTIVATION_RELU6 = 3


class QuantizedDense(keras.Layer):
    """Simulates TFLite's quantized FullyConnected op.

    The forward pass performs:
        1. Dequantize INT8 input to float32
        2. Dequantize INT8 weights to float32
        3. Matrix multiply + float32 bias
        4. Apply fused activation (if any)
        5. Fake-quantize output (quantize → dequantize with STE)

    Trainable parameters: bias, output_scale, output_zero_point.
    Frozen parameters: weight_int8, input/weight scales and zero-points.

    Args:
        weight_int8: INT8 weight values as numpy array, shape (out, in).
        bias_float: Float32 bias values, shape (out,).
        input_scale: Scale of the input activation tensor.
        input_zero_point: Zero point of the input activation tensor.
        weight_scale: Scale of the weight tensor (scalar for per-tensor).
        weight_zero_point: Zero point of the weight tensor.
        output_scale: Scale of the output activation tensor.
        output_zero_point: Zero point of the output activation tensor.
        fused_activation: TFLite fused activation code (0=none, 1=relu, 3=relu6).
        name: Layer name.
    """

    def __init__(
        self,
        weight_int8: np.ndarray,
        bias_float: np.ndarray,
        input_scale: float,
        input_zero_point: float,
        weight_scale: float,
        weight_zero_point: float,
        output_scale: float,
        output_zero_point: float,
        fused_activation: int = _FUSED_ACTIVATION_NONE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._weight_int8_data = weight_int8
        self._bias_float_data = bias_float
        self._input_scale = input_scale
        self._input_zero_point = input_zero_point
        self._weight_scale = weight_scale
        self._weight_zero_point = weight_zero_point
        self._output_scale = output_scale
        self._output_zero_point = output_zero_point
        self._fused_activation = fused_activation

    def build(self, input_shape):
        """Create the weights (bias, scale, zero_point) for the layer."""
        # Frozen INT8 weights — stored as float32 for computation
        self.weight_int8 = self.add_weight(
            name="weight_int8",
            shape=self._weight_int8_data.shape,
            initializer=keras.initializers.Constant(self._weight_int8_data.astype(np.float32)),
            trainable=False,
        )

        # Trainable float32 bias
        self.bias = self.add_weight(
            name="bias",
            shape=self._bias_float_data.shape,
            initializer=keras.initializers.Constant(self._bias_float_data),
            trainable=True,
        )

        # Input quantization params (frozen)
        self.input_scale = self.add_weight(
            name="input_scale",
            shape=(),
            initializer=keras.initializers.Constant(self._input_scale),
            trainable=False,
        )
        self.input_zero_point = self.add_weight(
            name="input_zero_point",
            shape=(),
            initializer=keras.initializers.Constant(self._input_zero_point),
            trainable=False,
        )

        # Weight quantization params (frozen)
        self.weight_scale = self.add_weight(
            name="weight_scale",
            shape=(),
            initializer=keras.initializers.Constant(self._weight_scale),
            trainable=False,
        )
        self.weight_zero_point = self.add_weight(
            name="weight_zero_point",
            shape=(),
            initializer=keras.initializers.Constant(self._weight_zero_point),
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

    def call(self, x):
        """Forward pass simulating TFLite's quantized FullyConnected.

        Args:
            x: Input tensor. Can be float32 (pre-dequantized) or
               simulated INT8 values in float32.

        Returns:
            Output tensor after fake-quantized dense computation.
        """
        # 1. Dequantize input: float = scale * (int8 - zero_point)
        input_float = self.input_scale * (x - self.input_zero_point)

        # 2. Dequantize weights: float = scale * (int8 - zero_point)
        weight_float = self.weight_scale * (self.weight_int8 - self.weight_zero_point)

        # 3. Matmul + bias (in float32, simulating INT32 accumulation)
        # TFLite FullyConnected: output = input @ weight^T + bias
        output = ops.matmul(input_float, ops.transpose(weight_float)) + self.bias

        # 4. Apply fused activation (before requantization)
        output = self._apply_fused_activation(output)

        # 5. Fake-quantize output (quantize → dequantize with STE)
        output = fake_quant._fake_quantize(output, self.output_scale, self.output_zero_point)

        return output

    def _apply_fused_activation(self, x):
        """Apply fused activation function."""
        if self._fused_activation == _FUSED_ACTIVATION_NONE:
            return x
        elif self._fused_activation == _FUSED_ACTIVATION_RELU:
            return ops.relu(x)
        elif self._fused_activation == _FUSED_ACTIVATION_RELU6:
            return ops.minimum(ops.relu(x), 6.0)
        elif self._fused_activation == _FUSED_ACTIVATION_RELU_N1_TO_1:
            return ops.clip(x, -1.0, 1.0)
        else:
            msg = f"Unsupported fused activation: {self._fused_activation}"
            raise ValueError(msg)

    def get_config(self):
        """Return the configuration dictionary for serialization of the layer."""
        config = super().get_config()
        config.update(
            {
                "input_scale": self._input_scale,
                "input_zero_point": self._input_zero_point,
                "weight_scale": self._weight_scale,
                "weight_zero_point": self._weight_zero_point,
                "output_scale": self._output_scale,
                "output_zero_point": self._output_zero_point,
                "fused_activation": self._fused_activation,
            }
        )
        return config


@registry.register_op("FULLY_CONNECTED")
def build_fully_connected(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a QuantizedDense layer from parsed TFLite operator info.

    TFLite FullyConnected inputs:
        [0] input tensor (INT8)
        [1] weight tensor (INT8)
        [2] bias tensor (INT32, optional)

    TFLite FullyConnected outputs:
        [0] output tensor (INT8)

    Args:
        op: Parsed operator info with input/output indices and options.
        tensors: All tensors in the graph.

    Returns:
        A configured QuantizedDense Keras layer.
    """
    input_tensor = tensors[op.input_indices[0]]
    weight_tensor = tensors[op.input_indices[1]]
    output_tensor = tensors[op.output_indices[0]]

    # Weight data must be available
    if weight_tensor.data is None:
        msg = f"Weight tensor '{weight_tensor.name}' has no data"
        raise ValueError(msg)

    # Extract quantization params
    input_quant = input_tensor.quantization
    weight_quant = weight_tensor.quantization
    output_quant = output_tensor.quantization

    if input_quant is None or weight_quant is None or output_quant is None:
        msg = "FullyConnected requires quantized input, weight, and output tensors"
        raise ValueError(msg)

    # Handle optional bias
    if len(op.input_indices) > 2 and op.input_indices[2] >= 0:
        bias_tensor = tensors[op.input_indices[2]]
        if bias_tensor.data is not None:
            # TFLite stores bias as INT32; convert to float32
            # bias_float = bias_int32 * (input_scale * weight_scale)
            bias_scale = float(input_quant.scales[0]) * weight_quant.scales.astype(np.float64)
            bias_float = bias_tensor.data.astype(np.float32) * bias_scale.astype(np.float32)
        else:
            output_units = weight_tensor.shape[0]
            bias_float = np.zeros(output_units, dtype=np.float32)
    else:
        output_units = weight_tensor.shape[0]
        bias_float = np.zeros(output_units, dtype=np.float32)

    fused_activation = op.options.get("fused_activation_function", _FUSED_ACTIVATION_NONE)

    return QuantizedDense(
        weight_int8=weight_tensor.data,
        bias_float=bias_float,
        input_scale=float(input_quant.scales[0]),
        input_zero_point=float(input_quant.zero_points[0]),
        weight_scale=float(weight_quant.scales[0]),
        weight_zero_point=float(weight_quant.zero_points[0]),
        output_scale=float(output_quant.scales[0]),
        output_zero_point=float(output_quant.zero_points[0]),
        fused_activation=fused_activation,
        name=f"quantized_dense_{op.output_indices[0]}",
    )
