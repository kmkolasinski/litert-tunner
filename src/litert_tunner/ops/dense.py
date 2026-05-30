"""FullyConnected (Dense) op implementation for litert_tunner.

Simulates TFLite's quantized FullyConnected op as a Keras layer.
The forward pass replicates the integer arithmetic in float32,
enabling gradient flow through trainable parameters (bias, scales,
zero-points).
"""

from __future__ import annotations

import typing

import keras
import numpy as np
from keras import ops

from litert_tunner.graph import types
from litert_tunner.ops import registry, utils
from litert_tunner.quantization import fake_quant


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
        weight_scale: float | np.ndarray,
        weight_zero_point: float | np.ndarray,
        output_scale: float,
        output_zero_point: float,
        fused_activation: int = utils.FUSED_ACTIVATION_NONE,
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
            initializer=keras.initializers.Constant(
                typing.cast(float, self._weight_int8_data.astype(np.float32))
            ),
            trainable=False,
        )

        # Trainable float32 bias
        self.bias = self.add_weight(
            name="bias",
            shape=self._bias_float_data.shape,
            initializer=keras.initializers.Constant(typing.cast(float, self._bias_float_data)),
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
        weight_scale_arr = np.asarray(self._weight_scale)
        self.weight_scale = self.add_weight(
            name="weight_scale",
            shape=weight_scale_arr.shape,
            initializer=keras.initializers.Constant(typing.cast(float, weight_scale_arr)),
            trainable=False,
        )
        weight_zp_arr = np.asarray(self._weight_zero_point)
        self.weight_zero_point = self.add_weight(
            name="weight_zero_point",
            shape=weight_zp_arr.shape,
            initializer=keras.initializers.Constant(typing.cast(float, weight_zp_arr)),
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
        input_float = fake_quant.dequantize_ste(x, self.input_scale, self.input_zero_point)

        # 2. Dequantize weights: float = scale * (int8 - zero_point)
        scale_expanded = utils.expand_dims_if_not_scalar(self.weight_scale, 1)
        zp_expanded = utils.expand_dims_if_not_scalar(self.weight_zero_point, 1)
        weight_float = fake_quant.dequantize_ste(self.weight_int8, scale_expanded, zp_expanded)

        # 3. Matmul + bias (in float32, simulating INT32 accumulation)
        # TFLite FullyConnected: output = input @ weight^T + bias
        output = ops.matmul(input_float, ops.transpose(weight_float)) + self.bias

        # 4. Apply fused activation (before requantization)
        output = utils.apply_fused_activation(output, self._fused_activation)

        # 5. Quantize output to simulated INT8 (with STE for gradient flow)
        # We use quantize_ste (not _fake_quantize) so the output stays in simulated
        # INT8 space. This ensures the next layer's dequantize step works correctly,
        # and the final DEQUANTIZE op converts back to float32.
        output = fake_quant.quantize_ste(output, self.output_scale, self.output_zero_point)

        return output

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

    def collect_write_ops(
        self,
        op: types.OperatorInfo,
        tensors: tuple[types.TensorInfo, ...],
    ) -> tuple[list[types.BufferWriteOp], list[types.QuantizationWriteOp]]:
        """Return flatbuffer write instructions for the FullyConnected layer.

        Writes back:
            - INT8 weights to the weight buffer
            - INT32 bias to the bias buffer (if present)
            - Quantization params for input, weight, and output tensors

        Args:
            op: The OperatorInfo that this layer was built from.
            tensors: All tensors in the graph.

        Returns:
            A tuple of (buffer_writes, quantization_writes).
        """
        buffer_writes: list[types.BufferWriteOp] = []
        quant_writes: list[types.QuantizationWriteOp] = []

        op_inputs = typing.cast(typing.Any, op.input_indices)
        op_outputs = typing.cast(typing.Any, op.output_indices)

        # Write weight_int8 buffer
        weight_int8 = utils.quantize_to_int8(self.weight_int8)
        weight_tensor_idx = op_inputs[1]
        buffer_writes.append(
            types.BufferWriteOp(tensor_index=weight_tensor_idx, data=bytes(weight_int8.tobytes()))
        )

        # Write bias buffer (if present)
        if len(op_inputs) > 2 and op_inputs[2] >= 0:
            bias_int32 = utils.quantize_bias_to_int32(
                self.bias, self.input_scale, self.weight_scale
            )
            buffer_writes.append(
                types.BufferWriteOp(tensor_index=op_inputs[2], data=bytes(bias_int32.tobytes()))
            )

        # Write quantization params
        input_tensor_idx = op_inputs[0]
        quant_writes.append(
            fake_quant.make_quant_write_op(
                input_tensor_idx, self.input_scale, self.input_zero_point
            )
        )
        quant_writes.append(
            fake_quant.make_quant_write_op(
                weight_tensor_idx, self.weight_scale, self.weight_zero_point
            )
        )
        output_tensor_idx = op_outputs[0]
        quant_writes.append(
            fake_quant.make_quant_write_op(
                output_tensor_idx, self.output_scale, self.output_zero_point
            )
        )

        return buffer_writes, quant_writes


@registry.register_op("FULLY_CONNECTED")
def build_fully_connected(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
    graph_def: types.GraphDef | None = None,
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
        graph_def: The parsed GraphDef.

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

    output_units = weight_tensor.shape[0]
    bias_float = utils.get_bias_float32(
        op=op,
        tensors=tensors,
        input_scale=float(input_quant.scales[0]),
        weight_scales=weight_quant.scales,
        output_units=output_units,
    )

    fused_activation = op.options.get("fused_activation_function", utils.FUSED_ACTIVATION_NONE)

    weight_scale_val = utils.get_quant_param_value(weight_quant.scales)
    weight_zp_val = utils.get_quant_param_value(weight_quant.zero_points)

    return QuantizedDense(
        weight_int8=weight_tensor.data,
        bias_float=bias_float,
        input_scale=float(input_quant.scales[0]),
        input_zero_point=float(input_quant.zero_points[0]),
        weight_scale=weight_scale_val,
        weight_zero_point=weight_zp_val,
        output_scale=float(output_quant.scales[0]),
        output_zero_point=float(output_quant.zero_points[0]),
        fused_activation=fused_activation,
        name=f"quantized_dense_{op.output_indices[0]}",
    )
