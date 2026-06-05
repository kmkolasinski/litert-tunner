"""FullyConnected (Dense) op implementation for litert_tunner.

Supports both quantized INT8 and float32 TFLite FullyConnected ops.

- **Quantized path** (``QuantizedDense``): Simulates TFLite's quantized
  FullyConnected op, replicating integer arithmetic in float32 with STE
  gradient flow through trainable parameters (bias, scales, zero-points).

- **Float path** (``FloatDense``): Thin wrapper around matmul + bias for
  float32 TFLite models. All weights are directly trainable.
"""

from __future__ import annotations

import typing

import keras
import numpy as np
from keras import ops

from litert_tunner.graph import types
from litert_tunner.ops import registry, utils

if typing.TYPE_CHECKING:
    from litert_tunner.ops.utils import TensorLike

    ShapeLike = tuple[int, ...] | list[int] | list[tuple[int, ...]]


class QuantizedDense(keras.Layer, types.Writable):
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

    def build(self, input_shape: ShapeLike) -> None:
        """Create the weights (bias, scale, zero_point) for the layer."""
        # INT8 weights — stored as float32 for computation.
        # Created as trainable so that the Keras Functional model graph
        # includes a gradient path. ``prepare_for_finetuning`` freezes
        # it by default; users opt-in with the ``weight_int8`` pattern.
        self.weight_int8 = self.add_weight(
            name="weight_int8",
            shape=self._weight_int8_data.shape,
            initializer=keras.initializers.Constant(
                typing.cast("float", self._weight_int8_data.astype(np.float32))
            ),
            trainable=True,
        )

        # Trainable float32 bias
        self.bias = self.add_weight(
            name="bias",
            shape=self._bias_float_data.shape,
            initializer=keras.initializers.Constant(typing.cast("float", self._bias_float_data)),
            trainable=True,
        )

        # Input quantization params (frozen)
        self.input_quant = utils.QuantizationVars(
            self,
            "input",
            self._input_scale,
            self._input_zero_point,
            trainable=False,
        )

        # Weight quantization params (trainable scale, frozen zero-point)
        self.weight_quant = utils.QuantizationVars(
            self,
            "weight",
            self._weight_scale,
            self._weight_zero_point,
            trainable=True,
        )

        # Output quantization params (frozen)
        self.output_quant = utils.QuantizationVars(
            self,
            "output",
            self._output_scale,
            self._output_zero_point,
            trainable=False,
        )

        super().build(input_shape)

    @property
    def bias_ste(self) -> TensorLike:
        """Returns the bias fake-quantized to INT32 using input and weight scales."""
        return utils.fake_quantize_bias(self.bias, self.input_quant.scale, self.weight_quant.scale)

    def call(self, x: TensorLike) -> TensorLike:
        """Forward pass simulating TFLite's quantized FullyConnected.

        Args:
            x: Input tensor. Can be float32 (pre-dequantized) or
               simulated INT8 values in float32.

        Returns:
            Output tensor after fake-quantized dense computation.
        """
        # 1. Dequantize input: float = scale * (int8 - zero_point)
        input_float = self.input_quant.dequantize(x)

        # 2. Dequantize weights: float = scale * (int8 - zero_point)
        # Always apply quantize_to_int8_ste to ensure valid INT8 values.
        # It's a no-op on already-valid integers but keeps the gradient path.
        weight_int8 = utils.quantize_to_int8_ste(self.weight_int8)

        scale_expanded = utils.expand_dims_if_not_scalar(self.weight_quant.scale, 1)
        zp_expanded = utils.expand_dims_if_not_scalar(self.weight_quant.zero_point, 1)
        weight_float = utils.dequantize_ste(weight_int8, scale_expanded, zp_expanded)

        # 3. Matmul + bias (in float32, simulating INT32 accumulation)
        # TFLite FullyConnected: output = input @ weight^T + bias
        output = ops.matmul(input_float, ops.transpose(weight_float)) + self.bias_ste

        # 4. Apply fused activation (before requantization)
        output = utils.apply_fused_activation(output, self._fused_activation)

        # 5. Quantize output to simulated INT8 (with STE for gradient flow)
        # We use quantize_ste (not _fake_quantize) so the output stays in simulated
        # INT8 space. This ensures the next layer's dequantize step works correctly,
        # and the final DEQUANTIZE op converts back to float32.
        return self.output_quant.quantize(output)

    def get_config(self):
        """Return the configuration dictionary for serialization of the layer."""
        config = super().get_config()
        config.update({
            "input_scale": self._input_scale,
            "input_zero_point": self._input_zero_point,
            "weight_scale": self._weight_scale,
            "weight_zero_point": self._weight_zero_point,
            "output_scale": self._output_scale,
            "output_zero_point": self._output_zero_point,
            "fused_activation": self._fused_activation,
        })
        return config

    def collect_write_ops(
        self,
        op: types.OperatorInfo,
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

        op_inputs = typing.cast("typing.Any", op.input_indices)
        op_outputs = typing.cast("typing.Any", op.output_indices)

        # Write weight_int8 buffer
        weight_int8 = utils.quantize_to_int8(self.weight_int8)
        weight_tensor_idx = op_inputs[1]
        buffer_writes.append(
            types.BufferWriteOp(tensor_index=weight_tensor_idx, data=bytes(weight_int8.tobytes()))
        )

        # Write bias buffer (if present)
        bias_index = 2
        if len(op_inputs) > bias_index and op_inputs[bias_index] >= 0:
            bias_int32 = utils.quantize_bias_to_int32(
                self.bias, self.input_quant.scale, self.weight_quant.scale
            )
            buffer_writes.append(
                types.BufferWriteOp(tensor_index=op_inputs[2], data=bytes(bias_int32.tobytes()))
            )
            quant_writes.append(
                utils.make_bias_quant_write_op(
                    tensor_index=op_inputs[2],
                    input_scale=self.input_quant.scale,
                    weight_scale=self.weight_quant.scale,
                )
            )

        # Write quantization params
        input_tensor_idx = op_inputs[0]
        quant_writes.append(self.input_quant.make_write_op(input_tensor_idx))
        quant_writes.append(self.weight_quant.make_write_op(weight_tensor_idx))
        output_tensor_idx = op_outputs[0]
        quant_writes.append(self.output_quant.make_write_op(output_tensor_idx))

        return buffer_writes, quant_writes


class FloatDense(keras.Layer, types.Writable):
    """Float32/Float16 TFLite FullyConnected op as a Keras layer.

    The forward pass performs a simple matmul + bias with an optional fused
    activation. No quantization/dequantization is involved.

    All weights (kernel and bias) are trainable by default.

    Args:
        kernel_data: Float32/Float16 weight values, shape (out_units, in_features).
        bias_data: Float32/Float16 bias values, shape (out_units,).
        fused_activation: TFLite fused activation code (0=none, 1=relu, 3=relu6).
        kernel_dtype: Original data type of the kernel in the flatbuffer.
        bias_dtype: Original data type of the bias in the flatbuffer.
        name: Layer name.
    """

    def __init__(
        self,
        kernel_data: np.ndarray,
        bias_data: np.ndarray,
        fused_activation: int = utils.FUSED_ACTIVATION_NONE,
        kernel_dtype: str = types.DTYPE_FLOAT32,
        bias_dtype: str = types.DTYPE_FLOAT32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._kernel_data = kernel_data.astype(np.float32)
        self._bias_data = bias_data.astype(np.float32)
        self._fused_activation = fused_activation
        self._kernel_dtype = kernel_dtype
        self._bias_dtype = bias_dtype

    def build(self, input_shape: ShapeLike) -> None:
        """Create trainable kernel and bias weights."""
        self.kernel = self.add_weight(
            name="kernel",
            shape=self._kernel_data.shape,
            initializer=keras.initializers.Constant(typing.cast("float", self._kernel_data)),
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=self._bias_data.shape,
            initializer=keras.initializers.Constant(typing.cast("float", self._bias_data)),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x: TensorLike) -> TensorLike:
        """Forward pass: matmul + bias + optional fused activation.

        Args:
            x: Float32 input tensor.

        Returns:
            Float32 output tensor.
        """
        # TFLite FullyConnected: output = input @ weight^T + bias
        output = ops.matmul(x, ops.transpose(self.kernel)) + self.bias
        return utils.apply_fused_activation(output, self._fused_activation)

    def get_config(self):
        """Return the configuration dictionary for serialization of the layer."""
        config = super().get_config()
        config.update({
            "fused_activation": self._fused_activation,
            "kernel_dtype": self._kernel_dtype,
            "bias_dtype": self._bias_dtype,
        })
        return config

    def collect_write_ops(
        self,
        op: types.OperatorInfo,
    ) -> tuple[list[types.BufferWriteOp], list[types.QuantizationWriteOp]]:
        """Return flatbuffer write instructions for the float32 FullyConnected layer.

        Writes back float32 kernel and float32 bias buffers. No quantization
        write ops are emitted since this is a float32 model.

        Args:
            op: The OperatorInfo that this layer was built from.

        Returns:
            A tuple of (buffer_writes, quantization_writes).
        """
        buffer_writes: list[types.BufferWriteOp] = []

        op_inputs = op.input_indices

        # Write kernel with original dtype
        kernel_np = typing.cast("np.ndarray", ops.convert_to_numpy(self.kernel)).astype(
            self._kernel_dtype
        )
        buffer_writes.append(
            types.BufferWriteOp(
                tensor_index=op_inputs[1],
                data=bytes(kernel_np.tobytes()),
            )
        )

        # Write bias (if present) with original dtype
        bias_index = 2
        if len(op_inputs) > bias_index and op_inputs[bias_index] >= 0:
            bias_np = typing.cast("np.ndarray", ops.convert_to_numpy(self.bias)).astype(
                self._bias_dtype
            )
            buffer_writes.append(
                types.BufferWriteOp(
                    tensor_index=op_inputs[2],
                    data=bytes(bias_np.tobytes()),
                )
            )

        return buffer_writes, []


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


@registry.register_op("FULLY_CONNECTED")
def build_fully_connected(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a Dense layer from parsed TFLite operator info.

    Dispatches to ``QuantizedDense`` for INT8 models or ``FloatDense`` for
    float32 models based on whether the input tensor has quantization params.

    TFLite FullyConnected inputs:
        [0] input tensor (INT8 or FLOAT32)
        [1] weight tensor (INT8 or FLOAT32)
        [2] bias tensor (INT32 or FLOAT32, optional)

    TFLite FullyConnected outputs:
        [0] output tensor (INT8 or FLOAT32)

    Args:
        op: Parsed operator info with input/output indices and options.
        tensors: All tensors in the graph.

    Returns:
        A configured QuantizedDense or FloatDense Keras layer.
    """
    input_tensor = tensors[op.input_indices[0]]
    weight_tensor = tensors[op.input_indices[1]]

    # Weight data must be available
    if weight_tensor.data is None:
        msg = f"Weight tensor '{weight_tensor.name}' has no data"
        raise ValueError(msg)

    if types.is_quantized(input_tensor):
        return _build_quantized_dense(op, tensors)
    return _build_float_dense(op, tensors)


def _build_quantized_dense(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> QuantizedDense:
    """Build a QuantizedDense layer for INT8 models."""
    input_tensor = tensors[op.input_indices[0]]
    weight_tensor = tensors[op.input_indices[1]]
    output_tensor = tensors[op.output_indices[0]]
    assert weight_tensor.data is not None  # validated by build_fully_connected  # noqa: S101

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


def _build_float_dense(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> FloatDense:
    """Build a FloatDense layer for float32 models."""
    weight_tensor = tensors[op.input_indices[1]]
    assert weight_tensor.data is not None  # validated by build_fully_connected  # noqa: S101
    output_units = weight_tensor.shape[0]

    bias_float = utils.get_float32_bias(
        op=op,
        tensors=tensors,
        output_units=output_units,
    )

    fused_activation = op.options.get("fused_activation_function", utils.FUSED_ACTIVATION_NONE)

    # Determine original dtypes for proper saving
    bias_index = 2
    bias_dtype = types.DTYPE_FLOAT32
    if len(op.input_indices) > bias_index and op.input_indices[bias_index] >= 0:
        bias_dtype = tensors[op.input_indices[bias_index]].dtype

    return FloatDense(
        kernel_data=weight_tensor.data,
        bias_data=bias_float,
        fused_activation=fused_activation,
        kernel_dtype=weight_tensor.dtype,
        bias_dtype=bias_dtype,
        name=f"float_dense_{op.output_indices[0]}",
    )
