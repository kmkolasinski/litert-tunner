"""Conv2D op implementation for litert_tunner.

Simulates TFLite's quantized Conv2D op as a Keras layer.
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

if typing.TYPE_CHECKING:
    from litert_tunner.ops.utils import TensorLike

    ShapeLike = tuple[int, ...] | list[int] | list[tuple[int, ...]]


# TFLite padding integer codes
_PADDING_SAME = 0
_PADDING_VALID = 1

_PADDING_MAP: dict[int, str] = {
    _PADDING_SAME: "same",
    _PADDING_VALID: "valid",
}


class QuantizedConv2D(keras.Layer, types.Writable):
    """Simulates TFLite's quantized Conv2D op.

    The forward pass performs:
        1. Dequantize INT8 input to float32
        2. Dequantize INT8 weights to float32 (per-channel)
        3. Conv2D + float32 bias
        4. Apply fused activation (if any)
        5. Fake-quantize output (quantize with STE)

    Trainable parameters: bias, output_scale, output_zero_point.
    Frozen parameters: weight_int8, input/weight scales and zero-points.

    Args:
        weight_int8: INT8 weight values, shape (out_ch, kH, kW, in_ch).
        bias_float: Float32 bias values, shape (out_ch,).
        input_scale: Scale of the input activation tensor.
        input_zero_point: Zero point of the input activation tensor.
        weight_scale: Per-channel weight scales, shape (out_ch,) or scalar.
        weight_zero_point: Per-channel weight zero points, shape (out_ch,) or scalar.
        output_scale: Scale of the output activation tensor.
        output_zero_point: Zero point of the output activation tensor.
        strides: Convolution strides as (sH, sW).
        padding: Padding string, "same" or "valid".
        dilation_rate: Dilation rate as (dH, dW).
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
        strides: tuple[int, int] = (1, 1),
        padding: str = "same",
        dilation_rate: tuple[int, int] = (1, 1),
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
        self._strides = strides
        self._padding = padding
        self._dilation_rate = dilation_rate
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
            trainable=False,
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
        """Forward pass simulating TFLite's quantized Conv2D.

        Args:
            x: Input tensor of shape (batch, H, W, C_in).

        Returns:
            Output tensor after fake-quantized convolution.
        """
        # 1. Dequantize input: float = scale * (int8 - zero_point)
        input_float = self.input_quant.dequantize(x)

        # 2. Dequantize weights (per-channel along output channel axis 0)
        # TFLite Conv2D weight shape: (out_ch, kH, kW, in_ch)
        # Scale shape: (out_ch,) → expand to (out_ch, 1, 1, 1)
        # Always apply quantize_to_int8_ste to ensure valid INT8 values.
        # It's a no-op on already-valid integers but keeps the gradient path.
        weight_int8 = utils.quantize_to_int8_ste(self.weight_int8)
        weight_scale = self.weight_quant.scale
        weight_zp = self.weight_quant.zero_point
        if len(weight_scale.shape) > 0:
            weight_scale = ops.reshape(weight_scale, (-1, 1, 1, 1))
            weight_zp = ops.reshape(weight_zp, (-1, 1, 1, 1))
        weight_float = utils.dequantize_ste(weight_int8, weight_scale, weight_zp)

        # 3. Conv2D + bias
        # Keras conv expects kernel shape (kH, kW, in_ch, out_ch) but TFLite stores
        # weights as (out_ch, kH, kW, in_ch). Transpose to Keras format.
        kernel = ops.transpose(weight_float, (1, 2, 3, 0))
        output = ops.conv(
            input_float,
            kernel,
            strides=typing.cast("typing.Any", self._strides),
            padding=self._padding,
            dilation_rate=typing.cast("typing.Any", self._dilation_rate),
        )
        output = output + self.bias_ste

        # 4. Apply fused activation (before requantization)
        output = utils.apply_fused_activation(output, self._fused_activation)

        # 5. Quantize output to simulated INT8 (with STE for gradient flow)
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
            "strides": self._strides,
            "padding": self._padding,
            "dilation_rate": self._dilation_rate,
            "fused_activation": self._fused_activation,
        })
        return config

    def collect_write_ops(
        self,
        op: types.OperatorInfo,
    ) -> tuple[list[types.BufferWriteOp], list[types.QuantizationWriteOp]]:
        """Return flatbuffer write instructions for the Conv2D layer.

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


def _map_padding(padding_code: int) -> str:
    """Map TFLite padding integer code to Keras padding string.

    Args:
        padding_code: Integer padding code (0=SAME, 1=VALID).

    Returns:
        Keras padding string.

    Raises:
        ValueError: If the padding code is not recognized.
    """
    if padding_code not in _PADDING_MAP:
        msg = f"Unsupported padding code: {padding_code}"
        raise ValueError(msg)
    return _PADDING_MAP[padding_code]


@registry.register_op("CONV_2D")
def build_conv2d(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a QuantizedConv2D layer from parsed TFLite operator info.

    TFLite Conv2D inputs:
        [0] input tensor (INT8), shape (batch, H, W, C_in)
        [1] weight tensor (INT8), shape (out_ch, kH, kW, C_in)
        [2] bias tensor (INT32, optional)

    TFLite Conv2D outputs:
        [0] output tensor (INT8)

    Args:
        op: Parsed operator info with input/output indices and options.
        tensors: All tensors in the graph.
        graph_def: The parsed GraphDef.

    Returns:
        A configured QuantizedConv2D Keras layer.
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
        msg = "Conv2D requires quantized input, weight, and output tensors"
        raise ValueError(msg)

    output_channels = weight_tensor.shape[0]
    bias_float = utils.get_bias_float32(
        op=op,
        tensors=tensors,
        input_scale=float(input_quant.scales[0]),
        weight_scales=weight_quant.scales,
        output_units=output_channels,
    )

    # Extract conv options
    fused_activation = op.options.get("fused_activation_function", utils.FUSED_ACTIVATION_NONE)
    padding_code = op.options.get("Padding", _PADDING_SAME)
    padding = _map_padding(padding_code)
    stride_h = op.options.get("StrideH", 1)
    stride_w = op.options.get("StrideW", 1)
    dilation_h = op.options.get("DilationHFactor", 1)
    dilation_w = op.options.get("DilationWFactor", 1)

    weight_scale_val = utils.get_quant_param_value(weight_quant.scales)
    weight_zp_val = utils.get_quant_param_value(weight_quant.zero_points)

    return QuantizedConv2D(
        weight_int8=weight_tensor.data,
        bias_float=bias_float,
        input_scale=float(input_quant.scales[0]),
        input_zero_point=float(input_quant.zero_points[0]),
        weight_scale=weight_scale_val,
        weight_zero_point=weight_zp_val,
        output_scale=float(output_quant.scales[0]),
        output_zero_point=float(output_quant.zero_points[0]),
        strides=(stride_h, stride_w),
        padding=padding,
        dilation_rate=(dilation_h, dilation_w),
        fused_activation=fused_activation,
        name=f"quantized_conv2d_{op.output_indices[0]}",
    )
