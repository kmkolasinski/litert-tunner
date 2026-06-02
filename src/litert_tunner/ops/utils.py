"""Utility helper functions for operators in litert_tunner.

These functions provide reusable logic for tensor reshaping, quantization parameters,
and converting tensors to types suitable for flatbuffer writing.
"""

from __future__ import annotations

import typing

import keras
import numpy as np
from keras import ops

from litert_tunner.graph import types

# Fused activation function codes from TFLite schema
FUSED_ACTIVATION_NONE = 0
FUSED_ACTIVATION_RELU = 1
FUSED_ACTIVATION_RELU_N1_TO_1 = 2
FUSED_ACTIVATION_RELU6 = 3

TensorLike = typing.Any


def expand_dims_if_not_scalar(tensor: TensorLike, axis: int) -> TensorLike:
    """Expands the dimensions of a tensor if it is not a scalar (ndim > 0).

    Args:
        tensor: The input tensor or variable.
        axis: The axis along which to expand.

    Returns:
        The tensor with expanded dimensions if not a scalar, otherwise the original tensor.
    """
    if len(tensor.shape) > 0:
        return ops.expand_dims(tensor, axis)
    return tensor


def quantize_to_int8(tensor: TensorLike) -> np.ndarray:
    """Converts a tensor to a rounded numpy INT8 array.

    Args:
        tensor: The input tensor to round and convert.

    Returns:
        NumPy array with dtype int8.
    """
    val = typing.cast("np.ndarray", ops.convert_to_numpy(tensor))
    return np.round(val).astype(np.int8)


def quantize_bias_to_int32(
    bias_tensor: TensorLike,
    input_scale_tensor: TensorLike,
    weight_scale_tensor: TensorLike,
) -> np.ndarray:
    """Quantizes a float32 bias tensor to INT32 using input and weight scales.

    Bias scale = input_scale * weight_scale.

    Args:
        bias_tensor: Float32 bias tensor.
        input_scale_tensor: Input scale tensor (scalar).
        weight_scale_tensor: Weight scale tensor (scalar or array).

    Returns:
        NumPy array with dtype int32.
    """
    bias_val = typing.cast("np.ndarray", ops.convert_to_numpy(bias_tensor))
    input_scale_val = float(typing.cast("typing.Any", ops.convert_to_numpy(input_scale_tensor)))
    weight_scale_val = np.asarray(
        typing.cast("np.ndarray", ops.convert_to_numpy(weight_scale_tensor))
    )
    bias_scale = input_scale_val * weight_scale_val
    return np.round(bias_val / bias_scale).astype(np.int32)


def get_quant_param_value(param_array: np.ndarray) -> float | np.ndarray:
    """Extracts a scalar float if the array has length 1, else returns float32 array.

    Args:
        param_array: The quantization parameter array (scales or zero points).

    Returns:
        A float scalar if length is 1, otherwise a NumPy float32 array.
    """
    if len(param_array) == 1:
        return float(param_array[0])
    return param_array.astype(np.float32)


def get_bias_float32(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
    input_scale: float,
    weight_scales: np.ndarray,
    output_units: int,
) -> np.ndarray:
    """Extracts the bias tensor as float32, or initializes it to zeros if not present.

    Args:
        op: The parsed operator info.
        tensors: All tensors in the graph.
        input_scale: The input quantization scale.
        weight_scales: NumPy array of weight quantization scales.
        output_units: The number of output channels/units to fallback to if bias is missing.

    Returns:
        NumPy array with dtype float32.
    """
    bias_index = 2
    if len(op.input_indices) > bias_index and op.input_indices[bias_index] >= 0:
        bias_tensor = tensors[op.input_indices[bias_index]]
        if bias_tensor.data is not None:
            # TFLite stores bias as INT32; convert to float32
            bias_scale = input_scale * weight_scales.astype(np.float64)
            return bias_tensor.data.astype(np.float32) * bias_scale.astype(np.float32)
    return np.zeros(output_units, dtype=np.float32)


def apply_fused_activation(x: TensorLike, fused_activation: int) -> TensorLike:
    """Applies a TFLite fused activation function to a tensor.

    Args:
        x: Input Keras tensor.
        fused_activation: Code of the fused activation function.

    Returns:
        The tensor after applying the activation.
    """
    if fused_activation == FUSED_ACTIVATION_NONE:
        return x
    if fused_activation == FUSED_ACTIVATION_RELU:
        return ops.relu(x)
    if fused_activation == FUSED_ACTIVATION_RELU6:
        return ops.minimum(ops.relu(x), 6.0)
    if fused_activation == FUSED_ACTIVATION_RELU_N1_TO_1:
        return ops.clip(x, -1.0, 1.0)
    msg = f"Unsupported fused activation: {fused_activation}"
    raise ValueError(msg)


# INT8 range constants
_INT8_MIN_F = -128.0
_INT8_MAX_F = 127.0
INT8_MIN = -128
INT8_MAX = 127

# Type aliases
TensorOrScalar = typing.Any


def quantize_int8(
    x: np.ndarray,
    scale: np.ndarray | float,
    zero_point: np.ndarray | int,
) -> np.ndarray:
    """Quantize float32 values to INT8 using TFLite's affine scheme.

    Formula: int8_value = clamp(round(x / scale) + zero_point, -128, 127)

    Args:
        x: Float32 values to quantize.
        scale: Quantization scale (per-tensor or per-channel).
        zero_point: Quantization zero point (per-tensor or per-channel).

    Returns:
        INT8 quantized values as int8 numpy array.
    """
    scaled = np.round(x / scale) + zero_point
    clamped = np.clip(scaled, INT8_MIN, INT8_MAX)
    return clamped.astype(np.int8)


def dequantize_float(
    x: np.ndarray,
    scale: np.ndarray | float,
    zero_point: np.ndarray | int,
) -> np.ndarray:
    """Dequantize INT8 values to float32 using TFLite's affine scheme.

    Formula: real_value = scale * (int8_value - zero_point)

    Args:
        x: INT8 quantized values.
        scale: Quantization scale (per-tensor or per-channel).
        zero_point: Quantization zero point (per-tensor or per-channel).

    Returns:
        Float32 dequantized values.
    """
    return scale * (x.astype(np.float32) - np.float32(zero_point))


def compute_requantize_multiplier(
    input_scale: float,
    weight_scale: np.ndarray | float,
    output_scale: float,
) -> np.ndarray | float:
    """Compute the requantization multiplier for fused ops.

    In TFLite, the accumulator (INT32) is rescaled to the output
    quantization domain using:
        multiplier = (input_scale * weight_scale) / output_scale

    Args:
        input_scale: Scale of the input activation tensor.
        weight_scale: Scale of the weight tensor (scalar or per-channel array).
        output_scale: Scale of the output activation tensor.

    Returns:
        Requantization multiplier (scalar or per-channel array).
    """
    return (input_scale * weight_scale) / output_scale


def _round_ste(x: TensorLike) -> TensorLike:
    """Round with Straight-Through Estimator.

    Forward: round(x)
    Backward: identity (gradient passes through unchanged)
    """
    return x + ops.stop_gradient(ops.round(x) - x)


def _clip_ste(x: TensorLike, min_val: float, max_val: float) -> TensorLike:
    """Clip with Straight-Through Estimator.

    Forward: clip(x, min_val, max_val)
    Backward: identity (gradient passes through unchanged)
    """
    return x + ops.stop_gradient(ops.clip(x, min_val, max_val) - x)


def quantize_ste(x: TensorLike, scale: TensorOrScalar, zero_point: TensorOrScalar) -> TensorLike:
    """Quantize float32 → simulated INT8 with STE gradients."""
    scaled = x / scale
    rounded = _round_ste(scaled)
    shifted = rounded + zero_point
    return _clip_ste(shifted, _INT8_MIN_F, _INT8_MAX_F)


def dequantize_ste(x: TensorLike, scale: TensorOrScalar, zero_point: TensorOrScalar) -> TensorLike:
    """Dequantize simulated INT8 values to float32.

    Formula: real_value = scale * (x - zero_point)
    """
    x_float = ops.cast(x, "float32")
    return scale * (x_float - zero_point)


def fake_quantize(x: TensorLike, scale: TensorOrScalar, zero_point: TensorOrScalar) -> TensorLike:
    """Fake quantize: quantize then dequantize with STE gradients."""
    quantized = quantize_ste(x, scale, zero_point)
    return dequantize_ste(quantized, scale, zero_point)


def to_float_list(tensor: TensorOrScalar) -> list[float]:
    """Converts a tensor to a list of float values.

    Args:
        tensor: The input tensor.

    Returns:
        A list of python float values.
    """
    arr = np.asarray(typing.cast("np.ndarray", ops.convert_to_numpy(tensor)))
    if arr.ndim == 0:
        return [float(arr)]
    return [float(x) for x in arr]


def to_int_list(tensor: TensorOrScalar) -> list[int]:
    """Converts a tensor to a list of rounded integer values.

    Args:
        tensor: The input tensor.

    Returns:
        A list of python int values.
    """
    arr = np.asarray(typing.cast("np.ndarray", ops.convert_to_numpy(tensor)))
    if arr.ndim == 0:
        return [int(np.round(arr))]
    return [int(np.round(x)) for x in arr]


def make_quant_write_op(
    tensor_index: int,
    scale_tensor: TensorOrScalar,
    zp_tensor: TensorOrScalar,
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


def extract_constant_input(
    input1_tensor: types.TensorInfo,
    input1_quant: types.QuantizationParams,
    input2_tensor: types.TensorInfo,
    input2_quant: types.QuantizationParams,
) -> tuple[np.ndarray | None, int]:
    """Extract and quantize a constant input tensor to simulated INT8.

    Returns:
        A tuple of (quantized_constant_data, constant_index). If neither input
        is constant, returns (None, -1).
    """
    for idx, (tensor, _quant) in enumerate(
        [(input1_tensor, input1_quant), (input2_tensor, input2_quant)]
    ):
        if tensor.data is not None:
            # Constant tensor data is already stored as INT8 in the flatbuffer.
            # Store it as float32 for computation (simulated INT8 values).
            return tensor.data.astype(np.float32), idx
    return None, -1


class QuantizationVars:
    """A container for quantization scale and zero_point variables."""

    def __init__(
        self,
        layer: keras.Layer,
        name: str,
        scale: np.ndarray | float,
        zero_point: np.ndarray | float,
        *,
        trainable: bool,
    ):
        name_scale = f"{name}_scale" if name else "scale"
        name_zp = f"{name}_zero_point" if name else "zero_point"
        self.scale = layer.add_weight(
            name=name_scale,
            shape=np.shape(scale),
            initializer=keras.initializers.Constant(typing.cast("float", scale)),
            trainable=trainable,
        )
        self.zero_point = layer.add_weight(
            name=name_zp,
            shape=np.shape(zero_point),
            initializer=keras.initializers.Constant(typing.cast("float", zero_point)),
            trainable=False,
        )

    def dequantize(self, x: TensorLike) -> TensorLike:
        """Dequantize an INT8 tensor to Float32."""
        return dequantize_ste(x, self.scale, self.zero_point)

    def quantize(self, x: TensorLike) -> TensorLike:
        """Quantize a Float32 tensor to INT8."""
        return quantize_ste(x, self.scale, self.zero_point)

    def make_write_op(self, tensor_index: int) -> types.QuantizationWriteOp:
        """Create a write operation for these variables."""
        return make_quant_write_op(tensor_index, self.scale, self.zero_point)
