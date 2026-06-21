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

# INT8 range constants
_INT8_MIN_F = -128.0
_INT8_MAX_F = 127.0

# Threshold for softplus inverse: above this value, softplus_inverse(x) ≈ x.
# Chosen to avoid overflow in exp() while maintaining float32 precision.
_SOFTPLUS_INV_THRESHOLD = 20.0
INT8_MIN = -128
INT8_MAX = 127

# Type aliases
TensorOrScalar = typing.Any
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


def get_float32_bias(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
    output_units: int,
) -> np.ndarray:
    """Extract bias as float32 from a float32 model (no dequantization needed).

    In float32 TFLite models, bias is already stored as FLOAT32. This helper
    simply reads the raw data, unlike ``get_bias_float32`` which dequantizes
    INT32 bias using input/weight scales.

    Args:
        op: The parsed operator info.
        tensors: All tensors in the graph.
        output_units: Fallback size if bias is absent.

    Returns:
        Float32 bias as a numpy array.
    """
    bias_index = 2
    if len(op.input_indices) > bias_index and op.input_indices[bias_index] >= 0:
        bias_tensor = tensors[op.input_indices[bias_index]]
        if bias_tensor.data is not None:
            return bias_tensor.data.astype(np.float32)
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


def round_to_int8_ndarray(tensor: TensorLike) -> np.ndarray:
    """Converts a tensor to a rounded numpy INT8 array.

    Args:
        tensor: The input tensor to round and convert.

    Returns:
        NumPy array with dtype int8.
    """
    val = typing.cast("np.ndarray", ops.convert_to_numpy(tensor))
    return np.clip(np.round(val), INT8_MIN, INT8_MAX).astype(np.int8)


def round_ste(x: TensorLike) -> TensorLike:
    """Round with Straight-Through Estimator.

    Forward: round(x)
    Backward: identity (gradient passes through unchanged)
    """
    return x + ops.stop_gradient(ops.round(x) - x)


def round_to_int8_ste(x: TensorLike) -> TensorLike:
    """Quantize float values to INT8 range with STE gradients.

    Differentiable equivalent of ``quantize_to_int8``. Rounds to nearest
    integer and clamps to [-128, 127]. Forward matches ``quantize_to_int8``
    exactly; backward uses STE (identity gradient, zeroed outside range).

    Args:
        x: Input tensor with float values representing INT8 weights.

    Returns:
        Tensor with values rounded and clamped to [-128, 127].
    """
    return ops.clip(round_ste(x), _INT8_MIN_F, _INT8_MAX_F)


def quantize_ste(x: TensorLike, scale: TensorOrScalar, zero_point: TensorOrScalar) -> TensorLike:
    """Quantize float32 → simulated INT8 with STE gradients.

    Uses hard clip so that gradients are zeroed for values that saturate
    outside the [-128, 127] range. This prevents the optimizer from wasting
    gradient budget pushing already-saturated values further out of range.
    """
    scaled = x / scale
    rounded = round_ste(scaled)
    shifted = rounded + zero_point
    return ops.clip(shifted, _INT8_MIN_F, _INT8_MAX_F)


def dequantize_ste(x: TensorLike, scale: TensorOrScalar, zero_point: TensorOrScalar) -> TensorLike:
    """Dequantize simulated INT8 values to float using STE gradients.

    Formula: real_value = scale * (x - zero_point)

    The input ``x`` is cast to match ``scale``'s dtype so that computation
    runs in the layer's compute dtype (e.g. float16 under mixed precision).
    """
    scale_tensor = ops.convert_to_tensor(scale)
    x_float = ops.cast(x, ops.dtype(scale_tensor))
    return scale_tensor * (x_float - zero_point)


def fake_quantize_bias(
    bias: TensorLike,
    input_scale: TensorOrScalar,
    weight_scale: TensorOrScalar,
) -> TensorLike:
    """Fake quantize the bias to INT32 using input and weight scales.

    In TFLite, bias is quantized to INT32 with scale = input_scale * weight_scale
    and zero_point = 0.
    """
    bias_scale = input_scale * weight_scale
    scaled_bias = bias / bias_scale
    rounded_bias = round_ste(scaled_bias)
    return rounded_bias * bias_scale


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


def make_bias_quant_write_op(
    tensor_index: int,
    input_scale: TensorOrScalar,
    weight_scale: TensorOrScalar,
) -> types.QuantizationWriteOp:
    """Helper to construct a QuantizationWriteOp for an INT32 bias tensor.

    TFLite requires the bias tensor to have a quantization parameter where:
        bias_scale = input_scale * weight_scale
        bias_zero_point = 0

    Args:
        tensor_index: The index of the bias tensor in the flatbuffer.
        input_scale: The input scale (scalar).
        weight_scale: The weight scale (scalar or array).

    Returns:
        A QuantizationWriteOp for the bias tensor.
    """
    input_scale_val = np.asarray(ops.convert_to_numpy(input_scale), dtype=np.float32)
    weight_scale_val = np.asarray(ops.convert_to_numpy(weight_scale), dtype=np.float32)
    bias_scale = input_scale_val * weight_scale_val

    scales = to_float_list(bias_scale)
    zps = [0] * len(scales)

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
    for idx, (tensor, _quant) in enumerate([
        (input1_tensor, input1_quant),
        (input2_tensor, input2_quant),
    ]):
        if tensor.data is not None:
            # Constant tensor data is already stored as INT8 in the flatbuffer.
            # Store it as float32 for computation (simulated INT8 values).
            return tensor.data.astype(np.float32), idx
    return None, -1


def _softplus_inverse(x: np.ndarray) -> np.ndarray:
    """Compute the inverse of softplus: log(exp(x) - 1).

    For large x (> 20), this is numerically equivalent to x itself,
    avoiding overflow in exp.

    Args:
        x: Input array. Must contain only positive values.

    Returns:
        Array y such that softplus(y) ≈ x.
    """
    x = np.asarray(x, dtype=np.float32)
    return np.where(x > _SOFTPLUS_INV_THRESHOLD, x, np.log(np.expm1(x)))


class QuantizationVars:
    """A container for quantization scale and zero_point variables.

    When ``trainable=True``, the scale is stored in a reparameterized form
    using the inverse softplus transform.  During the forward pass the
    actual scale is recovered via ``softplus(raw)``, which guarantees
    ``scale > 0`` regardless of how the optimizer updates the raw variable.
    This prevents division-by-zero in ``quantize_ste`` and ensures the
    dequantization formula ``scale * (int8 - zp)`` preserves its
    monotonicity.
    """

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

        self._trainable_scale = trainable

        if trainable:
            # Store inverse-softplus(scale) so the optimizer works in
            # unconstrained space and softplus maps back to (0, +∞).
            raw_scale = _softplus_inverse(np.asarray(scale, dtype=np.float32))
            self._scale_var = layer.add_weight(
                name=name_scale,
                shape=np.shape(scale),
                initializer=keras.initializers.Constant(typing.cast("float", raw_scale)),
                trainable=True,
            )
        else:
            self._scale_var = layer.add_weight(
                name=name_scale,
                shape=np.shape(scale),
                initializer=keras.initializers.Constant(typing.cast("float", scale)),
                trainable=False,
            )

        self.zero_point = layer.add_weight(
            name=name_zp,
            shape=np.shape(zero_point),
            initializer=keras.initializers.Constant(typing.cast("float", zero_point)),
            trainable=False,
        )

    @property
    def scale(self) -> TensorLike:
        """Returns the scale, applying softplus if the scale is trainable."""
        if self._trainable_scale:
            return ops.softplus(self._scale_var)
        return self._scale_var

    def dequantize(self, x: TensorLike) -> TensorLike:
        """Dequantize an INT8 tensor to Float32."""
        return dequantize_ste(x, self.scale, self.zero_point)

    def quantize(self, x: TensorLike) -> TensorLike:
        """Quantize a Float32 tensor to INT8."""
        return quantize_ste(x, self.scale, self.zero_point)

    def make_write_op(self, tensor_index: int) -> types.QuantizationWriteOp:
        """Create a write operation for these variables."""
        return make_quant_write_op(tensor_index, self.scale, self.zero_point)
