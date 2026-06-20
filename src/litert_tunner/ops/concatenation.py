"""CONCATENATION op implementation for litert_tunner.

Simulates TFLite's CONCATENATION op as a Keras layer.
Supports both fully-quantized INT8 and float32 (unquantized) models.

Quantized variant:
Concatenates multiple INT8 tensors along a specified axis.
When input quantization parameters differ from the output, each input
is requantized (dequantize → quantize) to the output's quantization
domain before concatenation. When they match, it's a passthrough.

Float32 variant:
Directly applies keras.ops.concatenate and any fused activation.
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


class QuantizedConcatenation(keras.Layer, types.Writable):
    """Simulates TFLite's quantized CONCATENATION op.

    The forward pass performs:
        1. For each input, dequantize INT8 → float32 using input quant params
        2. Concatenate along the specified axis in float32
        3. Apply fused activation (if any)
        4. Fake-quantize output to INT8

    When all inputs share the same quantization params as the output,
    the dequant/quant round-trip is still performed for gradient flow
    consistency.

    Trainable parameters: output_scale, output_zero_point.
    Frozen parameters: input scales and zero-points.

    Args:
        input_scales: Scales for each input tensor.
        input_zero_points: Zero-points for each input tensor.
        output_scale: Scale of the output tensor.
        output_zero_point: Zero-point of the output tensor.
        axis: Concatenation axis.
        fused_activation: TFLite fused activation code.
        constant_inputs: Dictionary mapping input index to its constant numpy array data.
        name: Layer name.
    """

    def __init__(
        self,
        input_scales: list[float],
        input_zero_points: list[float],
        output_scale: float,
        output_zero_point: float,
        axis: int = -1,
        fused_activation: int = utils.FUSED_ACTIVATION_NONE,
        constant_inputs: dict[int, np.ndarray] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._input_scales = input_scales
        self._input_zero_points = input_zero_points
        self._output_scale = output_scale
        self._output_zero_point = output_zero_point
        self._axis = axis
        self._fused_activation = fused_activation
        self._constant_inputs_data = constant_inputs or {}
        self._num_inputs = len(input_scales)

    def build(self, input_shape: ShapeLike) -> None:
        """Create quantization param variables for each input and output."""
        self.input_quants = []
        for i, (scale, zp) in enumerate(
            zip(self._input_scales, self._input_zero_points, strict=True)
        ):
            qv = utils.QuantizationVars(
                self,
                f"input{i}",
                scale,
                zp,
                trainable=False,
            )
            self.input_quants.append(qv)

        self.output_quant = utils.QuantizationVars(
            self,
            "output",
            self._output_scale,
            self._output_zero_point,
            trainable=False,
        )

        # Note: We store the constant weights in a list rather than a dict.
        # Storing Keras variables in a dict attribute (e.g. `self.constant_inputs = {}`)
        # was found to cause TFLite export to fail with a `NoneType` error during
        # model tracing. Using a list with a separate integer map works correctly.
        self.constant_weights = []
        self._constant_idx_to_weight_idx = {}
        for idx, data in self._constant_inputs_data.items():
            weight = self.add_weight(
                name=f"constant_input_{idx}",
                shape=data.shape,
                initializer=keras.initializers.Constant(
                    typing.cast("float", data.astype(np.float32))
                ),
                trainable=False,
            )
            self._constant_idx_to_weight_idx[idx] = len(self.constant_weights)
            self.constant_weights.append(weight)
        super().build(input_shape)

    def call(self, inputs: TensorLike | list[TensorLike] | tuple[TensorLike, ...]) -> TensorLike:
        """Forward pass simulating quantized CONCATENATION.

        Args:
            inputs: List of input tensors to concatenate.

        Returns:
            Concatenated and requantized output tensor.
        """
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        full_inputs = []
        dynamic_idx = 0
        for i in range(self._num_inputs):
            if i in self._constant_idx_to_weight_idx:
                weight_idx = self._constant_idx_to_weight_idx[i]
                full_inputs.append(self.constant_weights[weight_idx])
            else:
                full_inputs.append(inputs[dynamic_idx])
                dynamic_idx += 1

        # 1. Dequantize each input to float32
        float_inputs = [
            qv.dequantize(x) for x, qv in zip(full_inputs, self.input_quants, strict=True)
        ]
        # 2. Concatenate
        output_float = ops.concatenate(float_inputs, axis=self._axis)
        # 3. Fused activation
        output_float = utils.apply_fused_activation(output_float, self._fused_activation)
        # 4. Quantize to simulated INT8
        return self.output_quant.quantize(output_float)

    def get_config(self):
        """Return the configuration dictionary for serialization."""
        config = super().get_config()
        config.update({
            "input_scales": self._input_scales,
            "input_zero_points": self._input_zero_points,
            "output_scale": self._output_scale,
            "output_zero_point": self._output_zero_point,
            "axis": self._axis,
            "fused_activation": self._fused_activation,
        })
        return config

    def collect_write_ops(
        self,
        op: types.OperatorInfo,
    ) -> tuple[list[types.BufferWriteOp], list[types.QuantizationWriteOp]]:
        """Return flatbuffer write instructions for the CONCATENATION layer."""
        quant_writes: list[types.QuantizationWriteOp] = []
        for i, qv in enumerate(self.input_quants):
            quant_writes.append(qv.make_write_op(op.input_indices[i]))
        quant_writes.append(self.output_quant.make_write_op(op.output_indices[0]))
        return [], quant_writes


class FloatConcatenation(keras.Layer):
    """Float32 CONCATENATION op.

    Args:
        num_inputs: Total number of inputs (dynamic + constant).
        axis: Concatenation axis.
        fused_activation: TFLite fused activation code.
        constant_inputs: Dictionary mapping input index to its constant numpy array data.
        name: Layer name.
    """

    def __init__(
        self,
        num_inputs: int,
        axis: int = -1,
        fused_activation: int = utils.FUSED_ACTIVATION_NONE,
        constant_inputs: dict[int, np.ndarray] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._num_inputs = num_inputs
        self._axis = axis
        self._fused_activation = fused_activation
        self._constant_inputs_data = constant_inputs or {}

    def build(self, input_shape: ShapeLike) -> None:
        """Create variables for constant inputs."""
        # Note: We store the constant weights in a list rather than a dict.
        # Storing Keras variables in a dict attribute (e.g. `self.constant_inputs = {}`)
        # was found to cause TFLite export to fail with a `NoneType` error during
        # model tracing. Using a list with a separate integer map works correctly.
        self.constant_weights = []
        self._constant_idx_to_weight_idx = {}
        for idx, data in self._constant_inputs_data.items():
            weight = self.add_weight(
                name=f"constant_input_{idx}",
                shape=data.shape,
                initializer=keras.initializers.Constant(
                    typing.cast("float", data.astype(np.float32))
                ),
                trainable=False,
            )
            self._constant_idx_to_weight_idx[idx] = len(self.constant_weights)
            self.constant_weights.append(weight)
        super().build(input_shape)

    def call(self, inputs: TensorLike | list[TensorLike] | tuple[TensorLike, ...]) -> TensorLike:
        """Forward pass applying concatenation in float32."""
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        full_inputs = []
        dynamic_idx = 0
        for i in range(self._num_inputs):
            if i in self._constant_idx_to_weight_idx:
                weight_idx = self._constant_idx_to_weight_idx[i]
                full_inputs.append(self.constant_weights[weight_idx])
            else:
                full_inputs.append(inputs[dynamic_idx])
                dynamic_idx += 1

        output_float = ops.concatenate(full_inputs, axis=self._axis)
        return utils.apply_fused_activation(output_float, self._fused_activation)

    def get_config(self):
        """Return the configuration dictionary for serialization."""
        config = super().get_config()
        config.update({
            "num_inputs": self._num_inputs,
            "axis": self._axis,
            "fused_activation": self._fused_activation,
        })
        return config


@registry.register_op("CONCATENATION")
def build_concatenation(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a CONCATENATION layer from parsed TFLite operator info.

    TFLite CONCATENATION inputs:
        [0..N-1] input tensors to concatenate

    TFLite CONCATENATION outputs:
        [0] output tensor

    Args:
        op: Parsed operator info with input/output indices and options.
        tensors: All tensors in the graph.

    Returns:
        A configured Keras layer.
    """
    input_tensor = tensors[op.input_indices[0]]
    if types.is_quantized(input_tensor):
        return _build_quantized_concatenation(op, tensors)
    return _build_float_concatenation(op, tensors)


def _build_quantized_concatenation(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a QuantizedConcatenation layer."""
    output_tensor = tensors[op.output_indices[0]]
    output_quant = output_tensor.quantization

    if output_quant is None:
        msg = "CONCATENATION requires quantized output tensor"
        raise ValueError(msg)

    input_scales: list[float] = []
    input_zero_points: list[float] = []
    constant_inputs = {}
    for i, idx in enumerate(op.input_indices):
        input_tensor = tensors[idx]
        input_quant = input_tensor.quantization
        if input_quant is None:
            msg = f"CONCATENATION requires quantized input tensor at index {idx}"
            raise ValueError(msg)
        input_scales.append(float(input_quant.scales[0]))
        input_zero_points.append(float(input_quant.zero_points[0]))
        if input_tensor.data is not None:
            constant_inputs[i] = input_tensor.data.astype(np.float32)

    axis = op.options.get("Axis", -1)
    fused_activation = op.options.get("fused_activation_function", utils.FUSED_ACTIVATION_NONE)

    return QuantizedConcatenation(
        input_scales=input_scales,
        input_zero_points=input_zero_points,
        output_scale=float(output_quant.scales[0]),
        output_zero_point=float(output_quant.zero_points[0]),
        axis=axis,
        fused_activation=fused_activation,
        constant_inputs=constant_inputs,
        name=f"quantized_concatenation_{op.output_indices[0]}",
    )


def _build_float_concatenation(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a FloatConcatenation layer."""
    axis = op.options.get("Axis", -1)
    fused_activation = op.options.get("fused_activation_function", utils.FUSED_ACTIVATION_NONE)

    constant_inputs = {}
    for i, idx in enumerate(op.input_indices):
        input_tensor = tensors[idx]
        if input_tensor.data is not None:
            constant_inputs[i] = input_tensor.data.astype(np.float32)

    return FloatConcatenation(
        num_inputs=len(op.input_indices),
        axis=axis,
        fused_activation=fused_activation,
        constant_inputs=constant_inputs,
        name=f"float_concatenation_{op.output_indices[0]}",
    )
