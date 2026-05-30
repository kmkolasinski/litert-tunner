"""MEAN op implementation for litert_tunner.

Simulates TFLite's quantized MEAN op as a Keras layer.
MEAN is used by GlobalAveragePooling2D — it reduces spatial dimensions
by computing the mean, then requantizes the output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import keras
from keras import ops

from litert_tunner.graph import types
from litert_tunner.ops import registry, utils

if TYPE_CHECKING:
    from litert_tunner.ops.utils import TensorLike

    ShapeLike = tuple[int, ...] | list[int] | list[tuple[int, ...]]


class QuantizedMean(keras.Layer, types.Writable):
    """Simulates TFLite's quantized MEAN op.

    The forward pass performs:
        1. Dequantize INT8 input to float32
        2. Compute mean over specified axes
        3. Fake-quantize output (quantize with STE)

    Trainable parameters: output_scale, output_zero_point.
    Frozen parameters: input scale and zero-point.

    Args:
        axis: Axes over which to compute the mean.
        keep_dims: Whether to retain reduced dimensions.
        input_scale: Scale of the input activation tensor.
        input_zero_point: Zero point of the input activation tensor.
        output_scale: Scale of the output activation tensor.
        output_zero_point: Zero point of the output activation tensor.
        name: Layer name.
    """

    def __init__(
        self,
        axis: tuple[int, ...],
        *,
        keep_dims: bool,
        input_scale: float,
        input_zero_point: float,
        output_scale: float,
        output_zero_point: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._axis = axis
        self._keep_dims = keep_dims
        self._input_scale = input_scale
        self._input_zero_point = input_zero_point
        self._output_scale = output_scale
        self._output_zero_point = output_zero_point

    def build(self, input_shape: ShapeLike) -> None:
        """Create quantization params for the layer."""
        # Input quantization params (frozen)
        self.input_quant = utils.QuantizationVars(
            self,
            "input",
            self._input_scale,
            self._input_zero_point,
            trainable=False,
        )

        # Output quantization params (trainable)
        self.output_quant = utils.QuantizationVars(
            self,
            "output",
            self._output_scale,
            self._output_zero_point,
            trainable=True,
        )

        super().build(input_shape)

    def call(self, x: TensorLike) -> TensorLike:
        """Forward pass simulating TFLite's quantized MEAN.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after mean reduction and fake-quantization.
        """
        # 1. Dequantize input
        input_float = self.input_quant.dequantize(x)

        # 2. Compute mean over specified axes
        output = ops.mean(input_float, axis=self._axis, keepdims=self._keep_dims)

        # 3. Quantize output to simulated INT8
        return self.output_quant.quantize(output)

    def get_config(self):
        """Return the configuration dictionary for serialization of the layer."""
        config = super().get_config()
        config.update(
            {
                "axis": self._axis,
                "keep_dims": self._keep_dims,
                "input_scale": self._input_scale,
                "input_zero_point": self._input_zero_point,
                "output_scale": self._output_scale,
                "output_zero_point": self._output_zero_point,
            }
        )
        return config

    def collect_write_ops(
        self,
        op: types.OperatorInfo,
    ) -> tuple[list[types.BufferWriteOp], list[types.QuantizationWriteOp]]:
        """Return flatbuffer write instructions for the MEAN layer.

        Writes back quantization params for input and output tensors.

        Args:
            op: The OperatorInfo that this layer was built from.
            tensors: All tensors in the graph.

        Returns:
            A tuple of (buffer_writes, quantization_writes).
        """
        quant_writes: list[types.QuantizationWriteOp] = []
        quant_writes.append(self.input_quant.make_write_op(op.input_indices[0]))
        quant_writes.append(self.output_quant.make_write_op(op.output_indices[0]))
        return [], quant_writes


@registry.register_op("MEAN")
def build_mean(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a QuantizedMean layer from parsed TFLite operator info.

    TFLite MEAN inputs:
        [0] input tensor (INT8)
        [1] axis tensor (INT32, constant) — specifies reduction axes

    TFLite MEAN outputs:
        [0] output tensor (INT8)

    Args:
        op: Parsed operator info with input/output indices and options.
        tensors: All tensors in the graph.
        graph_def: The parsed GraphDef.

    Returns:
        A configured QuantizedMean Keras layer.
    """
    input_tensor = tensors[op.input_indices[0]]
    axis_tensor = tensors[op.input_indices[1]]
    output_tensor = tensors[op.output_indices[0]]

    # Axis data must be available (constant tensor)
    if axis_tensor.data is None:
        msg = f"Axis tensor '{axis_tensor.name}' has no data"
        raise ValueError(msg)

    axis = tuple(int(a) for a in axis_tensor.data.flatten())

    # Extract quantization params
    input_quant = input_tensor.quantization
    output_quant = output_tensor.quantization

    if input_quant is None or output_quant is None:
        msg = "MEAN requires quantized input and output tensors"
        raise ValueError(msg)

    keep_dims = bool(op.options.get("KeepDims", False))

    return QuantizedMean(
        axis=axis,
        keep_dims=keep_dims,
        input_scale=float(input_quant.scales[0]),
        input_zero_point=float(input_quant.zero_points[0]),
        output_scale=float(output_quant.scales[0]),
        output_zero_point=float(output_quant.zero_points[0]),
        name=f"quantized_mean_{op.output_indices[0]}",
    )
