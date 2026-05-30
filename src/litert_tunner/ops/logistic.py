"""LOGISTIC op implementation for litert_tunner."""

from __future__ import annotations

import keras
from keras import ops

from litert_tunner.graph import types
from litert_tunner.ops import registry
from litert_tunner.quantization import fake_quant


class QuantizedLogistic(keras.Layer):
    """Simulates TFLite's quantized LOGISTIC op.

    The forward pass performs:
        1. Dequantize INT8 input to float32
        2. Apply sigmoid
        3. Fake-quantize output to INT8
    """

    def __init__(
        self,
        input_scale: float,
        input_zero_point: float,
        output_scale: float,
        output_zero_point: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._input_scale = input_scale
        self._input_zero_point = input_zero_point
        self._output_scale = output_scale
        self._output_zero_point = output_zero_point

    def build(self, input_shape):
        """Create quantization params."""
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
        # LOGISTIC output quantization is hardcoded in TFLite (scale=1/256, zp=-128)
        # So we keep it frozen.
        self.output_scale = self.add_weight(
            name="output_scale",
            shape=(),
            initializer=keras.initializers.Constant(self._output_scale),
            trainable=False,
        )
        self.output_zero_point = self.add_weight(
            name="output_zero_point",
            shape=(),
            initializer=keras.initializers.Constant(self._output_zero_point),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        """Forward pass simulating quantized LOGISTIC."""
        # 1. Dequantize
        input_float = fake_quant.dequantize_ste(x, self.input_scale, self.input_zero_point)
        # 2. Sigmoid
        output_float = ops.sigmoid(input_float)
        # 3. Quantize to simulated INT8
        return fake_quant.quantize_ste(output_float, self.output_scale, self.output_zero_point)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
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
        tensors: tuple[types.TensorInfo, ...],
    ) -> tuple[list[types.BufferWriteOp], list[types.QuantizationWriteOp]]:
        """Return flatbuffer write instructions for the LOGISTIC layer."""
        quant_writes: list[types.QuantizationWriteOp] = []
        quant_writes.append(
            fake_quant.make_quant_write_op(
                op.input_indices[0], self.input_scale, self.input_zero_point
            )
        )
        quant_writes.append(
            fake_quant.make_quant_write_op(
                op.output_indices[0], self.output_scale, self.output_zero_point
            )
        )
        return [], quant_writes


@registry.register_op("LOGISTIC")
def build_logistic(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
    graph_def: types.GraphDef | None = None,
) -> keras.Layer:
    """Build a QuantizedLogistic layer from parsed TFLite operator info."""
    input_tensor = tensors[op.input_indices[0]]
    output_tensor = tensors[op.output_indices[0]]

    input_quant = input_tensor.quantization
    output_quant = output_tensor.quantization

    if input_quant is None or output_quant is None:
        msg = "LOGISTIC requires quantized input and output tensors"
        raise ValueError(msg)

    return QuantizedLogistic(
        input_scale=float(input_quant.scales[0]),
        input_zero_point=float(input_quant.zero_points[0]),
        output_scale=float(output_quant.scales[0]),
        output_zero_point=float(output_quant.zero_points[0]),
        name=f"quantized_logistic_{op.output_indices[0]}",
    )
