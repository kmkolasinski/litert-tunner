"""PACK op implementation for litert_tunner.

Simulates TFLite's PACK op as a Keras layer.
Packs multiple tensors into a single tensor along a new axis,
equivalent to ``np.stack`` / ``keras.ops.stack``.

In EfficientNet SE blocks, PACK often mixes dynamic activations
(e.g., batch dimension from STRIDED_SLICE) with constant values.
This layer embeds the constant values and inserts dynamic inputs
at the correct positions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import keras
from keras import ops

from litert_tunner.ops import registry

if TYPE_CHECKING:
    from litert_tunner.graph import types
    from litert_tunner.ops.utils import TensorLike

    ShapeLike = tuple[int, ...] | list[int] | list[tuple[int, ...]]


class QuantizedPack(keras.Layer):
    """Simulates TFLite's PACK op.

    Packs (stacks) multiple input tensors along a new axis. This is a
    passthrough for quantization — scale and zero-point are preserved.

    Supports mixed constant/dynamic inputs: constant values are embedded
    at build time, and dynamic inputs are inserted at the correct positions
    during the call.

    No trainable parameters. Does not implement ``Writable``.

    Args:
        axis: The axis along which to pack/stack the tensors.
        values_count: Expected total number of values (dynamic + constant).
        constant_map: Mapping from position index to constant float value.
            Dynamic inputs fill the remaining positions in order.
        name: Layer name.
    """

    def __init__(
        self,
        axis: int = 0,
        values_count: int = 2,
        constant_map: dict[int, float] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._axis = axis
        self._values_count = values_count
        self._constant_map = constant_map or {}

    def call(self, inputs: TensorLike) -> TensorLike:
        """Forward pass stacking input tensors.

        If only a single dynamic input is present, ``inputs`` is a single
        tensor rather than a list. The method handles both cases.

        Args:
            inputs: Single tensor or list of tensors (dynamic inputs only).

        Returns:
            Stacked output tensor with a new dimension at the specified axis.
        """
        # Normalize inputs to a list
        dynamic_inputs = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]

        # Assemble all values (dynamic + constant) in the correct order
        all_values: list[TensorLike] = []
        dynamic_idx = 0
        for pos in range(self._values_count):
            if pos in self._constant_map:
                all_values.append(
                    ops.cast(ops.convert_to_tensor(self._constant_map[pos]), "float32")
                )
            else:
                all_values.append(dynamic_inputs[dynamic_idx])
                dynamic_idx += 1

        # Expand each scalar along the target axis, then concatenate
        expanded = [ops.expand_dims(v, axis=self._axis) for v in all_values]
        return ops.concatenate(expanded, axis=self._axis)

    def get_config(self):
        """Return the configuration dictionary for serialization of the layer."""
        config = super().get_config()
        config.update({
            "axis": self._axis,
            "values_count": self._values_count,
            "constant_map": self._constant_map,
        })
        return config


@registry.register_op("PACK")
def build_pack(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a QuantizedPack layer from parsed TFLite operator info.

    TFLite PACK inputs:
        [0..N-1] input tensors to pack (may include constants)

    TFLite PACK outputs:
        [0] output tensor — packed along the specified axis

    Constant inputs (those with ``data is not None``) are embedded directly
    into the layer. Dynamic inputs are provided at call time.

    Args:
        op: Parsed operator info with input/output indices and options.
        tensors: All tensors in the graph.
        graph_def: The parsed GraphDef.

    Returns:
        A configured QuantizedPack Keras layer.
    """
    axis = op.options.get("Axis", 0)
    values_count = op.options.get("ValuesCount", len(op.input_indices))

    # Build constant map: position → constant value for inputs that have data
    constant_map: dict[int, float] = {}
    for pos, idx in enumerate(op.input_indices):
        if idx >= 0:
            tensor = tensors[idx]
            if tensor.data is not None:
                constant_map[pos] = float(tensor.data.flat[0])

    return QuantizedPack(
        axis=axis,
        values_count=values_count,
        constant_map=constant_map,
        name=f"quantized_pack_{op.output_indices[0]}",
    )
