"""STRIDED_SLICE op implementation for litert_tunner.

Simulates TFLite's STRIDED_SLICE op as a Keras layer.
Currently supports the common patterns used in EfficientNet's
Squeeze-and-Excitation blocks (scalar extraction with ShrinkAxisMask).
Can be extended for more complex slicing patterns as needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import keras

from litert_tunner.ops import registry

if TYPE_CHECKING:
    from litert_tunner.graph import types
    from litert_tunner.ops.utils import TensorLike

    ShapeLike = tuple[int, ...] | list[int] | list[tuple[int, ...]]

# Mask bit positions for TFLite StridedSlice
_BEGIN_MASK_BIT = 0
_END_MASK_BIT = 1
_ELLIPSIS_MASK_BIT = 2
_NEW_AXIS_MASK_BIT = 3
_SHRINK_AXIS_MASK_BIT = 4


class StridedSlice(keras.Layer):
    """Simulates TFLite's STRIDED_SLICE op.

    Extracts a strided slice from the input tensor. This is a passthrough
    for quantization — scale and zero-point are preserved.

    Currently handles the most common patterns:
        - Simple range slicing with begin/end/strides
        - Scalar extraction via ShrinkAxisMask

    No trainable parameters. Does not implement ``Writable``.

    Args:
        begin: Begin indices for each dimension.
        end: End indices for each dimension.
        strides: Strides for each dimension.
        begin_mask: Bitmask — if bit i is set, begin[i] is ignored.
        end_mask: Bitmask — if bit i is set, end[i] is ignored.
        ellipsis_mask: Bitmask — at most one bit may be set.
        new_axis_mask: Bitmask — inserts a new axis at bit positions.
        shrink_axis_mask: Bitmask — removes axis at bit positions (scalar extraction).
        name: Layer name.
    """

    def __init__(
        self,
        begin: tuple[int, ...],
        end: tuple[int, ...],
        strides: tuple[int, ...],
        begin_mask: int = 0,
        end_mask: int = 0,
        ellipsis_mask: int = 0,
        new_axis_mask: int = 0,
        shrink_axis_mask: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._begin = begin
        self._end = end
        self._strides = strides
        self._begin_mask = begin_mask
        self._end_mask = end_mask
        self._ellipsis_mask = ellipsis_mask
        self._new_axis_mask = new_axis_mask
        self._shrink_axis_mask = shrink_axis_mask

    def call(self, x: TensorLike) -> TensorLike:
        """Forward pass applying strided slice.

        Args:
            x: Input tensor.

        Returns:
            Sliced output tensor.
        """
        return self._apply_strided_slice(x)

    def _apply_strided_slice(self, x: TensorLike) -> TensorLike:
        """Apply strided slice using Python indexing on the tensor.

        Builds a tuple of slice objects and integer indices to perform the
        slicing. For dimensions with shrink_axis_mask set, uses an integer
        index to remove that dimension. Otherwise builds a proper slice with
        begin/end/stride, respecting begin_mask and end_mask.

        Args:
            x: Input tensor.

        Returns:
            Sliced tensor.
        """
        ndim = len(self._begin)
        slices: list[int | slice] = []

        for i in range(ndim):
            if self._shrink_axis_mask & (1 << i):
                # Scalar extraction — use integer index to remove the dimension
                slices.append(self._begin[i])
            else:
                # Build a normal slice
                start = None if (self._begin_mask & (1 << i)) else self._begin[i]
                stop = None if (self._end_mask & (1 << i)) else self._end[i]
                step = self._strides[i]
                slices.append(slice(start, stop, step if step != 1 else None))

        return x[tuple(slices)]

    def get_config(self):
        """Return the configuration dictionary for serialization of the layer."""
        config = super().get_config()
        config.update({
            "begin": self._begin,
            "end": self._end,
            "strides": self._strides,
            "begin_mask": self._begin_mask,
            "end_mask": self._end_mask,
            "ellipsis_mask": self._ellipsis_mask,
            "new_axis_mask": self._new_axis_mask,
            "shrink_axis_mask": self._shrink_axis_mask,
        })
        return config


@registry.register_op("STRIDED_SLICE")
def build_strided_slice(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a StridedSlice layer from parsed TFLite operator info.

    TFLite STRIDED_SLICE inputs:
        [0] input tensor
        [1] begin tensor (INT32, constant)
        [2] end tensor (INT32, constant)
        [3] strides tensor (INT32, constant)

    TFLite STRIDED_SLICE outputs:
        [0] output tensor

    Args:
        op: Parsed operator info with input/output indices and options.
        tensors: All tensors in the graph.
        graph_def: The parsed GraphDef.

    Returns:
        A configured QuantizedStridedSlice Keras layer.
    """
    begin_tensor = tensors[op.input_indices[1]]
    end_tensor = tensors[op.input_indices[2]]
    strides_tensor = tensors[op.input_indices[3]]

    if begin_tensor.data is None or end_tensor.data is None or strides_tensor.data is None:
        msg = "STRIDED_SLICE requires constant begin, end, and strides tensors"
        raise ValueError(msg)

    begin = tuple(int(v) for v in begin_tensor.data.flatten())
    end = tuple(int(v) for v in end_tensor.data.flatten())
    strides = tuple(int(v) for v in strides_tensor.data.flatten())

    begin_mask = op.options.get("BeginMask", 0)
    end_mask = op.options.get("EndMask", 0)
    ellipsis_mask = op.options.get("EllipsisMask", 0)
    new_axis_mask = op.options.get("NewAxisMask", 0)
    shrink_axis_mask = op.options.get("ShrinkAxisMask", 0)

    return StridedSlice(
        begin=begin,
        end=end,
        strides=strides,
        begin_mask=begin_mask,
        end_mask=end_mask,
        ellipsis_mask=ellipsis_mask,
        new_axis_mask=new_axis_mask,
        shrink_axis_mask=shrink_axis_mask,
        name=f"strided_slice_{op.output_indices[0]}",
    )
