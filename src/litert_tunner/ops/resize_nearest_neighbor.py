"""RESIZE_NEAREST_NEIGHBOR op implementation for litert_tunner.

Simulates TFLite's RESIZE_NEAREST_NEIGHBOR op as a Keras layer.
Resizes the spatial dimensions of an input tensor using nearest-neighbor
interpolation. This is a passthrough for quantization — scale and
zero-point are preserved unchanged.
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


class ResizeNearestNeighbor(keras.Layer):
    """Simulates TFLite's RESIZE_NEAREST_NEIGHBOR op.

    Performs nearest-neighbor interpolation to resize spatial dimensions.
    This is a passthrough for quantization — scale and zero-point
    are preserved unchanged.

    No trainable parameters. Does not implement ``Writable``.

    Args:
        target_height: Target height after resize.
        target_width: Target width after resize.
        align_corners: Whether to align corners during resize.
        half_pixel_centers: Whether to use half-pixel centers.
        name: Layer name.
    """

    def __init__(
        self,
        target_height: int,
        target_width: int,
        *,
        align_corners: bool = False,
        half_pixel_centers: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._target_height = target_height
        self._target_width = target_width
        self._align_corners = align_corners
        self._half_pixel_centers = half_pixel_centers

    def call(self, x: TensorLike) -> TensorLike:
        """Forward pass performing nearest-neighbor resize."""
        input_height = ops.shape(x)[1]
        input_width = ops.shape(x)[2]

        y_indices = ops.arange(self._target_height, dtype="float32")
        x_indices = ops.arange(self._target_width, dtype="float32")

        height_scale = ops.cast(input_height, "float32") / ops.cast(self._target_height, "float32")
        width_scale = ops.cast(input_width, "float32") / ops.cast(self._target_width, "float32")

        if self._align_corners and self._target_height > 1:
            height_scale = ops.cast(input_height - 1, "float32") / ops.cast(
                self._target_height - 1, "float32"
            )
        if self._align_corners and self._target_width > 1:
            width_scale = ops.cast(input_width - 1, "float32") / ops.cast(
                self._target_width - 1, "float32"
            )

        if self._half_pixel_centers:
            y_coords = ops.floor((y_indices + 0.5) * height_scale)
            x_coords = ops.floor((x_indices + 0.5) * width_scale)
        elif self._align_corners:
            y_coords = ops.round(y_indices * height_scale)
            x_coords = ops.round(x_indices * width_scale)
        else:
            y_coords = ops.floor(y_indices * height_scale)
            x_coords = ops.floor(x_indices * width_scale)

        y_coords = ops.cast(ops.clip(y_coords, 0.0, ops.cast(input_height - 1, "float32")), "int32")
        x_coords = ops.cast(ops.clip(x_coords, 0.0, ops.cast(input_width - 1, "float32")), "int32")

        output = ops.take(x, y_coords, axis=1)
        return ops.take(output, x_coords, axis=2)

    def get_config(self):
        """Return the configuration dictionary for serialization."""
        config = super().get_config()
        config.update({
            "target_height": self._target_height,
            "target_width": self._target_width,
            "align_corners": self._align_corners,
            "half_pixel_centers": self._half_pixel_centers,
        })
        return config


@registry.register_op("RESIZE_NEAREST_NEIGHBOR")
def build_resize_nearest_neighbor(
    op: types.OperatorInfo,
    tensors: tuple[types.TensorInfo, ...],
) -> keras.Layer:
    """Build a ResizeNearestNeighbor layer from parsed TFLite operator info.

    TFLite RESIZE_NEAREST_NEIGHBOR inputs:
        [0] input tensor (INT8), shape (batch, H, W, C)
        [1] size tensor (INT32, constant), shape (2,) — [target_height, target_width]

    TFLite RESIZE_NEAREST_NEIGHBOR outputs:
        [0] output tensor (INT8), shape (batch, target_H, target_W, C)

    Args:
        op: Parsed operator info with input/output indices and options.
        tensors: All tensors in the graph.

    Returns:
        A configured QuantizedResizeNearestNeighbor Keras layer.
    """
    # Extract target size from the constant size tensor
    size_tensor = tensors[op.input_indices[1]]
    if size_tensor.data is None:
        msg = "RESIZE_NEAREST_NEIGHBOR requires a constant size tensor"
        raise ValueError(msg)

    target_height = int(size_tensor.data.flat[0])
    target_width = int(size_tensor.data.flat[1])

    align_corners = op.options.get("AlignCorners", False)
    half_pixel_centers = op.options.get("HalfPixelCenters", False)

    return ResizeNearestNeighbor(
        target_height=target_height,
        target_width=target_width,
        align_corners=align_corners,
        half_pixel_centers=half_pixel_centers,
        name=f"resize_nn_{op.output_indices[0]}",
    )
