"""RESIZE_NEAREST_NEIGHBOR op implementation for litert_tunner.

Simulates TFLite's RESIZE_NEAREST_NEIGHBOR op as a Keras layer.
Resizes the spatial dimensions of an input tensor using nearest-neighbor
interpolation. This is a passthrough for quantization — scale and
zero-point are preserved unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import keras

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
        name: Layer name.
    """

    def __init__(
        self,
        target_height: int,
        target_width: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._target_height = target_height
        self._target_width = target_width

    def call(self, inputs: TensorLike, *, training: bool | None = None) -> TensorLike:
        if training:
            # https://github.com/keras-team/keras/issues/294
            # Use `ops.repeat` for `nearest` interpolation to enable XLA
            input_shape = keras.ops.shape(inputs)
            height_factor = self._target_height // input_shape[1]
            width_factor = self._target_width // input_shape[2]
            x = keras.ops.repeat(inputs, height_factor, axis=1)
            return keras.ops.repeat(x, width_factor, axis=2)

        return keras.ops.image.resize(
            inputs,
            size=(self._target_height, self._target_width),
            interpolation="nearest",
        )

    def get_config(self):
        """Return the configuration dictionary for serialization."""
        config = super().get_config()
        config.update({
            "target_height": self._target_height,
            "target_width": self._target_width,
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

    align_corners = op.options.get("AlignCorners")
    half_pixel_centers = op.options.get("HalfPixelCenters")

    if align_corners:
        msg = f"align_corners={align_corners} in ResizeNearestNeighbor op is not supported"
        raise NotImplementedError(msg)

    if half_pixel_centers is False:
        msg = (
            f"half_pixel_centers={half_pixel_centers} in ResizeNearestNeighbor op is not supported"
        )
        raise NotImplementedError(msg)

    return ResizeNearestNeighbor(
        target_height=target_height,
        target_width=target_width,
        name=f"resize_nn_{op.output_indices[0]}",
    )
