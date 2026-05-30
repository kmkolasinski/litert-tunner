"""Operations module for litert_tunner.

Triggers automatic registration of all operator builders.
"""

from __future__ import annotations

from litert_tunner.ops import (
    add,
    conv2d,
    dense,
    depthwise_conv2d,
    logistic,
    mean,
    mul,
    pack,
    pool,
    quantize_op,
    registry,
    reshape,
    shape_op,
    strided_slice,
    sub,
    utils,
)

__all__ = [
    "add",
    "conv2d",
    "dense",
    "depthwise_conv2d",
    "logistic",
    "mean",
    "mul",
    "pack",
    "pool",
    "quantize_op",
    "registry",
    "reshape",
    "shape_op",
    "strided_slice",
    "sub",
    "utils",
]
