"""Operations module for litert_tunner.

Triggers automatic registration of all operator builders.
"""

from __future__ import annotations

from litert_tunner.ops import (
    add,
    conv2d,
    dense,
    depthwise_conv2d,
    gelu,
    logistic,
    mean,
    mul,
    neg,
    pack,
    pool,
    quantize_op,
    registry,
    relu,
    reshape,
    rsqrt,
    shape_op,
    squared_difference,
    strided_slice,
    sub,
    utils,
)

__all__ = [
    "add",
    "conv2d",
    "dense",
    "depthwise_conv2d",
    "gelu",
    "logistic",
    "mean",
    "mul",
    "neg",
    "pack",
    "pool",
    "quantize_op",
    "registry",
    "relu",
    "reshape",
    "rsqrt",
    "shape_op",
    "squared_difference",
    "strided_slice",
    "sub",
    "utils",
]
