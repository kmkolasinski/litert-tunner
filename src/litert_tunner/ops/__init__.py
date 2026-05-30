"""Operations module for litert_tunner.

Triggers automatic registration of all operator builders.
"""

from __future__ import annotations

from litert_tunner.ops import add as add
from litert_tunner.ops import conv2d as conv2d
from litert_tunner.ops import dense as dense
from litert_tunner.ops import logistic as logistic
from litert_tunner.ops import mean as mean
from litert_tunner.ops import mul as mul
from litert_tunner.ops import pool as pool
from litert_tunner.ops import quantize_op as quantize_op
from litert_tunner.ops import registry as registry
from litert_tunner.ops import utils as utils

__all__ = [
    "add",
    "conv2d",
    "dense",
    "logistic",
    "mean",
    "mul",
    "pool",
    "quantize_op",
    "registry",
    "utils",
]
