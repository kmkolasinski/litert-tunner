"""Operations module for litert_tunner.

Triggers automatic registration of dense and quantize operators.
"""

from __future__ import annotations

from litert_tunner.ops import dense as dense
from litert_tunner.ops import quantize_op as quantize_op
from litert_tunner.ops import registry as registry
from litert_tunner.ops import utils as utils

__all__ = [
    "registry",
    "dense",
    "quantize_op",
    "utils",
]
