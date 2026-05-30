"""Flatbuffer module for litert_tunner.

Exposes parse_tflite and save_tflite.
"""

from __future__ import annotations

from litert_tunner.flatbuffer.parser import parse_tflite as parse_tflite
from litert_tunner.flatbuffer.writer import save_tflite as save_tflite
