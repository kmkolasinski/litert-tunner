"""Tests for litert_tunner.export.

This module contains unit tests verifying edge cases and different paths
in the litert_tunner export functionality, including float32/int8 export,
debugger runs, and dataset checks.
"""

from pathlib import Path
from typing import Any

import keras
import numpy as np
import pytest
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter

from litert_tunner import export


@pytest.fixture
def dummy_model() -> keras.Model:
    """Fixture providing a simple Keras model."""
    inputs = keras.Input(shape=(4,))
    outputs = keras.layers.Dense(2)(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)


@pytest.fixture
def rep_dataset() -> Any:
    """Fixture providing a representative dataset generator."""

    def _gen():
        rng = np.random.default_rng(42)
        for _ in range(3):
            yield [rng.uniform(-1.0, 1.0, (1, 4)).astype(np.float32)]

    return _gen


def test__export_int8_missing_dataset(dummy_model: keras.Model, tmp_path: Path):
    """Test ValueError raised when INT8 quantization missing representative dataset."""
    with pytest.raises(ValueError, match="representative_dataset must be provided"):
        export.export_litert_model(dummy_model, tmp_path / "model.tflite", quantization="int8")


def test__export_float32(dummy_model: keras.Model, tmp_path: Path):
    """Test successful float32 export without dataset."""
    model_path = tmp_path / "model.tflite"
    debugger = export.export_litert_model(dummy_model, model_path, quantization="float32")
    assert debugger is None
    assert model_path.exists()
    assert model_path.name == "model.tflite"

    # Verify I/O is float32
    interp = Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    assert interp.get_input_details()[0]["dtype"] == np.float32


def test__export_float32_debugger_raises(dummy_model: keras.Model, tmp_path: Path):
    """Test ValueError raised when debugger used with float32 quantization."""
    with pytest.raises(ValueError, match="Debugger only supported for int8 quantization"):
        export.export_litert_model(
            dummy_model, tmp_path / "model.tflite", quantization="float32", run_debugger=True
        )


def test__export_int8_success(dummy_model: keras.Model, tmp_path: Path, rep_dataset: Any):
    """Test successful INT8 export."""
    model_path = tmp_path / "model.tflite"
    debugger = export.export_litert_model(
        dummy_model, model_path, quantization="int8", representative_dataset=rep_dataset
    )
    assert debugger is None
    assert model_path.exists()
    assert not (tmp_path / "quantization_stats.csv").exists()


def test__export_int8_float_io(dummy_model: keras.Model, tmp_path: Path, rep_dataset: Any):
    """Test float_io flag correctly sets input/output types."""
    # Test float_io=True (default is float32 I/O for INT8)
    path_float = tmp_path / "float_io" / "model.tflite"
    debugger1 = export.export_litert_model(
        dummy_model, path_float, representative_dataset=rep_dataset, float_io=True
    )
    assert debugger1 is None
    interp_float = Interpreter(model_path=str(path_float))
    interp_float.allocate_tensors()
    assert interp_float.get_input_details()[0]["dtype"] == np.float32

    # Test float_io=False (INT8 I/O for INT8)
    path_int8 = tmp_path / "int8_io" / "model.tflite"
    debugger2 = export.export_litert_model(
        dummy_model, path_int8, representative_dataset=rep_dataset, float_io=False
    )
    assert debugger2 is None
    interp_int8 = Interpreter(model_path=str(path_int8))
    interp_int8.allocate_tensors()
    assert interp_int8.get_input_details()[0]["dtype"] == np.int8


def test__export_int8_with_debugger(dummy_model: keras.Model, tmp_path: Path, rep_dataset: Any):
    """Test running debugger dumps stats and returns debugger instance."""
    model_path = tmp_path / "model.tflite"
    debugger = export.export_litert_model(
        dummy_model,
        model_path,
        quantization="int8",
        representative_dataset=rep_dataset,
        run_debugger=True,
    )
    assert model_path.exists()
    assert (tmp_path / "quantization_stats.csv").exists()

    assert isinstance(debugger, tf.lite.experimental.QuantizationDebugger)


def test__export_int8_denylisted_ops(dummy_model: keras.Model, tmp_path: Path, rep_dataset: Any):
    """Test exporting with denylisted_ops via debugger."""
    model_path = tmp_path / "model.tflite"
    debugger = export.export_litert_model(
        dummy_model,
        model_path,
        quantization="int8",
        representative_dataset=rep_dataset,
        run_debugger=True,
        denylisted_ops=["CONV_2D"],
    )
    assert model_path.exists()
    assert (tmp_path / "quantization_stats.csv").exists()

    assert isinstance(debugger, tf.lite.experimental.QuantizationDebugger)


def test__export_int8_denylisted_nodes(dummy_model: keras.Model, tmp_path: Path, rep_dataset: Any):
    """Test exporting with denylisted_nodes via debugger."""
    model_path = tmp_path / "model.tflite"
    debugger = export.export_litert_model(
        dummy_model,
        model_path,
        quantization="int8",
        representative_dataset=rep_dataset,
        run_debugger=True,
        denylisted_nodes=["StatefulPartitionedCall/sequential/dense/MatMul"],
    )
    assert model_path.exists()
    assert (tmp_path / "quantization_stats.csv").exists()

    assert isinstance(debugger, tf.lite.experimental.QuantizationDebugger)


def test__export_denylisted_ops_without_debugger_raises(dummy_model: keras.Model, tmp_path: Path):
    """Test ValueError raised when denylisted_ops provided without debugger."""
    msg = "denylisted_ops and denylisted_nodes can only be provided when run_debugger is True"
    with pytest.raises(ValueError, match=msg):
        export.export_litert_model(
            dummy_model,
            tmp_path / "model.tflite",
            denylisted_ops=["CONV_2D"],  # type: ignore  # noqa: PGH003
        )

    with pytest.raises(ValueError, match=msg):
        export.export_litert_model(
            dummy_model,
            tmp_path / "model.tflite",
            denylisted_nodes=["StatefulPartitionedCall/sequential/dense/MatMul"],  # type: ignore  # noqa: PGH003
        )
