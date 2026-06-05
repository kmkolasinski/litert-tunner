"""Tests for operations with constant inputs."""

from collections.abc import Callable
from pathlib import Path

import keras
import numpy as np

import litert_tunner
from litert_tunner import testing_utils
from tests import conftest


def test__mul_with_constant_input(temp_model_dir: Path, run_interpreter: Callable):
    """Test loading and running a model with MUL having a constant input."""
    inputs = keras.Input(shape=(4,))
    constant_val = np.array([0.5, 1.5, -2.0, 1.0], dtype=np.float32)
    x = inputs * constant_val
    model = keras.Model(inputs=inputs, outputs=x)

    model_path = temp_model_dir / "mul_constant.tflite"
    conftest.export_quantized_tflite_model((4,), model, float_io=True, output_path=model_path)

    rng = np.random.default_rng(42)
    x_test = rng.uniform(-1.0, 1.0, (5, 4)).astype(np.float32)
    litert_outputs = run_interpreter(model_path, x_test)

    keras_model = litert_tunner.load_model(str(model_path))
    keras_outputs = keras_model.predict(x_test)
    testing_utils.assert_cosine_similarity(keras_outputs, litert_outputs)
    np.testing.assert_allclose(litert_outputs, keras_outputs, atol=1e-3)


def test__add_with_constant_input(temp_model_dir: Path, run_interpreter: Callable):
    """Test loading and running a model with ADD having a constant input."""
    inputs = keras.Input(shape=(4,))
    constant_val = np.array([0.5, 1.5, -2.0, 1.0], dtype=np.float32)
    x = inputs + constant_val
    model = keras.Model(inputs=inputs, outputs=x)

    model_path = temp_model_dir / "add_constant.tflite"
    conftest.export_quantized_tflite_model((4,), model, float_io=True, output_path=model_path)

    rng = np.random.default_rng(42)
    x_test = rng.uniform(-1.0, 1.0, (5, 4)).astype(np.float32)
    litert_outputs = run_interpreter(model_path, x_test)

    keras_model = litert_tunner.load_model(str(model_path))
    keras_outputs = keras_model.predict(x_test)
    testing_utils.assert_cosine_similarity(keras_outputs, litert_outputs)
    np.testing.assert_allclose(litert_outputs, keras_outputs, atol=1e-3)


def test__mul_with_constant_input_float32(temp_model_dir: Path, run_interpreter: Callable):
    """Test loading and running a float32 model with MUL having a constant input."""
    inputs = keras.Input(shape=(4,))
    constant_val = np.array([0.5, 1.5, -2.0, 1.0], dtype=np.float32)
    x = inputs * constant_val
    model = keras.Model(inputs=inputs, outputs=x)

    model_path = temp_model_dir / "mul_constant_f32.tflite"
    conftest.export_float32_tflite_model((4,), model, output_path=model_path)

    rng = np.random.default_rng(42)
    x_test = rng.uniform(-1.0, 1.0, (5, 4)).astype(np.float32)
    litert_outputs = run_interpreter(model_path, x_test)

    keras_model = litert_tunner.load_model(str(model_path))
    keras_outputs = keras_model.predict(x_test)
    testing_utils.assert_cosine_similarity(keras_outputs, litert_outputs)
    np.testing.assert_allclose(litert_outputs, keras_outputs, atol=1e-5)


def test__add_with_constant_input_float32(temp_model_dir: Path, run_interpreter: Callable):
    """Test loading and running a float32 model with ADD having a constant input."""
    inputs = keras.Input(shape=(4,))
    constant_val = np.array([0.5, 1.5, -2.0, 1.0], dtype=np.float32)
    x = inputs + constant_val
    model = keras.Model(inputs=inputs, outputs=x)

    model_path = temp_model_dir / "add_constant_f32.tflite"
    conftest.export_float32_tflite_model((4,), model, output_path=model_path)

    rng = np.random.default_rng(42)
    x_test = rng.uniform(-1.0, 1.0, (5, 4)).astype(np.float32)
    litert_outputs = run_interpreter(model_path, x_test)

    keras_model = litert_tunner.load_model(str(model_path))
    keras_outputs = keras_model.predict(x_test)
    testing_utils.assert_cosine_similarity(keras_outputs, litert_outputs)
    np.testing.assert_allclose(litert_outputs, keras_outputs, atol=1e-5)
