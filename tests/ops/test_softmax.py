"""Tests for SOFTMAX operator."""

import keras
import numpy as np
import pytest

from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests import conftest
from tests.ops import op_test_utils


@pytest.fixture
def float_softmax_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal SOFTMAX op with float32 I/O (no quantization)."""
    input_tensor = op_test_utils.make_tensor(
        name="input_f32", index=0, shape=(1, 4), dtype=types.DTYPE_FLOAT32, quantization=None
    )

    output_tensor = op_test_utils.make_tensor(
        name="output_f32", index=1, shape=(1, 4), dtype=types.DTYPE_FLOAT32, quantization=None
    )

    tensors = (input_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="SOFTMAX",
        input_indices=(0,),
        output_indices=(1,),
    )
    return op, tensors


@pytest.fixture
def softmax_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal SOFTMAX op with INT8 I/O."""
    input_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[-5])
    input_tensor = op_test_utils.make_tensor(
        name="input_int8", index=0, shape=(1, 4), dtype=types.DTYPE_INT8, quantization=input_quant
    )

    # TFLite hardcodes SOFTMAX output: scale=1/256, zp=-128
    output_quant = op_test_utils.make_quant_params(scales=[1.0 / 256.0], zero_points=[-128])
    output_tensor = op_test_utils.make_tensor(
        name="output_int8", index=1, shape=(1, 4), dtype=types.DTYPE_INT8, quantization=output_quant
    )

    tensors = (input_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="SOFTMAX",
        input_indices=(0,),
        output_indices=(1,),
    )
    return op, tensors


class TestSoftmaxBuild:
    def test__softmax_is_registered(self):
        assert "SOFTMAX" in registry.registered_ops()

    def test__build_returns_keras_layer(self, softmax_setup):
        op, tensors = softmax_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__build_layer_name_contains_output_index(self, softmax_setup):
        op, tensors = softmax_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}")


class TestSoftmaxCall:
    def test__output_shape_matches_expected(self, softmax_setup):
        op, tensors = softmax_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (1, 4)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (1, 4))

    def test__softmax_output_sums_approximately_to_one(self, softmax_setup):
        """Verify softmax output values sum approximately to 1 after dequantization."""
        op, tensors = softmax_setup
        input_data = np.array([[-2, 1, 0, 3]], dtype=np.float32)
        _, output = op_test_utils.build_and_call(op, tensors, input_data)

        # Dequantize output: scale=1/256, zp=-128 → real = (1/256) * (x - (-128))
        output_scale = 1.0 / 256.0
        output_zp = -128.0
        dequantized = output_scale * (output - output_zp)

        # Softmax outputs should sum to ~1.0
        np.testing.assert_allclose(dequantized.sum(axis=-1), 1.0, atol=0.02)


class TestSoftmaxTrainableWeights:
    def test__trainable_weights(self, softmax_setup):
        """SOFTMAX has no trainable weights (output quant is frozen)."""
        op, tensors = softmax_setup
        input_data = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_trainable_weight_names(layer, set())

    def test__non_trainable_weights(self, softmax_setup):
        op, tensors = softmax_setup
        input_data = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_non_trainable_weight_names(
            layer,
            {
                "input_scale",
                "input_zero_point",
                "output_scale",
                "output_zero_point",
            },
        )


class TestSoftmaxWriteOps:
    def test__is_writable(self, softmax_setup):
        op, tensors = softmax_setup
        input_data = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_layer_is_writable(layer)

    def test__write_ops_counts(self, softmax_setup):
        op, tensors = softmax_setup
        input_data = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_collect_write_ops(
            layer,
            op,
            expected_buffer_writes=0,
            expected_quant_writes=2,
        )

    def test__write_ops_quant_indices(self, softmax_setup):
        op, tensors = softmax_setup
        input_data = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, input_data)
        _, quant_writes = layer.collect_write_ops(op)
        op_test_utils.assert_quant_write_tensor_indices(
            quant_writes, {op.input_indices[0], op.output_indices[0]}
        )


class TestFloatSoftmaxBuild:
    def test__float_softmax_build_returns_keras_layer(self, float_softmax_setup):
        op, tensors = float_softmax_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__float_softmax_layer_name_contains_output_index(self, float_softmax_setup):
        op, tensors = float_softmax_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}")


class TestFloatSoftmaxCall:
    def test__float_softmax_output_shape(self, float_softmax_setup):
        op, tensors = float_softmax_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (1, 4)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (1, 4))

    def test__float_softmax_formula_matches_numpy(self, float_softmax_setup):
        op, tensors = float_softmax_setup
        input_data = np.array([[-2, 1, 0, 3]], dtype=np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)

        # Calculate expected softmax
        exp_x = np.exp(input_data - np.max(input_data, axis=-1, keepdims=True))
        expected = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        np.testing.assert_allclose(output, expected, atol=1e-5)


class TestFloatSoftmaxTrainableWeights:
    def test__float_softmax_trainable_weights(self, float_softmax_setup):
        op, tensors = float_softmax_setup
        input_data = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_trainable_weight_names(layer, set())

    def test__float_softmax_non_trainable_weights(self, float_softmax_setup):
        op, tensors = float_softmax_setup
        input_data = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_non_trainable_weight_names(layer, set())


class TestFloatSoftmaxWriteOps:
    def test__float_softmax_not_writable(self, float_softmax_setup):
        op, tensors = float_softmax_setup
        input_data = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_layer_not_writable(layer)


@pytest.mark.parametrize("quantization", ["int8", "float32"])
def test__softmax_integration(temp_model_dir, run_interpreter, quantization: str):
    """Integration test: build a Keras model that produces a TFLite SOFTMAX op."""
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(4,))
    x = keras.layers.Dense(8)(inputs)
    outputs = keras.layers.Softmax()(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    input_shape = (1, 4)

    output_path = temp_model_dir / f"{quantization}_softmax_integration.tflite"
    conftest.export_tflite_model(
        input_shape=input_shape[1:],
        model=model,
        quantization=quantization,
        float_io=True,
        output_path=output_path,
    )

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, input_shape).astype(np.float32)

    op_test_utils.verify_model_outputs(output_path, x_train, run_interpreter)

    op_test_utils.verify_model_contains_operator(output_path, "SOFTMAX")
