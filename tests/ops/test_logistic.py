"""Tests for LOGISTIC operator."""

import keras
import numpy as np
import pytest

from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests import conftest
from tests.ops import op_test_utils


@pytest.fixture
def logistic_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal LOGISTIC op with INT8 I/O."""
    input_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[-5])
    input_tensor = op_test_utils.make_tensor(
        name="input_int8", index=0, shape=(1, 4), dtype=types.DTYPE_INT8, quantization=input_quant
    )

    output_quant = op_test_utils.make_quant_params(scales=[1 / 256.0], zero_points=[-128])
    output_tensor = op_test_utils.make_tensor(
        name="output_int8", index=1, shape=(1, 4), dtype=types.DTYPE_INT8, quantization=output_quant
    )

    tensors = (input_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="LOGISTIC",
        input_indices=(0,),
        output_indices=(1,),
    )
    return op, tensors


class TestLogisticBuild:
    def test__logistic_is_registered(self):
        assert "LOGISTIC" in registry.registered_ops()

    def test__build_returns_keras_layer(self, logistic_setup):
        op, tensors = logistic_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__build_layer_name_contains_output_index(self, logistic_setup):
        op, tensors = logistic_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}")


class TestLogisticCall:
    def test__output_shape_matches_expected(self, logistic_setup):
        op, tensors = logistic_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (1, 4)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (1, 4))

    def test__logistic_formula_matches_expected(self, logistic_setup):
        """Verify logistic computation produces correct simulated INT8 output."""
        op, tensors = logistic_setup
        input_data = np.array([[-5, 5, -15, 15]], dtype=np.float32)

        _, output = op_test_utils.build_and_call(op, tensors, input_data)

        deq = np.array([[0.0, 1.0, -1.0, 2.0]], dtype=np.float32)
        sigm = 1.0 / (1.0 + np.exp(-deq))
        expected = np.round(sigm * 256.0) - 128.0

        np.testing.assert_allclose(output, expected, atol=1e-5)


class TestLogisticTrainableWeights:
    def test__trainable_weights(self, logistic_setup):
        op, tensors = logistic_setup
        inputs = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_trainable_weight_names(layer, set())

    def test__non_trainable_weights(self, logistic_setup):
        op, tensors = logistic_setup
        inputs = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_non_trainable_weight_names(
            layer,
            {
                "input_scale",
                "input_zero_point",
                "output_scale",
                "output_zero_point",
            },
        )


class TestLogisticWriteOps:
    def test__is_writable(self, logistic_setup):
        op, tensors = logistic_setup
        inputs = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_layer_is_writable(layer)

    def test__write_ops_counts(self, logistic_setup):
        op, tensors = logistic_setup
        inputs = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_collect_write_ops(
            layer,
            op,
            expected_buffer_writes=0,
            expected_quant_writes=2,
        )

    def test__write_ops_quant_indices(self, logistic_setup):
        op, tensors = logistic_setup
        inputs = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        _, quant_writes = layer.collect_write_ops(op)
        op_test_utils.assert_quant_write_tensor_indices(
            quant_writes, {op.input_indices[0], op.output_indices[0]}
        )


@pytest.fixture
def float_logistic_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal LOGISTIC op with float32 I/O (no quantization)."""
    input_tensor = op_test_utils.make_tensor(
        name="input_f32", index=0, shape=(1, 4), dtype=types.DTYPE_FLOAT32, quantization=None
    )

    output_tensor = op_test_utils.make_tensor(
        name="output_f32", index=1, shape=(1, 4), dtype=types.DTYPE_FLOAT32, quantization=None
    )

    tensors = (input_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="LOGISTIC",
        input_indices=(0,),
        output_indices=(1,),
    )
    return op, tensors


class TestFloatLogisticBuild:
    def test__float_logistic_build_returns_keras_layer(self, float_logistic_setup):
        """Builder must return a Keras layer for float32 inputs."""
        op, tensors = float_logistic_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__float_logistic_layer_name_contains_output_index(self, float_logistic_setup):
        """Layer name must end with output tensor index."""
        op, tensors = float_logistic_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}")


class TestFloatLogisticCall:
    def test__float_logistic_output_shape(self, float_logistic_setup):
        """Output shape must match expected shape."""
        op, tensors = float_logistic_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (1, 4)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (1, 4))

    def test__float_logistic_formula_matches_numpy(self, float_logistic_setup):
        """Float32 op output must match numpy reference computation."""
        op, tensors = float_logistic_setup
        input_data = np.array([[-5, 5, -15, 15]], dtype=np.float32)
        _, output = op_test_utils.build_and_call(op, tensors, input_data)

        expected = 1.0 / (1.0 + np.exp(-input_data))
        np.testing.assert_allclose(output, expected, atol=1e-5)


class TestFloatLogisticTrainableWeights:
    def test__float_logistic_trainable_weights(self, float_logistic_setup):
        op, tensors = float_logistic_setup
        inputs = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_trainable_weight_names(layer, set())

    def test__float_logistic_non_trainable_weights(self, float_logistic_setup):
        op, tensors = float_logistic_setup
        inputs = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_non_trainable_weight_names(layer, set())


class TestFloatLogisticWriteOps:
    def test__float_logistic_not_writable(self, float_logistic_setup):
        op, tensors = float_logistic_setup
        inputs = np.zeros((1, 4), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_layer_not_writable(layer)


@pytest.mark.parametrize("quantization", ["int8", "float32"])
def test__logistic_integration(temp_model_dir, run_interpreter, quantization: str):
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(4,))
    outputs = keras.layers.Activation("sigmoid")(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    input_shape = (1, 4)

    output_path = temp_model_dir / f"{quantization}_logistic_integration.tflite"
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

    op_test_utils.verify_model_contains_operator(output_path, "LOGISTIC")
