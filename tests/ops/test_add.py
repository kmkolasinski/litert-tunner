"""Tests for ADD operator."""

import keras
import numpy as np
import pytest

from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests import conftest
from tests.ops import op_test_utils


@pytest.fixture
def add_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal ADD op with INT8 I/O."""
    input1_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[-5])
    input1_tensor = op_test_utils.make_tensor(
        name="input1_int8", index=0, shape=(1, 4), dtype=types.DTYPE_INT8, quantization=input1_quant
    )

    input2_quant = op_test_utils.make_quant_params(scales=[0.2], zero_points=[0])
    input2_tensor = op_test_utils.make_tensor(
        name="input2_int8", index=1, shape=(1, 4), dtype=types.DTYPE_INT8, quantization=input2_quant
    )

    output_quant = op_test_utils.make_quant_params(scales=[0.5], zero_points=[10])
    output_tensor = op_test_utils.make_tensor(
        name="output_int8", index=2, shape=(1, 4), dtype=types.DTYPE_INT8, quantization=output_quant
    )

    tensors = (input1_tensor, input2_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="ADD",
        input_indices=(0, 1),
        output_indices=(2,),
    )
    return op, tensors


class TestAddBuild:
    def test__add_is_registered(self):
        assert "ADD" in registry.registered_ops()

    def test__build_returns_keras_layer(self, add_setup):
        op, tensors = add_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__build_layer_name_contains_output_index(self, add_setup):
        op, tensors = add_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}")


class TestAddCall:
    def test__output_shape_matches_expected(self, add_setup):
        op, tensors = add_setup
        rng = np.random.default_rng(42)
        input_data = [
            rng.uniform(-1.0, 1.0, (1, 4)).astype(np.float32),
            rng.uniform(-1.0, 1.0, (1, 4)).astype(np.float32),
        ]
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (1, 4))

    def test__add_formula_matches_expected(self, add_setup):
        """Verify add computation produces correct simulated INT8 output."""
        op, tensors = add_setup
        input1_data = np.array([[-2, 1, 0, 3]], dtype=np.float32)
        input2_data = np.array([[10, -10, 0, 20]], dtype=np.float32)

        _, output = op_test_utils.build_and_call(op, tensors, [input1_data, input2_data])

        expected = np.array([[15.0, 7.0, 11.0, 20.0]], dtype=np.float32)
        np.testing.assert_allclose(output, expected, atol=1e-5)


class TestAddTrainableWeights:
    def test__trainable_weights(self, add_setup):
        op, tensors = add_setup
        inputs = [np.zeros((1, 4), dtype=np.float32), np.zeros((1, 4), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_trainable_weight_names(layer, set())

    def test__non_trainable_weights(self, add_setup):
        op, tensors = add_setup
        inputs = [np.zeros((1, 4), dtype=np.float32), np.zeros((1, 4), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_non_trainable_weight_names(
            layer,
            {
                "input1_scale",
                "input1_zero_point",
                "input2_scale",
                "input2_zero_point",
                "output_scale",
                "output_zero_point",
            },
        )


class TestAddWriteOps:
    def test__is_writable(self, add_setup):
        op, tensors = add_setup
        inputs = [np.zeros((1, 4), dtype=np.float32), np.zeros((1, 4), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_layer_is_writable(layer)

    def test__write_ops_counts(self, add_setup):
        op, tensors = add_setup
        inputs = [np.zeros((1, 4), dtype=np.float32), np.zeros((1, 4), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_collect_write_ops(
            layer,
            op,
            expected_buffer_writes=0,
            expected_quant_writes=3,
        )

    def test__write_ops_quant_indices(self, add_setup):
        op, tensors = add_setup
        inputs = [np.zeros((1, 4), dtype=np.float32), np.zeros((1, 4), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        _, quant_writes = layer.collect_write_ops(op)
        op_test_utils.assert_quant_write_tensor_indices(
            quant_writes, {op.input_indices[0], op.input_indices[1], op.output_indices[0]}
        )


@pytest.fixture
def float_add_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal ADD op with float32 I/O (no quantization)."""
    input1_tensor = op_test_utils.make_tensor(
        name="input1_f32", index=0, shape=(1, 4), dtype=types.DTYPE_FLOAT32, quantization=None
    )

    input2_tensor = op_test_utils.make_tensor(
        name="input2_f32", index=1, shape=(1, 4), dtype=types.DTYPE_FLOAT32, quantization=None
    )

    output_tensor = op_test_utils.make_tensor(
        name="output_f32", index=2, shape=(1, 4), dtype=types.DTYPE_FLOAT32, quantization=None
    )

    tensors = (input1_tensor, input2_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="ADD",
        input_indices=(0, 1),
        output_indices=(2,),
    )
    return op, tensors


class TestFloatAddBuild:
    def test__float_add_build_returns_keras_layer(self, float_add_setup):
        """Builder must return a Keras layer for float32 inputs."""
        op, tensors = float_add_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__float_add_layer_name_contains_output_index(self, float_add_setup):
        """Layer name must end with output tensor index."""
        op, tensors = float_add_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}")


class TestFloatAddCall:
    def test__float_add_output_shape(self, float_add_setup):
        """Output shape must match expected shape."""
        op, tensors = float_add_setup
        rng = np.random.default_rng(42)
        input_data = [
            rng.uniform(-1.0, 1.0, (1, 4)).astype(np.float32),
            rng.uniform(-1.0, 1.0, (1, 4)).astype(np.float32),
        ]
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (1, 4))

    def test__float_add_formula_matches_numpy(self, float_add_setup):
        """Float32 op output must match numpy reference computation."""
        op, tensors = float_add_setup
        input1_data = np.array([[-2, 1, 0, 3]], dtype=np.float32)
        input2_data = np.array([[10, -10, 0, 20]], dtype=np.float32)
        _, output = op_test_utils.build_and_call(op, tensors, [input1_data, input2_data])

        expected = input1_data + input2_data
        np.testing.assert_allclose(output, expected, atol=1e-5)


class TestFloatAddTrainableWeights:
    def test__float_add_trainable_weights(self, float_add_setup):
        op, tensors = float_add_setup
        inputs = [np.zeros((1, 4), dtype=np.float32), np.zeros((1, 4), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_trainable_weight_names(layer, set())

    def test__float_add_non_trainable_weights(self, float_add_setup):
        op, tensors = float_add_setup
        inputs = [np.zeros((1, 4), dtype=np.float32), np.zeros((1, 4), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_non_trainable_weight_names(layer, set())


class TestFloatAddWriteOps:
    def test__float_add_not_writable(self, float_add_setup):
        op, tensors = float_add_setup
        inputs = [np.zeros((1, 4), dtype=np.float32), np.zeros((1, 4), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_layer_not_writable(layer)


@pytest.mark.parametrize("dtype_policy", ["float32", "mixed_float16"])
@pytest.mark.parametrize("quantization", ["int8", "float32"])
def test__add_integration(temp_model_dir, run_interpreter, quantization: str, dtype_policy: str):
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(4,))
    x = keras.layers.Dense(4)(inputs)
    y = keras.layers.Dense(4)(inputs)
    outputs = keras.layers.Add()([x, y])
    model = keras.Model(inputs=inputs, outputs=outputs)
    input_shape = (1, 4)

    output_path = temp_model_dir / f"{quantization}_{dtype_policy}_add_integration.tflite"
    conftest.export_tflite_model(
        input_shape=input_shape[1:],
        model=model,
        quantization=quantization,
        float_io=True,
        output_path=output_path,
    )

    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1.0, 1.0, input_shape).astype(np.float32)

    original_policy = keras.config.dtype_policy()
    try:
        keras.config.set_dtype_policy(dtype_policy)
        atol = op_test_utils.get_default_atol(dtype_policy)
        op_test_utils.verify_model_outputs(output_path, x_train, run_interpreter, atol=atol)
    finally:
        keras.config.set_dtype_policy(original_policy)

    op_test_utils.verify_model_contains_operator(output_path, "ADD")
