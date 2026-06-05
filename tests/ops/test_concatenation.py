"""Tests for CONCATENATION operator."""

import keras
import numpy as np
import pytest

from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests import conftest
from tests.ops import op_test_utils


@pytest.fixture
def concat_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal CONCATENATION op with INT8 I/O."""
    input1_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[-5])
    input1_tensor = op_test_utils.make_tensor(
        name="input1_int8", index=0, shape=(1, 4), dtype=types.DTYPE_INT8, quantization=input1_quant
    )

    input2_quant = op_test_utils.make_quant_params(scales=[0.2], zero_points=[0])
    input2_tensor = op_test_utils.make_tensor(
        name="input2_int8", index=1, shape=(1, 3), dtype=types.DTYPE_INT8, quantization=input2_quant
    )

    output_quant = op_test_utils.make_quant_params(scales=[0.5], zero_points=[10])
    output_tensor = op_test_utils.make_tensor(
        name="output_int8", index=2, shape=(1, 7), dtype=types.DTYPE_INT8, quantization=output_quant
    )

    tensors = (input1_tensor, input2_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="CONCATENATION",
        input_indices=(0, 1),
        output_indices=(2,),
        options={"Axis": 1},
    )
    return op, tensors


class TestConcatenationBuild:
    def test__concatenation_is_registered(self):
        assert "CONCATENATION" in registry.registered_ops()

    def test__build_returns_keras_layer(self, concat_setup):
        op, tensors = concat_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__build_layer_name_contains_output_index(self, concat_setup):
        op, tensors = concat_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}")

    def test__build_raises_without_quantization(self, concat_setup):
        op, tensors = concat_setup
        tensors_list = list(tensors)
        tensors_list[2] = op_test_utils.make_tensor(
            name="output_float", index=2, shape=(1, 7), dtype=types.DTYPE_FLOAT32, quantization=None
        )
        with pytest.raises(ValueError, match="requires quantized output tensor"):
            op_test_utils.build_layer_from_registry(op, tuple(tensors_list))


class TestConcatenationCall:
    def test__output_shape_matches_expected(self, concat_setup):
        op, tensors = concat_setup
        rng = np.random.default_rng(42)
        input_data = [
            rng.uniform(-1.0, 1.0, (1, 4)).astype(np.float32),
            rng.uniform(-1.0, 1.0, (1, 3)).astype(np.float32),
        ]
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (1, 7))

    def test__concatenation_formula_matches_expected(self, concat_setup):
        """Verify concatenation computation produces correct simulated INT8 output."""
        op, tensors = concat_setup
        # Simulated int8 inputs.
        # input1_scale = 0.1, input1_zp = -5
        # 0 in input1 means: 0.1 * (0 - (-5)) = 0.5
        # 1 in input1 means: 0.1 * (1 - (-5)) = 0.6
        input1_data = np.array([[0, 1, 2, 3]], dtype=np.float32)

        # input2_scale = 0.2, input2_zp = 0
        # 10 in input2 means: 0.2 * (10 - 0) = 2.0
        input2_data = np.array([[10, 20, 30]], dtype=np.float32)

        _, output = op_test_utils.build_and_call(op, tensors, [input1_data, input2_data])

        # Concatenated float: [0.5, 0.6, 0.7, 0.8, 2.0, 4.0, 6.0]
        # output_scale = 0.5, output_zp = 10
        # quantize: round value (val / 0.5) + 10
        # [0.5 / 0.5 + 10 = 11, ...]
        expected = np.array([[11.0, 11.0, 11.0, 12.0, 14.0, 18.0, 22.0]], dtype=np.float32)

        np.testing.assert_allclose(output, expected, atol=1e-5)


class TestConcatenationTrainableWeights:
    def test__trainable_weights(self, concat_setup):
        op, tensors = concat_setup
        inputs = [np.zeros((1, 4), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_trainable_weight_names(layer, set())

    def test__non_trainable_weights(self, concat_setup):
        op, tensors = concat_setup
        inputs = [np.zeros((1, 4), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_non_trainable_weight_names(
            layer,
            {
                "input0_scale",
                "input0_zero_point",
                "input1_scale",
                "input1_zero_point",
                "output_scale",
                "output_zero_point",
            },
        )


class TestConcatenationWriteOps:
    def test__is_writable(self, concat_setup):
        op, tensors = concat_setup
        inputs = [np.zeros((1, 4), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_layer_is_writable(layer)

    def test__write_ops_counts(self, concat_setup):
        op, tensors = concat_setup
        inputs = [np.zeros((1, 4), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_collect_write_ops(
            layer,
            op,
            expected_buffer_writes=0,
            expected_quant_writes=3,
        )

    def test__write_ops_quant_indices(self, concat_setup):
        op, tensors = concat_setup
        inputs = [np.zeros((1, 4), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        _, quant_writes = layer.collect_write_ops(op)
        op_test_utils.assert_quant_write_tensor_indices(
            quant_writes, {op.input_indices[0], op.input_indices[1], op.output_indices[0]}
        )


@pytest.fixture
def float_concatenation_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal CONCATENATION op with float32 I/O (no quantization)."""
    input1_tensor = op_test_utils.make_tensor(
        name="input1_f32", index=0, shape=(1, 4), dtype=types.DTYPE_FLOAT32, quantization=None
    )

    input2_tensor = op_test_utils.make_tensor(
        name="input2_f32", index=1, shape=(1, 3), dtype=types.DTYPE_FLOAT32, quantization=None
    )

    output_tensor = op_test_utils.make_tensor(
        name="output_f32", index=2, shape=(1, 7), dtype=types.DTYPE_FLOAT32, quantization=None
    )

    tensors = (input1_tensor, input2_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="CONCATENATION",
        input_indices=(0, 1),
        output_indices=(2,),
        options={"Axis": 1},
    )
    return op, tensors


class TestFloatConcatenationBuild:
    def test__float_concatenation_build_returns_keras_layer(self, float_concatenation_setup):
        """Builder must return a Keras layer for float32 inputs."""
        op, tensors = float_concatenation_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__float_concatenation_layer_name_contains_output_index(self, float_concatenation_setup):
        """Layer name must end with output tensor index."""
        op, tensors = float_concatenation_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}")


class TestFloatConcatenationCall:
    def test__float_concatenation_output_shape(self, float_concatenation_setup):
        """Output shape must match expected shape."""
        op, tensors = float_concatenation_setup
        rng = np.random.default_rng(42)
        input_data = [
            rng.uniform(-1.0, 1.0, (1, 4)).astype(np.float32),
            rng.uniform(-1.0, 1.0, (1, 3)).astype(np.float32),
        ]
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (1, 7))

    def test__float_concatenation_formula_matches_numpy(self, float_concatenation_setup):
        """Float32 op output must match numpy reference computation."""
        op, tensors = float_concatenation_setup
        rng = np.random.default_rng(42)
        input1_data = rng.uniform(-1.0, 1.0, (1, 4)).astype(np.float32)
        input2_data = rng.uniform(-1.0, 1.0, (1, 3)).astype(np.float32)

        _layer, output = op_test_utils.build_and_call(op, tensors, [input1_data, input2_data])

        expected = np.concatenate([input1_data, input2_data], axis=1)
        np.testing.assert_allclose(output, expected, atol=1e-5)


class TestFloatConcatenationTrainableWeights:
    def test__float_concatenation_trainable_weights(self, float_concatenation_setup):
        op, tensors = float_concatenation_setup
        dummy_input = [np.zeros((1, 4), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, dummy_input)
        op_test_utils.assert_trainable_weight_names(layer, set())

    def test__float_concatenation_non_trainable_weights(self, float_concatenation_setup):
        op, tensors = float_concatenation_setup
        dummy_input = [np.zeros((1, 4), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, dummy_input)
        op_test_utils.assert_non_trainable_weight_names(layer, set())

    def test__float_concatenation_not_writable(self, float_concatenation_setup):
        op, tensors = float_concatenation_setup
        dummy_input = [np.zeros((1, 4), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, dummy_input)
        op_test_utils.assert_layer_not_writable(layer)


@pytest.mark.parametrize("quantization", ["int8", "float32"])
def test__concatenation_integration(temp_model_dir, run_interpreter, quantization: str):
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(4,))
    x = keras.layers.Dense(4)(inputs)
    y = keras.layers.Dense(3)(inputs)
    outputs = keras.layers.Concatenate(axis=-1)([x, y])
    model = keras.Model(inputs=inputs, outputs=outputs)
    input_shape = (1, 4)

    output_path = temp_model_dir / f"{quantization}_concatenation_integration.tflite"
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

    op_test_utils.verify_model_contains_operator(output_path, "CONCATENATION")
