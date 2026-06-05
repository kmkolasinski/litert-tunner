"""Tests for DIV operator."""

import keras
import numpy as np
import pytest

from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests import conftest
from tests.ops import op_test_utils


@pytest.fixture
def div_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal DIV op with INT8 I/O."""
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
        op_type="DIV",
        input_indices=(0, 1),
        output_indices=(2,),
    )
    return op, tensors


class TestDivBuild:
    def test__div_is_registered(self):
        assert "DIV" in registry.registered_ops()

    def test__build_returns_keras_layer(self, div_setup):
        op, tensors = div_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__build_layer_name_contains_output_index(self, div_setup):
        op, tensors = div_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}")


class TestDivCall:
    def test__output_shape_matches_expected(self, div_setup):
        op, tensors = div_setup
        rng = np.random.default_rng(42)
        input_data = [
            rng.uniform(-1.0, 1.0, (1, 4)).astype(np.float32),
            rng.uniform(0.5, 2.0, (1, 4)).astype(np.float32),
        ]
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (1, 4))

    def test__div_formula_matches_expected(self, div_setup):
        """Verify div computation produces correct simulated INT8 output."""
        op, tensors = div_setup
        # Use non-zero divisors to avoid division by zero
        input1_data = np.array([[-2, 1, 4, 3]], dtype=np.float32)
        input2_data = np.array([[10, -10, 20, 5]], dtype=np.float32)

        _, output = op_test_utils.build_and_call(op, tensors, [input1_data, input2_data])

        # Manual computation:
        # Dequant input1: scale=0.1, zp=-5 → real = 0.1 * (x - (-5)) = 0.1 * (x + 5)
        #   [-2+5, 1+5, 4+5, 3+5] * 0.1 = [0.3, 0.6, 0.9, 0.8]
        # Dequant input2: scale=0.2, zp=0 → real = 0.2 * (x - 0) = 0.2 * x
        #   [10, -10, 20, 5] * 0.2 = [2.0, -2.0, 4.0, 1.0]
        # Divide: [0.3/2.0, 0.6/(-2.0), 0.9/4.0, 0.8/1.0]  # noqa: ERA001
        #       = [0.15, -0.3, 0.225, 0.8]
        # Quantize output: scale=0.5, zp=10 → int8 = round(real/0.5) + 10
        #   [round(0.3)+10, round(-0.6)+10, round(0.45)+10, round(1.6)+10]  # noqa: ERA001
        #   = [10, 9, 10, 12]
        expected = np.array([[10.0, 9.0, 10.0, 12.0]], dtype=np.float32)
        np.testing.assert_allclose(output, expected, atol=1e-5)


class TestDivTrainableWeights:
    def test__trainable_weights(self, div_setup):
        op, tensors = div_setup
        inputs = [np.ones((1, 4), dtype=np.float32), np.ones((1, 4), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_trainable_weight_names(layer, set())

    def test__non_trainable_weights(self, div_setup):
        op, tensors = div_setup
        inputs = [np.ones((1, 4), dtype=np.float32), np.ones((1, 4), dtype=np.float32)]
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


class TestDivWriteOps:
    def test__is_writable(self, div_setup):
        op, tensors = div_setup
        inputs = [np.ones((1, 4), dtype=np.float32), np.ones((1, 4), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_layer_is_writable(layer)

    def test__write_ops_counts(self, div_setup):
        op, tensors = div_setup
        inputs = [np.ones((1, 4), dtype=np.float32), np.ones((1, 4), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_collect_write_ops(
            layer,
            op,
            expected_buffer_writes=0,
            expected_quant_writes=3,
        )

    def test__write_ops_quant_indices(self, div_setup):
        op, tensors = div_setup
        inputs = [np.ones((1, 4), dtype=np.float32), np.ones((1, 4), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        _, quant_writes = layer.collect_write_ops(op)
        op_test_utils.assert_quant_write_tensor_indices(
            quant_writes, {op.input_indices[0], op.input_indices[1], op.output_indices[0]}
        )


@pytest.fixture
def float_div_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal DIV op with float32 I/O (no quantization)."""
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
        op_type="DIV",
        input_indices=(0, 1),
        output_indices=(2,),
    )
    return op, tensors


class TestFloatDivBuild:
    def test__float_div_build_returns_keras_layer(self, float_div_setup):
        """Builder must return a Keras layer for float32 inputs."""
        op, tensors = float_div_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__float_div_layer_name_contains_output_index(self, float_div_setup):
        """Layer name must end with output tensor index."""
        op, tensors = float_div_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}")


class TestFloatDivCall:
    def test__float_div_output_shape(self, float_div_setup):
        """Output shape must match expected shape."""
        op, tensors = float_div_setup
        rng = np.random.default_rng(42)
        input_data = [
            rng.uniform(-1.0, 1.0, (1, 4)).astype(np.float32),
            rng.uniform(0.5, 2.0, (1, 4)).astype(np.float32),
        ]
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (1, 4))

    def test__float_div_formula_matches_numpy(self, float_div_setup):
        """Float32 op output must match numpy reference computation."""
        op, tensors = float_div_setup
        input1_data = np.array([[-2, 1, 4, 3]], dtype=np.float32)
        input2_data = np.array([[10, -10, 20, 5]], dtype=np.float32)
        _, output = op_test_utils.build_and_call(op, tensors, [input1_data, input2_data])

        expected = input1_data / input2_data
        np.testing.assert_allclose(output, expected, atol=1e-5)


class TestFloatDivTrainableWeights:
    def test__float_div_trainable_weights(self, float_div_setup):
        op, tensors = float_div_setup
        inputs = [np.ones((1, 4), dtype=np.float32), np.ones((1, 4), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_trainable_weight_names(layer, set())

    def test__float_div_non_trainable_weights(self, float_div_setup):
        op, tensors = float_div_setup
        inputs = [np.ones((1, 4), dtype=np.float32), np.ones((1, 4), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_non_trainable_weight_names(layer, set())


class TestFloatDivWriteOps:
    def test__float_div_not_writable(self, float_div_setup):
        op, tensors = float_div_setup
        inputs = [np.ones((1, 4), dtype=np.float32), np.ones((1, 4), dtype=np.float32)]
        layer, _ = op_test_utils.build_and_call(op, tensors, inputs)
        op_test_utils.assert_layer_not_writable(layer)


@pytest.mark.parametrize("quantization", ["int8", "float32"])
def test__div_integration(temp_model_dir, run_interpreter, quantization: str):
    """Integration test: build a Keras model that produces a TFLite DIV op."""
    keras.utils.set_random_seed(42)

    # Two-branch single-input model: numerator / denominator.
    # sigmoid on the denominator guarantees positive values (maps to LOGISTIC).
    inputs = keras.Input(shape=(4,))
    x = keras.layers.Dense(4)(inputs)
    y = keras.layers.Dense(4, activation="sigmoid")(inputs)
    outputs = keras.layers.Lambda(lambda args: args[0] / args[1])([x, y])
    model = keras.Model(inputs=inputs, outputs=outputs)
    input_shape = (1, 4)

    output_path = temp_model_dir / f"{quantization}_div_integration.tflite"
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

    op_test_utils.verify_model_contains_operator(output_path, "DIV")
