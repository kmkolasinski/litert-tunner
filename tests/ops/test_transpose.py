"""Tests for TRANSPOSE operator."""

import keras
import numpy as np
import pytest

from litert_tunner.graph import types
from litert_tunner.ops import registry
from tests import conftest
from tests.ops import op_test_utils


@pytest.fixture
def transpose_setup() -> tuple[types.OperatorInfo, tuple[types.TensorInfo, ...]]:
    """Create a minimal TRANSPOSE op."""
    input_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[-5])
    input_tensor = op_test_utils.make_tensor(
        name="input_int8",
        index=0,
        shape=(1, 2, 3),
        dtype=types.DTYPE_INT8,
        quantization=input_quant,
    )

    perm_data = np.array([0, 2, 1], dtype=np.int32)
    perm_tensor = op_test_utils.make_tensor(
        name="perm_int32", index=1, shape=(3,), dtype=types.DTYPE_INT32, data=perm_data
    )

    output_quant = op_test_utils.make_quant_params(scales=[0.1], zero_points=[-5])
    output_tensor = op_test_utils.make_tensor(
        name="output_int8",
        index=2,
        shape=(1, 3, 2),
        dtype=types.DTYPE_INT8,
        quantization=output_quant,
    )

    tensors = (input_tensor, perm_tensor, output_tensor)
    op = op_test_utils.make_operator(
        op_type="TRANSPOSE",
        input_indices=(0, 1),
        output_indices=(2,),
    )
    return op, tensors


class TestTransposeBuild:
    def test__transpose_is_registered(self):
        assert "TRANSPOSE" in registry.registered_ops()

    def test__build_returns_keras_layer(self, transpose_setup):
        op, tensors = transpose_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        assert isinstance(layer, keras.Layer)

    def test__build_layer_name_contains_output_index(self, transpose_setup):
        op, tensors = transpose_setup
        layer = op_test_utils.build_layer_from_registry(op, tensors)
        output_idx = op.output_indices[0]
        assert layer.name.endswith(f"_{output_idx}")

    def test__build_raises_without_constant_perm(self, transpose_setup):
        op, tensors = transpose_setup
        tensors_list = list(tensors)
        tensors_list[1] = op_test_utils.make_tensor(
            name="perm_int32",
            index=1,
            shape=(3,),
            dtype=types.DTYPE_INT32,
            data=None,
        )
        with pytest.raises(ValueError, match="requires a constant permutation tensor"):
            op_test_utils.build_layer_from_registry(op, tuple(tensors_list))


class TestTransposeCall:
    def test__output_shape_matches_expected(self, transpose_setup):
        op, tensors = transpose_setup
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, (1, 2, 3)).astype(np.float32)
        _layer, output = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_output_shape(output, (1, 3, 2))

    def test__transpose_formula_matches_expected(self, transpose_setup):
        op, tensors = transpose_setup
        input_data = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32)
        _, output = op_test_utils.build_and_call(op, tensors, input_data)
        expected = np.transpose(input_data, axes=(0, 2, 1))
        np.testing.assert_allclose(output, expected, atol=1e-5)

    def test__not_writable(self, transpose_setup):
        op, tensors = transpose_setup
        input_data = np.zeros((1, 2, 3), dtype=np.float32)
        layer, _ = op_test_utils.build_and_call(op, tensors, input_data)
        op_test_utils.assert_layer_not_writable(layer)


@pytest.mark.parametrize("quantization", ["int8", "float32"])
def test__transpose_integration(temp_model_dir, run_interpreter, quantization: str):
    keras.utils.set_random_seed(42)

    inputs = keras.Input(shape=(2, 3))
    # Note: Keras Permute layer axes are 1-based because it ignores the batch dim
    outputs = keras.layers.Permute(dims=(2, 1))(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    input_shape = (1, 2, 3)

    output_path = temp_model_dir / f"{quantization}_transpose_integration.tflite"
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
    op_test_utils.verify_model_contains_operator(output_path, "TRANSPOSE")
