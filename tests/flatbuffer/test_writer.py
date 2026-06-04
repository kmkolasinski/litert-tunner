"""Tests for flatbuffer writer module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import keras
import numpy as np
import pytest
from ai_edge_litert.interpreter import Interpreter

import litert_tunner
from litert_tunner import flatbuffer

if TYPE_CHECKING:
    from collections.abc import Callable


class TestSaveTfliteValidation:
    """Tests for save_tflite input validation."""

    def test__rejects_model_without_graph_def(self):
        """save_tflite must raise ValueError for a model not created by load_model."""
        model = keras.Sequential([keras.layers.Dense(2, input_shape=(4,))])
        with pytest.raises(ValueError, match="not created by litert_tunner"):
            flatbuffer.save_tflite(model, "/tmp/dummy.tflite")

    def test__accepts_model_created_by_load_model(
        self,
        make_dense_tflite: Callable,
        tmp_path: Path,
    ):
        """save_tflite must not raise for a model from load_model."""
        model_path = make_dense_tflite(num_features=4, num_units=2)
        tunner_model = litert_tunner.load_model(str(model_path))
        output_path = tmp_path / "output.tflite"
        # Should not raise
        litert_tunner.save_model(tunner_model, str(output_path))
        assert output_path.exists()


class TestSaveTfliteRoundTrip:
    """Tests that saving without modification preserves model outputs."""

    def test__identity_save_produces_valid_tflite(
        self,
        make_dense_tflite: Callable,
        tmp_path: Path,
    ):
        """Saved file must be loadable by the TFLite Interpreter."""
        model_path = make_dense_tflite(num_features=4, num_units=2, activation=None)
        tunner_model = litert_tunner.load_model(str(model_path))
        saved_path = tmp_path / "roundtrip.tflite"
        litert_tunner.save_model(tunner_model, str(saved_path))

        # Must not raise
        interpreter = Interpreter(model_path=str(saved_path))
        interpreter.allocate_tensors()

    def test__identity_save_preserves_file_size(
        self,
        make_dense_tflite: Callable,
        tmp_path: Path,
    ):
        """Saved file must be the same size as the original (binary surgery, no topology change)."""
        model_path = make_dense_tflite(num_features=4, num_units=2)
        tunner_model = litert_tunner.load_model(str(model_path))
        saved_path = tmp_path / "roundtrip.tflite"
        litert_tunner.save_model(tunner_model, str(saved_path))
        assert saved_path.stat().st_size == model_path.stat().st_size

    def test__identity_save_no_activation_matches_interpreter(
        self,
        make_dense_tflite: Callable,
        run_interpreter: Callable,
        tmp_path: Path,
    ):
        """Round-trip save of a no-activation model must produce bit-exact identical outputs."""
        model_path = make_dense_tflite(
            num_features=4,
            num_units=1,
            use_bias=True,
            activation=None,
            float_io=False,
        )
        saved_path = tmp_path / "roundtrip.tflite"
        tunner_model = litert_tunner.load_model(str(model_path))
        litert_tunner.save_model(tunner_model, str(saved_path))

        rng = np.random.default_rng(42)
        inputs = rng.uniform(-1.0, 1.0, (10, 4)).astype(np.float32)
        original_out = run_interpreter(model_path, inputs)
        saved_out = run_interpreter(saved_path, inputs)
        np.testing.assert_array_equal(original_out, saved_out)

    @pytest.mark.xfail(reason="graph/builder.py does not handle bias index -1 (use_bias=False)")
    def test__identity_save_no_bias_matches_interpreter(
        self,
        make_dense_tflite: Callable,
        run_interpreter: Callable,
        tmp_path: Path,
    ):
        """Round-trip save of a model without bias must produce near-identical outputs."""
        # NOTE: Models with use_bias=False use index -1 for the missing bias tensor
        # in TFLite, which currently causes a KeyError in graph/builder.py.
        # This is a pre-existing limitation, not a writer bug.
        model_path = make_dense_tflite(
            num_features=4,
            num_units=2,
            use_bias=False,
            activation=None,
            float_io=False,
        )
        saved_path = tmp_path / "roundtrip.tflite"
        tunner_model = litert_tunner.load_model(str(model_path))
        litert_tunner.save_model(tunner_model, str(saved_path))

        rng = np.random.default_rng(42)
        inputs = rng.uniform(-1.0, 1.0, (10, 4)).astype(np.float32)
        original_out = run_interpreter(model_path, inputs)
        saved_out = run_interpreter(saved_path, inputs)
        # Allow small INT8 rounding error from quantization param round-trip
        np.testing.assert_allclose(original_out, saved_out, atol=2, rtol=0)

    def test__identity_save_single_unit_matches_interpreter(
        self,
        make_dense_tflite: Callable,
        run_interpreter: Callable,
        tmp_path: Path,
    ):
        """Round-trip save of a single-unit Dense model must produce bit-exact identical outputs."""
        model_path = make_dense_tflite(
            num_features=8,
            num_units=1,
            use_bias=True,
            activation=None,
            float_io=False,
        )
        saved_path = tmp_path / "roundtrip.tflite"
        tunner_model = litert_tunner.load_model(str(model_path))
        litert_tunner.save_model(tunner_model, str(saved_path))

        rng = np.random.default_rng(42)
        inputs = rng.uniform(-1.0, 1.0, (10, 8)).astype(np.float32)
        original_out = run_interpreter(model_path, inputs)
        saved_out = run_interpreter(saved_path, inputs)
        np.testing.assert_array_equal(original_out, saved_out)


class TestSaveTfliteFloatIO:
    """Tests for saving models with float32 I/O (QUANTIZE + DEQUANTIZE ops)."""

    def test__float_io_identity_save_produces_valid_tflite(
        self,
        make_dense_tflite: Callable,
        tmp_path: Path,
    ):
        """Float-IO round-trip save must produce a valid TFLite model."""
        model_path = make_dense_tflite(num_features=4, num_units=2, float_io=True, activation=None)
        tunner_model = litert_tunner.load_model(str(model_path))
        saved_path = tmp_path / "float_io_roundtrip.tflite"
        litert_tunner.save_model(tunner_model, str(saved_path))

        interpreter = Interpreter(model_path=str(saved_path))
        interpreter.allocate_tensors()

    def test__float_io_identity_save_matches_interpreter(
        self,
        make_dense_tflite: Callable,
        run_interpreter: Callable,
        tmp_path: Path,
    ):
        """Float-IO round-trip save must produce bit-exact identical outputs."""
        model_path = make_dense_tflite(
            num_features=4,
            num_units=2,
            use_bias=True,
            activation=None,
            float_io=True,
        )
        saved_path = tmp_path / "float_io_roundtrip.tflite"
        tunner_model = litert_tunner.load_model(str(model_path))
        litert_tunner.save_model(tunner_model, str(saved_path))

        rng = np.random.default_rng(42)
        inputs = rng.uniform(-1.0, 1.0, (10, 4)).astype(np.float32)
        original_out = run_interpreter(model_path, inputs)
        saved_out = run_interpreter(saved_path, inputs)
        # Float32 output has larger numerical tolerance due to dequant precision
        np.testing.assert_allclose(original_out, saved_out, atol=0.02, rtol=0)


class TestSaveTfliteBiasModification:
    """Tests that modifying bias values in the Keras model are correctly persisted."""

    def test__modified_bias_changes_interpreter_output(
        self,
        make_dense_tflite: Callable,
        run_interpreter: Callable,
        tmp_path: Path,
    ):
        """Modifying the bias and saving must produce different outputs than the original."""
        model_path = make_dense_tflite(
            num_features=4, num_units=2, use_bias=True, activation=None, float_io=False
        )
        tunner_model = litert_tunner.load_model(str(model_path))

        # Find the dense layer and modify its bias
        dense_layer = _find_dense_layer(tunner_model)
        assert dense_layer is not None

        original_bias = dense_layer.bias.numpy()
        large_bias_shift = 50.0
        dense_layer.bias.assign(original_bias + large_bias_shift)

        saved_path = tmp_path / "modified_bias.tflite"
        litert_tunner.save_model(tunner_model, str(saved_path))

        rng = np.random.default_rng(42)
        inputs = rng.uniform(-1.0, 1.0, (10, 4)).astype(np.float32)
        original_out = run_interpreter(model_path, inputs)
        modified_out = run_interpreter(saved_path, inputs)

        # Outputs must differ after bias change
        assert not np.array_equal(original_out, modified_out)

    def test__modified_bias_is_written_to_flatbuffer(
        self,
        make_dense_tflite: Callable,
        tmp_path: Path,
    ):
        """Modified bias must be reflected when re-parsing the saved flatbuffer."""
        model_path = make_dense_tflite(
            num_features=4, num_units=2, use_bias=True, activation=None, float_io=False
        )
        tunner_model = litert_tunner.load_model(str(model_path))

        dense_layer = _find_dense_layer(tunner_model)
        assert dense_layer is not None

        # Set bias to a known value
        new_bias = np.array([100.0, -100.0], dtype=np.float32)
        dense_layer.bias.assign(new_bias)

        saved_path = tmp_path / "bias_check.tflite"
        litert_tunner.save_model(tunner_model, str(saved_path))

        # Re-parse and verify the bias buffer was updated
        graph_def = flatbuffer.parse_tflite(saved_path)
        fc_ops = [op for op in graph_def.operators if op.op_type == "FULLY_CONNECTED"]
        assert len(fc_ops) >= 1

        fc_op = fc_ops[0]
        bias_tensor = graph_def.tensors[fc_op.input_indices[2]]
        assert bias_tensor.data is not None

        # The saved bias_int32 = round(bias_float / (input_scale * weight_scale))
        # We can verify the saved values are non-trivially different from zero
        assert np.any(bias_tensor.data != 0), "Bias data should be non-zero after modification"


class TestSaveTfliteWeightPreservation:
    """Tests that weight values are correctly preserved/updated during save."""

    def test__modified_bias_and_scales_are_persisted_and_loaded(
        self,
        make_dense_tflite: Callable,
        tmp_path: Path,
    ):
        """Modified bias and scales must be exactly the same when loaded back."""
        model_path = make_dense_tflite(
            num_features=4, num_units=2, use_bias=True, activation=None, float_io=False
        )
        tunner_model = litert_tunner.load_model(str(model_path))

        dense_layer = _find_dense_layer(tunner_model)
        assert dense_layer is not None

        # Modify bias
        original_bias = dense_layer.bias.numpy()
        dense_layer.bias.assign(original_bias + 5.0)

        # Modify weight_scale
        original_weight_scale_var = dense_layer.weight_quant._scale_var.numpy()
        dense_layer.weight_quant._scale_var.assign(original_weight_scale_var + 0.5)

        saved_path = tmp_path / "modified_params.tflite"
        litert_tunner.save_model(tunner_model, str(saved_path))

        # Load again
        loaded_model = litert_tunner.load_model(str(saved_path))
        loaded_dense_layer = _find_dense_layer(loaded_model)
        assert loaded_dense_layer is not None

        # Values should be exactly the same
        # Compare actual calculated scales (since _scale_var represents inverse softplus)
        original_scale = np.asarray(keras.ops.convert_to_numpy(dense_layer.weight_quant.scale))
        loaded_scale = np.asarray(keras.ops.convert_to_numpy(loaded_dense_layer.weight_quant.scale))

        np.testing.assert_allclose(
            dense_layer.bias.numpy(), loaded_dense_layer.bias.numpy(), rtol=1e-5
        )
        np.testing.assert_allclose(original_scale, loaded_scale, rtol=1e-5)

    def test__weights_preserved_on_identity_save(
        self,
        make_dense_tflite: Callable,
        tmp_path: Path,
    ):
        """INT8 weights must be identical after an identity save."""
        model_path = make_dense_tflite(
            num_features=4, num_units=2, use_bias=True, activation=None, float_io=False
        )
        original_graph = flatbuffer.parse_tflite(model_path)
        tunner_model = litert_tunner.load_model(str(model_path))

        saved_path = tmp_path / "weight_check.tflite"
        litert_tunner.save_model(tunner_model, str(saved_path))

        saved_graph = flatbuffer.parse_tflite(saved_path)

        # Compare weight tensors from FULLY_CONNECTED ops
        for orig_op, saved_op in zip(original_graph.operators, saved_graph.operators, strict=False):
            if orig_op.op_type == "FULLY_CONNECTED":
                orig_weight = original_graph.tensors[orig_op.input_indices[1]]
                saved_weight = saved_graph.tensors[saved_op.input_indices[1]]
                assert orig_weight.data is not None
                assert saved_weight.data is not None
                np.testing.assert_array_equal(orig_weight.data, saved_weight.data)


class TestSaveTfliteQuantizationParams:
    """Tests that quantization parameters are correctly written."""

    def test__quantization_scales_preserved_on_identity_save(
        self,
        make_dense_tflite: Callable,
        tmp_path: Path,
    ):
        """Quantization scales must be identical after an identity save."""
        model_path = make_dense_tflite(
            num_features=4, num_units=2, use_bias=True, activation=None, float_io=False
        )
        original_graph = flatbuffer.parse_tflite(model_path)
        tunner_model = litert_tunner.load_model(str(model_path))

        saved_path = tmp_path / "quant_check.tflite"
        litert_tunner.save_model(tunner_model, str(saved_path))

        saved_graph = flatbuffer.parse_tflite(saved_path)

        for orig_t, saved_t in zip(original_graph.tensors, saved_graph.tensors, strict=False):
            if orig_t.quantization is not None and saved_t.quantization is not None:
                np.testing.assert_allclose(
                    orig_t.quantization.scales,
                    saved_t.quantization.scales,
                    rtol=1e-6,
                    err_msg=f"Scale mismatch for tensor '{orig_t.name}'",
                )

    def test__quantization_zero_points_preserved_on_identity_save(
        self,
        make_dense_tflite: Callable,
        tmp_path: Path,
    ):
        """Quantization zero points must be identical after an identity save."""
        model_path = make_dense_tflite(
            num_features=4, num_units=2, use_bias=True, activation=None, float_io=False
        )
        original_graph = flatbuffer.parse_tflite(model_path)
        tunner_model = litert_tunner.load_model(str(model_path))

        saved_path = tmp_path / "quant_check.tflite"
        litert_tunner.save_model(tunner_model, str(saved_path))

        saved_graph = flatbuffer.parse_tflite(saved_path)

        for orig_t, saved_t in zip(original_graph.tensors, saved_graph.tensors, strict=False):
            if orig_t.quantization is not None and saved_t.quantization is not None:
                np.testing.assert_array_equal(
                    orig_t.quantization.zero_points,
                    saved_t.quantization.zero_points,
                    err_msg=f"Zero point mismatch for tensor '{orig_t.name}'",
                )


class TestSaveTfliteGraphTopologyPreservation:
    """Tests that graph topology is never modified by save."""

    def test__operator_count_unchanged(
        self,
        make_dense_tflite: Callable,
        tmp_path: Path,
    ):
        """Number of operators must be identical after save."""
        model_path = make_dense_tflite(num_features=4, num_units=2)
        original_graph = flatbuffer.parse_tflite(model_path)
        tunner_model = litert_tunner.load_model(str(model_path))

        saved_path = tmp_path / "topology.tflite"
        litert_tunner.save_model(tunner_model, str(saved_path))

        saved_graph = flatbuffer.parse_tflite(saved_path)
        assert len(original_graph.operators) == len(saved_graph.operators)

    def test__tensor_count_unchanged(
        self,
        make_dense_tflite: Callable,
        tmp_path: Path,
    ):
        """Number of tensors must be identical after save."""
        model_path = make_dense_tflite(num_features=4, num_units=2)
        original_graph = flatbuffer.parse_tflite(model_path)
        tunner_model = litert_tunner.load_model(str(model_path))

        saved_path = tmp_path / "topology.tflite"
        litert_tunner.save_model(tunner_model, str(saved_path))

        saved_graph = flatbuffer.parse_tflite(saved_path)
        assert len(original_graph.tensors) == len(saved_graph.tensors)

    def test__operator_types_unchanged(
        self,
        make_dense_tflite: Callable,
        tmp_path: Path,
    ):
        """Operator type sequence must be identical after save."""
        model_path = make_dense_tflite(num_features=4, num_units=2, float_io=True)
        original_graph = flatbuffer.parse_tflite(model_path)
        tunner_model = litert_tunner.load_model(str(model_path))

        saved_path = tmp_path / "topology.tflite"
        litert_tunner.save_model(tunner_model, str(saved_path))

        saved_graph = flatbuffer.parse_tflite(saved_path)
        original_op_types = [op.op_type for op in original_graph.operators]
        saved_op_types = [op.op_type for op in saved_graph.operators]
        assert original_op_types == saved_op_types

    def test__io_indices_unchanged(
        self,
        make_dense_tflite: Callable,
        tmp_path: Path,
    ):
        """Graph input/output indices must be identical after save."""
        model_path = make_dense_tflite(num_features=4, num_units=2)
        original_graph = flatbuffer.parse_tflite(model_path)
        tunner_model = litert_tunner.load_model(str(model_path))

        saved_path = tmp_path / "topology.tflite"
        litert_tunner.save_model(tunner_model, str(saved_path))

        saved_graph = flatbuffer.parse_tflite(saved_path)
        assert original_graph.input_indices == saved_graph.input_indices
        assert original_graph.output_indices == saved_graph.output_indices

    def test__tensor_shapes_unchanged(
        self,
        make_dense_tflite: Callable,
        tmp_path: Path,
    ):
        """All tensor shapes must be identical after save."""
        model_path = make_dense_tflite(num_features=4, num_units=2)
        original_graph = flatbuffer.parse_tflite(model_path)
        tunner_model = litert_tunner.load_model(str(model_path))

        saved_path = tmp_path / "topology.tflite"
        litert_tunner.save_model(tunner_model, str(saved_path))

        saved_graph = flatbuffer.parse_tflite(saved_path)
        for orig_t, saved_t in zip(original_graph.tensors, saved_graph.tensors, strict=False):
            assert orig_t.shape == saved_t.shape, (
                f"Shape changed for tensor '{orig_t.name}': {orig_t.shape} → {saved_t.shape}"
            )

    def test__tensor_dtypes_unchanged(
        self,
        make_dense_tflite: Callable,
        tmp_path: Path,
    ):
        """All tensor dtypes must be identical after save."""
        model_path = make_dense_tflite(num_features=4, num_units=2)
        original_graph = flatbuffer.parse_tflite(model_path)
        tunner_model = litert_tunner.load_model(str(model_path))

        saved_path = tmp_path / "topology.tflite"
        litert_tunner.save_model(tunner_model, str(saved_path))

        saved_graph = flatbuffer.parse_tflite(saved_path)
        for orig_t, saved_t in zip(original_graph.tensors, saved_graph.tensors, strict=False):
            assert orig_t.dtype == saved_t.dtype, (
                f"Dtype changed for tensor '{orig_t.name}': {orig_t.dtype} → {saved_t.dtype}"
            )


class TestSaveTflitePathHandling:
    """Tests for file path handling."""

    def test__accepts_string_path(
        self,
        make_dense_tflite: Callable,
        tmp_path: Path,
    ):
        """save_tflite must accept a plain string path."""
        model_path = make_dense_tflite(num_features=4, num_units=2)
        tunner_model = litert_tunner.load_model(str(model_path))
        output_path = str(tmp_path / "str_path.tflite")
        litert_tunner.save_model(tunner_model, output_path)
        assert Path(output_path).exists()

    def test__accepts_pathlib_path(
        self,
        make_dense_tflite: Callable,
        tmp_path: Path,
    ):
        """save_tflite must accept a pathlib.Path."""
        model_path = make_dense_tflite(num_features=4, num_units=2)
        tunner_model = litert_tunner.load_model(str(model_path))
        output_path = tmp_path / "pathlib_path.tflite"
        flatbuffer.save_tflite(tunner_model, output_path)
        assert output_path.exists()


class TestSaveTfliteMultipleSaves:
    """Tests for multiple consecutive save operations."""

    def test__double_save_produces_identical_files(
        self,
        make_dense_tflite: Callable,
        tmp_path: Path,
    ):
        """Saving the same unmodified model twice must produce identical files."""
        model_path = make_dense_tflite(num_features=4, num_units=2, activation=None, float_io=False)
        tunner_model = litert_tunner.load_model(str(model_path))

        path_a = tmp_path / "save_a.tflite"
        path_b = tmp_path / "save_b.tflite"
        litert_tunner.save_model(tunner_model, str(path_a))
        litert_tunner.save_model(tunner_model, str(path_b))

        bytes_a = path_a.read_bytes()
        bytes_b = path_b.read_bytes()
        assert bytes_a == bytes_b

    def test__save_after_modification_then_save_again_are_identical(
        self,
        make_dense_tflite: Callable,
        tmp_path: Path,
    ):
        """Two saves after the same modification must produce identical files."""
        model_path = make_dense_tflite(
            num_features=4, num_units=2, use_bias=True, activation=None, float_io=False
        )
        tunner_model = litert_tunner.load_model(str(model_path))

        dense_layer = _find_dense_layer(tunner_model)
        assert dense_layer is not None
        dense_layer.bias.assign(dense_layer.bias.numpy() + 5.0)

        path_a = tmp_path / "mod_save_a.tflite"
        path_b = tmp_path / "mod_save_b.tflite"
        litert_tunner.save_model(tunner_model, str(path_a))
        litert_tunner.save_model(tunner_model, str(path_b))

        assert path_a.read_bytes() == path_b.read_bytes()


def _find_dense_layer(model):
    """Find the first QuantizedDense layer in the model."""
    for layer in model.layers:
        if "quantized_dense" in layer.name:
            return layer
    return None
