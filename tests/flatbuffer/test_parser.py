"""Tests for flatbuffer parser module."""

from collections.abc import Callable
from pathlib import Path

import keras
import numpy as np
import pytest
import tflite

from litert_tunner import flatbuffer
from litert_tunner.flatbuffer import parser
from litert_tunner.graph import types
from tests import conftest


class TestParseTfliteBasic:
    """Tests for basic parse_tflite functionality on a real INT8 model."""

    def test__returns_graph_def(self, make_dense_tflite: Callable):
        """parse_tflite must return a GraphDef instance."""
        model_path = make_dense_tflite(num_features=4, num_units=1)
        graph_def = flatbuffer.parse_tflite(model_path)
        assert isinstance(graph_def, types.GraphDef)

    def test__tensors_are_non_empty(self, make_dense_tflite: Callable):
        """Parsed graph must contain at least one tensor."""
        model_path = make_dense_tflite(num_features=4, num_units=1)
        graph_def = flatbuffer.parse_tflite(model_path)
        assert len(graph_def.tensors) > 0

    def test__operators_are_non_empty(self, make_dense_tflite: Callable):
        """Parsed graph must contain at least one operator."""
        model_path = make_dense_tflite(num_features=4, num_units=1)
        graph_def = flatbuffer.parse_tflite(model_path)
        assert len(graph_def.operators) > 0

    def test__has_input_and_output_indices(self, make_dense_tflite: Callable):
        """Graph must have non-empty input and output index tuples."""
        model_path = make_dense_tflite(num_features=4, num_units=1)
        graph_def = flatbuffer.parse_tflite(model_path)
        assert len(graph_def.input_indices) >= 1
        assert len(graph_def.output_indices) >= 1

    def test__preserves_raw_model_bytes(self, make_dense_tflite: Callable):
        """raw_model_bytes must match the original file contents."""
        model_path = make_dense_tflite(num_features=4, num_units=1)
        with model_path.open("rb") as f:
            expected_bytes = f.read()
        graph_def = flatbuffer.parse_tflite(model_path)
        assert graph_def.raw_model_bytes == expected_bytes

    def test__accepts_string_path(self, make_dense_tflite: Callable):
        """parse_tflite must accept a plain string path."""
        model_path = make_dense_tflite(num_features=4, num_units=1)
        graph_def = flatbuffer.parse_tflite(str(model_path))
        assert isinstance(graph_def, types.GraphDef)

    def test__accepts_pathlib_path(self, make_dense_tflite: Callable):
        """parse_tflite must accept a pathlib.Path."""
        model_path = make_dense_tflite(num_features=4, num_units=1)
        graph_def = flatbuffer.parse_tflite(Path(model_path))
        assert isinstance(graph_def, types.GraphDef)


class TestParseTfliteTensors:
    """Tests for tensor parsing details."""

    def test__tensor_names_are_strings(self, make_dense_tflite: Callable):
        """All tensor names must be decoded to str, not bytes."""
        model_path = make_dense_tflite(num_features=4, num_units=2)
        graph_def = flatbuffer.parse_tflite(model_path)
        for tensor in graph_def.tensors:
            assert isinstance(tensor.name, str)

    def test__tensor_indices_are_sequential(self, make_dense_tflite: Callable):
        """Tensor index field must match its position in the tensors tuple."""
        model_path = make_dense_tflite(num_features=4, num_units=2)
        graph_def = flatbuffer.parse_tflite(model_path)
        for i, tensor in enumerate(graph_def.tensors):
            assert tensor.index == i

    def test__tensor_shapes_are_tuples(self, make_dense_tflite: Callable):
        """All tensor shapes must be tuples of ints."""
        model_path = make_dense_tflite(num_features=4, num_units=2)
        graph_def = flatbuffer.parse_tflite(model_path)
        for tensor in graph_def.tensors:
            assert isinstance(tensor.shape, tuple)

    def test__tensor_dtypes_are_valid(self, make_dense_tflite: Callable):
        """All tensor dtype strings must be one of the supported constants."""
        valid_dtypes = {types.DTYPE_INT8, types.DTYPE_INT32, types.DTYPE_FLOAT32}
        model_path = make_dense_tflite(num_features=4, num_units=2)
        graph_def = flatbuffer.parse_tflite(model_path)
        for tensor in graph_def.tensors:
            assert tensor.dtype in valid_dtypes, f"Unexpected dtype: {tensor.dtype}"

    def test__weight_tensor_has_data(self, make_dense_tflite: Callable):
        """Weight tensors (with buffer data) must have non-None data arrays."""
        model_path = make_dense_tflite(num_features=4, num_units=2)
        graph_def = flatbuffer.parse_tflite(model_path)
        weight_tensors = [t for t in graph_def.tensors if t.data is not None]
        assert len(weight_tensors) > 0, "Expected at least one tensor with data (weights/biases)"

    def test__activation_tensor_has_no_data(self, make_dense_tflite: Callable):
        """Activation (intermediate/input/output) tensors should have data=None."""
        model_path = make_dense_tflite(num_features=4, num_units=2)
        graph_def = flatbuffer.parse_tflite(model_path)
        input_tensor = graph_def.tensors[graph_def.input_indices[0]]
        assert input_tensor.data is None, "Input activation tensor should not carry buffer data"

    def test__weight_data_shape_matches_tensor_shape(self, make_dense_tflite: Callable):
        """Buffer data arrays must be reshaped to match the declared tensor shape."""
        model_path = make_dense_tflite(num_features=4, num_units=3)
        graph_def = flatbuffer.parse_tflite(model_path)
        for tensor in graph_def.tensors:
            if tensor.data is not None and tensor.shape:
                assert tensor.data.shape == tensor.shape, (
                    f"Tensor '{tensor.name}': data shape {tensor.data.shape} "
                    f"!= declared shape {tensor.shape}"
                )


class TestParseTfliteQuantization:
    """Tests for quantization parameter parsing."""

    def test__int8_tensors_have_quantization_params(self, make_dense_tflite: Callable):
        """All INT8 tensors in a fully quantized model should have quantization params."""
        model_path = make_dense_tflite(num_features=4, num_units=2)
        graph_def = flatbuffer.parse_tflite(model_path)
        int8_tensors = [t for t in graph_def.tensors if t.dtype == types.DTYPE_INT8]
        assert len(int8_tensors) > 0, "Expected at least one INT8 tensor"
        for tensor in int8_tensors:
            assert tensor.quantization is not None, (
                f"INT8 tensor '{tensor.name}' missing quantization params"
            )

    def test__quantization_scales_are_positive(self, make_dense_tflite: Callable):
        """All quantization scales must be positive float32 values."""
        model_path = make_dense_tflite(num_features=4, num_units=2)
        graph_def = flatbuffer.parse_tflite(model_path)
        for tensor in graph_def.tensors:
            if tensor.quantization is not None:
                assert tensor.quantization.scales.dtype == np.float32
                assert np.all(tensor.quantization.scales > 0), (
                    f"Tensor '{tensor.name}': scales must be positive, "
                    f"got {tensor.quantization.scales}"
                )

    def test__quantization_zero_points_are_int32(self, make_dense_tflite: Callable):
        """Zero points must be stored as int32 arrays."""
        model_path = make_dense_tflite(num_features=4, num_units=2)
        graph_def = flatbuffer.parse_tflite(model_path)
        for tensor in graph_def.tensors:
            if tensor.quantization is not None:
                assert tensor.quantization.zero_points.dtype == np.int32

    def test__per_tensor_quantization_has_single_scale(self, make_dense_tflite: Callable):
        """Per-tensor quantized tensors (activations) should have exactly one scale."""
        model_path = make_dense_tflite(num_features=4, num_units=2)
        graph_def = flatbuffer.parse_tflite(model_path)
        input_tensor = graph_def.tensors[graph_def.input_indices[0]]
        assert input_tensor.quantization is not None
        assert len(input_tensor.quantization.scales) == 1
        assert len(input_tensor.quantization.zero_points) == 1


class TestParseTfliteOperators:
    """Tests for operator parsing."""

    def test__dense_model_has_fully_connected_op(self, make_dense_tflite: Callable):
        """A Dense-only model must contain a FULLY_CONNECTED operator."""
        model_path = make_dense_tflite(num_features=4, num_units=1)
        graph_def = flatbuffer.parse_tflite(model_path)
        op_types = [op.op_type for op in graph_def.operators]
        assert "FULLY_CONNECTED" in op_types

    def test__operator_input_output_indices_are_valid(self, make_dense_tflite: Callable):
        """All operator input/output indices must reference existing tensors."""
        model_path = make_dense_tflite(num_features=4, num_units=2)
        graph_def = flatbuffer.parse_tflite(model_path)
        num_tensors = len(graph_def.tensors)
        for op in graph_def.operators:
            for idx in op.input_indices:
                assert 0 <= idx < num_tensors, (
                    f"Op '{op.op_type}': input index {idx} out of range [0, {num_tensors})"
                )
            for idx in op.output_indices:
                assert 0 <= idx < num_tensors, (
                    f"Op '{op.op_type}': output index {idx} out of range [0, {num_tensors})"
                )

    def test__operator_options_is_dict(self, make_dense_tflite: Callable):
        """Operator options must be a dictionary."""
        model_path = make_dense_tflite(num_features=4, num_units=1)
        graph_def = flatbuffer.parse_tflite(model_path)
        for op in graph_def.operators:
            assert isinstance(op.options, dict)

    def test__float_io_model_has_quantize_dequantize_ops(self, make_dense_tflite: Callable):
        """A float32 I/O model should contain QUANTIZE and DEQUANTIZE wrapper ops."""
        model_path = make_dense_tflite(num_features=4, num_units=2, float_io=True)
        graph_def = flatbuffer.parse_tflite(model_path)
        op_types = [op.op_type for op in graph_def.operators]
        assert "QUANTIZE" in op_types, "Expected QUANTIZE op for float32 input conversion"
        assert "DEQUANTIZE" in op_types, "Expected DEQUANTIZE op for float32 output conversion"


class TestParseTfliteDenseModelStructure:
    """End-to-end structural validation for a Dense model graph."""

    def test__fully_connected_has_weight_and_bias_inputs(self, make_dense_tflite: Callable):
        """FULLY_CONNECTED op should have 3 inputs: activation, weight, bias."""
        model_path = make_dense_tflite(num_features=4, num_units=2, use_bias=True)
        graph_def = flatbuffer.parse_tflite(model_path)
        fc_ops = [op for op in graph_def.operators if op.op_type == "FULLY_CONNECTED"]
        assert len(fc_ops) >= 1
        fc_op = fc_ops[0]
        assert len(fc_op.input_indices) == 3, (
            f"Expected 3 inputs (activation, weight, bias), got {len(fc_op.input_indices)}"
        )

    def test__weight_tensor_is_int8(self, make_dense_tflite: Callable):
        """The weight tensor of FULLY_CONNECTED must be INT8."""
        model_path = make_dense_tflite(num_features=4, num_units=2, use_bias=True)
        graph_def = flatbuffer.parse_tflite(model_path)
        fc_ops = [op for op in graph_def.operators if op.op_type == "FULLY_CONNECTED"]
        fc_op = fc_ops[0]
        weight_tensor = graph_def.tensors[fc_op.input_indices[1]]
        assert weight_tensor.dtype == types.DTYPE_INT8

    def test__bias_tensor_is_int32(self, make_dense_tflite: Callable):
        """The bias tensor of FULLY_CONNECTED must be INT32."""
        model_path = make_dense_tflite(num_features=4, num_units=2, use_bias=True)
        graph_def = flatbuffer.parse_tflite(model_path)
        fc_ops = [op for op in graph_def.operators if op.op_type == "FULLY_CONNECTED"]
        fc_op = fc_ops[0]
        bias_tensor = graph_def.tensors[fc_op.input_indices[2]]
        assert bias_tensor.dtype == types.DTYPE_INT32

    def test__weight_shape_matches_dense_config(self, make_dense_tflite: Callable):
        """Weight tensor shape must reflect (num_units, num_features) for FullyConnected."""
        num_features = 6
        num_units = 3
        model_path = make_dense_tflite(num_features=num_features, num_units=num_units)
        graph_def = flatbuffer.parse_tflite(model_path)
        fc_ops = [op for op in graph_def.operators if op.op_type == "FULLY_CONNECTED"]
        fc_op = fc_ops[0]
        weight_tensor = graph_def.tensors[fc_op.input_indices[1]]
        assert weight_tensor.shape == (num_units, num_features), (
            f"Expected weight shape ({num_units}, {num_features}), got {weight_tensor.shape}"
        )


class TestParseTfliteErrors:
    """Tests for error handling in parse_tflite."""

    def test__raises_on_nonexistent_file(self, tmp_path: Path):
        """Must raise FileNotFoundError for a missing file."""
        with pytest.raises(FileNotFoundError):
            flatbuffer.parse_tflite(tmp_path / "nonexistent.tflite")

    def test__raises_on_invalid_flatbuffer(self, tmp_path: Path):
        """Must raise an error when given garbage bytes instead of a valid flatbuffer."""
        garbage_path = tmp_path / "garbage.tflite"
        garbage_path.write_bytes(b"this is not a valid flatbuffer")
        with pytest.raises(Exception):  # noqa: B017
            flatbuffer.parse_tflite(garbage_path)


class TestParseTfliteDeterminism:
    """Tests that parsing is deterministic."""

    def test__parsing_same_file_twice_gives_equal_graphs(self, make_dense_tflite: Callable):
        """Two calls to parse_tflite on the same file must produce identical GraphDefs."""
        model_path = make_dense_tflite(num_features=4, num_units=2)
        graph_def_1 = flatbuffer.parse_tflite(model_path)
        graph_def_2 = flatbuffer.parse_tflite(model_path)

        assert len(graph_def_1.tensors) == len(graph_def_2.tensors)
        assert len(graph_def_1.operators) == len(graph_def_2.operators)
        assert graph_def_1.input_indices == graph_def_2.input_indices
        assert graph_def_1.output_indices == graph_def_2.output_indices

        for t1, t2 in zip(graph_def_1.tensors, graph_def_2.tensors, strict=False):
            assert t1.name == t2.name
            assert t1.shape == t2.shape
            assert t1.dtype == t2.dtype
            if t1.data is not None:
                assert t2.data is not None
                np.testing.assert_array_equal(t1.data, t2.data)
            else:
                assert t2.data is None

        for op1, op2 in zip(graph_def_1.operators, graph_def_2.operators, strict=False):
            assert op1.op_type == op2.op_type
            assert op1.input_indices == op2.input_indices
            assert op1.output_indices == op2.output_indices


class TestTensorTypeMaps:
    """Tests for _TENSOR_TYPE_MAP, _NUMPY_DTYPE_MAP, and helper functions."""

    # All tensor types that must be supported (tflite code, dtype string, numpy dtype).
    _EXPECTED_TYPES = (
        (tflite.TensorType.FLOAT16, types.DTYPE_FLOAT16, np.float16),
        (tflite.TensorType.FLOAT32, types.DTYPE_FLOAT32, np.float32),
        (tflite.TensorType.FLOAT64, types.DTYPE_FLOAT64, np.float64),
        (tflite.TensorType.INT8, types.DTYPE_INT8, np.int8),
        (tflite.TensorType.INT16, types.DTYPE_INT16, np.int16),
        (tflite.TensorType.INT32, types.DTYPE_INT32, np.int32),
        (tflite.TensorType.INT64, types.DTYPE_INT64, np.int64),
        (tflite.TensorType.UINT8, types.DTYPE_UINT8, np.uint8),
        (tflite.TensorType.UINT16, types.DTYPE_UINT16, np.uint16),
        (tflite.TensorType.UINT32, types.DTYPE_UINT32, np.uint32),
        (tflite.TensorType.UINT64, types.DTYPE_UINT64, np.uint64),
        (tflite.TensorType.BOOL, types.DTYPE_BOOL, np.bool_),
    )

    @pytest.mark.parametrize(
        ("tflite_code", "expected_str", "_np_dtype"),
        _EXPECTED_TYPES,
        ids=[t[1] for t in _EXPECTED_TYPES],
    )
    def test__tensor_type_map_contains_entry(self, tflite_code, expected_str, _np_dtype):
        """_TENSOR_TYPE_MAP must contain every supported tflite type code."""
        assert tflite_code in parser._TENSOR_TYPE_MAP
        assert parser._TENSOR_TYPE_MAP[tflite_code] == expected_str

    @pytest.mark.parametrize(
        ("tflite_code", "expected_str", "_np_dtype"),
        _EXPECTED_TYPES,
        ids=[t[1] for t in _EXPECTED_TYPES],
    )
    def test__map_tensor_type_returns_correct_string(self, tflite_code, expected_str, _np_dtype):
        """_map_tensor_type must return the correct dtype string for each code."""
        assert parser._map_tensor_type(tflite_code) == expected_str

    @pytest.mark.parametrize(
        ("tflite_code", "_dtype_str", "expected_np"),
        _EXPECTED_TYPES,
        ids=[t[1] for t in _EXPECTED_TYPES],
    )
    def test__numpy_dtype_map_contains_entry(self, tflite_code, _dtype_str, expected_np):
        """_NUMPY_DTYPE_MAP must contain every supported tflite type code."""
        assert tflite_code in parser._NUMPY_DTYPE_MAP
        assert parser._NUMPY_DTYPE_MAP[tflite_code] == expected_np

    @pytest.mark.parametrize(
        ("tflite_code", "_dtype_str", "expected_np"),
        _EXPECTED_TYPES,
        ids=[t[1] for t in _EXPECTED_TYPES],
    )
    def test__get_numpy_dtype_returns_correct_dtype(self, tflite_code, _dtype_str, expected_np):
        """get_numpy_dtype must return the correct numpy dtype for each code."""
        assert parser.get_numpy_dtype(tflite_code) == expected_np

    def test__map_tensor_type_raises_on_unsupported(self):
        """_map_tensor_type must raise ValueError for an unknown type code."""
        with pytest.raises(ValueError, match="Unsupported tensor type"):
            parser._map_tensor_type(9999)

    def test__get_numpy_dtype_raises_on_unsupported(self):
        """get_numpy_dtype must raise ValueError for an unknown type code."""
        with pytest.raises(ValueError, match="Unsupported tensor type"):
            parser.get_numpy_dtype(9999)

    def test__tensor_type_and_numpy_dtype_maps_have_same_keys(self):
        """Both maps must cover exactly the same set of tflite type codes."""
        assert set(parser._TENSOR_TYPE_MAP.keys()) == set(parser._NUMPY_DTYPE_MAP.keys())


class TestBuiltinOptionsMap:
    """Tests for _BUILTIN_OPTIONS_MAP completeness and consistency."""

    # All entries that must be present in _BUILTIN_OPTIONS_MAP,
    # grouped by project phase for clarity.
    _EXPECTED_OPTIONS = (
        # Phase 1: Dense
        (tflite.BuiltinOptions.FullyConnectedOptions, tflite.FullyConnectedOptions),
        # Phase 2: Convolutions
        (tflite.BuiltinOptions.Conv2DOptions, tflite.Conv2DOptions),
        (tflite.BuiltinOptions.DepthwiseConv2DOptions, tflite.DepthwiseConv2DOptions),
        (tflite.BuiltinOptions.TransposeConvOptions, tflite.TransposeConvOptions),
        (tflite.BuiltinOptions.Conv3DOptions, tflite.Conv3DOptions),
        # Phase 3: Pooling & Reshape
        (tflite.BuiltinOptions.Pool2DOptions, tflite.Pool2DOptions),
        (tflite.BuiltinOptions.ReshapeOptions, tflite.ReshapeOptions),
        (tflite.BuiltinOptions.SqueezeOptions, tflite.SqueezeOptions),
        (tflite.BuiltinOptions.ExpandDimsOptions, tflite.ExpandDimsOptions),
        (tflite.BuiltinOptions.SliceOptions, tflite.SliceOptions),
        (tflite.BuiltinOptions.StridedSliceOptions, tflite.StridedSliceOptions),
        (tflite.BuiltinOptions.PackOptions, tflite.PackOptions),
        (tflite.BuiltinOptions.UnpackOptions, tflite.UnpackOptions),
        (tflite.BuiltinOptions.TransposeOptions, tflite.TransposeOptions),
        (tflite.BuiltinOptions.TileOptions, tflite.TileOptions),
        (tflite.BuiltinOptions.GatherOptions, tflite.GatherOptions),
        (tflite.BuiltinOptions.GatherNdOptions, tflite.GatherNdOptions),
        # Phase 4: Normalization & Skip connections
        (tflite.BuiltinOptions.AddOptions, tflite.AddOptions),
        (tflite.BuiltinOptions.SubOptions, tflite.SubOptions),
        (tflite.BuiltinOptions.MulOptions, tflite.MulOptions),
        (tflite.BuiltinOptions.DivOptions, tflite.DivOptions),
        (tflite.BuiltinOptions.L2NormOptions, tflite.L2NormOptions),
        (tflite.BuiltinOptions.ReducerOptions, tflite.ReducerOptions),
        (tflite.BuiltinOptions.BatchToSpaceNDOptions, tflite.BatchToSpaceNDOptions),
        (tflite.BuiltinOptions.SpaceToBatchNDOptions, tflite.SpaceToBatchNDOptions),
        (tflite.BuiltinOptions.SpaceToDepthOptions, tflite.SpaceToDepthOptions),
        (tflite.BuiltinOptions.DepthToSpaceOptions, tflite.DepthToSpaceOptions),
        # Phase 5: Advanced
        (tflite.BuiltinOptions.SoftmaxOptions, tflite.SoftmaxOptions),
        (tflite.BuiltinOptions.ConcatenationOptions, tflite.ConcatenationOptions),
        (tflite.BuiltinOptions.PadOptions, tflite.PadOptions),
        (tflite.BuiltinOptions.PadV2Options, tflite.PadV2Options),
        (tflite.BuiltinOptions.MirrorPadOptions, tflite.MirrorPadOptions),
        (tflite.BuiltinOptions.SplitOptions, tflite.SplitOptions),
        (tflite.BuiltinOptions.SplitVOptions, tflite.SplitVOptions),
        (tflite.BuiltinOptions.ResizeBilinearOptions, tflite.ResizeBilinearOptions),
        (tflite.BuiltinOptions.ResizeNearestNeighborOptions, tflite.ResizeNearestNeighborOptions),
        (tflite.BuiltinOptions.CastOptions, tflite.CastOptions),
        (tflite.BuiltinOptions.MaximumMinimumOptions, tflite.MaximumMinimumOptions),
        (tflite.BuiltinOptions.LeakyReluOptions, tflite.LeakyReluOptions),
        (tflite.BuiltinOptions.ArgMaxOptions, tflite.ArgMaxOptions),
        (tflite.BuiltinOptions.ArgMinOptions, tflite.ArgMinOptions),
        (tflite.BuiltinOptions.BatchMatMulOptions, tflite.BatchMatMulOptions),
        # Quantization-related
        (tflite.BuiltinOptions.QuantizeOptions, tflite.QuantizeOptions),
        (tflite.BuiltinOptions.DequantizeOptions, tflite.DequantizeOptions),
        (tflite.BuiltinOptions.FakeQuantOptions, tflite.FakeQuantOptions),
    )

    @pytest.mark.parametrize(
        ("opt_code", "opt_class"),
        _EXPECTED_OPTIONS,
        ids=[cls.__name__ for _, cls in _EXPECTED_OPTIONS],
    )
    def test__builtin_options_map_contains_entry(self, opt_code, opt_class):
        """_BUILTIN_OPTIONS_MAP must map each option type code to its class."""
        assert opt_code in parser._BUILTIN_OPTIONS_MAP
        assert parser._BUILTIN_OPTIONS_MAP[opt_code] is opt_class

    def test__builtin_options_map_keys_are_ints(self):
        """All keys in _BUILTIN_OPTIONS_MAP must be integers (BuiltinOptions codes)."""
        for key in parser._BUILTIN_OPTIONS_MAP:
            assert isinstance(key, int), f"Key {key} is {type(key)}, expected int"

    def test__builtin_options_map_values_are_classes(self):
        """All values in _BUILTIN_OPTIONS_MAP must be classes (not instances)."""
        for key, val in parser._BUILTIN_OPTIONS_MAP.items():
            assert isinstance(val, type), f"Value for key {key} is {type(val)}, expected a class"

    def test__builtin_options_map_does_not_contain_none(self):
        """NONE (0) must not be a key — ops with no options return {} directly."""
        assert tflite.BuiltinOptions.NONE not in parser._BUILTIN_OPTIONS_MAP


class TestBuiltinOptionsParsing:
    """Tests that _parse_builtin_options extracts fused_activation_function correctly."""

    def test__fully_connected_options_parsed(self, make_dense_tflite: Callable):
        """FullyConnected op options must include fused_activation_function."""
        model_path = make_dense_tflite(num_features=4, num_units=2, activation="relu")
        graph_def = flatbuffer.parse_tflite(model_path)
        fc_ops = [op for op in graph_def.operators if op.op_type == "FULLY_CONNECTED"]
        assert len(fc_ops) >= 1
        fc_op = fc_ops[0]
        assert "fused_activation_function" in fc_op.options

    def test__no_activation_gives_zero_fused(self, make_dense_tflite: Callable):
        """FullyConnected with no activation must have fused_activation_function=0 (NONE)."""
        model_path = make_dense_tflite(num_features=4, num_units=2, activation=None)
        graph_def = flatbuffer.parse_tflite(model_path)
        fc_ops = [op for op in graph_def.operators if op.op_type == "FULLY_CONNECTED"]
        fc_op = fc_ops[0]
        assert fc_op.options.get("fused_activation_function") == 0

    def test__relu_activation_gives_nonzero_fused(self, make_dense_tflite: Callable):
        """FullyConnected with relu activation must have fused_activation_function > 0."""
        model_path = make_dense_tflite(num_features=4, num_units=2, activation="relu")
        graph_def = flatbuffer.parse_tflite(model_path)
        fc_ops = [op for op in graph_def.operators if op.op_type == "FULLY_CONNECTED"]
        fc_op = fc_ops[0]
        assert fc_op.options.get("fused_activation_function", 0) > 0


class TestParseTfliteFloat16:
    """Tests for parsing float16 optimized TFLite models."""

    def test__dequantize_on_constant_is_eagerly_evaluated(self, tmp_path: Path):
        """Test that DEQUANTIZE op on a float16 constant is eagerly evaluated."""
        # 1. Create a simple model
        inputs = keras.Input(shape=(4,))
        x = keras.layers.Dense(2, use_bias=True)(inputs)
        model = keras.Model(inputs=inputs, outputs=x)

        # 2. Export to float16 TFLite
        model_path = tmp_path / "float16.tflite"
        conftest.export_float16_tflite_model(
            input_shape=(4,),
            model=model,
            output_path=model_path,
        )

        # 3. Parse the model
        graph_def = parser.parse_tflite(model_path)

        # 4. Verify DEQUANTIZE is removed
        op_types = [op.op_type for op in graph_def.operators]
        assert "DEQUANTIZE" not in op_types, "DEQUANTIZE op should be eagerly evaluated and removed"

        # 5. Verify FULLY_CONNECTED weights are float32 but keep float16 buffer index
        fc_ops = [op for op in graph_def.operators if op.op_type == "FULLY_CONNECTED"]
        assert len(fc_ops) == 1
        fc_op = fc_ops[0]
        weight_tensor = graph_def.tensors[fc_op.input_indices[1]]

        assert weight_tensor.data is not None, "Weight data should be eagerly evaluated"
        assert weight_tensor.data.dtype == "float32", "Data should be cast to float32"
        assert weight_tensor.dtype == types.DTYPE_FLOAT16, (
            "Original FLOAT16 dtype should be preserved"
        )
