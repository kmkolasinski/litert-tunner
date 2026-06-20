# AGENTS.md — Guidelines for AI Agents Working on `litert-tunner`

## Project Vision

**litert-tunner** fine-tunes fully-quantized INT8 LiteRT (TFLite) models
*after* export. It parses an INT8 flatbuffer, reconstructs it as a Keras 3
model with differentiable quantization simulation, fine-tunes float32
parameters, and writes them back — **without altering graph topology**.

## Target API

```python
import litert_tunner
from litert_tunner import distillation

tunner_model = litert_tunner.load_model("model_int8.tflite")
distillation.prepare_for_finetuning(tunner_model, trainable_pattern=".*bias")
trainer = distillation.Trainer(student_model=tunner_model, teacher_model=teacher_model)
trainer.compile(optimizer=..., loss=..., metrics=...)
trainer.fit(train_ds, validation_data=val_ds, epochs=5)
litert_tunner.save_model(tunner_model, "model_int8_finetuned.tflite")
```

## Quantization

| Component   | LiteRT storage | Keras repr | Trainable |
| ----------- | -------------- | ---------- | --------- |
| Weights     | INT8           | INT8 as-is | Yes       |
| Biases      | INT32          | Float32    | Yes       |
| Scales      | Float32        | Float32    | No        |
| Zero-points | INT8/INT32     | Float32    | No        |

**Dequantize:** `real_value = scale * (int8_value - zero_point)`
**Quantize:** `int8_value = clamp(round(real_value / scale) + zero_point, -128, 127)`

Fused activations (NONE=0, RELU=1, RELU_N1_TO_1=2, RELU6=3) defined in
`ops/utils.py`, applied via `apply_fused_activation()`.

## Op Implementation Pattern

1. **Keras Layer** in `src/litert_tunner/ops/<op>.py`:
   - `__init__`: Store numpy data. `build`: Create weights. `call`: Dequant →
     float32 compute → fused activation → quantize (STE). `collect_write_ops`:
     Return `BufferWriteOp`/`QuantizationWriteOp`.
1. **Builder function** decorated with `@registry.register_op("OP_TYPE")`:
   - Extract tensors/params from `OperatorInfo`, use `ops/utils.py` helpers.

**Key `ops/utils.py` helpers:**

- STE quant/dequant: `dequantize_ste`, `quantize_ste`, `fake_quantize`,
  `quantize_to_int8_ste`, `fake_quantize_bias`
- Non-STE: `dequantize_float`, `quantize_int8`, `quantize_to_int8`,
  `quantize_bias_to_int32`
- Fused activation: `apply_fused_activation`
- Param extraction: `get_bias_float32`, `get_float32_bias`,
  `get_quant_param_value`, `extract_constant_input`
- Write-back: `make_quant_write_op`, `make_bias_quant_write_op`
- Compute: `compute_requantize_multiplier`, `get_padding`,
  `compute_output_shape`, `validate_per_channel_quantization`
- Conversion: `to_float_list`, `to_int_list`, `expand_dims_if_not_scalar`
- Class: `QuantizationVars` — container for scale/zero_point Keras variables
  with optional softplus reparameterization.

## Code Organization

```text
src/litert_tunner/
├── __init__.py              # Public API: load_model, save_model, distillation
├── export.py                # TFLite model export utilities
├── logging.py / testing_utils.py
├── distillation/            # Trainer, losses, metrics, prepare_for_finetuning
│                            # losses: kl_loss, mse_loss, cosine_loss
├── flatbuffer/              # parser.py (parse_tflite), writer.py (save_tflite)
├── graph/                   # types.py (QuantizationParams, TensorInfo, OperatorInfo,
│                            #   GraphDef, BufferWriteOp, QuantizationWriteOp, Writable)
│                            # builder.py (build_keras_model: GraphDef → Keras Functional)
└── ops/                     # registry.py, utils.py, one file per op
    ├── dense.py, conv2d.py, depthwise_conv2d.py, transpose_conv.py
    ├── add.py, sub.py, mul.py, div.py
    ├── concatenation.py, mean.py, pool.py
    ├── softmax.py, logistic.py, relu.py, gelu.py
    ├── neg.py, rsqrt.py, squared_difference.py
    ├── reshape.py, transpose.py, pack.py, strided_slice.py
    ├── resize_nearest_neighbor.py, quantize_op.py, shape_op.py
    ├── expand_dims.py, pad.py, tile.py
```

```text
tests/
├── conftest.py              # Fixtures: make_dense_tflite, make_mlp_tflite,
│                            #   make_resnet_tflite, make_backbone_tflite,
│                            #   make_efficientnetb0_tflite, make_float32_dense_tflite,
│                            #   run_interpreter, compute_gradient, temp_model_dir
│                            # Helpers: export_quantized_tflite_model,
│                            #   export_float32_tflite_model, export_float16_tflite_model
├── test_load_save_roundtrip.py / test_finetuning_e2e.py / test_export.py
├── distillation/            # test_trainer.py, test_losses.py, test_metrics.py
├── flatbuffer/              # test_parser.py, test_writer.py, test_parse_write.py
├── graph/                   # test_builder.py
├── ops/                     # op_test_utils.py + one test file per op
└── networks/                # test_mlp.py, test_resnet.py, test_efficientnet.py, ...
```

## Coding Standards

- **Python ≥ 3.11** — modern syntax (`|` unions, `match`).
- **Type hints** on all signatures. **Google-style docstrings** on public API.
- **Line length** 100 chars. **Linting**: `ruff check` + `ruff format`.
- **No magic numbers** — use constants/enums.
- **RNG**: `np.random.default_rng(seed)` only. Never `np.random.seed`.
- **Dataclasses**: `frozen=True` for immutable types.
- **Keras 3 only**: Use `keras.ops` everywhere. No `tf.`/`jax.`/`torch.` in
  production code (TF allowed in tests for converter/interpreter).

### Import Style (Google-style)

```python
# ✅ Import modules, use dotted access
from litert_tunner import flatbuffer
from litert_tunner.graph import types

tensor = types.TensorInfo(...)

# ❌ Never import symbols directly
from litert_tunner.graph.types import TensorInfo  # NO
```

### Test Naming — double underscore prefix

```python
def test__dense_output_matches_interpreter(): ...  # ✅
def test_dense_output_matches_interpreter(): ...  # ❌
```

### Environment & Commands

- Activate: `source .venv/bin/activate`
- Package mgmt: `uv pip` (not `pip`)
- Run tests: `.venv/bin/python -m pytest <path> -v`
- Lint: `make precommit` (runs `pre-commit run --all-files` — ruff, interrogate, ty, etc.)
- Run all tests: `make test` (pytest with `-n 4 --forked` + coverage)
- Type check: `npx -y pyright --pythonpath .venv/bin/python`

## Testing Pattern

Network-level tests (primary correctness validation):

```python
def test__mlp_forward(make_mlp_tflite, run_interpreter):
    model_path = make_mlp_tflite(input_size=4, hidden_sizes=[8])
    litert_out = run_interpreter(model_path, x_train)
    keras_model = litert_tunner.load_model(str(model_path))
    keras_out = keras_model.predict(x_train)
    np.testing.assert_allclose(litert_out, keras_out, atol=1e-3)
    # Save round-trip
    litert_tunner.save_model(keras_model, str(model_path))
    saved_out = run_interpreter(model_path, x_train)
    np.testing.assert_allclose(litert_out, saved_out, atol=1e-5)
```

Op-level tests use `tests/ops/op_test_utils.py`: `make_tensor`, `make_operator`,
`make_quant_params`, `build_and_call`, `build_layer_from_registry`,
`assert_trainable_weight_names`, `assert_non_trainable_weight_names`,
`assert_layer_is_writable`, `assert_layer_not_writable`,
`assert_output_shape`, `assert_collect_write_ops`,
`assert_quant_write_tensor_indices`, `assert_buffer_write_tensor_indices`,
`verify_model_outputs`, `verify_model_contains_operator`.

## New Feature Checklist

- [ ] Implement `keras.Layer` in `src/litert_tunner/ops/<op>.py` using `ops/utils.py`
- [ ] Register with `@registry.register_op("OP_TYPE")`
- [ ] Add import in `ops/__init__.py`
- [ ] Implement `Writable` if op has trainable params for flatbuffer write-back
- [ ] Write op-level tests in `tests/ops/test_<op>.py`
- [ ] Write/extend network-level test in `tests/networks/`
- [ ] Run `make precommit` — linting passes
- [ ] Run `.venv/bin/python -m pytest` — all tests pass
- [ ] Update this file if architecture/conventions change

## Hard Rules

- **Never modify graph topology on save** — only buffer data changes
- **No ops without tests** — untested ops cause silent bugs
- **No model-specific logic** — everything driven by parsed graph
- **No heavy dependencies** — keep library lightweight
- **No TF/JAX/PyTorch in production code** — `keras.ops` only
- **No legacy `np.random`** — use `default_rng(seed)` only
- **No direct symbol imports** — Google-style module imports only
- **No timeout polling for tests** — always wait for completion
