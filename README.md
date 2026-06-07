# litert-tunner

Fine-tune fully-quantized **INT8 LiteRT** (TFLite) models *after* export —
without retraining from scratch.

## What is this?

Standard Quantization-Aware Training (QAT) requires inserting fake-quantization
nodes *during* training, which is invasive, framework-specific, and often
impractical when you only have access to the exported `.tflite` file.

**litert-tunner** takes a different approach:

1. **Parse** an already-exported INT8 `.tflite` flatbuffer.
1. **Reconstruct** the computation graph as a trainable **Keras 3** model with
   differentiable quantization simulation (STE-based fake-quant).
1. **Fine-tune** selected float32 parameters (biases, scales, zero-points) using
   any Keras optimizer and loss function.
1. **Write** the updated parameters back into the flatbuffer — without altering
   the graph topology.

The result is a valid `.tflite` model that runs on any LiteRT-compatible device
(Android, iOS, embedded, Edge TPU) with improved accuracy.

## Key Assumptions & Scope

- **INT8 only** — the library targets fully-quantized INT8 models (per-tensor
  and per-channel quantization). Float16 or dynamic-range models are not
  supported.
- **LiteRT → Keras** — we convert *from* `.tflite` *to* Keras, not the other
  direction. The Keras model is a training-time replica, not a general-purpose
  converter.
- **Graph topology is immutable** — `save_model` updates *only* buffer data
  (weights, biases, scales, zero-points). The graph structure is never modified,
  guaranteeing the output is always a valid LiteRT model.
- **Backend-agnostic** — production code uses `keras.ops` exclusively (no
  direct TensorFlow, JAX, or PyTorch calls). TF is only required in dev
  dependencies for model export in tests.
- **Supported ops** — see [Supported Operations](#supported-operations) below.
  Unsupported ops will raise an error at load time.

## Installation

### From source (recommended for development)

```bash
# Clone the repository
git clone https://github.com/kmkolasinski/litert-tunner.git
cd litert-tunner

# Create a virtual environment and install uv
make venv
make init

# Install the project in editable mode with dev dependencies
make install
```

### As a dependency (pip)

```bash
pip install litert-tunner
```

> **Note:** The core library depends on `keras>=3.0`, `numpy`, `tflite>=2.18.0`,
> and `flatbuffers`. Development/test dependencies (`tensorflow`, `ai-edge-litert`,
> `pytest`, `ruff`, etc.) are only needed when running tests or exporting models.

## Quickstart

### Standard Fine-Tuning

```python
import litert_tunner

# 1. Load an INT8 LiteRT model → trainable Keras replica
model = litert_tunner.load_model("model_int8.tflite")

# 2. Inference — should match LiteRT Interpreter output
predictions = model.predict(inputs)

# 3. Prepare for fine-tuning (freeze everything except biases & scales)
litert_tunner.prepare_for_finetuning(model, trainable_pattern=".*bias")

# 4. Fine-tune with any Keras optimizer / loss
model.compile(optimizer="adam", loss="mse")
model.fit(x_train, y_train, epochs=5)

# 5. Export — writes updated parameters back into the flatbuffer
litert_tunner.save_model(model, "model_int8_finetuned.tflite")
```

### Distillation Fine-Tuning (Recommended)

```python
import litert_tunner

# 1. Load an INT8 LiteRT model → trainable Keras replica
tunner_model = litert_tunner.load_model("model_int8.tflite")

# 2. Prepare for fine-tuning
litert_tunner.prepare_for_finetuning(tunner_model, trainable_pattern=".*bias")

# 3. Fine-tune using the Trainer wrapper (handles distillation and weight drift)
trainer = litert_tunner.Trainer(
    student_model=tunner_model,
    teacher_model=teacher_model,  # The original float32 model
)
trainer.compile(optimizer="adam", loss="mse")
trainer.fit(train_ds, validation_data=val_ds, epochs=5)

# 5. Export — writes updated parameters back into the flatbuffer
litert_tunner.save_model(tunner_model, "model_int8_finetuned.tflite")
```

For a complete, runnable end-to-end example with a real dataset, see the
[notebooks/quickstart_finetuning.ipynb](notebooks/quickstart_finetuning.ipynb)
notebook.

## API Reference

### Core Functions

| Function                               | Description                                                       |
| -------------------------------------- | ----------------------------------------------------------------- |
| `litert_tunner.load_model`             | Parse a `.tflite` file and return a trainable Keras Model replica |
| `litert_tunner.save_model`             | Write updated parameters back to a `.tflite` file                 |
| `litert_tunner.prepare_for_finetuning` | Freeze all variables except those matching a regex pattern        |
| `litert_tunner.Trainer`                | Distillation trainer (teacher–student) with L2 weight drift loss  |

### Testing Utilities

| Function                                                | Description                                  |
| ------------------------------------------------------- | -------------------------------------------- |
| `litert_tunner.assert_cosine_similarity`                | Assert cosine similarity between two arrays  |
| `litert_tunner.assert_allclose_with_mismatch_tolerance` | Assert allclose with a max mismatch fraction |

## Supported Operations

The following LiteRT ops are currently supported:

| Category         | Operations                                              |
| ---------------- | ------------------------------------------------------- |
| **Linear**       | `FULLY_CONNECTED`, `CONV_2D`, `DEPTHWISE_CONV_2D`       |
| **Arithmetic**   | `ADD`, `SUB`, `MUL`, `DIV`, `SQUARED_DIFFERENCE`, `NEG` |
| **Activation**   | `RELU`, `GELU`, `LOGISTIC`, `SOFTMAX`                   |
| **Pooling**      | `AVERAGE_POOL_2D`, `MAX_POOL_2D`, `MEAN`                |
| **Reshape**      | `RESHAPE`, `TRANSPOSE`, `PACK`, `STRIDED_SLICE`         |
| **Resize**       | `RESIZE_NEAREST_NEIGHBOR`                               |
| **Quantization** | `QUANTIZE`, `DEQUANTIZE`                                |
| **Other**        | `CONCATENATION`, `RSQRT`, `SHAPE`                       |

Fused activations (`RELU`, `RELU6`, `RELU_N1_TO_1`) are supported on
linear ops (`FULLY_CONNECTED`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `ADD`, etc.).

## Architecture

```text
src/litert_tunner/
├── __init__.py              # Public API: load_model, save_model, Trainer, etc.
├── trainer.py               # Trainer wrapper + prepare_for_finetuning
├── testing_utils.py         # Public testing helpers
├── flatbuffer/              # Flatbuffer parsing & serialization
│   ├── parser.py            # Parse .tflite → GraphDef
│   └── writer.py            # Write updated params back to .tflite
├── graph/                   # Internal graph representation
│   ├── types.py             # TensorInfo, OperatorInfo, GraphDef, etc.
│   └── builder.py           # Convert GraphDef → Keras Functional model
└── ops/                     # Operation registry & implementations
    ├── registry.py           # @register_op decorator
    ├── utils.py              # Shared helpers: STE quant/dequant, fused activations
    └── *.py                  # One file per supported op
```

## How It Works

### Quantization Formulas

All fake-quantization simulation uses the standard TFLite affine scheme:

**Dequantize (INT8 → Float32):**

```python
real_value = scale * (int8_value - zero_point)
```

**Quantize (Float32 → INT8):**

```python
int8_value = clamp(round(real_value / scale) + zero_point, -128, 127)
```

Gradients flow through quantization via the **Straight-Through Estimator**
(STE): the gradient of the rounding operation is approximated as 1.

### Training Flow

```text
           ┌───────────────────────────────────┐
           │  INT8 .tflite model (flatbuffer)  │
           └──────────────┬────────────────────┘
                          │ litert_tunner.load_model()
                          ▼
           ┌───────────────────────────────────┐
           │   Keras 3 Model (float32 params)  │
           │   with STE fake-quant simulation  │
           └──────────────┬────────────────────┘
                          │ prepare_for_finetuning()
                          │ model.compile() / model.fit()
                          ▼
           ┌───────────────────────────────────┐
           │     Fine-tuned Keras Model        │
           └──────────────┬────────────────────┘
                          │ litert_tunner.save_model()
                          ▼
           ┌───────────────────────────────────┐
           │ Updated INT8 .tflite (same graph) │
           └───────────────────────────────────┘
```

## License

[MIT](LICENSE) — Copyright © 2026 Krzysztof Kolasinski
