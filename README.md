# litert-tunner

Fine-tune fully-quantized **INT8 LiteRT** (TFLite) models *after* export — without retraining from scratch.

## What is this?

**litert-tunner** parses an exported INT8 `.tflite` model, reconstructs it as a Keras 3 model with simulated quantization, lets you fine-tune parameters (biases, scales, weights), and writes them back into the flatbuffer. Graph topology stays intact.

> **Note:** Even though this library simulates the quantization process during fine-tuning, in some cases the resulting INT8 model may still perform worse than a full-precision float32 model.

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

*(Core dependencies: `keras>=3.0`, `numpy`, `tflite>=2.18.0`, `flatbuffers`. `tensorflow` and `ai-edge-litert` are only needed for dev/tests).*

## Quickstart

For a complete, runnable end-to-end example, see the **[Example Notebook](notebooks/quickstart_finetuning.ipynb)**.

```python
import litert_tunner
from litert_tunner import distillation

# 1. Load an INT8 LiteRT model → trainable Keras replica
model = litert_tunner.load_model("model_int8.tflite")

# 2. Inference — should match LiteRT Interpreter output
predictions = model.predict(inputs)

# 3. Prepare for fine-tuning (freeze everything except biases & scales)
distillation.prepare_for_finetuning(model, trainable_pattern=".*bias")

# 4. Fine-tune with any Keras optimizer / loss
model.compile(optimizer="adam", loss="mse")
model.fit(x_train, y_train, epochs=5)

# 5. Export — writes updated parameters back into the flatbuffer
litert_tunner.save_model(model, "model_int8_finetuned.tflite")
```

### Distillation Fine-Tuning (Recommended)

```python
import litert_tunner
from litert_tunner import distillation

# 1. Load an INT8 LiteRT model → trainable Keras replica
student_model = litert_tunner.load_model("model_int8.tflite")
teacher_model = litert_tunner.load_model("model_float32.tflite")

# 2. Freeze everything except biases
distillation.prepare_for_finetuning(student_model, trainable_pattern=".*bias")

# 3. Fine-tune using Trainer (handles distillation & weight drift)
trainer = distillation.Trainer(
    student_model=student_model,
    teacher_model=teacher_model,  # Original float32 model
    distillation_loss_fn=distillation.kl_loss,  # Optional: defaults to mse_loss
)
trainer.compile(optimizer="adam")
trainer.fit(train_ds, validation_data=val_ds, epochs=5)

# 4. Save updated parameters to flatbuffer
litert_tunner.save_model(student_model, "model_int8_finetuned.tflite")
```

## Supported Operations

- **Linear:** `FULLY_CONNECTED`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `TRANSPOSE_CONV`
- **Arithmetic:** `ADD`, `SUB`, `MUL`, `DIV`, `SQUARED_DIFFERENCE`, `NEG`
- **Activation:** `RELU`, `GELU`, `LOGISTIC`, `SOFTMAX`
- **Pooling:** `AVERAGE_POOL_2D`, `MAX_POOL_2D`, `MEAN`
- **Reshape/Resize:** `RESHAPE`, `TRANSPOSE`, `PACK`, `STRIDED_SLICE`, `RESIZE_NEAREST_NEIGHBOR`, `SHAPE`, `EXPAND_DIMS`, `PAD`, `TILE`
- **Other:** `CONCATENATION`, `RSQRT`, `QUANTIZE`, `DEQUANTIZE`

Fused activations (`RELU`, `RELU6`, `RELU_N1_TO_1`) are supported.

## License

[MIT](LICENSE) — Copyright © 2026 Krzysztof Kolasinski
