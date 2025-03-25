# 🚀 Guide: Converting NeMo ONNX ASR Models to TensorRT and Deploying with Triton Inference Server

This tutorial provides detailed instructions on converting a NeMo ASR model from ONNX to TensorRT format for optimized deployment on NVIDIA Triton Inference Server.

## 📌 Why TensorRT?

### Advantages of TensorRT:
- Significantly faster inference (typically 2-5x faster than ONNX Runtime).
- Reduced GPU memory consumption (20-40% savings).
- GPU-specific optimizations for NVIDIA hardware.
- Effective scaling for production deployments.

## 🔧 Step-by-Step Conversion from ONNX to TensorRT

### 🟢 Step 1: Prepare Environment

Use the NVIDIA Docker container (`nvcr.io/nvidia/pytorch:24.05-py3`):

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/pytorch:24.05-py3 /bin/bash
```

Install required tools:

```bash
pip install onnx onnxruntime-gpu==1.19 tensorrt==10.0.1
```

Ensure CUDA and TensorRT are available in the container.

### 🟢 Step 2: Convert ONNX to TensorRT

Use NVIDIA's official tool `trtexec`:

```bash
trtexec \
  --onnx=model.onnx \
  --saveEngine=model.plan \
  --fp16 \
  --workspace=4096 \
  --explicitBatch \
  --minShapes=signal:1x80x128 \
  --optShapes=signal:4x80x512 \
  --maxShapes=signal:8x80x2048
```

Parameters explained:
- `--fp16`: Enables FP16 precision for faster inference.
- `--workspace`: Allocates GPU memory (in MB) for optimization.
- `--explicitBatch`: Enables explicit batch dimension handling.
- `minShapes`, `optShapes`, `maxShapes`: Defines dynamic input shapes for optimal performance.

### 🟢 Step 3: Validate the TensorRT Model

Check model inference:

```bash
trtexec --loadEngine=model.plan --shapes=signal:4x80x500
```

Ensure the model runs without errors.

## Next Steps
- Deploy modules to Triton.
- Benchmark performance.
- Integrate modules into your application pipeline.