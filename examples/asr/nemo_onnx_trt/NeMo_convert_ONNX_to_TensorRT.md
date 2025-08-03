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

It is important to use the same versions of CUDA drivers and TensorRT packages that will be used on the inference instance. This ensures compatibility and stability of the model's performance. Make sure that the versions of CUDA and TensorRT in your environment match those installed on the inference server. You can verify this using the commands `nvcc --version` for CUDA and `dpkg -l | grep tensorrt` for TensorRT within the container.

In these examples, we will use Triton Inference Server version `nvcr.io/nvidia/tritonserver:25.02-py3`. Therefore, we must convert our ONNX file to TensorRT using the same container of this version. This will ensure compatibility and stability of the model's performance on the Triton server.

Use the NVIDIA Docker container (`nvcr.io/nvidia/tritonserver:25.02-py3`):

We need to mount the directory containing our ONNX file into the container. This allows the container to access the ONNX file for conversion to TensorRT.

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/tritonserver:25.02-py3
```

Ensure CUDA and TensorRT are available in the container.

### 🟢 Step 2: Convert ONNX to TensorRT

Use NVIDIA's official tool `trtexec`:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=model.onnx \
  --saveEngine=model.plan \
  --minShapes=signal:1x80x128 \
  --optShapes=signal:4x80x512 \
  --maxShapes=signal:8x80x2048
```

Parameters explained:
- `onnx` : Path to onnx model
- `saveEngine`: Path to TensorRT model
- `minShapes`, `optShapes`, `maxShapes`: Defines dynamic input shapes for optimal performance.

### 🟢 Step 3: Validate the TensorRT Model

Check model inference:

```bash
/usr/src/tensorrt/bin/trtexec --loadEngine=model.plan --shapes=signal:4x80x500
```

Ensure the model runs without errors.

## Next Steps
- [Deploy modules to Triton](Triton_deployment.md)
- [Benchmark performance](Testing_and_ModelAnalyzer.md)
- Integrate modules into your application pipeline.