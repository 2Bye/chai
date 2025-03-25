# 🚀 NeMo ASR Deployment (ONNX & TensorRT) on Triton

This guide provides comprehensive examples and benchmarks for deploying optimized NeMo ASR models using ONNX and TensorRT within Triton Inference Server. We will also compare performance using Triton's built-in profiler (model-analyzer).

## 📖 Navigation

- **[1. Convert NeMo ASR model to ONNX](NeMo_convert_to_ONNX.ipynb)**  
  Detailed notebook guiding through exporting the NeMo model to ONNX.

- **[2. Convert ONNX model to TensorRT](NeMo_convert_ONNX_to_TensorRT.md)**  
  Step-by-step instructions to optimize the ONNX model with TensorRT.

- **[3. Deploy on Triton Inference Server](Triton_deployment.md)**  
  Instructions and examples for structuring and deploying the models in Triton.

---

_All actions described were performed within the Docker container **nvcr.io/nvidia/pytorch:24.05-py3**_