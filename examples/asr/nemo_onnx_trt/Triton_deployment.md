# 🚀 Triton Deployment for NeMo ASR (ONNX & TensorRT)

This guide covers detailed steps to deploy NeMo ASR models using NVIDIA Triton Inference Server. It includes preprocessing, ONNX/TensorRT inference modules, postprocessing (CTC decoding), and creating an ensemble pipeline.

For detailed Triton documentation, visit the [official Triton repository](https://github.com/triton-inference-server/server).

---

## 🗂 Recommended Directory Structure

Organize your Triton model [repository](model_repository) as follows:

```
model_repository/
├── preprocessing_english_stt
│   ├── config.pbtxt
│   └── 1/
│       └── model.py
├── onnx_english_stt
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
├── tensorrt_english_stt
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan
├── postprocessing_english_stt
│   ├── config.pbtxt
│   └── 1/
│       └── model.py
├── ensemble_english_stt
│   └── config.pbtxt
└── ensemble_tensorrt_english_stt
    └── config.pbtxt
```

---

## 🟢 1. Preprocessing Module (Python Backend)

This module converts raw audio signals into Mel spectrograms using Triton's Python Backend, adapted for batch inference.

### Implementation (`model.py`)

*[model.py file](model_repository/preprocessing_english_stt/1/model.py)*

### Configuration (`config.pbtxt`)

*[config file](model_repository/preprocessing_english_stt/config.pbtxt)*

---

## 🟢 2. ASR Inference Module (ONNX & TensorRT)

This module performs acoustic model inference.

### ONNX Configuration (`config.pbtxt`)

*[config file](model_repository/onnx_english_stt/config.pbtxt)*

### TensorRT Configuration (`config.pbtxt`)

*[config file](model_repository/tensorrt_english_stt/config.pbtxt)*

---

## 🟢 3. Postprocessing Module (CTC Decoder)

Decodes ASR acoustic model logits into readable text.

### Implementation (`model.py`)

*[model.py file](model_repository/postprocessing_english_stt/1/model.py)*

### Configuration (`config.pbtxt`)

*[config file](model_repository/postprocessing_english_stt/config.pbtxt)*

---

## 🟢 4. Ensemble Pipelines

Ensemble models enable chaining multiple Triton models into a single inference request. Below are configurations for ONNX and TensorRT ensemble pipelines.

### ONNX Ensemble Configuration (`config.pbtxt`)

*[config file](model_repository/ensemble_english_stt/config.pbtxt)*

### TensorRT Ensemble Configuration (`config.pbtxt`)

*[config file](model_repository/ensemble_tensorrt_english_stt/config.pbtxt)*

---

## 🟢 5. Deploying Triton Inference Server

### Dockerfile
Create a Dockerfile with the following [content](Dockerfile):

```Dockerfile
FROM nvcr.io/nvidia/tritonserver:24.08-py3

RUN pip install git+https://github.com/NVIDIA/NeMo.git@v2.2.0rc3#egg=nemo_toolkit[asr]
RUN pip install tensorrt==10.0.1

COPY . /workspace

EXPOSE 8000
EXPOSE 8001
EXPOSE 8002

CMD ["tritonserver", "--model-repository=/workspace/model_repository"]
```

### Build and Run Docker Container

Build your Docker container:

```sh
docker build . -t triton_example
```

Run the container:

```sh
docker run --gpus all -d --name {container_name} -p 8000:8000 -p 8001:8001 -p 8002:8002 --shm-size=1g triton_example:latest
```

This structured setup ensures optimized performance and maintainability for NeMo ASR models deployed on Triton Inference Server.