# ⚙️ Triton Model Analyzer: Performance Profiling Guide

This guide explains how to profile NeMo ASR models served with **Triton Inference Server** using **Triton Model Analyzer**. It includes setting up the profiling container, writing the configuration file, and interpreting key parameters for optimizing performance and GPU utilization.

For full documentation, refer to the official repositories:

* [Model Analyzer Config Guide](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/config.md)
* [Perf Analyzer Reference](https://github.com/triton-inference-server/perf_analyzer/blob/main/README.md)

---

## 🛠 Prerequisites

Ensure you have already deployed a Triton Inference Server container of version `25.02` or any matching version with your models. The **Model Analyzer** SDK container should match **exactly the same release version** as your Triton Server.

---

## 🧱 Recommended Directory Layout

```
project_root/
├── model_repository/             # your original model repo
├── output_models/                # profiler-generated optimized models
├── profile_results/              # metrics, logs, and reports
└── analyzer_config.yaml          # model analyzer configuration file
```

---

## 🚀 Launching the Model Analyzer Container

Use the SDK container for your Triton version. Mount the model repository and profiling output paths. Also mount the Docker socket if using `docker` launch mode.

```bash
docker run -it --gpus all \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /home/user/byebye:/home/user/byebye \
    --net=host \
    nvcr.io/nvidia/tritonserver:25.02-py3-sdk
```

Inside the container, you can now run profiling commands using:

```bash
model-analyzer --config-file /home/user/byebye/analyzer_config.yaml
```

---

## 📄 Sample Configuration (`analyzer_config.yaml`)

```yaml
model_repository: /home/user/byebye/model_repository

profile_models:
  - ensemble_english_stt_tensorrt

output_model_repository_path: output_models
export_path: profile_results
override_output_model_repository: true

triton_launch_mode: docker
triton_docker_image: triton_example
triton_docker_shm_size: 10G

perf_analyzer_flags:
  shape: audio_signal:160000
  measurement-request-count: 50
  measurement-interval: 500
  measurement-mode: time_windows

run_config_search_mode: quick
num_configs_per_model: 3

run_config_search_min_model_batch_size: 1
run_config_search_max_model_batch_size: 8
run_config_search_min_instance_count: 3
run_config_search_max_instance_count: 15
run_config_search_max_concurrency: 64

constraints:
  gpu_used_memory:
    max: 42000
```

---

## 🔍 Key Parameters Explained

### 🔹 `profile_models`

Defines which models to analyze. You can list multiple models:

```yaml
profile_models:
  - ensemble_english_stt_tensorrt
  - onnx_english_stt
```

### 🔹 `run_config_search_*`

Model Analyzer will generate different combinations of batch size and instance count:

* `run_config_search_min_instance_count`: minimum number of instances to test
* `run_config_search_max_instance_count`: maximum number
* `run_config_search_min_model_batch_size`: smallest batch size
* `run_config_search_max_model_batch_size`: largest batch size
* `num_configs_per_model`: how many top-performing configs to keep

### 🔹 `perf_analyzer_flags`

Settings for low-level request simulation:

* `shape`: mandatory — specifies input tensor shape (e.g., audio length in samples)
* `measurement-request-count`: how many requests per measurement
* `measurement-interval`: time window for each sample (ms)
* `measurement-mode`: `time_windows` is most robust

Other flags like `concurrency-range`, `output-shared-memory-size`, or `shared-memory` can be added as needed.

### 🔹 `triton_launch_mode`

* `docker`: each config runs in a separate Triton container
* `remote`: connect to an existing running server (use `triton_http_endpoint`)

### 🔹 `triton_docker_image`

Image used if launch mode is `docker`. Must match your model environment.

---

## 📈 Output

After a successful run of model-analyzer, PDF reports are automatically generated in the export_path directory (e.g., profile_results/), one per profiled model.

Each PDF contains (See example in [my PDF file](results.pdf)):

📊 Graphs showing latency, throughput, GPU memory usage, and instance utilization across different configurations
🔍 Tables with full performance metrics
✅ Highlighted best-performing configurations based on your constraints

---

## 📚 Additional Notes

* If your model input shape is dynamic, you **must** specify the `shape` manually.
* You can tune GPU constraints (e.g., `gpu_used_memory.max`) to fit available resources.
* It is highly recommended to set `override_output_model_repository: true` during experimentation.

---

## 🧭 Next Steps

* Deploy top models to production using your existing Triton container
* Monitor live metrics using `/metrics` endpoint and integrate with Prometheus
