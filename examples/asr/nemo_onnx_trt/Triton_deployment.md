# 🚀 Triton Deployment for NeMo ASR (ONNX & TensorRT)

This guide covers detailed steps to deploy NeMo ASR models using NVIDIA Triton Inference Server. It includes preprocessing, ONNX/TensorRT inference modules, postprocessing (CTC decoding), and creating an ensemble pipeline.

For detailed Triton documentation, visit the [official Triton repository](https://github.com/triton-inference-server/server).

---

## 🗂 Directory Structure

Organize your Triton model repository as follows:

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
└── ensemble_english_stt
    └── config.pbtxt
```

---

## 🟢 1. Preprocessing Module (Python Backend)

This module converts raw audio signals into Mel spectrograms using Triton's Python Backend, adapted for batch inference.

### Implementation (`model.py`)

```python
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor

class TritonPythonModel:
    def initialize(self, args):
        """Called once when the model is loaded."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.preprocessor = AudioToMelSpectrogramPreprocessor(features=80)
        self.preprocessor.to(self.device)

    def execute(self, requests):
        responses = []

        for request in requests:
            audio_batch = pb_utils.get_input_tensor_by_name(request, "audio_signal").as_numpy()

            audio_lengths = [len(audio) for audio in audio_batch]
            max_length = max(audio_lengths)

            padded_audio = [
                np.pad(audio, (0, max_length - len(audio)), mode='constant') for audio in audio_batch
            ]

            audio_signal = torch.from_numpy(np.array(padded_audio)).float().to(self.device)
            audio_signal_len = torch.tensor(audio_lengths, dtype=torch.float32).to(self.device)

            processed_signal, _ = self.preprocessor(
                input_signal=audio_signal, length=audio_signal_len
            )

            processed_signal_tensor = pb_utils.Tensor("processed_signal", processed_signal.cpu().numpy())

            inference_response = pb_utils.InferenceResponse(output_tensors=[processed_signal_tensor])
            responses.append(inference_response)

        return responses

    def finalize(self):
        pass
```

### Configuration (`config.pbtxt`)

```protobuf
ame: "preprocessing_english_stt"
backend: "python"

max_batch_size: 8

dynamic_batching {
  max_queue_delay_microseconds: 10000
}

instance_group [
  {
    kind: KIND_CPU
    count: 1
  }
]

input [
  {
    name: "audio_signal"
    data_type: TYPE_FP64
    dims: [-1]
  }
]

output [
  {
    name: "processed_signal"
    data_type: TYPE_FP32
    dims: [80, -1]
  },
  {
    name: "processed_signal_length"
    data_type: TYPE_FP32
    dims: [1]
  }
]
```

---

## 🟢 2. ASR Inference Module (ONNX & TensorRT)

This module performs acoustic model inference.

### ONNX Configuration (`config.pbtxt`)

```protobuf
name: "onnx_english_stt"
backend: "onnxruntime"

max_batch_size: 8

dynamic_batching {
  max_queue_delay_microseconds: 10000
}

instance_group [
  {
    kind: KIND_GPU
    gpus: [ 0 ]
    count: 2
  }
]

input [
    {
        name: "signal",
        data_type: TYPE_FP32,
        dims: [80, -1]
    }
]

output [
    {
        name: "output",
        data_type: TYPE_FP32,
        dims: [-1, 1025],
    }
]
```

### TensorRT Configuration (`config.pbtxt`)

```protobuf
name: "tensorrt_english_stt"
platform: "tensorrt_plan"

max_batch_size: 8

dynamic_batching {
  max_queue_delay_microseconds: 10000
}

instance_group [
  {
    kind: KIND_GPU
    gpus: [ 0 ]
    count: 2
  }
]

input [
    {
        name: "signal",
        data_type: TYPE_FP32,
        dims: [80, -1]
    }
]

output [
    {
        name: "output",
        data_type: TYPE_FP32,
        dims: [-1, 1025],
    }
]
```

---

## 🟢 3. Postprocessing Module (CTC Decoder)

Decodes ASR acoustic model logits into readable text.

### Implementation (`model.py`)

```python
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
import nemo.collections.asr as nemo_asr

class TritonPythonModel:
    def initialize(self, args):
        asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
            model_name="nvidia/stt_en_fastconformer_ctc_large",
            map_location='cpu'
        )
        asr_model.ctc_decoding.cfg.strategy = 'greedy_batch'
        self.ctc_decoder = asr_model.decoding.ctc_decoder_predictions_tensor
        del asr_model

    def execute(self, requests):
        responses = []

        for request in requests:
            asr_output = pb_utils.get_input_tensor_by_name(request, "asr_output").as_numpy()
            ctc_output_tensor = torch.from_numpy(asr_output)
            decoded_texts = self.ctc_decoder(ctc_output_tensor)

            decoded_texts_numpy = np.array(decoded_texts[0], dtype=np.object)

            output_tensor = pb_utils.Tensor("decoded_texts", decoded_texts_numpy)

            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses

    def finalize(self):
        pass
```

### Configuration (`config.pbtxt`)

```protobuf
name: "postprocessing_english_stt"
backend: "python"

max_batch_size: 8

dynamic_batching {
  max_queue_delay_microseconds: 10000
}

instance_group [
  {
    kind: KIND_CPU
    count: 1
  }
]

input [
  {
    name: "asr_output"
    data_type: TYPE_FP32
    dims: [-1, 1025]
  }
]

output [
  {
    name: "decoded_texts"
    data_type: TYPE_STRING
    dims: [1]
  }
]
```

---

## 🟢 4. Ensemble Pipelines

Ensemble models enable chaining multiple Triton models into a single inference request. Below are configurations for ONNX and TensorRT ensemble pipelines.

### ONNX Ensemble Configuration (`config.pbtxt`)

```protobuf
name: "ensemble_english_stt"
platform: "ensemble"
max_batch_size: 8

input [
  {
    name: "audio_signal"
    data_type: TYPE_FP64
    dims: [-1]
  }
]

output [
  {
    name: "decoded_texts"
    data_type: TYPE_STRING
    dims: [1]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "preprocessing_english_stt"
      model_version: -1
      input_map {
        key: "audio_signal"
        value: "audio_signal"
      }
      output_map {
        key: "processed_signal"
        value: "signal"
      }
    },
    {
      model_name: "onnx_english_stt"
      model_version: -1
      input_map {
        key: "signal"
        value: "signal"
      }
      output_map {
        key: "output"
        value: "asr_output"
      }
    },
    {
      model_name: "postprocessing_english_stt"
      model_version: -1
      input_map {
        key: "asr_output"
        value: "asr_output"
      }
      output_map {
        key: "decoded_texts"
        value: "decoded_texts"
      }
    }
  ]
}
```

### TensorRT Ensemble Configuration (`config.pbtxt`)

```protobuf
name: "ensemble_tensorrt_english_stt"
platform: "ensemble"
max_batch_size: 8

input [
  {
    name: "audio_signal"
    data_type: TYPE_FP64
    dims: [-1]
  }
]

output [
  {
    name: "decoded_texts"
    data_type: TYPE_STRING
    dims: [1]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "preprocessing_english_stt"
      model_version: -1
      input_map {
        key: "audio_signal"
        value: "audio_signal"
      }
      output_map {
        key: "processed_signal"
        value: "signal"
      }
    },
    {
      model_name: "tensorrt_english_stt"
      model_version: -1
      input_map {
        key: "signal"
        value: "signal"
      }
      output_map {
        key: "output"
        value: "asr_output"
      }
    },
    {
      model_name: "postprocessing_english_stt"
      model_version: -1
      input_map {
        key: "asr_output"
        value: "asr_output"
      }
      output_map {
        key: "decoded_texts"
        value: "decoded_texts"
      }
    }
  ]
}
```

This structured setup ensures optimized performance and maintainability for NeMo ASR models deployed on Triton Inference Server.