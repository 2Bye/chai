import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor

class TritonPythonModel:
    def initialize(self, args):
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